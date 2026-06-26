"""Regression tests for diarization hardening."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from dv_backend.adapters.diarization.service import (
    PyannoteCommunity1Backend,
    resolve_pyannote_local_model,
)
from dv_backend.pyannote_vendor import PYANNOTE_REQUIRED_WEIGHTS


def _stub_pyannote_model_dir(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.yaml").write_text("x: 1", encoding="utf-8")
    for rel in PYANNOTE_REQUIRED_WEIGHTS:
        path = model_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stub")


from dv_backend.checkpoint_compat import (
    ASR_ALIGNMENT_SCHEMA_VERSION,
    asr_checkpoint_fingerprint,
    diarize_checkpoint_is_stale,
    diarization_settings_fingerprint,
    validate_asr_for_diarization,
)
from dv_backend.checkpoints import PIPELINE_STEPS, checkpoint_path, load_checkpoint, save_checkpoint
from dv_backend.config import AppConfig
from dv_backend.database import Database
from dv_backend.diarization_models import AttributedSegment, DiarizationOptions, SpeakerAssignmentConfig, SpeakerProfile
from dv_backend.diarization_second_pass import derive_second_pass_windows
from dv_backend.errors import AppError
from dv_backend.gpu_lease import gpu_lease, gpu_lease_holder, reset_gpu_lease_for_tests
from dv_backend.jobs import JobService
from dv_backend.speaker_review_service import (
    complete_speaker_review,
    merge_speakers,
    update_voice_mapping,
)
from dv_backend.speaker_samples import generate_speaker_sample_files, select_sample_candidates
from dv_backend.api import create_app


def _segment(**kwargs) -> AttributedSegment:
    defaults = {
        "index": 0,
        "start": 0.0,
        "end": 3.0,
        "text": "你好",
        "speaker_id": "SPK_01",
        "speaker_confidence": 0.9,
        "speaker_coverage": 0.9,
        "speaker_margin": 0.7,
        "overlap_ratio": 0.0,
        "flags": [],
        "unit_count": 1,
    }
    defaults.update(kwargs)
    return AttributedSegment(**defaults)


def _insert_legacy_job(database: Database, job_id: str, *, normalize_completed: bool) -> None:
    now = "2025-01-01T00:00:00+00:00"
    legacy_steps = ("resolve", "download", "extract_audio", "vad", "asr", "normalize_segments")
    with database.connection:
        database.connection.execute(
            "INSERT INTO jobs (id, source_url, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (
                job_id,
                f"https://www.douyin.com/video/{job_id}",
                "completed" if normalize_completed else "running",
                now,
                now,
            ),
        )
        for position, name in enumerate(legacy_steps):
            if name == "normalize_segments":
                status = "completed" if normalize_completed else "pending"
            else:
                status = "completed"
            database.connection.execute(
                "INSERT INTO job_steps (job_id, name, position, status, checkpoint_path) VALUES (?, ?, ?, ?, ?)",
                (job_id, name, position, status, ""),
            )


def _create_completed_job(service: JobService, job_id: str = "job-rerun") -> None:
    config = AppConfig(service.data_dir)
    config.ensure_directories()
    now = "2026-01-01T00:00:00+00:00"
    with service.database.connection:
        service.database.connection.execute(
            "INSERT INTO jobs (id, source_url, status, created_at, updated_at) VALUES (?, ?, 'completed', ?, ?)",
            (job_id, "https://www.douyin.com/video/123", now, now),
        )
        service.database.connection.executemany(
            "INSERT INTO job_steps (job_id, name, position, status, checkpoint_path) VALUES (?, ?, ?, 'completed', ?)",
            [
                (
                    job_id,
                    name,
                    position,
                    str(checkpoint_path(service.data_dir, job_id, name)),
                )
                for position, name in enumerate(PIPELINE_STEPS)
            ],
        )
    for step_name in PIPELINE_STEPS:
        save_checkpoint(service.data_dir, job_id, step_name, {"step_name": step_name, "job_id": job_id})


@pytest.fixture
def job_service(tmp_path: Path) -> JobService:
    database = Database(tmp_path / "app.db")
    database.migrate()
    return JobService(database, tmp_path)


def test_select_sample_candidates_bounds_and_quality() -> None:
    segments = [
        _segment(start=0.0, end=1.0, speaker_confidence=0.95),
        _segment(start=2.0, end=6.0, speaker_confidence=0.92),
        _segment(start=8.0, end=12.0, speaker_confidence=0.91, flags=["overlap_speech"]),
    ]
    candidates = select_sample_candidates(
        segments,
        speaker_id="SPK_01",
        review_confidence_threshold=0.65,
    )
    assert len(candidates) == 1
    assert 2.0 <= candidates[0]["duration_sec"] <= 8.0


def test_generate_speaker_sample_files_serializes_paths(tmp_path: Path) -> None:
    job_id = "job-sample"
    job_dir = tmp_path / "jobs" / job_id
    audio = job_dir / "artifacts" / "audio_16k.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"wav")
    ffmpeg = tmp_path / "ffmpeg.exe"
    ffmpeg.write_text("")

    profiles = [
        SpeakerProfile(
            speaker_id="SPK_01",
            total_speech_sec=10.0,
            confidence=0.9,
            tts_voice="Xuân Vĩnh",
        )
    ]
    segments = [_segment(start=1.0, end=5.0)]

    def fake_ffmpeg(cmd: list[str]) -> None:
        output = Path(cmd[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"RIFF")

    updated = generate_speaker_sample_files(
        job_dir=job_dir,
        audio_path=audio,
        ffmpeg_path=ffmpeg,
        profiles=profiles,
        segments=segments,
        review_confidence_threshold=0.65,
        run_ffmpeg=fake_ffmpeg,
    )
    assert updated[0].representative_samples
    sample = updated[0].representative_samples[0]
    assert sample["artifact_path"].startswith("artifacts/diarization/speaker_samples/")
    assert sample["playback_url"].endswith(".wav")
    assert (job_dir / sample["artifact_path"]).is_file()


def test_validate_asr_rejects_missing_alignment_when_diarization_enabled() -> None:
    with pytest.raises(AppError) as exc:
        validate_asr_for_diarization(
            {"schema_version": 1, "segments": [{"text": "你好", "start": 0, "end": 1}]},
            speaker_diarization_enabled=True,
        )
    assert exc.value.info.code == "INCOMPATIBLE_ASR_ALIGNMENT"


def test_diarize_checkpoint_stale_on_settings_change() -> None:
    asr_cp = {
        "schema_version": ASR_ALIGNMENT_SCHEMA_VERSION,
        "segments": [{"text": "你好", "start": 0, "end": 1}],
        "aligned_units": [{"text": "你", "start": 0, "end": 0.5}],
    }
    settings_a = {"speaker_diarization": True, "diarization_backend": "pyannote_community_1"}
    settings_b = {**settings_a, "diarization_backend": "funasr_campp"}
    diarize_cp = {
        "asr_fingerprint": asr_checkpoint_fingerprint(asr_cp),
        "settings_fingerprint": diarization_settings_fingerprint(settings_a),
    }
    assert not diarize_checkpoint_is_stale(diarize_cp, asr_cp, settings_a)
    assert diarize_checkpoint_is_stale(diarize_cp, asr_cp, settings_b)


def test_migration_legacy_job_without_normalize_completed(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    database.migrate()
    _insert_legacy_job(database, "legacy-1", normalize_completed=False)
    database.migrate()
    row = database.connection.execute(
        "SELECT status FROM job_steps WHERE job_id = ? AND name = 'diarize'",
        ("legacy-1",),
    ).fetchone()
    assert row["status"] == "pending"


def test_migration_legacy_job_with_normalize_completed(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    database.migrate()
    _insert_legacy_job(database, "legacy-2", normalize_completed=True)
    database.migrate()
    row = database.connection.execute(
        "SELECT status FROM job_steps WHERE job_id = ? AND name = 'diarize'",
        ("legacy-2",),
    ).fetchone()
    assert row["status"] == "completed"


def test_rerun_from_asr_invalidates_diarize(job_service: JobService) -> None:
    job_id = "job-rerun-asr"
    _create_completed_job(job_service, job_id)
    keep_steps = list(PIPELINE_STEPS[: PIPELINE_STEPS.index("asr") + 1])
    job = job_service.rerun(job_id, keep_steps)
    diarize = next(item for item in job.steps if item.name == "diarize")
    assert diarize.status == "pending"
    assert not checkpoint_path(job_service.data_dir, job_id, "diarize").is_file()


def test_rerun_from_diarize_keeps_asr(job_service: JobService) -> None:
    job_id = "job-rerun-diarize"
    _create_completed_job(job_service, job_id)
    keep_steps = list(PIPELINE_STEPS[: PIPELINE_STEPS.index("diarize")])
    job = job_service.rerun(job_id, keep_steps)
    asr = next(item for item in job.steps if item.name == "asr")
    diarize = next(item for item in job.steps if item.name == "diarize")
    normalize = next(item for item in job.steps if item.name == "normalize_segments")
    assert asr.status == "completed"
    assert diarize.status == "pending"
    assert normalize.status == "pending"


def test_voice_mapping_invalidation_does_not_include_diarize() -> None:
    diarize_cp = {
        "speaker_profiles": [{"speaker_id": "SPK_01", "tts_voice": "A"}],
        "speaker_manual_overrides": {},
    }
    result = update_voice_mapping(diarize_cp, "SPK_01", "Xuân Vĩnh")
    assert "diarize" not in result["invalidated_steps"]
    assert result["resume_from"] == "tts"


def test_merge_speakers_invalidates_from_normalize() -> None:
    diarize_cp = {
        "speaker_profiles": [
            {"speaker_id": "SPK_01"},
            {"speaker_id": "SPK_02"},
        ],
        "segments": [{"diarization_speaker_id": "SPK_01"}],
        "speaker_manual_overrides": {},
    }
    result = merge_speakers(diarize_cp, "SPK_01", "SPK_02")
    assert result["resume_from"] == "normalize_segments"
    assert "translate" in result["invalidated_steps"]


def test_complete_speaker_review_is_idempotent(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    database.migrate()
    job_id = "review-job"
    job_dir = tmp_path / "jobs" / job_id / "artifacts" / "diarization"
    job_dir.mkdir(parents=True)
    (job_dir / "diagnostics.json").write_text("{}", encoding="utf-8")
    save_checkpoint(
        tmp_path,
        job_id,
        "diarize",
        {"manual_review_completed": True, "review_required": False, "speaker_profiles": []},
    )
    runner = MagicMock()
    result = complete_speaker_review(
        data_dir=tmp_path,
        job_id=job_id,
        job_status="completed",
        database=database,
        runner=runner,
    )
    assert result["status"] == "already_completed"
    runner.start_job.assert_not_called()


def test_gpu_lease_released_when_cancelled_mid_diarize() -> None:
    reset_gpu_lease_for_tests()

    class Cancelled(Exception):
        pass

    with pytest.raises(Cancelled):
        with gpu_lease("job-x:diarize"):
            raise Cancelled("cancelled")
    assert gpu_lease_holder() is None
    reset_gpu_lease_for_tests()


def test_pyannote_local_model_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(AppError) as exc:
        resolve_pyannote_local_model(str(tmp_path / "pyannote"))
    assert exc.value.info.code == "PYANNOTE_MODEL_NOT_BOOTSTRAPPED"


def test_pyannote_local_model_available(tmp_path: Path) -> None:
    model_dir = tmp_path / "pyannote" / "speaker-diarization-community-1"
    _stub_pyannote_model_dir(model_dir)
    resolved = resolve_pyannote_local_model(str(tmp_path / "pyannote"))
    assert resolved == model_dir.resolve()
    assert resolved.is_absolute()


def test_second_pass_windows_merge_nearby_segments() -> None:
    segments = [
        _segment(start=1.0, end=2.0, speaker_confidence=0.2, flags=["low_confidence"]),
        _segment(start=2.1, end=3.0, speaker_confidence=0.25, flags=["boundary_ambiguous"]),
    ]
    windows = derive_second_pass_windows(segments, SpeakerAssignmentConfig())
    assert len(windows) == 1


def _mock_pyannote_pipeline(*, load_error: Exception | None = None) -> MagicMock:
    mock_output = MagicMock()
    mock_output.itertracks.return_value = iter([])
    mock_instance = MagicMock(return_value=mock_output)
    mock_pipeline_cls = MagicMock()
    if load_error is not None:
        mock_pipeline_cls.from_pretrained.side_effect = load_error
    else:
        mock_pipeline_cls.from_pretrained.return_value = mock_instance
    mock_audio_module = MagicMock()
    mock_audio_module.Pipeline = mock_pipeline_cls
    mock_audio_module.__version__ = "3.3.0"
    return mock_audio_module


def test_pyannote_loads_local_model_without_remote(tmp_path: Path, monkeypatch) -> None:
    mock_audio_module = _mock_pyannote_pipeline()
    monkeypatch.setitem(sys.modules, "pyannote.audio", mock_audio_module)
    monkeypatch.setitem(sys.modules, "pyannote", MagicMock(audio=mock_audio_module))
    monkeypatch.setattr(
        "dv_backend.adapters.diarization.service._load_audio_waveform_dict",
        lambda _path: {"waveform": MagicMock(), "sample_rate": 16000},
    )

    model_dir = tmp_path / "pyannote" / "speaker-diarization-community-1"
    _stub_pyannote_model_dir(model_dir)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")

    backend = PyannoteCommunity1Backend()
    options = DiarizationOptions(model_cache_dir=str(tmp_path / "pyannote"), device="cpu")
    result = backend.diarize(audio, options)
    mock_audio_module.Pipeline.from_pretrained.assert_called_once()
    assert str(model_dir) in mock_audio_module.Pipeline.from_pretrained.call_args.args[0]
    assert result.regular.metadata.get("offline_local_load") is True


def test_pyannote_local_model_load_failure(tmp_path: Path, monkeypatch) -> None:
    mock_audio_module = _mock_pyannote_pipeline(load_error=RuntimeError("corrupt checkpoint"))
    monkeypatch.setitem(sys.modules, "pyannote.audio", mock_audio_module)
    monkeypatch.setitem(sys.modules, "pyannote", MagicMock(audio=mock_audio_module))

    model_dir = tmp_path / "pyannote" / "speaker-diarization-community-1"
    _stub_pyannote_model_dir(model_dir)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")

    backend = PyannoteCommunity1Backend()
    options = DiarizationOptions(model_cache_dir=str(tmp_path / "pyannote"), device="cpu")
    with pytest.raises(AppError) as exc:
        backend.diarize(audio, options)
    assert exc.value.info.code == "PYANNOTE_LOAD_FAILED"


def test_api_speaker_sample_route(tmp_path: Path) -> None:
    client = TestClient(create_app(AppConfig(tmp_path)))
    create = client.post("/api/jobs", json={"source_url": "https://www.douyin.com/video/sample"})
    job_id = create.json()["id"]
    sample_dir = tmp_path / "jobs" / job_id / "artifacts" / "diarization" / "speaker_samples"
    sample_dir.mkdir(parents=True)
    sample_file = sample_dir / "SPK_01_01.wav"
    sample_file.write_bytes(b"RIFF")
    response = client.get(f"/api/jobs/{job_id}/diarization/samples/SPK_01_01.wav")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")


def test_enable_diarization_requires_aligned_units() -> None:
    asr_cp = {
        "schema_version": ASR_ALIGNMENT_SCHEMA_VERSION,
        "segments": [{"text": "你好", "start": 0, "end": 1}],
        "aligned_units": [],
    }
    with pytest.raises(AppError) as exc:
        validate_asr_for_diarization(asr_cp, speaker_diarization_enabled=True)
    assert exc.value.info.code == "INCOMPATIBLE_ASR_ALIGNMENT"


def test_diarize_stale_when_asr_fingerprint_changes() -> None:
    asr_v1 = {
        "schema_version": ASR_ALIGNMENT_SCHEMA_VERSION,
        "segments": [{"text": "你好", "start": 0, "end": 1}],
        "aligned_units": [{"text": "你", "start": 0, "end": 0.5}],
    }
    asr_v2 = {
        **asr_v1,
        "aligned_units": [{"text": "好", "start": 0.5, "end": 1.0}],
    }
    settings = {"speaker_diarization": True, "diarization_backend": "pyannote_community_1"}
    diarize_cp = {
        "asr_fingerprint": asr_checkpoint_fingerprint(asr_v1),
        "settings_fingerprint": diarization_settings_fingerprint(settings),
    }
    assert not diarize_checkpoint_is_stale(diarize_cp, asr_v1, settings)
    assert diarize_checkpoint_is_stale(diarize_cp, asr_v2, settings)
