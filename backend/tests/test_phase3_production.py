"""Phase 3 production validation tests."""

from __future__ import annotations

import json
import wave
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dv_backend.checkpoints import PIPELINE_STEPS, load_checkpoint, save_checkpoint
from dv_backend.config import AppConfig
from dv_backend.database import Database
from dv_backend.dubbing_quality_score import score_segment_quality
from dv_backend.experiment_comparability import validate_experiment_comparability
from dv_backend.jobs import JobService
from dv_backend.release_quality_gate import evaluate_release_gate
from dv_backend.timing_experiment import clone_job_prefix, validate_fixed_settings_match
from dv_backend.timing_qc_metrics import compute_timing_qc_metrics
from dv_backend.utterance_policy import classify_utterance_length, should_skip_rewrite_for_short_utterance
from dv_backend.voice_duration_profile import update_voice_profile_from_sample
from dv_backend.voice_profile_policy import blend_profiles, effective_voice_profile


@pytest.fixture
def env(tmp_path: Path):
    database = Database(tmp_path / "app.db")
    database.migrate()
    config = AppConfig(tmp_path)
    config.ensure_directories()
    jobs = JobService(database, tmp_path)
    return config, database, jobs


def _seed_source_job(jobs: JobService, job_id: str | None = None) -> str:
    video = jobs.data_dir / "sample.mp4"
    video.write_bytes(b"video-bytes-for-test")
    job = jobs.create_imported(video, original_filename="sample.mp4")
    job_dir = jobs.data_dir / "jobs" / job.id
    artifacts = job_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "original.mp4").write_bytes(b"video-bytes-for-test")
    (artifacts / "vocals_16k.wav").write_bytes(b"RIFF" + b"\x00" * 64)

    for step in PIPELINE_STEPS[: PIPELINE_STEPS.index("translate")]:
        save_checkpoint(
            jobs.data_dir,
            job.id,
            step,
            {"job_id": job.id, "step_name": step, "segments": [{"index": 0, "text": "hello", "start": 0, "end": 2}] if step == "normalize_segments" else {}},
        )
        with jobs.database.connection:
            jobs.database.connection.execute(
                "UPDATE job_steps SET status = 'completed' WHERE job_id = ? AND name = ?",
                (job.id, step),
            )
    norm = load_checkpoint(jobs.data_dir, job.id, "normalize_segments")
    norm["segments"] = [{"index": 0, "text": "源文本", "start": 0.0, "end": 2.0, "original_duration": 2.0}]
    save_checkpoint(jobs.data_dir, job.id, "normalize_segments", norm)
    return job.id


def test_clone_job_does_not_touch_source(env) -> None:
    config, database, jobs = env
    source_id = _seed_source_job(jobs)
    source_artifacts = list((config.data_dir / "jobs" / source_id / "artifacts").iterdir())
    clone_id = clone_job_prefix(jobs, source_id, label="baseline")
    assert clone_id != source_id
    assert (config.data_dir / "jobs" / clone_id / "artifacts" / "original.mp4").is_file()
    assert load_checkpoint(config.data_dir, clone_id, "normalize_segments") is not None
    assert load_checkpoint(config.data_dir, clone_id, "translate") is None
    assert len(list((config.data_dir / "jobs" / source_id / "artifacts").iterdir())) >= len(source_artifacts)


def test_comparison_invalid_when_voice_differs(env, tmp_path: Path) -> None:
    config, _, _ = env
    result = validate_experiment_comparability(
        config.data_dir,
        "job-a",
        "job-b",
        baseline_settings={"tts_backend": "omnivoice", "omnivoice_model": "m1"},
        experiment_settings={"tts_backend": "edge", "edge_tts_voice": "vi-VN-HoaiMyNeural"},
    )
    assert result["comparison_valid"] is False
    assert "voice_identity" in result["differences"] or any("fixed_setting" in d for d in result["differences"])


def test_quality_score_speech_trim_is_danger() -> None:
    result = score_segment_quality({"translation": "x", "speech_trimmed": True})
    assert result["quality_severity"] == "danger"


def test_release_gate_blocks_speech_trim() -> None:
    segments = [{"translation": "a", "tts_duration": 1.0, "speech_trimmed": True}]
    gate = evaluate_release_gate(segments, metrics=compute_timing_qc_metrics(segments))
    assert gate["passed"] is False
    assert "speech_trim_count" in gate["blocking"]


def test_short_utterance_skips_rewrite() -> None:
    segment = {"timing_profile": {"speech_target_duration": 0.6}, "translation": "Ừ."}
    assert classify_utterance_length(segment) == "short"
    assert should_skip_rewrite_for_short_utterance(segment, "slightly_short") is True


def test_voice_profile_rejects_repaired(env) -> None:
    config, _, _ = env
    settings = {"voice_duration_profile_enabled": True, "tts_backend": "omnivoice", "omnivoice_model": "m"}
    update_voice_profile_from_sample(
        settings,
        text="Xin chào các bạn",
        speech_duration_sec=1.2,
        data_dir=config.data_dir,
        from_repaired_audio=True,
    )
    profile = effective_voice_profile(settings, data_dir=config.data_dir)
    assert profile.get("profile_source", "").startswith("default")


def test_blend_profiles_deterministic() -> None:
    default = {"syllables_per_second": 4.0}
    learned = {"syllables_per_second": 5.0, "samples": 10}
    a = blend_profiles(default, learned, weight=0.5)
    b = blend_profiles(default, learned, weight=0.5)
    assert a == b


def test_import_review_validation(tmp_path: Path) -> None:
    from scripts.import_timing_review import aggregate_human_metrics, validate_review

    payload = {
        "experiment_id": "exp1",
        "segments": {"0": {"naturalness": 4, "timing": 5, "preferred": "experiment"}},
    }
    assert not validate_review(payload)
    agg = aggregate_human_metrics([payload])
    assert agg["mean_naturalness"] == 4.0


def test_validate_fixed_settings_match() -> None:
    assert not validate_fixed_settings_match(
        {"tts_backend": "omnivoice"},
        {"tts_backend": "omnivoice"},
        {"tts_backend": "omnivoice"},
    )
