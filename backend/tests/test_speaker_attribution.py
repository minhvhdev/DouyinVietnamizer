"""Unit tests for speaker attribution and diarization helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dv_backend.checkpoints import PIPELINE_STEPS, checkpoint_path, save_checkpoint
from dv_backend.database import Database
from dv_backend.diarization_models import (
    DiarizationTimeline,
    DiarizationTurn,
    SpeakerAssignmentConfig,
)
from dv_backend.gpu_lease import gpu_lease, reset_gpu_lease_for_tests
from dv_backend.jobs import JobService
from dv_backend.settings import SettingsService
from dv_backend.speaker_attribution import (
    attribute_speakers,
    compare_timelines,
    compute_overlap_stats,
    coverage_margin,
    detect_overlap_regions,
    map_speakers_by_overlap,
    merge_attributed_units,
    speaker_coverage_for_unit,
    temporal_overlap,
)
from dv_backend.speaker_profiles import build_speaker_profiles, remap_segments_to_voice_slots
from dv_backend.speaker_review import should_require_speaker_review
from dv_backend.diarization_models import AttributedUnit, DiarizationDiagnostics


def _timeline(turns: list[tuple[str, float, float]]) -> DiarizationTimeline:
    return DiarizationTimeline(
        backend="test",
        model="test",
        device="cpu",
        turns=[
            DiarizationTurn(speaker_id=speaker, start=start, end=end)
            for speaker, start, end in turns
        ],
    )


def test_temporal_overlap_and_coverage() -> None:
    assert temporal_overlap(0.0, 1.0, 0.5, 1.5) == pytest.approx(0.5)
    coverage = speaker_coverage_for_unit(
        0.0,
        1.0,
        [DiarizationTurn(speaker_id="SPK_01", start=0.0, end=0.6)],
    )
    assert coverage["SPK_01"] == pytest.approx(0.6)


def test_exclusive_speaker_selection() -> None:
    regular = _timeline([("SPK_01", 0.0, 1.0), ("SPK_02", 1.0, 2.0)])
    exclusive = _timeline([("SPK_01", 0.0, 1.0), ("SPK_02", 1.0, 2.0)])
    stats = compute_overlap_stats(0.2, 0.8, regular.turns, exclusive.turns)
    assert stats.exclusive_speaker == "SPK_01"


def test_overlap_detection_and_flags() -> None:
    regular = _timeline([("SPK_01", 0.0, 1.2), ("SPK_02", 0.8, 2.0)])
    exclusive = _timeline([("SPK_01", 0.0, 1.0), ("SPK_02", 1.0, 2.0)])
    config = SpeakerAssignmentConfig(overlap_flag_threshold=0.2)
    attributed = attribute_speakers(
        [{"text": "你", "start": 0.9, "end": 1.1}],
        regular,
        exclusive,
        config,
    )
    assert attributed.units[0].flags
    assert "overlap_speech" in attributed.units[0].flags or attributed.units[0].overlap_ratio > 0


def test_coverage_margin_calculation() -> None:
    cov, margin, speaker = coverage_margin({"SPK_01": 0.8, "SPK_02": 0.1})
    assert speaker == "SPK_01"
    assert cov == pytest.approx(0.8)
    assert margin == pytest.approx(0.7)


def test_merge_attributed_units_respects_speaker_and_gap() -> None:
    units = [
        AttributedUnit(text="你", start=0.0, end=0.2, speaker_id="SPK_01", speaker_confidence=0.9),
        AttributedUnit(text="好", start=0.25, end=0.5, speaker_id="SPK_01", speaker_confidence=0.9),
        AttributedUnit(text="吗", start=1.0, end=1.2, speaker_id="SPK_02", speaker_confidence=0.8),
    ]
    segments = merge_attributed_units(units, SpeakerAssignmentConfig())
    assert len(segments) == 2
    assert segments[0].text == "你好"
    assert segments[1].speaker_id == "SPK_02"


def test_map_speakers_by_overlap() -> None:
    primary = _timeline([("SPK_01", 0.0, 2.0), ("SPK_02", 2.0, 4.0)])
    secondary = _timeline([("SPK_A", 0.0, 2.0), ("SPK_B", 2.0, 4.0)])
    mapping = map_speakers_by_overlap(primary, secondary)
    assert mapping["SPK_01"] == "SPK_A"
    assert mapping["SPK_02"] == "SPK_B"


def test_compare_timelines_agreement() -> None:
    primary = _timeline([("SPK_01", 0.0, 2.0)])
    secondary = _timeline([("SPK_01", 0.0, 2.0)])
    ratio, disagree = compare_timelines(primary, secondary)
    assert ratio > 0.5
    assert disagree >= 0.0


def test_speaker_profile_aggregation_and_slot_remap() -> None:
    from dv_backend.diarization_models import AttributedSegment

    segments = [
        AttributedSegment(
            index=0,
            start=0.0,
            end=2.0,
            text="你好",
            speaker_id="SPK_01",
            speaker_confidence=0.9,
        ),
        AttributedSegment(
            index=1,
            start=2.0,
            end=3.0,
            text="嗯",
            speaker_id="SPK_02",
            speaker_confidence=0.4,
            flags=["low_confidence"],
        ),
    ]
    profiles = build_speaker_profiles(
        segments,
        [],
        SpeakerAssignmentConfig(profile_min_seconds=3.0),
    )
    short_profile = next(p for p in profiles if p.speaker_id == "SPK_02")
    assert short_profile.below_profile_threshold
    remapped = remap_segments_to_voice_slots(segments, profiles)
    assert remapped[0]["speaker_id"] in {"0", "1"}


def test_review_trigger_logic() -> None:
    from dv_backend.diarization_models import AttributedSegment

    segments = [
        AttributedSegment(
            index=0,
            start=0.0,
            end=5.0,
            text="长句",
            speaker_id="SPK_01",
            speaker_confidence=0.2,
            flags=["low_confidence"],
        )
    ]
    diagnostics = DiarizationDiagnostics(
        backend_used="pyannote_community_1",
        overlap_ratio=0.3,
        speaker_count=1,
    )
    required, reasons = should_require_speaker_review(
        segments,
        diagnostics,
        SpeakerAssignmentConfig(),
        min_speakers=1,
        max_speakers=6,
    )
    assert required
    assert reasons


def test_gpu_lease_released_on_exception() -> None:
    reset_gpu_lease_for_tests()
    with pytest.raises(RuntimeError):
        with gpu_lease("test-owner"):
            raise RuntimeError("boom")
    reset_gpu_lease_for_tests()
    with gpu_lease("test-owner-2"):
        pass
    reset_gpu_lease_for_tests()


def test_database_migration_adds_diarize_step(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    database.migrate()
    job_service = JobService(database, tmp_path)
    job = job_service.create("https://www.douyin.com/video/123")
    step_names = [step.name for step in job.steps]
    assert "diarize" in step_names
    assert step_names.index("diarize") == PIPELINE_STEPS.index("diarize")


def test_rerun_invalidates_downstream_after_diarize(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    database.migrate()
    job_service = JobService(database, tmp_path)
    job = job_service.create("https://www.douyin.com/video/123")
    job_id = job.id
    for step_name in PIPELINE_STEPS[: PIPELINE_STEPS.index("diarize") + 1]:
        save_checkpoint(tmp_path, job_id, step_name, {"step_name": step_name, "ok": True})
        with database.connection:
            database.connection.execute(
                "UPDATE job_steps SET status = 'completed' WHERE job_id = ? AND name = ?",
                (job_id, step_name),
            )
    with database.connection:
        database.connection.execute(
            "UPDATE jobs SET status = 'completed' WHERE id = ?",
            (job_id,),
        )
    job_service.rerun(job_id, list(PIPELINE_STEPS[: PIPELINE_STEPS.index("diarize")]))
    refreshed = job_service.get(job_id)
    diarize = next(step for step in refreshed.steps if step.name == "diarize")
    normalize = next(step for step in refreshed.steps if step.name == "normalize_segments")
    assert diarize.status == "pending"
    assert normalize.status == "pending"
    assert not checkpoint_path(tmp_path, job_id, "diarize").is_file()


def test_settings_defaults_include_diarization_keys(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    database.migrate()
    settings = SettingsService(database).get_all()
    assert settings["diarization_backend"] == "pyannote_community_1"
    assert settings["speaker_assignment_min_coverage"] == 0.75


def test_backward_compatible_old_asr_checkpoint_without_diarize(tmp_path: Path) -> None:
    from dv_backend.checkpoints import load_checkpoint
    from dv_backend.pipeline_diarize import _legacy_checkpoint_from_asr

    database = Database(tmp_path / "app.db")
    database.migrate()
    job_service = JobService(database, tmp_path)
    job = job_service.create("https://www.douyin.com/video/456")
    save_checkpoint(
        tmp_path,
        job.id,
        "asr",
        {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "你好", "speaker_id": "0", "speaker_confidence": 0.8}
            ]
        },
    )
    settings = SettingsService(database)
    settings.update({"speaker_diarization": True})
    asr_cp = load_checkpoint(tmp_path, job.id, "asr")
    legacy = _legacy_checkpoint_from_asr(asr_cp, settings.get_raw_all(), job_id=job.id)
    assert legacy["segments"]
    assert legacy["legacy"] is True
