from __future__ import annotations

import json
from pathlib import Path

from dv_backend.duration_diagnostics import (
    annotate_segments_duration_diagnostics,
    normalize_repair_action,
    safe_duration,
    segment_duration_diagnostics,
)
from dv_backend.subtitle_timing import (
    hash_subtitle_track_body,
    load_canonical_subtitle_track,
    write_canonical_subtitle_track,
)


def test_safe_duration_rejects_nan_inf_negative() -> None:
    assert safe_duration(float("nan")) is None
    assert safe_duration(float("inf")) is None
    assert safe_duration(-1.0) is None
    assert safe_duration(1.25) == 1.25


def test_invalid_budget_ratio_is_null_not_inf() -> None:
    diag = segment_duration_diagnostics(
        {
            "duration_budget": 0,
            "tts_duration": 2.0,
            "repaired_duration": 2.0,
        }
    )
    assert diag["tts_vs_budget_ratio"] is None
    assert diag["repaired_vs_budget_ratio"] is None
    assert diag["duration_miss"] is None


def test_duration_miss_without_placement_shift() -> None:
    diag = segment_duration_diagnostics(
        {
            "start": 1.0,
            "placement_start": 1.0,
            "placement_drift_sec": 0.0,
            "duration_budget": 1.0,
            "tts_duration": 1.8,
            "repaired_duration": 1.8,
            "repaired_method": "none",
        }
    )
    assert diag["duration_miss"] is True
    assert diag["placement_shifted"] is False
    assert "duration_miss" in diag["issues"]
    assert "placement_shift" not in diag["issues"]


def test_placement_shift_without_duration_miss() -> None:
    diag = segment_duration_diagnostics(
        {
            "start": 1.0,
            "placement_start": 2.2,
            "placement_drift_sec": 1.2,
            "duration_budget": 1.0,
            "tts_duration": 1.0,
            "repaired_duration": 1.0,
            "timing_status": "SHIFTED",
        }
    )
    assert diag["duration_miss"] is False
    assert diag["placement_shifted"] is True
    assert diag["placement_shift_sec"] == 1.2
    assert diag["placement_shift_cause"] == "soft_schedule_or_source_conflict"
    assert "placement_shift" in diag["issues"]
    assert "duration_miss" not in diag["issues"]


def test_conflict_propagation_marks_followers() -> None:
    previous = {
        "index": 0,
        "timing_overflow_sec": 0.4,
        "timing_needs_compact": True,
        "repaired_duration": 3.0,
        "duration_budget": 2.0,
    }
    follower = {
        "index": 1,
        "start": 2.0,
        "placement_start": 3.1,
        "placement_drift_sec": 1.1,
        "timing_status": "SHIFTED",
        "duration_budget": 1.0,
        "repaired_duration": 1.0,
        "tts_duration": 1.0,
    }
    diag = segment_duration_diagnostics(follower, previous=previous)
    assert diag["placement_shift_cause"] == "previous_repaired_overflow"
    assert diag["duration_miss"] is False


def test_normalize_repair_action() -> None:
    assert normalize_repair_action({"repaired_method": "time_stretch"}) == "tempo"
    assert normalize_repair_action({"repaired_method": "llm_shorten"}) == "rewrite"
    assert normalize_repair_action({"repaired_method": "none"}) == "none"


def test_annotate_does_not_drop_source_fields() -> None:
    segments = [
        {
            "index": 0,
            "start": 0.0,
            "end": 1.0,
            "duration_budget": 1.0,
            "tts_duration": 1.0,
            "repaired_duration": 1.0,
            "placement_start": 0.0,
            "placement_drift_sec": 0.0,
        }
    ]
    before = json.dumps(segments[0], sort_keys=True)
    annotate_segments_duration_diagnostics(segments)
    after_core = {k: v for k, v in segments[0].items() if k != "duration_diagnostics"}
    assert json.dumps(after_core, sort_keys=True) == before
    assert "duration_diagnostics" in segments[0]


def test_subtitle_track_error_marks_qc_failed_status() -> None:
    """subtitle_track_error must surface as explicit failed status, not silent pass."""
    metrics = {
        "subtitle_track_error": "hash_mismatch",
        "subtitle_track_status": "failed",
    }
    assert metrics["subtitle_track_status"] == "failed"
    assert metrics["subtitle_track_error"]
    assert metrics.get("subtitle_track_status") != "ok"


def test_artifact_integrity_hash_and_tamper(tmp_path: Path) -> None:
    cues = [{"start": 0.0, "end": 1.0, "text": "A"}]
    write_canonical_subtitle_track(tmp_path, cues=cues)
    loaded = load_canonical_subtitle_track(tmp_path)
    assert loaded is not None
    assert loaded["content_hash"] == hash_subtitle_track_body(cues=cues)

    path = tmp_path / "artifacts" / "subtitle_track.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cues"][0]["text"] = "Tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_canonical_subtitle_track(tmp_path) is None
