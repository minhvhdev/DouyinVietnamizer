"""Tests for conflict-cluster timing repair helpers."""

from __future__ import annotations

from dv_backend.timing_conflict_repair import find_conflict_clusters
from dv_backend.timing_placement import enforce_zero_overlap_placements, schedule_soft_placements


def test_find_conflict_clusters_from_overflow_and_overlap() -> None:
    segments = [
        {
            "index": 0,
            "start": 0.0,
            "end": 2.0,
            "placement_start": 0.0,
            "repaired_duration": 3.5,
            "timing_overflow_sec": 1.0,
        },
        {
            "index": 1,
            "start": 2.2,
            "end": 4.0,
            "placement_start": 2.2,
            "repaired_duration": 1.5,
            "timing_overflow_sec": 0.0,
        },
        {
            "index": 2,
            "start": 8.0,
            "end": 9.0,
            "placement_start": 8.0,
            "repaired_duration": 0.8,
            "timing_overflow_sec": 0.0,
        },
    ]
    clusters = find_conflict_clusters(segments)
    assert clusters
    assert clusters[0][:2] == [0, 1]
    # Segment 2 is separated by long silence and should not join the cluster.
    assert all(2 not in cluster for cluster in clusters)


def test_find_conflict_clusters_caps_long_runs() -> None:
    segments = []
    for i in range(12):
        segments.append(
            {
                "index": i,
                "start": float(i) * 1.0,
                "end": float(i) * 1.0 + 0.9,
                "placement_start": float(i) * 1.0,
                "repaired_duration": 1.5,
                "timing_overflow_sec": 0.5,
            }
        )
    clusters = find_conflict_clusters(segments)
    assert clusters
    assert all(len(cluster) <= 5 for cluster in clusters)
    assert max(max(cluster) for cluster in clusters) == 11


def test_absorbed_unit_does_not_create_false_overlap() -> None:
    from dv_backend.timing_placement import segments_with_voiced_overlap

    segments = [
        {
            "index": 71,
            "placement_start": 300.89,
            "repaired_duration": 3.2,
            "tts_spoken_text": "anh em suýt chạy gãy chân.",
        },
        {
            "index": 73,
            "placement_start": 308.27,
            "repaired_duration": 0.0,
            "tts_duration": 2.45,
            "no_speech": True,
            "tts_spoken_text": "",
        },
        {
            "index": 72,
            "placement_start": 304.38,
            "repaired_duration": 3.7,
            "tts_spoken_text": "Nhớ đãi một bữa.",
        },
    ]
    assert segments_with_voiced_overlap(segments) == []


def test_enforce_zero_overlap_placements() -> None:
    segments = [
        {
            "index": 0,
            "start": 0.0,
            "preferred_placement_start": 0.0,
            "repaired_duration": 3.0,
        },
        {
            "index": 1,
            "start": 1.0,
            "preferred_placement_start": 1.0,
            "repaired_duration": 1.0,
        },
    ]
    schedule_soft_placements(segments)
    enforce_zero_overlap_placements(segments)
    end0 = float(segments[0]["placement_start"]) + float(segments[0]["repaired_duration"])
    assert float(segments[1]["placement_start"]) + 1e-6 >= end0
    assert segments[1]["timing_status"] == "UNRESOLVED_TIMING"
