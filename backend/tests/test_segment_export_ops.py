from __future__ import annotations

from copy import deepcopy

import pytest

from dv_backend.segment_edit_plan import EditableSegment, SegmentEditPlan
from dv_backend.segment_export_ops import materialize_target_segments


def editable(
    segment_id: str,
    *,
    start_ms: int,
    end_ms: int,
    text: str,
    source_index: int | None,
) -> EditableSegment:
    return EditableSegment(
        segment_id=segment_id,
        start_ms=start_ms,
        end_ms=end_ms,
        spoken_text=text,
        source_text=None,
        origin="pipeline" if source_index is not None else "user",
        source_segment_index=source_index,
    )


def pipeline_segment(index: int, *, text: str, raw: str, repaired: str) -> dict:
    return {
        "index": index,
        "start": float(index),
        "end": float(index + 1),
        "translation": text,
        "tts_spoken_text": text,
        "tts_raw_path": raw,
        "tts_path": repaired,
        "repaired_duration": 0.8,
        "placement_start": float(index),
    }


def test_materialize_target_uses_stable_ids_and_draft_order() -> None:
    applied = [
        editable("a", start_ms=0, end_ms=1000, text="A", source_index=0),
        editable("b", start_ms=1000, end_ms=2000, text="B", source_index=1),
    ]
    plan = SegmentEditPlan(
        plan_version=2,
        applied_plan_version=1,
        applied_segments=applied,
        draft_segments=[
            editable("b", start_ms=1200, end_ms=2200, text="B", source_index=1),
            editable("new", start_ms=2300, end_ms=3000, text="Mới", source_index=None),
            editable("a", start_ms=0, end_ms=1000, text="A mới", source_index=0),
        ],
    )
    source = [
        pipeline_segment(0, text="A", raw="raw-a.wav", repaired="rep-a.wav"),
        pipeline_segment(1, text="B", raw="raw-b.wav", repaired="rep-b.wav"),
    ]
    source_before = deepcopy(source)

    target = materialize_target_segments(
        plan,
        source_segments=source,
        synthesized_by_id={
            "new": {
                **pipeline_segment(1, text="Mới", raw="raw-new.wav", repaired="raw-new.wav"),
            },
            "a": {
                **pipeline_segment(2, text="A mới", raw="raw-a2.wav", repaired="raw-a2.wav"),
            },
        },
        repaired_by_id={
            "b": {
                **pipeline_segment(0, text="B", raw="raw-b.wav", repaired="rep-b2.wav"),
            },
            "new": {
                **pipeline_segment(1, text="Mới", raw="raw-new.wav", repaired="rep-new.wav"),
            },
            "a": {
                **pipeline_segment(2, text="A mới", raw="raw-a2.wav", repaired="rep-a2.wav"),
            },
        },
    )

    assert [segment["segment_id"] for segment in target] == ["b", "new", "a"]
    assert [segment["index"] for segment in target] == [0, 1, 2]
    assert target[0]["tts_raw_path"] == "raw-b.wav"
    assert target[0]["tts_path"] == "rep-b2.wav"
    assert target[0]["start"] == pytest.approx(1.2)
    assert target[2]["tts_raw_path"] == "raw-a2.wav"
    assert target[2]["translation"] == "A mới"
    assert source == source_before
