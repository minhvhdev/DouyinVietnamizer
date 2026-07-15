from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from dv_backend.segment_edit_plan import (
    EditableSegment,
    PlanVersionConflictError,
    SegmentEditPlan,
    diff_applied_to_draft,
    replace_draft,
    validate_segments,
)


def segment(
    segment_id: str,
    *,
    start_ms: int = 0,
    end_ms: int = 1000,
    spoken_text: str = "Xin chào",
    source_text: str | None = "你好",
    origin: str = "pipeline",
    source_segment_index: int | None = 0,
) -> EditableSegment:
    return EditableSegment(
        segment_id=segment_id,
        start_ms=start_ms,
        end_ms=end_ms,
        spoken_text=spoken_text,
        source_text=source_text,
        origin=origin,
        source_segment_index=source_segment_index,
    )


def plan(*segments: EditableSegment) -> SegmentEditPlan:
    return SegmentEditPlan(
        plan_version=3,
        applied_plan_version=2,
        applied_segments=list(segments),
        draft_segments=list(segments),
    )


def test_unchanged_diff_is_empty_but_all_tts_is_reusable() -> None:
    segments = [segment("a"), segment("b", start_ms=1000, end_ms=2000)]

    diff = diff_applied_to_draft(segments, deepcopy(segments))

    assert diff.has_changes is False
    assert diff.structural_changed is False
    assert diff.deltas == []
    assert diff.requires_tts_segment_ids == set()
    assert diff.requires_duration_check_segment_ids == set()
    assert diff.reusable_tts_segment_ids == {"a", "b"}


def test_text_change_requires_tts_and_duration_only_for_that_segment() -> None:
    applied = [segment("a"), segment("b", start_ms=1000, end_ms=2000)]
    draft = deepcopy(applied)
    draft[1] = draft[1].model_copy(update={"spoken_text": "Đã sửa"})

    diff = diff_applied_to_draft(applied, draft)

    assert [delta.segment_id for delta in diff.deltas] == ["b"]
    assert diff.deltas[0].text_changed is True
    assert diff.requires_tts_segment_ids == {"b"}
    assert diff.requires_duration_check_segment_ids == {"b"}
    assert diff.reusable_tts_segment_ids == {"a"}


def test_timing_change_reuses_tts_but_requires_duration_check() -> None:
    applied = [segment("a")]
    draft = [applied[0].model_copy(update={"end_ms": 1200})]

    diff = diff_applied_to_draft(applied, draft)

    assert diff.deltas[0].timing_changed is True
    assert diff.requires_tts_segment_ids == set()
    assert diff.requires_duration_check_segment_ids == {"a"}
    assert diff.reusable_tts_segment_ids == {"a"}


def test_text_and_timing_change_does_not_mark_tts_reusable() -> None:
    applied = [segment("a")]
    draft = [applied[0].model_copy(update={"spoken_text": "Mới", "end_ms": 1200})]

    diff = diff_applied_to_draft(applied, draft)

    assert diff.requires_tts_segment_ids == {"a"}
    assert diff.requires_duration_check_segment_ids == {"a"}
    assert diff.reusable_tts_segment_ids == set()


def test_add_in_middle_preserves_existing_identity_and_reusable_tts() -> None:
    applied = [segment("a"), segment("b", start_ms=1000, end_ms=2000)]
    added = segment(
        "new",
        start_ms=500,
        end_ms=900,
        spoken_text="Đoạn mới",
        source_text=None,
        origin="user",
        source_segment_index=None,
    )

    diff = diff_applied_to_draft(applied, [applied[0], added, applied[1]])

    assert diff.requires_tts_segment_ids == {"new"}
    assert diff.reusable_tts_segment_ids == {"a", "b"}
    assert diff.structural_changed is True
    assert next(delta for delta in diff.deltas if delta.segment_id == "new").added is True
    assert not next(delta for delta in diff.deltas if delta.segment_id == "b").text_changed


def test_delete_in_middle_does_not_turn_later_segment_into_text_change() -> None:
    applied = [
        segment("a"),
        segment("b", start_ms=1000, end_ms=2000),
        segment("c", start_ms=2000, end_ms=3000),
    ]

    diff = diff_applied_to_draft(applied, [applied[0], applied[2]])

    by_id = {delta.segment_id: delta for delta in diff.deltas}
    assert diff.deleted_segment_ids == {"b"}
    assert by_id["b"].deleted is True
    assert by_id["c"].text_changed is False
    assert "c" in diff.reusable_tts_segment_ids


def test_reorder_is_structural_without_requiring_tts() -> None:
    applied = [segment("a"), segment("b", start_ms=1000, end_ms=2000)]

    diff = diff_applied_to_draft(applied, [applied[1], applied[0]])

    assert diff.structural_changed is True
    assert diff.requires_tts_segment_ids == set()
    assert diff.reusable_tts_segment_ids == {"a", "b"}
    assert all(delta.order_changed for delta in diff.deltas)


def test_add_and_delete_do_not_change_remaining_identity() -> None:
    applied = [segment("a"), segment("b", start_ms=1000, end_ms=2000)]
    added = segment(
        "c",
        start_ms=1000,
        end_ms=2000,
        spoken_text="C",
        source_text=None,
        origin="user",
        source_segment_index=None,
    )

    diff = diff_applied_to_draft(applied, [applied[0], added])

    assert diff.deleted_segment_ids == {"b"}
    assert diff.requires_tts_segment_ids == {"c"}
    assert "a" in diff.reusable_tts_segment_ids


def test_source_text_only_change_does_not_require_processing() -> None:
    applied = [segment("a")]
    draft = [applied[0].model_copy(update={"source_text": "您好"})]

    diff = diff_applied_to_draft(applied, draft)

    assert diff.has_changes is False
    assert diff.deltas == []
    assert diff.reusable_tts_segment_ids == {"a"}


@pytest.mark.parametrize(
    "segments",
    [
        [segment("a"), segment("a", start_ms=1000, end_ms=2000)],
        [segment(" ")],
        [segment("a", start_ms=-1)],
        [segment("a", start_ms=1000, end_ms=1000)],
        [segment("a", spoken_text=" \t ")],
        [segment("a", source_segment_index=None)],
        [
            segment(
                "a",
                origin="user",
                source_text=None,
                source_segment_index=0,
            )
        ],
    ],
)
def test_invalid_segments_are_rejected(segments: list[EditableSegment]) -> None:
    with pytest.raises(ValueError):
        validate_segments(segments)


def test_invalid_origin_is_rejected_by_model() -> None:
    with pytest.raises(ValidationError):
        segment("a", origin="external")


def test_version_mismatch_raises_dedicated_exception() -> None:
    current = plan(segment("a"))

    with pytest.raises(PlanVersionConflictError) as exc_info:
        replace_draft(current, current.draft_segments, expected_plan_version=2)

    assert exc_info.value.expected == 2
    assert exc_info.value.current == 3


def test_noop_replace_draft_does_not_increment_version() -> None:
    current = plan(segment("a"))

    updated = replace_draft(current, deepcopy(current.draft_segments), expected_plan_version=3)

    assert updated.plan_version == 3
    assert updated == current
    assert updated is not current


def test_changed_draft_increments_once_and_preserves_applied_snapshot() -> None:
    current = plan(segment("a"))
    proposed = [current.draft_segments[0].model_copy(update={"spoken_text": "Đã sửa"})]

    updated = replace_draft(current, proposed, expected_plan_version=3)

    assert updated.plan_version == 4
    assert updated.applied_plan_version == 2
    assert updated.applied_segments == current.applied_segments
    assert updated.draft_segments == proposed


def test_reverting_draft_to_applied_produces_empty_export_diff() -> None:
    base = segment("a")
    current = SegmentEditPlan(
        plan_version=4,
        applied_plan_version=2,
        applied_segments=[base],
        draft_segments=[base.model_copy(update={"spoken_text": "Đã sửa"})],
    )

    reverted = replace_draft(current, current.applied_segments, expected_plan_version=4)
    diff = diff_applied_to_draft(reverted.applied_segments, reverted.draft_segments)

    assert reverted.plan_version == 5
    assert diff.has_changes is False


def test_serialization_round_trip_preserves_diff_semantics() -> None:
    current = plan(segment("a"))
    current = replace_draft(
        current,
        [current.draft_segments[0].model_copy(update={"end_ms": 1200})],
        expected_plan_version=3,
    )

    restored = SegmentEditPlan.model_validate_json(current.model_dump_json())

    assert diff_applied_to_draft(
        restored.applied_segments,
        restored.draft_segments,
    ) == diff_applied_to_draft(current.applied_segments, current.draft_segments)


def test_operations_do_not_mutate_inputs() -> None:
    applied = [segment("a"), segment("b", start_ms=1000, end_ms=2000)]
    draft = deepcopy(applied)
    current = plan(*applied)
    applied_before = deepcopy(applied)
    draft_before = deepcopy(draft)
    plan_before = current.model_copy(deep=True)

    diff_applied_to_draft(applied, draft)
    replace_draft(current, draft, expected_plan_version=3)

    assert applied == applied_before
    assert draft == draft_before
    assert current == plan_before


def test_diff_delta_order_is_deterministic() -> None:
    applied = [
        segment("b"),
        segment("a", start_ms=1000, end_ms=2000),
        segment("deleted", start_ms=2000, end_ms=3000),
    ]
    added = segment(
        "added",
        start_ms=3000,
        end_ms=4000,
        spoken_text="Mới",
        source_text=None,
        origin="user",
        source_segment_index=None,
    )
    draft = [
        applied[1].model_copy(update={"spoken_text": "A mới"}),
        applied[0],
        added,
    ]

    first = diff_applied_to_draft(applied, draft)
    second = diff_applied_to_draft(deepcopy(applied), deepcopy(draft))

    assert [delta.segment_id for delta in first.deltas] == [
        delta.segment_id for delta in second.deltas
    ]
