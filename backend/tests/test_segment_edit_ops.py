from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dv_backend.checkpoints import checkpoint_path, load_checkpoint, save_checkpoint
from dv_backend.segment_edit_ops import (
    CorruptSegmentEditPlanError,
    EditableSegmentInput,
    InvalidJobStateError,
    SegmentEditPlanService,
    SegmentSourceUnavailableError,
    UnknownSegmentIdError,
)
from dv_backend.segment_edit_plan import PlanVersionConflictError


class FakeJobs:
    def __init__(self, status: str = "completed") -> None:
        self.status = status

    def get(self, job_id: str) -> SimpleNamespace:
        return SimpleNamespace(id=job_id, status=self.status)


def legacy_segment(
    index: int,
    *,
    start: float = 1.0,
    end: float = 2.0,
    translation: str = "Xin chào",
    segment_id: str | None = None,
) -> dict:
    result = {
        "index": index,
        "start": start,
        "end": end,
        "translation": translation,
        "text": "你好",
    }
    if segment_id is not None:
        result["segment_id"] = segment_id
    return result


def service(tmp_path: Path, *, status: str = "completed") -> SegmentEditPlanService:
    return SegmentEditPlanService(tmp_path, FakeJobs(status))


def test_initializes_completed_legacy_job_once_with_deterministic_ids(tmp_path: Path) -> None:
    save_checkpoint(
        tmp_path,
        "job-1",
        "duration_repair",
        {"segments": [legacy_segment(0), legacy_segment(1, start=2.0, end=3.0)]},
    )

    first = service(tmp_path).get_or_create("job-1")
    second = service(tmp_path).get_or_create("job-1")

    assert first == second
    assert first.plan_version == first.applied_plan_version == 1
    assert first.applied_segments == first.draft_segments
    assert [item.start_ms for item in first.draft_segments] == [1000, 2000]
    assert len({item.segment_id for item in first.draft_segments}) == 2


def test_legacy_source_precedence_prefers_first_complete_checkpoint(tmp_path: Path) -> None:
    save_checkpoint(
        tmp_path,
        "job-1",
        "align_final_dub",
        {"segments": [{"index": 0, "start": 0.0, "end": 1.0}]},
    )
    save_checkpoint(
        tmp_path,
        "job-1",
        "duration_repair",
        {"segments": [legacy_segment(0, translation="Duration")]},
    )
    save_checkpoint(
        tmp_path,
        "job-1",
        "tts",
        {"segments": [legacy_segment(0, translation="TTS")]},
    )

    plan = service(tmp_path).get_or_create("job-1")

    assert plan.draft_segments[0].spoken_text == "Duration"


def test_existing_unique_stable_id_is_preserved(tmp_path: Path) -> None:
    save_checkpoint(
        tmp_path,
        "job-1",
        "align_final_dub",
        {"segments": [legacy_segment(0, segment_id="stable-a")]},
    )

    plan = service(tmp_path).get_or_create("job-1")

    assert plan.draft_segments[0].segment_id == "stable-a"


def test_existing_canonical_plan_is_not_remigrated(tmp_path: Path) -> None:
    save_checkpoint(
        tmp_path,
        "job-1",
        "translate",
        {"segments": [legacy_segment(0, translation="First")]},
    )
    edit_service = service(tmp_path)
    first = edit_service.get_or_create("job-1")
    save_checkpoint(
        tmp_path,
        "job-1",
        "translate",
        {"segments": [legacy_segment(0, translation="Changed legacy")]},
    )

    second = edit_service.get_or_create("job-1")

    assert second == first


def test_corrupt_canonical_checkpoint_is_not_overwritten(tmp_path: Path) -> None:
    canonical = checkpoint_path(tmp_path, "job-1", "segment_edit_plan")
    canonical.parent.mkdir(parents=True)
    canonical.write_text("{broken", encoding="utf-8")
    save_checkpoint(
        tmp_path,
        "job-1",
        "translate",
        {"segments": [legacy_segment(0)]},
    )

    with pytest.raises(CorruptSegmentEditPlanError):
        service(tmp_path).get_or_create("job-1")

    assert canonical.read_text(encoding="utf-8") == "{broken"


def test_rejects_non_completed_job_and_missing_legacy_source(tmp_path: Path) -> None:
    with pytest.raises(InvalidJobStateError):
        service(tmp_path, status="running").get_or_create("job-1")
    with pytest.raises(SegmentSourceUnavailableError):
        service(tmp_path).get_or_create("job-1")


def test_save_preserves_existing_provenance_and_adds_server_owned_segment(
    tmp_path: Path,
) -> None:
    save_checkpoint(
        tmp_path,
        "job-1",
        "translate",
        {"segments": [legacy_segment(0)]},
    )
    edit_service = service(tmp_path)
    initial = edit_service.get_or_create("job-1")
    old = initial.draft_segments[0]

    updated = edit_service.save_draft(
        "job-1",
        expected_plan_version=1,
        segments=[
            EditableSegmentInput(
                segment_id=old.segment_id,
                start_ms=100,
                end_ms=900,
                spoken_text="Sửa",
            ),
            EditableSegmentInput(
                segment_id=None,
                start_ms=1000,
                end_ms=1500,
                spoken_text="Mới",
            ),
        ],
    )

    existing, added = updated.draft_segments
    assert updated.plan_version == 2
    assert existing.source_text == old.source_text
    assert existing.source_segment_index == old.source_segment_index
    assert existing.origin == "pipeline"
    assert added.segment_id
    assert added.origin == "user"
    assert added.source_text is None
    assert added.source_segment_index is None


def test_unknown_explicit_id_is_rejected_without_write(tmp_path: Path) -> None:
    save_checkpoint(
        tmp_path,
        "job-1",
        "translate",
        {"segments": [legacy_segment(0)]},
    )
    edit_service = service(tmp_path)
    initial = edit_service.get_or_create("job-1")
    stored_before = load_checkpoint(tmp_path, "job-1", "segment_edit_plan")

    with pytest.raises(UnknownSegmentIdError):
        edit_service.save_draft(
            "job-1",
            expected_plan_version=initial.plan_version,
            segments=[
                EditableSegmentInput(
                    segment_id="client-invented",
                    start_ms=0,
                    end_ms=1000,
                    spoken_text="Không hợp lệ",
                )
            ],
        )

    assert load_checkpoint(tmp_path, "job-1", "segment_edit_plan") == stored_before


def test_delete_and_reorder_are_full_list_replacement(tmp_path: Path) -> None:
    save_checkpoint(
        tmp_path,
        "job-1",
        "translate",
        {
            "segments": [
                legacy_segment(0),
                legacy_segment(1, start=2.0, end=3.0),
                legacy_segment(2, start=3.0, end=4.0),
            ]
        },
    )
    edit_service = service(tmp_path)
    initial = edit_service.get_or_create("job-1")
    first, _, third = initial.draft_segments

    updated = edit_service.save_draft(
        "job-1",
        expected_plan_version=1,
        segments=[
            EditableSegmentInput(
                segment_id=third.segment_id,
                start_ms=third.start_ms,
                end_ms=third.end_ms,
                spoken_text=third.spoken_text,
            ),
            EditableSegmentInput(
                segment_id=first.segment_id,
                start_ms=first.start_ms,
                end_ms=first.end_ms,
                spoken_text=first.spoken_text,
            ),
        ],
    )

    assert [item.segment_id for item in updated.draft_segments] == [
        third.segment_id,
        first.segment_id,
    ]


def test_noop_save_does_not_write_or_increment_version(tmp_path: Path) -> None:
    save_checkpoint(
        tmp_path,
        "job-1",
        "translate",
        {"segments": [legacy_segment(0)]},
    )
    edit_service = service(tmp_path)
    initial = edit_service.get_or_create("job-1")
    current = initial.draft_segments[0]

    with patch("dv_backend.segment_edit_ops.save_checkpoint") as save_mock:
        updated = edit_service.save_draft(
            "job-1",
            expected_plan_version=1,
            segments=[
                EditableSegmentInput(
                    segment_id=current.segment_id,
                    start_ms=current.start_ms,
                    end_ms=current.end_ms,
                    spoken_text=current.spoken_text,
                )
            ],
        )

    assert updated.plan_version == 1
    save_mock.assert_not_called()


def test_failed_write_preserves_previous_checkpoint(tmp_path: Path) -> None:
    save_checkpoint(
        tmp_path,
        "job-1",
        "translate",
        {"segments": [legacy_segment(0)]},
    )
    edit_service = service(tmp_path)
    initial = edit_service.get_or_create("job-1")
    current = initial.draft_segments[0]
    stored_before = load_checkpoint(tmp_path, "job-1", "segment_edit_plan")

    with (
        patch(
            "dv_backend.segment_edit_ops.save_checkpoint",
            side_effect=OSError("disk full"),
        ),
        pytest.raises(OSError, match="disk full"),
    ):
        edit_service.save_draft(
            "job-1",
            expected_plan_version=1,
            segments=[
                EditableSegmentInput(
                    segment_id=current.segment_id,
                    start_ms=current.start_ms,
                    end_ms=current.end_ms,
                    spoken_text="Changed",
                )
            ],
        )

    assert load_checkpoint(tmp_path, "job-1", "segment_edit_plan") == stored_before


def test_concurrent_same_version_only_one_save_succeeds(tmp_path: Path) -> None:
    save_checkpoint(
        tmp_path,
        "job-1",
        "translate",
        {"segments": [legacy_segment(0)]},
    )
    edit_service = service(tmp_path)
    current = edit_service.get_or_create("job-1").draft_segments[0]

    def update(text: str) -> str:
        try:
            edit_service.save_draft(
                "job-1",
                expected_plan_version=1,
                segments=[
                    EditableSegmentInput(
                        segment_id=current.segment_id,
                        start_ms=current.start_ms,
                        end_ms=current.end_ms,
                        spoken_text=text,
                    )
                ],
            )
            return "saved"
        except PlanVersionConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(update, ["One", "Two"]))

    assert sorted(results) == ["conflict", "saved"]


def test_response_diff_sets_are_sorted_lists(tmp_path: Path) -> None:
    save_checkpoint(
        tmp_path,
        "job-1",
        "translate",
        {
            "segments": [
                legacy_segment(0),
                legacy_segment(1, start=2.0, end=3.0),
            ]
        },
    )
    edit_service = service(tmp_path)
    plan = edit_service.get_or_create("job-1")
    payload = edit_service.response_payload(plan)

    assert payload["diff"]["reusable_tts_segment_ids"] == sorted(
        payload["diff"]["reusable_tts_segment_ids"]
    )
