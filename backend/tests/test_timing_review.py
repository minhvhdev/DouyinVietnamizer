from pathlib import Path

import pytest

from dv_backend.errors import AppError
from dv_backend.timing_review import estimate_words_to_remove, mark_infeasible_at_cap
from dv_backend.timing_review_ops import submit_timing_review_edits


def test_estimate_words_to_remove_range() -> None:
    result = estimate_words_to_remove(
        text="một hai ba bốn năm sáu bảy tám chín mười",
        fitted_duration=5.0,
        available_duration=3.0,
        overflow_seconds=2.0,
    )
    assert result["overflow_seconds"] == 2.0
    assert result["estimated_words_to_remove"] >= 3
    assert result["estimated_words_to_remove_min"] == result["estimated_words_to_remove"]
    assert result["estimated_words_to_remove_max"] >= result["estimated_words_to_remove_min"]


def test_estimate_words_to_remove_ignores_silence_pad_dilution() -> None:
    """Silence padding must not make the UI under-estimate word cuts."""
    text = "một hai ba bốn năm sáu bảy tám"
    optimistic = estimate_words_to_remove(
        text=text,
        fitted_duration=4.0,  # includes 2s silence
        available_duration=2.0,
        overflow_seconds=2.0,
        speech_duration=None,
    )
    speech_aware = estimate_words_to_remove(
        text=text,
        fitted_duration=4.0,
        available_duration=2.0,
        overflow_seconds=2.0,
        speech_duration=2.0,
    )
    assert speech_aware["estimated_words_to_remove"] >= optimistic["estimated_words_to_remove"]
    assert speech_aware["estimated_words_to_remove"] >= 4


def test_mark_infeasible_at_cap_sets_review_fields() -> None:
    segment = {
        "index": 3,
        "tts_spoken_text": "đây là một câu dịch khá dài để mô phỏng overflow",
        "timing_overflow_sec": 1.2,
        "timing_available_duration": 2.0,
        "repaired_duration": 3.2,
        "soft_speed_factor": 1.0,
    }
    assert mark_infeasible_at_cap(segment, absolute_max_rate=1.2) is True
    assert segment["timing_status"] == "timing_review_required"
    assert segment["timing_review_reason"] == "infeasible_at_cap"
    assert segment["needs_review"] is True
    assert segment["release_blocking"] is True
    assert segment["estimated_words_to_remove"] >= 1
    assert float(segment["required_speed"]) > 1.2


def test_mark_infeasible_skips_when_max_speed_would_fit() -> None:
    segment = {
        "index": 4,
        "tts_spoken_text": "hơi dài một chút",
        "timing_overflow_sec": 0.3,
        "timing_available_duration": 2.0,
        "repaired_duration": 2.3,
        "soft_speed_factor": 1.0,
        "timing_status": "OVERFLOW",
        "needs_review": True,
    }
    assert mark_infeasible_at_cap(segment, absolute_max_rate=1.2) is False
    assert "needs_review" not in segment
    assert "timing_review_reason" not in segment


def test_mark_infeasible_clears_stale_review_when_healthy() -> None:
    segment = {
        "index": 2,
        "tts_spoken_text": "câu ngắn",
        "timing_overflow_sec": 0.0,
        "timing_available_duration": 2.0,
        "repaired_duration": 1.2,
        "timing_status": "timing_review_required",
        "timing_review_reason": "infeasible_at_cap",
        "needs_review": True,
        "release_blocking": True,
    }
    assert mark_infeasible_at_cap(segment, absolute_max_rate=1.2) is False
    assert "needs_review" not in segment
    assert "timing_review_reason" not in segment
    assert "release_blocking" not in segment
    assert segment["timing_status"] == "OK"


def test_list_timing_review_ignores_stale_needs_review_without_overflow() -> None:
    from dv_backend.timing_review import flag_infeasible_segments, list_timing_review_segments

    segments = [
        {
            "index": 1,
            "tts_spoken_text": "đã vừa khung",
            "timing_overflow_sec": 0.0,
            "timing_available_duration": 3.77,
            "repaired_duration": 2.84,
            "soft_speed_factor": 1.2,
            "needs_review": True,
            "timing_status": "timing_review_required",
            "timing_review_reason": "infeasible_at_cap",
            "estimated_words_to_remove": 1,
            "overflow_seconds": 0.0,
        },
        {
            "index": 2,
            "tts_spoken_text": "vẫn dài hơn khung sau max speed",
            "timing_overflow_sec": 0.8,
            "timing_available_duration": 2.0,
            "repaired_duration": 2.8,
            "soft_speed_factor": 1.2,
            "timing_status": "timing_review_required",
            "timing_review_reason": "infeasible_at_cap",
        },
    ]
    flag_infeasible_segments(segments, absolute_max_rate=1.2)
    rows = list_timing_review_segments(segments)
    assert [row["index"] for row in rows] == [2]
    assert float(rows[0]["overflow_seconds"]) > 0.15
    assert float(rows[0]["repaired_duration"]) > float(rows[0]["timing_available_duration"])


def test_list_excludes_when_duration_already_fits_available() -> None:
    """Critical: dài < khung must never appear as 'needs shorten' with Thừa~0."""
    from dv_backend.timing_review import flag_infeasible_segments, list_timing_review_segments

    segments = [
        {
            "index": 17,
            "tts_spoken_text": "Hắn không muốn đi, ta lại không muốn hắn tiếp tục ăn thịt người.",
            # Stale schedule overflow — pre-speed / not recomputed.
            "timing_overflow_sec": 0.9,
            "timing_available_duration": 3.77,
            "repaired_duration": 2.84,
            "soft_speed_factor": 1.2,
            "estimated_words_to_remove": 1,
            "estimated_words_to_remove_min": 1,
            "overflow_seconds": 0.0,
            "needs_review": True,
            "timing_status": "timing_review_required",
            "timing_review_reason": "infeasible_at_cap",
            "release_blocking": True,
        }
    ]
    flag_infeasible_segments(segments, absolute_max_rate=1.2)
    rows = list_timing_review_segments(segments)
    assert rows == []
    assert segments[0].get("needs_review") is not True
    assert "estimated_words_to_remove" not in segments[0]


def test_list_excludes_when_max_speed_would_fit_even_if_raw_overflow() -> None:
    from dv_backend.timing_review import flag_infeasible_segments, list_timing_review_segments

    segments = [
        {
            "index": 4,
            "tts_spoken_text": "hơi dài một chút nhưng 1.2x đủ",
            "timing_overflow_sec": 0.3,
            "timing_available_duration": 2.0,
            "repaired_duration": 2.3,
            "soft_speed_factor": 1.0,
            "estimated_words_to_remove": 2,
        }
    ]
    flag_infeasible_segments(segments, absolute_max_rate=1.2)
    assert list_timing_review_segments(segments) == []


def test_mark_infeasible_at_cap_uses_duration_minus_available_not_stale_overflow() -> None:
    segment = {
        "index": 18,
        "tts_spoken_text": "đã vừa sau speed",
        "timing_overflow_sec": 1.5,  # stale
        "timing_available_duration": 3.77,
        "repaired_duration": 2.84,
        "soft_speed_factor": 1.2,
        "needs_review": True,
        "timing_status": "timing_review_required",
        "estimated_words_to_remove": 1,
        "overflow_seconds": 0.0,
    }
    assert mark_infeasible_at_cap(segment, absolute_max_rate=1.2) is False
    assert "needs_review" not in segment
    assert "estimated_words_to_remove" not in segment
    assert "overflow_seconds" not in segment


def test_rebase_cue_texts_to_spoken_overrides_asr_words() -> None:
    from dv_backend.subtitle_timing import _rebase_cue_texts_to_spoken

    cues = [
        {"start": 1.0, "end": 1.8, "text": "asr sai"},
        {"start": 1.8, "end": 2.6, "text": "toàn bộ"},
    ]
    out = _rebase_cue_texts_to_spoken(
        cues,
        "Xin chào thế giới",
        language="vi",
        settings={"subtitle_max_chars_per_cue": 40},
    )
    joined = " ".join(c["text"] for c in out)
    assert "asr" not in joined.lower()
    assert "Xin chào" in joined or "chào" in joined.lower()


def test_checkpoint_for_review_prefers_align_final_dub_when_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dv_backend import timing_review_ops as ops

    align_cp = {
        "segments": [
            {
                "index": 0,
                "start": 0.0,
                "end": 2.0,
                "placement_start": 0.0,
                "repaired_duration": 2.0,
            },
            {
                "index": 1,
                "start": 3.0,
                "end": 5.0,
                "placement_start": 4.2,
                "repaired_duration": 1.8,
            },
            {
                "index": 2,
                "start": 6.0,
                "end": 8.0,
                "placement_start": 6.0,
                "repaired_duration": 2.0,
            },
        ]
    }
    repair_cp = {
        "segments": [
            {
                "index": 1,
                "start": 3.0,
                "end": 5.0,
                "placement_start": 3.0,
                "repaired_duration": 1.8,
            }
        ]
    }

    def _load(_data_dir, _job_id, step):
        if step == "align_final_dub":
            return align_cp
        if step == "duration_repair":
            return repair_cp
        return None

    monkeypatch.setattr(ops, "load_checkpoint", _load)

    class _Cfg:
        data_dir = tmp_path

    step, cp = ops._checkpoint_for_review(_Cfg(), "job-1")  # type: ignore[arg-type]
    assert step == "align_final_dub"
    assert len(cp["segments"]) == 3


def test_timing_review_rows_include_effective_playback_fields() -> None:
    from dv_backend.timing_review import flag_infeasible_segments, list_timing_review_segments

    segments = [
        {
            "index": 1,
            "start": 3.0,
            "end": 5.0,
            "placement_start": 4.2,
            "placement_end": 6.0,
            "tts_spoken_text": "đoạn bị dời và vẫn quá dài sau max speed",
            "timing_available_duration": 1.2,
            "repaired_duration": 2.4,
            "soft_speed_factor": 1.2,
        }
    ]
    flag_infeasible_segments(segments, absolute_max_rate=1.2)
    rows = list_timing_review_segments(segments, timing_stage="align_final_dub")
    assert len(rows) == 1
    assert rows[0]["effective_start"] == 4.2
    assert rows[0]["source_start"] == 3.0
    assert rows[0]["timing_stage"] == "align_final_dub"


def test_segment_without_dub_words_still_has_timing_diagnostics() -> None:
    from dv_backend.timing_placement import segment_timing_diagnostics

    payload = segment_timing_diagnostics(
        {
            "index": 2,
            "start": 6.0,
            "end": 8.0,
            "placement_start": 6.0,
            "repaired_duration": 2.0,
        },
        timing_stage="align_final_dub",
    )
    assert payload["effective_start"] == 6.0
    assert payload["effective_end"] == 8.0


def test_plan_version_conflict_rejects_stale_edit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dv_backend import timing_review_ops as ops

    job_id = "job-plan-lock"
    cp = {
        "segments": [
            {
                "index": 1,
                "tts_spoken_text": "text cũ dài hơn cần rút",
                "translation": "text cũ dài hơn cần rút",
                "plan_version": 3,
                "start": 0.0,
                "end": 2.0,
            }
        ]
    }

    monkeypatch.setattr(ops, "_checkpoint_for_review", lambda *_a, **_k: ("duration_repair", cp))
    monkeypatch.setattr(ops, "_load_settings", lambda *_a, **_k: {"edge_tts_overflow_speed_hard_max": 1.2})
    monkeypatch.setattr(ops, "resolve_tool_path", lambda *_a, **_k: tmp_path / "ffmpeg")
    monkeypatch.setattr(ops, "load_checkpoint", lambda *_a, **_k: None)
    monkeypatch.setattr(ops, "save_checkpoint", lambda *_a, **_k: None)

    class _Cfg:
        data_dir = tmp_path

    with pytest.raises(AppError) as raised:
        submit_timing_review_edits(
            config=_Cfg(),  # type: ignore[arg-type]
            database=object(),  # type: ignore[arg-type]
            runner=object(),
            job_id=job_id,
            edits=[{"index": 1, "spoken_text": "text rút", "expected_plan_version": 2}],
            resume_pipeline=False,
        )
    assert raised.value.status_code == 409
    assert raised.value.info.code == "PLAN_VERSION_CONFLICT"
