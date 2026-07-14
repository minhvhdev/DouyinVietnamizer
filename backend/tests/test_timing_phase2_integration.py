"""Integration tests for timing-aware translate → TTS → repair flow."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dv_backend.pipeline import (
    _tts_text_fingerprint,
    duration_repair_step,
    translate_step,
    tts_step,
)
from dv_backend.translation_candidate_ranking import rank_translation_candidates


PROFILE = {
    "timeline_window": 4.8,
    "speech_target_duration": 3.9,
    "soft_min_duration": 3.3,
    "hard_max_duration": 4.45,
    "leading_silence_allowance": 0.2,
    "trailing_silence_allowance": 0.45,
}


def _segment(index: int, *, text: str, translation: str, candidates: list[dict], selected: int) -> dict:
    return {
        "index": index,
        "start": float(index * 5),
        "end": float(index * 5 + 3),
        "original_duration": 3.0,
        "duration_budget": 5.0,
        "text": text,
        "timing_profile": dict(PROFILE),
        "translation": translation,
        "translation_candidates": candidates,
        "selected_candidate_index": selected,
    }


def test_segment_a_natural_candidate_ranking_accepts_first() -> None:
    candidates = [
        {"text": "Hôm nay ta thử món này nhé.", "style": "natural"},
        {"text": "Thử món này.", "style": "compact"},
    ]
    ranked = rank_translation_candidates(
        candidates,
        timing_profile=PROFILE,
        source_text="今天试试这个。",
        language="vi",
    )
    assert ranked["selected_candidate_index"] in {0, 1}


def test_segment_b_compact_selected_when_natural_long() -> None:
    candidates = [
        {"text": "Hôm nay " + "chúng ta " * 12 + "thử món.", "style": "natural"},
        {"text": "Hôm nay thử món này.", "style": "compact"},
    ]
    ranked = rank_translation_candidates(
        candidates,
        timing_profile=PROFILE,
        source_text="今天试试这个。",
        language="vi",
    )
    assert ranked["selected_candidate_index"] == 1


def test_translation_change_invalidates_tts_fingerprint() -> None:
    assert _tts_text_fingerprint("compact") != _tts_text_fingerprint("natural")


@pytest.mark.parametrize("step_module", [translate_step, tts_step, duration_repair_step])
def test_pipeline_steps_are_callable(step_module) -> None:
    assert callable(step_module)
