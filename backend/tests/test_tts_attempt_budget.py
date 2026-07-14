"""Tests for unified TTS attempt budget."""

from __future__ import annotations

from dv_backend.tts_attempt_budget import TtsAttemptBudget, budget_from_settings


def test_candidate_and_rewrite_respect_total_limit() -> None:
    budget = TtsAttemptBudget(max_total_syntheses=3, max_candidate_attempts=2, max_rewrite_attempts=1)
    budget.record_candidate()
    budget.record_candidate()
    assert budget.can_try_candidate() is False
    assert budget.can_rewrite() is True
    budget.record_rewrite()
    assert budget.remaining == 0
    assert budget.can_synthesize() is False


def test_cache_hit_does_not_consume_synth() -> None:
    budget = TtsAttemptBudget(max_total_syntheses=3)
    budget.record_cache_hit()
    assert budget.used == 0
    assert budget.cache_hits == 1


def test_budget_from_settings() -> None:
    budget = budget_from_settings({"timing_max_tts_attempts": 4, "timing_max_candidate_tts_attempts": 3})
    assert budget.max_total_syntheses == 4
    assert budget.max_candidate_attempts == 3
