"""Unified TTS synthesis attempt budget across TTS step and duration repair."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TtsAttemptBudget:
    max_total_syntheses: int = 3
    max_candidate_attempts: int = 2
    max_rewrite_attempts: int = 1
    used: int = 0
    candidate_attempts: int = 0
    rewrite_attempts: int = 0
    cache_hits: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.max_total_syntheses - self.used)

    def can_synthesize(self) -> bool:
        return self.remaining > 0

    def can_try_candidate(self) -> bool:
        return self.can_synthesize() and self.candidate_attempts < self.max_candidate_attempts

    def can_rewrite(self) -> bool:
        return self.can_synthesize() and self.rewrite_attempts < self.max_rewrite_attempts

    def record_candidate(self) -> None:
        self.candidate_attempts += 1
        self.used += 1

    def record_rewrite(self) -> None:
        self.rewrite_attempts += 1
        self.used += 1

    def record_repair_resynth(self) -> None:
        self.used += 1

    def record_cache_hit(self) -> None:
        self.cache_hits += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_total_syntheses": self.max_total_syntheses,
            "used": self.used,
            "remaining": self.remaining,
            "candidate_attempts": self.candidate_attempts,
            "rewrite_attempts": self.rewrite_attempts,
            "cache_hits": self.cache_hits,
        }


def budget_from_settings(settings: dict[str, Any]) -> TtsAttemptBudget:
    return TtsAttemptBudget(
        max_total_syntheses=max(1, int(settings.get("timing_max_tts_attempts", 3) or 3)),
        max_candidate_attempts=max(1, int(settings.get("timing_max_candidate_tts_attempts", 2) or 2)),
        max_rewrite_attempts=max(0, int(settings.get("timing_max_llm_rewrite_attempts", 1) or 1)),
    )
