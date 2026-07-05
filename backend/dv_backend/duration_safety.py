from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class StretchDecision:
    factor: float
    risk: str
    allowed: bool
    warning: str | None = None


def classify_stretch(factor: float, *, max_safe: float = 1.25, explicit_allow_danger: bool = False) -> StretchDecision:
    value = max(0.01, float(factor))
    effective = value if value >= 1.0 else 1.0 / value
    if effective <= 1.10:
        return StretchDecision(value, "normal", True)
    if effective <= max_safe:
        return StretchDecision(value, "allowed", True)
    if effective <= 1.35:
        return StretchDecision(value, "warning", True, "prefer_rewrite_before_large_stretch")
    return StretchDecision(value, "danger", bool(explicit_allow_danger), "stretch_factor_exceeds_safe_policy")


def tail_has_speech(samples: Sequence[float], *, sample_rate: int, tail_ms: int = 200, threshold: float = 0.02) -> bool:
    if sample_rate <= 0 or not samples:
        return False
    count = max(1, int(sample_rate * tail_ms / 1000.0))
    tail = list(samples[-count:])
    if not tail:
        return False
    energy = sum(float(sample) * float(sample) for sample in tail) / len(tail)
    return energy ** 0.5 > threshold
