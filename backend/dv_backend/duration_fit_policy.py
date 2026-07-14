"""Duration fit classification and tempo policy for natural dubbing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .duration_safety import StretchDecision, classify_stretch
from .timing_profile import timing_profile_from_segment
from .utterance_policy import classify_utterance_length, short_utterance_abs_tolerance


@dataclass(frozen=True)
class DurationFitPolicy:
    good_ratio: float = 0.08
    good_abs_sec: float = 0.18
    slight_ratio: float = 0.15
    slight_abs_sec: float = 0.35
    preferred_tempo_min: float = 0.94
    preferred_tempo_max: float = 1.08
    warning_tempo_min: float = 0.90
    warning_tempo_max: float = 1.12


def policy_from_settings(settings: dict[str, Any]) -> DurationFitPolicy:
    return DurationFitPolicy(
        good_ratio=float(settings.get("timing_good_ratio", 0.08) or 0.08),
        good_abs_sec=float(settings.get("timing_good_abs_sec", 0.18) or 0.18),
        slight_ratio=float(settings.get("timing_slight_ratio", 0.15) or 0.15),
        slight_abs_sec=float(settings.get("timing_slight_abs_sec", 0.35) or 0.35),
        preferred_tempo_min=float(settings.get("timing_preferred_tempo_min", 0.94) or 0.94),
        preferred_tempo_max=float(settings.get("timing_preferred_tempo_max", 1.08) or 1.08),
        warning_tempo_min=float(settings.get("timing_warning_tempo_min", 0.90) or 0.90),
        warning_tempo_max=float(settings.get("timing_warning_tempo_max", 1.12) or 1.12),
    )


def _tolerance(target: float, *, ratio: float, abs_sec: float) -> float:
    if target <= 0:
        return abs_sec
    return max(abs_sec, target * ratio)


def classify_duration_fit(
    speech_duration: float,
    timing_profile: dict[str, float],
    *,
    policy: DurationFitPolicy | None = None,
    segment: dict | None = None,
) -> str:
    policy = policy or DurationFitPolicy()
    target = float(timing_profile.get("speech_target_duration") or 0.0)
    hard_max = float(timing_profile.get("hard_max_duration") or 0.0)
    soft_min = float(timing_profile.get("soft_min_duration") or 0.0)
    speech = max(0.0, float(speech_duration))

    if target <= 0:
        return "unknown"

    good_abs = policy.good_abs_sec
    slight_abs = policy.slight_abs_sec
    if segment is not None and classify_utterance_length(segment) == "short":
        good_abs = short_utterance_abs_tolerance(good_abs)
        slight_abs = short_utterance_abs_tolerance(slight_abs)

    # Gap-based fit: the only hard failures are overflowing the free window before the next
    # segment (hard_max) or leaving abnormally long silence (below soft_min). Anything that
    # sits inside [soft_min, hard_max] is acceptable even if it is longer than the original
    # speech duration, because the extra room is real free space, not an overlap.
    if hard_max > 0 and speech > hard_max:
        return "too_long"
    if soft_min > 0 and speech < soft_min:
        return "too_short"

    good_tol = _tolerance(target, ratio=policy.good_ratio, abs_sec=good_abs)
    if abs(speech - target) <= good_tol:
        return "good"

    # Within the allowed window but not exactly on target: report direction for telemetry
    # only. These are still acceptable (see acceptable_duration_fit) and must not force a
    # compress/stretch when the dub already fits the free space.
    if speech > target:
        return "slightly_long"
    return "slightly_short"


def acceptable_duration_fit(fit: str) -> bool:
    return fit in {"good", "slightly_short", "slightly_long"}


def tempo_factor_for_duration(current: float, target: float) -> float:
    if current <= 0 or target <= 0:
        return 1.0
    return max(0.5, min(2.0, current / target))


def clamp_automatic_tempo(
    factor: float,
    *,
    policy: DurationFitPolicy | None = None,
    user_global_speed: float = 1.0,
) -> tuple[float, str]:
    """Return (automatic_tempo, risk_label). User global speed is applied separately."""
    policy = policy or DurationFitPolicy()
    automatic = float(factor)
    if user_global_speed > 0 and abs(user_global_speed - 1.0) > 0.001:
        automatic = automatic / user_global_speed

    if policy.preferred_tempo_min <= automatic <= policy.preferred_tempo_max:
        return round(automatic, 3), "preferred"
    if policy.warning_tempo_min <= automatic <= policy.warning_tempo_max:
        return round(automatic, 3), "warning"
    return round(automatic, 3), "danger"


def classify_stretch_with_policy(
    factor: float,
    *,
    settings: dict[str, Any],
    explicit_allow_danger: bool = False,
) -> StretchDecision:
    policy = policy_from_settings(settings)
    max_safe = float(settings.get("exact_timing_max_safe_stretch", 1.12) or 1.12)
    max_safe = min(max_safe, policy.warning_tempo_max)
    return classify_stretch(factor, max_safe=max_safe, explicit_allow_danger=explicit_allow_danger)


def effective_timing_target(segment: dict[str, Any]) -> dict[str, float]:
    return timing_profile_from_segment(segment)


def should_lengthen_for_timing(
    speech_duration: float,
    segment: dict[str, Any],
    *,
    policy: DurationFitPolicy | None = None,
) -> bool:
    """Do not lengthen merely to fill timeline window trailing silence."""
    profile = timing_profile_from_segment(segment)
    fit = classify_duration_fit(speech_duration, profile, policy=policy)
    if fit in {"good", "slightly_short", "slightly_long"}:
        return False
    return fit == "too_short"


def should_shorten_for_timing(
    speech_duration: float,
    segment: dict[str, Any],
    *,
    policy: DurationFitPolicy | None = None,
) -> bool:
    profile = timing_profile_from_segment(segment)
    fit = classify_duration_fit(speech_duration, profile, policy=policy)
    # Only shorten when the dub actually overflows the free window before the next segment.
    # A "slightly_long" dub still fits and must be kept to preserve full content.
    return fit == "too_long"


def duration_fit_decision_trace(
    speech_duration: float,
    timing_profile: dict[str, float],
    *,
    policy: DurationFitPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or DurationFitPolicy()
    target = float(timing_profile.get("speech_target_duration") or 0.0)
    hard_max = float(timing_profile.get("hard_max_duration") or 0.0)
    classification = classify_duration_fit(speech_duration, timing_profile, policy=policy)
    tolerance_ms = round(
        _tolerance(target, ratio=policy.good_ratio, abs_sec=policy.good_abs_sec) * 1000
    )
    action = "accept_without_rewrite"
    if classification == "too_long":
        action = "candidate_retry_or_shorten"
    elif classification == "too_short":
        action = "accept_if_silence_valid"
    elif classification in {"slightly_long", "slightly_short"}:
        action = "accept_without_rewrite"
    return {
        "classification": classification,
        "speech_duration": round(float(speech_duration), 3),
        "speech_target": round(target, 3),
        "hard_max": round(hard_max, 3),
        "tolerance_ms": tolerance_ms,
        "action": action,
    }
