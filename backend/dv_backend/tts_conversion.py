from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TtsConversionResult:
    strategy: str
    fallback_reason: str | None
    process_count: int
    wall_time_ms: int
    inputs: int


def conversion_strategy_from_settings(settings: dict) -> str:
    """Return the chosen conversion strategy. Default to lazy_mix for lower TTS conversion overhead."""
    raw = str(settings.get("tts_conversion_strategy", "lazy_mix") or "lazy_mix").strip().lower()
    if raw not in {"per_segment", "lazy_mix"}:
        return "lazy_mix"
    return raw


def describe(result: TtsConversionResult) -> dict:
    return {
        "conversion_strategy": result.strategy,
        "conversion_input_count": result.inputs,
        "conversion_wall_time_ms": result.wall_time_ms,
        "conversion_process_count": result.process_count,
        "conversion_fallback_reason": result.fallback_reason,
    }
