from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable


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


def _ffmpeg_path(config, pipeline_module) -> Path:
    resolve = getattr(pipeline_module, "resolve_tool_path", None)
    if resolve is None:
        raise RuntimeError("resolve_tool_path unavailable in pipeline module")
    return resolve(config, "ffmpeg")


def convert_segments_per_segment(
    config,
    pipeline_module,
    job_id,
    runner,
    segments: Iterable[dict],
    *,
    sample_rate: int = 48000,
    channels: int = 2,
    fallback_reason: str | None = None,
) -> TtsConversionResult:
    from .pipeline import _convert_tts_to_final_wav  # type: ignore[attr-defined]

    tts_dir = Path(config.data_dir) / "jobs" / job_id / "artifacts" / "tts"
    ffmpeg = _ffmpeg_path(config, pipeline_module)
    started = time.perf_counter()
    process_count = 0
    inputs = 0
    for segment in segments:
        idx = segment["index"]
        repaired = tts_dir / f"tts_repaired_{idx}.wav"
        if not repaired.is_file():
            repaired = tts_dir / f"tts_{idx}.wav"
        if not repaired.is_file():
            continue
        final = tts_dir / f"tts_{idx}.wav"
        if repaired.resolve() != final.resolve():
            final.unlink(missing_ok=True)
            _convert_tts_to_final_wav(ffmpeg, repaired, final, job_id, runner)
            process_count += 1
        inputs += 1
    return TtsConversionResult(
        strategy="per_segment",
        fallback_reason=fallback_reason,
        process_count=process_count,
        wall_time_ms=round((time.perf_counter() - started) * 1000),
        inputs=inputs,
    )


def convert_segments(
    config,
    pipeline_module,
    job_id,
    runner,
    segments: list[dict],
    *,
    settings: dict | None = None,
) -> TtsConversionResult:
    """Convert repaired TTS segments to canonical layout or defer conversion to mix."""
    strategy = conversion_strategy_from_settings(settings or {})
    if strategy == "per_segment":
        return convert_segments_per_segment(config, pipeline_module, job_id, runner, segments)
    # lazy_mix: defer conversion; the mix step will resample natively.
    return TtsConversionResult(
        strategy="lazy_mix",
        fallback_reason=None,
        process_count=0,
        wall_time_ms=0,
        inputs=len(list(segments)),
    )


def describe(result: TtsConversionResult) -> dict:
    return {
        "conversion_strategy": result.strategy,
        "conversion_input_count": result.inputs,
        "conversion_wall_time_ms": result.wall_time_ms,
        "conversion_process_count": result.process_count,
        "conversion_fallback_reason": result.fallback_reason,
    }
