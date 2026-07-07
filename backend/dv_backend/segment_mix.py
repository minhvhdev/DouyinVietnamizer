"""FFmpeg filter helpers for smooth narration segment boundaries."""

from __future__ import annotations

from typing import Any

SEGMENT_FADE_IN_SEC = 0.012
SEGMENT_FADE_OUT_SEC = 0.028
SEGMENT_BOUNDARY_MARGIN_SEC = 0.025
AMIX_DROPOUT_TRANSITION_SEC = 0.04


def scaled_segment_fades(
    duration_sec: float,
    *,
    fade_in: float = SEGMENT_FADE_IN_SEC,
    fade_out: float = SEGMENT_FADE_OUT_SEC,
) -> tuple[float, float, float]:
    """Return fade-in duration, fade-out duration, and fade-out start time."""
    duration = max(0.05, float(duration_sec))
    total_fade = fade_in + fade_out
    if duration < total_fade * 1.5:
        scale = duration / (total_fade * 1.5)
        fade_in *= scale
        fade_out *= scale
    fade_out_start = max(0.0, duration - fade_out)
    return fade_in, fade_out, fade_out_start


def annotate_segment_mix_caps(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cap each segment so it cannot bleed into the next placement slot."""
    ordered = sorted(entries, key=lambda item: float(item["placement_start"]))
    for index, entry in enumerate(ordered):
        if index + 1 < len(ordered):
            gap = float(ordered[index + 1]["placement_start"]) - float(entry["placement_start"])
            entry["max_duration"] = max(0.05, gap - SEGMENT_BOUNDARY_MARGIN_SEC)
        else:
            entry["max_duration"] = None
    return ordered


def effective_clip_duration(clip_duration: float, max_duration: float | None) -> float:
    duration = max(0.05, float(clip_duration))
    if max_duration is not None:
        duration = min(duration, max(0.05, float(max_duration)))
    return duration


def build_narration_segment_filter(
    input_index: int,
    *,
    placement_start: float,
    clip_duration: float,
    max_duration: float | None,
) -> str:
    """Build per-segment resample/trim/fade/delay chain for narration mixing."""
    effective = effective_clip_duration(clip_duration, max_duration)
    fade_in, fade_out, fade_out_start = scaled_segment_fades(effective)
    delay_ms = max(0, round(float(placement_start) * 1000))
    label = f"seg{input_index}"

    filters = [
        "aresample=48000",
        "aformat=sample_fmts=s16:channel_layouts=stereo",
    ]
    if max_duration is not None and float(clip_duration) > effective + 0.001:
        filters.append(f"atrim=0:{effective:.3f}")
    filters.extend(
        [
            f"afade=t=in:st=0:d={fade_in:.3f}",
            f"afade=t=out:st={fade_out_start:.3f}:d={fade_out:.3f}",
            f"adelay={delay_ms}:all=1",
        ]
    )
    return f"[{input_index}:a]{','.join(filters)}[{label}]"


def build_narration_amix_filter(input_count: int) -> str:
    labels = "".join(f"[seg{index}]" for index in range(input_count))
    return (
        f"{labels}amix=inputs={input_count}:duration=longest:"
        f"dropout_transition={AMIX_DROPOUT_TRANSITION_SEC:.3f}:normalize=0[narration]"
    )
