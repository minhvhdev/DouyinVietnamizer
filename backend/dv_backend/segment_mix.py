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
    """Annotate next-slot room for telemetry; does not imply hard clipping."""
    ordered = sorted(entries, key=lambda item: float(item["placement_start"]))
    for index, entry in enumerate(ordered):
        if index + 1 < len(ordered):
            gap = float(ordered[index + 1]["placement_start"]) - float(entry["placement_start"])
            entry["max_duration"] = max(0.05, gap - SEGMENT_BOUNDARY_MARGIN_SEC)
        else:
            entry["max_duration"] = None
        clip = max(0.05, float(entry.get("clip_duration") or 0.0))
        cap = entry.get("max_duration")
        if cap is not None and clip > float(cap) + 0.001:
            entry["mix_would_clip_sec"] = round(clip - float(cap), 3)
        else:
            entry["mix_would_clip_sec"] = 0.0
    return ordered


def effective_clip_duration(
    clip_duration: float,
    max_duration: float | None,
    *,
    allow_hard_clip: bool = False,
) -> float:
    """Return playable duration. Hard clip only when explicitly allowed (legacy)."""
    duration = max(0.05, float(clip_duration))
    if allow_hard_clip and max_duration is not None:
        duration = min(duration, max(0.05, float(max_duration)))
    return duration


def build_narration_segment_filter(
    input_index: int,
    *,
    placement_start: float,
    clip_duration: float,
    max_duration: float | None,
    allow_hard_clip: bool = False,
) -> str:
    """Build per-segment resample/fade/delay chain for narration mixing.

    Default policy (ChatGPT TL): never silently atrim voiced audio to fit the next slot.
    Soft placement / speed / compact must resolve overflow before mix.
    """
    effective = effective_clip_duration(
        clip_duration,
        max_duration,
        allow_hard_clip=allow_hard_clip,
    )
    fade_in, fade_out, fade_out_start = scaled_segment_fades(effective)
    delay_ms = max(0, round(float(placement_start) * 1000))
    label = f"seg{input_index}"

    filters = [
        "aresample=48000",
        "aformat=sample_fmts=s16:channel_layouts=stereo",
    ]
    if (
        allow_hard_clip
        and max_duration is not None
        and float(clip_duration) > effective + 0.001
    ):
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


def format_mix_target_duration(target_duration_sec: float) -> str:
    """High-precision duration string for ffmpeg atrim (avoid coarse :.1f truncation)."""
    target = float(target_duration_sec)
    if not (target > 0) or target == float("inf"):
        raise ValueError(f"target_duration_sec must be a finite positive number, got {target_duration_sec!r}")
    return f"{target:.6f}"


def duration_lock_audio_chain(target_duration_sec: float) -> str:
    """Pad with silence then trim so a stream is exactly the video target length."""
    target = format_mix_target_duration(target_duration_sec)
    return f"apad,atrim=0:{target},asetpts=PTS-STARTPTS"


def build_background_narration_mix_filter(
    *,
    duck: bool,
    target_duration_sec: float,
) -> str:
    """Build bg+narration mix graph locked to video stream duration.

    Both inputs are pad/trimmed to the target *before* amix so a short extracted
    WAV cannot decide final length via ``duration=first``.
    """
    lock = duration_lock_audio_chain(target_duration_sec)
    bg = (
        f"[0:a]loudnorm=I=-24:TP=-4:LRA=7,alimiter=limit=0.72,{lock}[bg];"
        f"[1:a]loudnorm=I=-16:TP=-1.5:LRA=7,alimiter=limit=0.96,{lock}[fg];"
    )
    if duck:
        return (
            bg
            + "[fg]asplit=2[fg1][fg2];"
            + "[bg][fg1]sidechaincompress=threshold=0.015:ratio=12:"
            + "attack=12:release=350[ducked];"
            + "[ducked][fg2]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[mixed]"
        )
    return bg + "[bg][fg]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[mixed]"
