from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import wave

logger = logging.getLogger(__name__)


def _candidate_names(ffmpeg_path: Path | None) -> list[Path]:
    candidates: list[Path] = []
    env_path = os.environ.get("DV_FFPROBE_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    if ffmpeg_path is not None and ffmpeg_path.name:
        stem = ffmpeg_path.stem
        suffix = ffmpeg_path.suffix
        sibling = ffmpeg_path.with_name(f"ffprobe{suffix}")
        candidates.append(sibling)
        if ffmpeg_path.parent:
            candidates.append(ffmpeg_path.parent / "ffprobe")
    vendor_dir = os.environ.get("DV_VENDOR_DIR", "").strip()
    if vendor_dir:
        root = Path(vendor_dir).expanduser()
        candidates.append(root / "ffprobe.exe")
        candidates.append(root / "ffmpeg" / "ffprobe.exe")
        candidates.append(root / "ffprobe")
    candidates.append(Path("ffprobe"))
    return candidates


def _resolve_ffprobe(ffmpeg_path: Path | None) -> tuple[Path | None, str | None]:
    for candidate in _candidate_names(ffmpeg_path):
        try:
            resolved = candidate.expanduser().resolve() if candidate.is_absolute() else shutil.which(str(candidate))
        except OSError:
            resolved = None
        target = resolved if resolved is not None else candidate
        if not target:
            continue
        if Path(target).is_file():
            return Path(target), str(candidate)
    return None, None


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        return frames / float(rate)


def get_audio_duration(
    path: Path,
    *,
    ffprobe_path: Path | None = None,
    timeout: float = 10.0,
) -> float:
    resolved, source = (ffprobe_path, "explicit") if ffprobe_path is not None else _resolve_ffprobe(None)
    if resolved is None:
        return wav_duration(path)
    try:
        result = subprocess.run(
            [
                str(resolved),
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=True,
        )
        payload = json.loads(result.stdout or "{}")
        duration = float((payload.get("format") or {}).get("duration") or 0.0)
        if duration > 0:
            return duration
    except Exception:
        pass
    return wav_duration(path)


def _parse_positive_duration(value: object) -> float | None:
    if value is None or value == "N/A":
        return None
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    if duration > 0 and duration != float("inf"):
        return duration
    return None


def get_video_stream_duration(
    path: Path,
    *,
    ffprobe_path: Path | None = None,
    ffmpeg_path: Path | None = None,
    timeout: float = 10.0,
) -> float:
    """Return duration of the first video stream (v:0). Never use audio stream duration.

    Fallback order:
      1) streams[v:0].duration
      2) streams[v:0].duration_ts * time_base
      3) format.duration (only when stream-level duration is unavailable)
    """
    resolved = ffprobe_path
    source = "explicit"
    if resolved is None:
        resolved, source = _resolve_ffprobe(ffmpeg_path)
    if resolved is None:
        raise RuntimeError(f"ffprobe not found while probing video duration for {path}")

    try:
        result = subprocess.run(
            [
                str(resolved),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=duration,duration_ts,time_base:format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=True,
        )
    except Exception as exc:
        raise RuntimeError(f"ffprobe failed for video duration ({path}): {exc}") from exc

    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found while probing duration for {path}")

    stream = streams[0]
    duration = _parse_positive_duration(stream.get("duration"))
    if duration is not None:
        logger.debug("video duration from stream.duration=%.6f path=%s", duration, path)
        return duration

    duration_ts = stream.get("duration_ts")
    time_base = stream.get("time_base")
    if duration_ts is not None and time_base:
        try:
            num_s, den_s = str(time_base).split("/", 1)
            num = float(num_s)
            den = float(den_s)
            if den != 0:
                duration = float(duration_ts) * (num / den)
                if duration > 0 and duration != float("inf"):
                    logger.debug(
                        "video duration from duration_ts*time_base=%.6f path=%s",
                        duration,
                        path,
                    )
                    return duration
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    duration = _parse_positive_duration((payload.get("format") or {}).get("duration"))
    if duration is not None:
        logger.info(
            "video duration fallback to format.duration=%.6f path=%s ffprobe=%s",
            duration,
            path,
            source,
        )
        return duration

    raise RuntimeError(
        f"Unable to resolve positive video stream duration for {path} (ffprobe={source})"
    )


def resolved_probe_path(ffmpeg_path: Path | None) -> Path | None:
    resolved, _ = _resolve_ffprobe(ffmpeg_path)
    return resolved
