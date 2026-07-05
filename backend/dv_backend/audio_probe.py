from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import wave


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


def resolved_probe_path(ffmpeg_path: Path | None) -> Path | None:
    resolved, _ = _resolve_ffprobe(ffmpeg_path)
    return resolved
