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


# --- OmniVoice clone/auto diagnostics (read-only metrics) ---

import hashlib
import math
from typing import Any, Sequence

DIAG_ENV = "DV_OMNIVOICE_DIAGNOSTICS"
DIAG_DIR_ENV = "DV_OMNIVOICE_DIAGNOSTICS_DIR"
CAPTURE_INPUTS_ENV = "DV_OMNIVOICE_DIAGNOSTICS_CAPTURE_INPUTS"
_NEAR_ZERO = 1e-6
_SPEECH_ABS_THRESHOLD = 0.01


def diagnostics_enabled() -> bool:
    value = str(os.environ.get(DIAG_ENV, "") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def capture_inputs_enabled() -> bool:
    value = str(os.environ.get(CAPTURE_INPUTS_ENV, "") or "").strip().lower()
    return diagnostics_enabled() and value in {"1", "true", "yes", "on"}


def diagnostics_dir(default: Path | None = None) -> Path:
    configured = str(os.environ.get(DIAG_DIR_ENV, "") or "").strip()
    if configured:
        path = Path(configured)
    elif default is not None:
        path = default
    else:
        path = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".") / "omnivoice_diagnostics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def short_hash(value: str | bytes | None, *, length: int = 12) -> str:
    raw = value if isinstance(value, bytes) else str(value or "").encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()[: max(4, int(length))]


def file_content_hash(path: str | Path | None, *, length: int = 12) -> str | None:
    if not path:
        return None
    target = Path(path)
    if not target.is_file():
        return None
    try:
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()[: max(4, int(length))]
    except OSError:
        return None


def probe_waveform(
    samples: Sequence[float] | Any,
    *,
    sample_rate: int,
    channels: int = 1,
) -> dict[str, Any]:
    """Probe an in-memory mono/interleaved float waveform in [-1, 1] range."""
    try:
        rate = max(1, int(sample_rate or 1))
        channel_count = max(1, int(channels or 1))
        values = [float(x) for x in list(samples)]
        if not values:
            return _empty_wav_probe(sample_rate=rate, channels=channel_count, error="empty_waveform")
        frame_count = len(values) // channel_count
        if channel_count > 1 and frame_count > 0:
            mono: list[float] = []
            for index in range(frame_count):
                start = index * channel_count
                frame = values[start : start + channel_count]
                mono.append(sum(frame) / len(frame))
            values = mono
        peak = max(abs(v) for v in values)
        mean_sq = sum(v * v for v in values) / len(values)
        rms = math.sqrt(mean_sq)
        nonzero = sum(1 for v in values if abs(v) > _NEAR_ZERO) / len(values)
        duration = len(values) / float(rate)
        speech = bool(rms >= 0.002 and duration > 0.05)
        return {
            "ok": True,
            "duration_sec": round(duration, 6),
            "sample_rate": rate,
            "channels": channel_count,
            "peak_abs": round(peak, 8),
            "rms": round(rms, 8),
            "nonzero_ratio": round(nonzero, 6),
            "speech_detected": speech,
            "suspect": _is_suspect_wav(duration=duration, peak_abs=peak, rms=rms, speech_detected=speech),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 — diagnostics must never raise to callers
        return _empty_wav_probe(sample_rate=int(sample_rate or 0), channels=int(channels or 1), error=str(exc))


def probe_wav_path(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return _empty_wav_probe(error="missing_path")
    target = Path(path)
    if not target.is_file():
        return _empty_wav_probe(error="missing_file", path=str(target))
    try:
        from .final_dub_alignment import wav_has_detectable_speech

        with wave.open(str(target), "rb") as handle:
            sample_rate = int(handle.getframerate() or 0)
            channels = int(handle.getnchannels() or 1)
            sample_width = int(handle.getsampwidth() or 0)
            frame_count = int(handle.getnframes() or 0)
            raw = handle.readframes(frame_count)
        if sample_rate <= 0 or frame_count <= 0 or not raw:
            return _empty_wav_probe(
                sample_rate=sample_rate,
                channels=channels,
                error="empty_wav",
                path=str(target),
            )
        if sample_width != 2:
            duration = frame_count / float(sample_rate)
            speech = False
            try:
                speech = wav_has_detectable_speech(target)
            except Exception:  # noqa: BLE001
                speech = False
            return {
                "ok": True,
                "duration_sec": round(duration, 6),
                "sample_rate": sample_rate,
                "channels": channels,
                "peak_abs": None,
                "rms": None,
                "nonzero_ratio": None,
                "speech_detected": bool(speech),
                "suspect": duration <= 0.05 or not speech,
                "error": f"unsupported_sample_width:{sample_width}",
                "path": str(target),
                "file_size": target.stat().st_size,
            }

        total = 0.0
        peak = 0
        nonzero = 0
        count = len(raw) // 2
        for index in range(0, len(raw) - 1, 2):
            sample = int.from_bytes(raw[index : index + 2], "little", signed=True)
            abs_sample = abs(sample)
            if abs_sample > peak:
                peak = abs_sample
            total += sample * sample
            if abs_sample > 0:
                nonzero += 1
        peak_abs = peak / 32768.0
        rms = math.sqrt(total / count) / 32768.0 if count else 0.0
        duration = frame_count / float(sample_rate)
        try:
            speech = wav_has_detectable_speech(target)
        except Exception:  # noqa: BLE001
            speech = bool(rms >= 0.002 and duration > 0.05)
        envelope = _pcm16_envelope_metrics(raw, sample_rate=sample_rate, channels=channels)
        payload = {
            "ok": True,
            "duration_sec": round(duration, 6),
            "sample_rate": sample_rate,
            "channels": channels,
            "peak_abs": round(peak_abs, 8),
            "rms": round(rms, 8),
            "nonzero_ratio": round(nonzero / count, 6) if count else 0.0,
            "speech_detected": bool(speech),
            "suspect": _is_suspect_wav(
                duration=duration,
                peak_abs=peak_abs,
                rms=rms,
                speech_detected=bool(speech),
            ),
            "error": None,
            "path": str(target),
            "file_size": target.stat().st_size,
            "dtype": "pcm_s16le",
        }
        payload.update(envelope)
        return payload
    except Exception as exc:  # noqa: BLE001
        return _empty_wav_probe(error=str(exc), path=str(target))


def _is_suspect_wav(*, duration: float, peak_abs: float, rms: float, speech_detected: bool) -> bool:
    return (
        duration <= 0.05
        or peak_abs <= _NEAR_ZERO
        or rms <= _NEAR_ZERO
        or not speech_detected
    )


def _pcm16_envelope_metrics(raw: bytes, *, sample_rate: int, channels: int) -> dict[str, Any]:
    if sample_rate <= 0 or not raw:
        return {}
    threshold = int(_SPEECH_ABS_THRESHOLD * 32768)
    frame_count = (len(raw) // 2) // max(1, channels)
    voiced = [False] * frame_count
    clipped = 0
    dc_sum = 0.0
    sample_count = 0
    for frame_index in range(frame_count):
        peak = 0
        for channel in range(channels):
            offset = (frame_index * channels + channel) * 2
            sample = int.from_bytes(raw[offset : offset + 2], "little", signed=True)
            abs_sample = abs(sample)
            if abs_sample > peak:
                peak = abs_sample
            if abs_sample >= 32767:
                clipped += 1
            dc_sum += sample
            sample_count += 1
        voiced[frame_index] = peak >= threshold
    leading = 0
    while leading < frame_count and not voiced[leading]:
        leading += 1
    trailing = 0
    index = frame_count - 1
    while index >= 0 and not voiced[index]:
        trailing += 1
        index -= 1
    speech_frames = sum(1 for flag in voiced if flag)
    duration = frame_count / float(sample_rate)
    speech_duration = speech_frames / float(sample_rate)
    return {
        "leading_silence_sec": round(leading / float(sample_rate), 6),
        "trailing_silence_sec": round(trailing / float(sample_rate), 6),
        "detectable_speech_duration_sec": round(speech_duration, 6),
        "silence_ratio": round(1.0 - (speech_frames / frame_count), 6) if frame_count else 1.0,
        "clipped_sample_ratio": round(clipped / sample_count, 6) if sample_count else 0.0,
        "dc_offset": round((dc_sum / sample_count) / 32768.0, 8) if sample_count else 0.0,
        "sample_width_bits": 16,
        "duration_sec_from_envelope": round(duration, 6),
    }


def probe_text_metrics(text: str | None) -> dict[str, Any]:
    value = str(text or "")
    normalized = " ".join(value.replace("\ufeff", "").split())
    words = [part for part in normalized.split(" ") if part]
    control_chars = sum(1 for ch in value if ord(ch) < 32 and ch not in "\t\n\r")
    return {
        "ref_text_chars": len(value),
        "ref_text_words": len(words),
        "ref_text_normalized_sha256": short_hash(normalized, length=16),
        "ref_text_unicode_form": "NFC",
        "ref_text_contains_bom": value.startswith("\ufeff") or "\ufeff" in value,
        "ref_text_contains_control_chars": control_chars > 0,
        "ref_text_control_char_count": control_chars,
    }


def _empty_wav_probe(
    *,
    sample_rate: int = 0,
    channels: int = 0,
    error: str | None = None,
    path: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "duration_sec": 0.0,
        "sample_rate": sample_rate,
        "channels": channels,
        "peak_abs": 0.0,
        "rms": 0.0,
        "nonzero_ratio": 0.0,
        "speech_detected": False,
        "suspect": True,
        "error": error,
    }
    if path is not None:
        payload["path"] = path
    return payload
