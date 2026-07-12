"""Subtitle cue timing: one display chunk at a time, optional TTS ASR alignment."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import wave
from pathlib import Path
from typing import Any, Callable

from .adapters.vad_silencedetect import detect_speech_regions_silencedetect, silencedetect_filter
from .segment_mix import annotate_segment_mix_caps, effective_clip_duration

logger = logging.getLogger(__name__)

SUBTITLE_MAX_CHARS_PER_CUE = 58
SUBTITLE_MIN_DURATION_FOR_TTS_ASR = 2.5
SUBTITLE_MIN_CHUNKS_FOR_TTS_ASR = 2
SUBTITLE_MIN_CUE_DURATION = 0.25
SUBTITLE_SPEECH_DETECT_NOISE_DB = -40.0
SUBTITLE_SPEECH_DETECT_MIN_SILENCE_SEC = 0.25

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…。！？;])\s*")
_COMMA_SPLIT_RE = re.compile(r"\s*,\s*")


def segment_subtitle_start(segment: dict) -> float:
    placement = segment.get("placement_start")
    if placement is not None:
        return float(placement)
    return float(segment["start"])


def segment_subtitle_end(segment: dict) -> float:
    start = segment_subtitle_start(segment)
    playback = segment.get("subtitle_playback_duration")
    if playback is not None:
        return start + float(playback)
    repaired_duration = segment.get("repaired_duration")
    if repaired_duration is not None:
        return start + float(repaired_duration)
    if segment.get("end") is not None:
        return float(segment["end"])
    return start + float(segment.get("duration_budget", 1.0))


def annotate_subtitle_playback_windows(segments: list[dict[str, Any]]) -> None:
    """Cap subtitle windows to match narration mix clip boundaries."""
    ordered = sorted(segments, key=lambda item: int(item.get("index", 0) or 0))
    entries: list[dict[str, Any]] = []
    for segment in ordered:
        entries.append(
            {
                "segment": segment,
                "placement_start": segment_subtitle_start(segment),
                "clip_duration": float(
                    segment.get("repaired_duration")
                    or segment.get("tts_duration")
                    or 0.0
                ),
            }
        )
    annotate_segment_mix_caps(entries)
    for entry in entries:
        segment = entry["segment"]
        effective = effective_clip_duration(entry["clip_duration"], entry.get("max_duration"))
        segment["subtitle_playback_duration"] = round(effective, 3)


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def _clip_aligned_units(
    units: list[dict[str, Any]],
    *,
    max_duration: float,
) -> list[dict[str, Any]]:
    if max_duration <= 0:
        return []
    clipped: list[dict[str, Any]] = []
    for unit in units:
        text = str(unit.get("text") or "").strip()
        if not text:
            continue
        try:
            start = max(0.0, float(unit.get("start", 0.0) or 0.0))
            end = max(start, float(unit.get("end", start) or start))
        except (TypeError, ValueError):
            continue
        if start >= max_duration:
            continue
        end = min(end, max_duration)
        if end <= start:
            end = min(max_duration, start + 0.05)
        clipped.append({"text": text, "start": round(start, 3), "end": round(end, 3)})
    return clipped


def _speech_duration_from_units(units: list[dict[str, Any]], wav_duration: float) -> float | None:
    if not units or wav_duration <= 0:
        return None
    try:
        speech_end = max(float(unit.get("end", 0.0) or 0.0) for unit in units)
    except (TypeError, ValueError):
        return None
    speech_end = min(max(speech_end, SUBTITLE_MIN_CUE_DURATION), wav_duration)
    if speech_end <= SUBTITLE_MIN_CUE_DURATION:
        return None
    return speech_end


def _detect_wav_speech_duration(wav_path: Path, ffmpeg_path: Path) -> float | None:
    duration = _wav_duration(wav_path)
    if duration <= SUBTITLE_MIN_CUE_DURATION:
        return None
    cmd = [
        str(ffmpeg_path),
        "-hide_banner",
        "-i",
        str(wav_path),
        "-af",
        silencedetect_filter(SUBTITLE_SPEECH_DETECT_NOISE_DB, SUBTITLE_SPEECH_DETECT_MIN_SILENCE_SEC),
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError:
        return None
    regions = detect_speech_regions_silencedetect(
        wav_path,
        total_duration=duration,
        stderr=f"{completed.stdout}\n{completed.stderr}",
    )
    if not regions:
        return None
    speech_end = max(float(region.get("end", 0.0) or 0.0) for region in regions)
    speech_end = min(max(speech_end, SUBTITLE_MIN_CUE_DURATION), duration)
    return speech_end if speech_end > SUBTITLE_MIN_CUE_DURATION else None


def resolve_subtitle_speech_window(
    *,
    window_start: float,
    window_end: float,
    wav_path: Path | None,
    ffmpeg_path: Path | None,
    aligned_units: list[dict[str, Any]] | None = None,
) -> tuple[float, float]:
    """Return [start, end) for subtitle timing based on speech, not tail silence pad."""
    start = float(window_start)
    end = float(window_end)
    full_duration = max(SUBTITLE_MIN_CUE_DURATION, end - start)
    if wav_path is None or not wav_path.is_file():
        return start, end

    wav_duration = _wav_duration(wav_path)
    speech_duration = _speech_duration_from_units(aligned_units or [], wav_duration)
    if speech_duration is None and ffmpeg_path is not None and ffmpeg_path.is_file():
        speech_duration = _detect_wav_speech_duration(wav_path, ffmpeg_path)

    if speech_duration is None:
        return start, end

    # Map speech inside the clip file onto the segment placement window.
    if wav_duration > SUBTITLE_MIN_CUE_DURATION:
        speech_duration = min(speech_duration, wav_duration)
        scaled = full_duration * (speech_duration / wav_duration)
    else:
        scaled = speech_duration
    scaled = max(SUBTITLE_MIN_CUE_DURATION, min(full_duration, scaled))
    return start, start + scaled


def _units_cache_path(cache_dir: Path, wav_path: Path) -> Path:
    return cache_dir / f"{wav_path.stem}_units.json"


def _load_cached_units(cache_dir: Path, wav_path: Path) -> list[dict[str, Any]] | None:
    cache_path = _units_cache_path(cache_dir, wav_path)
    if not cache_path.is_file() or cache_path.stat().st_mtime < wav_path.stat().st_mtime:
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    units = payload.get("aligned_units")
    if not isinstance(units, list):
        return None
    return units


def _store_cached_units(cache_dir: Path, wav_path: Path, units: list[dict[str, Any]]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _units_cache_path(cache_dir, wav_path)
    cache_path.write_text(
        json.dumps({"aligned_units": units}, ensure_ascii=False),
        encoding="utf-8",
    )


def split_translation_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    cleaned = re.sub(r"\s*\n+\s*", ". ", cleaned)
    if not cleaned:
        return []
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(cleaned) if part.strip()]
    return parts or [cleaned]


def _split_on_commas(text: str) -> list[str]:
    parts = [part.strip() for part in _COMMA_SPLIT_RE.split(text) if part.strip()]
    if len(parts) <= 1:
        return parts
    merged: list[str] = []
    buffer = ""
    for part in parts:
        candidate = f"{buffer}, {part}".strip(", ") if buffer else part
        if len(candidate) <= SUBTITLE_MAX_CHARS_PER_CUE or not buffer:
            buffer = candidate
            continue
        merged.append(buffer)
        buffer = part
    if buffer:
        merged.append(buffer)
    return merged


def _split_word_chunks(text: str, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return [text.strip()] if text.strip() else []
    chunks: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = word
    if current:
        chunks.append(current)
    return chunks


def split_for_subtitle_display(text: str) -> list[str]:
    """Split translation into chunks shown one at a time on screen."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return []

    sentences = split_translation_sentences(cleaned)
    chunks: list[str] = []
    for sentence in sentences:
        if len(sentence) <= SUBTITLE_MAX_CHARS_PER_CUE:
            chunks.append(sentence)
            continue
        comma_parts = _split_on_commas(sentence)
        if len(comma_parts) > 1:
            chunks.extend(comma_parts)
            continue
        chunks.extend(_split_word_chunks(sentence, SUBTITLE_MAX_CHARS_PER_CUE))

    if not chunks:
        return [cleaned]
    return [chunk for chunk in chunks if chunk.strip() and chunk.strip() != "."]


def _chunk_weights(chunks: list[str]) -> list[int]:
    return [max(1, len(chunk.replace(" ", ""))) for chunk in chunks]


def allocate_proportional_cues(chunks: list[str], start: float, end: float) -> list[dict[str, Any]]:
    if not chunks:
        return []
    window_start = float(start)
    window_end = float(end)
    duration = max(SUBTITLE_MIN_CUE_DURATION, window_end - window_start)
    if len(chunks) == 1:
        return [{"start": window_start, "end": window_end, "text": chunks[0]}]

    weights = _chunk_weights(chunks)
    total_weight = sum(weights)
    cursor = window_start
    cues: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        portion = duration * weights[index] / total_weight
        cue_end = window_end if index == len(chunks) - 1 else cursor + portion
        cue_end = max(cursor + SUBTITLE_MIN_CUE_DURATION, cue_end)
        cues.append({"start": cursor, "end": cue_end, "text": chunk})
        cursor = cue_end
    cues[-1]["end"] = window_end
    return cues


def _normalize_match_text(text: str) -> str:
    return re.sub(r"[\s\d\W_]+", "", text.lower(), flags=re.UNICODE)


def map_chunks_to_asr_timeline(
    chunks: list[str],
    aligned_units: list[dict[str, Any]],
    *,
    window_start: float,
    window_duration: float,
) -> list[dict[str, Any]] | None:
    if not chunks or not aligned_units or window_duration <= 0:
        return None

    unit_spans: list[tuple[int, int, float, float]] = []
    cursor = 0
    for unit in aligned_units:
        text = _normalize_match_text(str(unit.get("text") or ""))
        if not text:
            continue
        try:
            unit_start = float(unit.get("start", 0.0) or 0.0)
            unit_end = float(unit.get("end", unit_start) or unit_start)
        except (TypeError, ValueError):
            continue
        if unit_end <= unit_start:
            unit_end = unit_start + 0.05
        unit_spans.append((cursor, cursor + len(text), unit_start, unit_end))
        cursor += len(text)

    if not unit_spans:
        return None

    chunk_spans: list[tuple[int, int]] = []
    cursor = 0
    for chunk in chunks:
        norm = _normalize_match_text(chunk)
        length = max(1, len(norm))
        chunk_spans.append((cursor, cursor + length))
        cursor += length

    total_chars = max(1, cursor)
    cues: list[dict[str, Any]] = []
    for chunk, (char_start, char_end) in zip(chunks, chunk_spans, strict=True):
        rel_start = char_start / total_chars
        rel_end = char_end / total_chars

        matching_units = [
            span
            for span in unit_spans
            if span[1] > char_start and span[0] < char_end
        ]
        if matching_units:
            t_start = matching_units[0][2]
            t_end = matching_units[-1][3]
        else:
            t_start = rel_start * window_duration
            t_end = rel_end * window_duration

        abs_start = window_start + t_start
        abs_end = window_start + t_end
        if abs_end - abs_start < SUBTITLE_MIN_CUE_DURATION:
            abs_end = abs_start + SUBTITLE_MIN_CUE_DURATION
        cues.append({"start": abs_start, "end": abs_end, "text": chunk})

    if cues:
        cues[0]["start"] = window_start
        cues[-1]["end"] = window_start + window_duration
    return cues


def enforce_monotonic_cues(
    cues: list[dict[str, Any]],
    *,
    window_start: float,
    window_end: float,
) -> list[dict[str, Any]]:
    """Rescale ASR-aligned cues so they stay sequential inside one segment window."""
    if not cues:
        return []
    ordered = sorted(cues, key=lambda item: (float(item["start"]), float(item["end"])))
    window_start = float(window_start)
    window_end = float(window_end)
    window_duration = max(SUBTITLE_MIN_CUE_DURATION, window_end - window_start)
    if len(ordered) == 1:
        return [{"start": window_start, "end": window_end, "text": str(ordered[0]["text"])}]

    raw_durations = [
        max(SUBTITLE_MIN_CUE_DURATION, float(cue["end"]) - float(cue["start"]))
        for cue in ordered
    ]
    total_raw = sum(raw_durations)
    scale = window_duration / total_raw if total_raw > window_duration else 1.0
    cursor = window_start
    normalized: list[dict[str, Any]] = []
    for index, (cue, raw_duration) in enumerate(zip(ordered, raw_durations, strict=True)):
        duration = max(SUBTITLE_MIN_CUE_DURATION, raw_duration * scale)
        end = window_end if index == len(ordered) - 1 else min(window_end, cursor + duration)
        normalized.append({"start": cursor, "end": end, "text": str(cue["text"])})
        cursor = end
    normalized[-1]["end"] = window_end
    return normalized


def _asr_cues_are_usable(
    cues: list[dict[str, Any]],
    *,
    window_duration: float,
    chunk_count: int,
) -> bool:
    if not cues or window_duration <= 0 or chunk_count <= 0:
        return False
    durations = [float(cue["end"]) - float(cue["start"]) for cue in cues]
    expected_avg = window_duration / chunk_count
    rapid_limit = max(3, int(round(len(cues) * 0.35)))
    rapid_count = sum(
        1 for duration in durations if duration <= SUBTITLE_MIN_CUE_DURATION + 0.01
    )
    if rapid_count > rapid_limit:
        return False
    min_ratio = 0.2 if chunk_count >= 8 else 0.3
    if min(durations) < max(SUBTITLE_MIN_CUE_DURATION, expected_avg * min_ratio):
        return False
    return True


def resolve_overlapping_cues(cues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not cues:
        return []
    ordered = sorted(cues, key=lambda item: (float(item["start"]), float(item["end"])))
    resolved: list[dict[str, Any]] = []
    for cue in ordered:
        start = float(cue["start"])
        end = float(cue["end"])
        if end <= start:
            continue
        if resolved:
            prev = resolved[-1]
            prev_start = float(prev["start"])
            prev_end = float(prev["end"])
            if start < prev_end - 0.02:
                trimmed_end = max(prev_start + SUBTITLE_MIN_CUE_DURATION, start)
                if trimmed_end <= prev_start:
                    resolved.pop()
                else:
                    prev["end"] = trimmed_end
        if end <= start:
            end = start + SUBTITLE_MIN_CUE_DURATION
        resolved.append({"start": start, "end": end, "text": str(cue["text"])})
    return resolved


def _resolve_tts_wav(job_dir: Path, segment: dict) -> Path | None:
    tts_dir = job_dir / "artifacts" / "tts"
    idx = segment.get("index")
    if idx is None:
        return None
    repaired = tts_dir / f"tts_repaired_{idx}.wav"
    if repaired.is_file():
        return repaired
    for key in ("tts_path", "tts_raw_path"):
        candidate = segment.get(key)
        if candidate:
            path = Path(candidate)
            if path.is_file():
                return path
    fallback = tts_dir / f"tts_{idx}.wav"
    return fallback if fallback.is_file() else None


def _ensure_asr_audio(
    source: Path,
    target: Path,
    ffmpeg_path: Path,
    *,
    trim_duration: float | None = None,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ffmpeg_path),
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
    ]
    if trim_duration is not None and trim_duration > SUBTITLE_MIN_CUE_DURATION:
        cmd.extend(["-af", f"atrim=0:{trim_duration:.3f}"])
    cmd.extend(["-c:a", "pcm_s16le", str(target)])
    subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def _resolve_asr_device(settings: dict[str, Any]) -> str:
    from .hardware import resolve_inference_device

    return resolve_inference_device(str(settings.get("qwen3_device", "cuda:0") or "cuda:0"))


def transcribe_tts_clip_for_subtitles(
    wav_path: Path,
    *,
    vendor_dir: Path,
    settings: dict[str, Any],
    language: str,
    ffmpeg_path: Path,
    cache_dir: Path,
    transcribe_fn: Callable[..., Any],
) -> list[dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = _load_cached_units(cache_dir, wav_path)
    if cached is not None:
        wav_duration = _wav_duration(wav_path)
        return _clip_aligned_units(cached, max_duration=wav_duration)

    wav_duration = _wav_duration(wav_path)
    speech_duration = _detect_wav_speech_duration(wav_path, ffmpeg_path)
    trim_duration = None
    if speech_duration is not None and speech_duration < wav_duration * 0.85:
        trim_duration = speech_duration + 0.05

    prepared = cache_dir / f"{wav_path.stem}_16k.wav"
    if not prepared.is_file() or prepared.stat().st_mtime < wav_path.stat().st_mtime:
        _ensure_asr_audio(wav_path, prepared, ffmpeg_path, trim_duration=trim_duration)

    asr_kwargs = {
        "vendor_dir": vendor_dir,
        "asr_model": str(settings.get("qwen3_asr_model", "") or ""),
        "aligner_model": str(settings.get("qwen3_aligner_model", "") or ""),
        "device": _resolve_asr_device(settings),
        "language": language,
        "speaker_diarization": False,
        "include_alignment": True,
    }
    result = transcribe_fn(prepared, **asr_kwargs)
    units: list[dict[str, Any]] = []
    if isinstance(result, dict):
        units = list(result.get("aligned_units") or [])
        if not units:
            units = [
                {
                    "text": str(segment.get("text") or ""),
                    "start": float(segment.get("start", 0.0) or 0.0),
                    "end": float(segment.get("end", 0.0) or 0.0),
                }
                for segment in result.get("segments") or []
                if str(segment.get("text") or "").strip()
            ]
    clip_duration = trim_duration or wav_duration
    units = _clip_aligned_units(units, max_duration=clip_duration)
    _store_cached_units(cache_dir, wav_path, units)
    return units


def build_segment_subtitle_cues(
    segment: dict[str, Any],
    *,
    job_dir: Path | None,
    settings: dict[str, Any] | None,
    vendor_dir: Path | None,
    ffmpeg_path: Path | None,
    transcribe_fn: Callable[..., Any] | None,
    tts_asr_align: bool,
) -> list[dict[str, Any]]:
    translation = str(segment.get("translation") or "").strip()
    if not translation:
        return []

    chunks = split_for_subtitle_display(translation)
    window_start = segment_subtitle_start(segment)
    window_end = segment_subtitle_end(segment)
    wav_path = _resolve_tts_wav(job_dir, segment) if job_dir is not None else None
    speech_start, speech_end = resolve_subtitle_speech_window(
        window_start=window_start,
        window_end=window_end,
        wav_path=wav_path,
        ffmpeg_path=ffmpeg_path,
    )
    speech_duration = max(SUBTITLE_MIN_CUE_DURATION, speech_end - speech_start)

    should_try_asr = (
        tts_asr_align
        and len(chunks) >= SUBTITLE_MIN_CHUNKS_FOR_TTS_ASR
        and speech_duration >= SUBTITLE_MIN_DURATION_FOR_TTS_ASR
        and job_dir is not None
        and settings is not None
        and vendor_dir is not None
        and ffmpeg_path is not None
        and transcribe_fn is not None
    )
    if should_try_asr and wav_path is not None:
        try:
            from .dubbing_languages import dub_language_config, dub_language_from_settings

            language = dub_language_config(dub_language_from_settings(settings))["label_en"]
            cache_dir = job_dir / "artifacts" / "subtitle_asr"
            units = transcribe_tts_clip_for_subtitles(
                wav_path,
                vendor_dir=vendor_dir,
                settings=settings,
                language=language,
                ffmpeg_path=ffmpeg_path,
                cache_dir=cache_dir,
                transcribe_fn=transcribe_fn,
            )
            speech_start, speech_end = resolve_subtitle_speech_window(
                window_start=window_start,
                window_end=window_end,
                wav_path=wav_path,
                ffmpeg_path=ffmpeg_path,
                aligned_units=units,
            )
            speech_duration = max(SUBTITLE_MIN_CUE_DURATION, speech_end - speech_start)
            aligned = map_chunks_to_asr_timeline(
                chunks,
                units,
                window_start=speech_start,
                window_duration=speech_duration,
            )
            if aligned:
                normalized = enforce_monotonic_cues(
                    aligned,
                    window_start=speech_start,
                    window_end=speech_end,
                )
                if _asr_cues_are_usable(
                    normalized,
                    window_duration=speech_duration,
                    chunk_count=len(chunks),
                ):
                    return normalized
                logger.info(
                    "Subtitle ASR alignment low quality for segment %s; using proportional timing.",
                    segment.get("index"),
                )
        except Exception as exc:
            logger.warning(
                "Subtitle TTS ASR alignment failed for segment %s: %s",
                segment.get("index"),
                exc,
            )

    return allocate_proportional_cues(chunks, speech_start, speech_end)


def build_subtitle_cues(
    segments: list[dict],
    *,
    job_dir: Path | None = None,
    settings: dict[str, Any] | None = None,
    vendor_dir: Path | None = None,
    ffmpeg_path: Path | None = None,
    transcribe_fn: Callable[..., Any] | None = None,
    tts_asr_align: bool = False,
) -> list[dict[str, Any]]:
    annotate_subtitle_playback_windows(segments)
    cues: list[dict[str, Any]] = []
    for segment in segments:
        cues.extend(
            build_segment_subtitle_cues(
                segment,
                job_dir=job_dir,
                settings=settings,
                vendor_dir=vendor_dir,
                ffmpeg_path=ffmpeg_path,
                transcribe_fn=transcribe_fn,
                tts_asr_align=tts_asr_align,
            )
        )
    return resolve_overlapping_cues(cues)
