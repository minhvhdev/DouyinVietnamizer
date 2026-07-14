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
SUBTITLE_MAX_LINES = 2
SUBTITLE_MAX_CHARS_PER_LINE = 40
SUBTITLE_DUB_MIN_CUE_DURATION_MS = 700
SUBTITLE_MAX_CUE_DURATION_MS = 5500
SUBTITLE_MIN_GAP_MS = 50
SUBTITLE_PAUSE_SPLIT_THRESHOLD_MS = 550
SUBTITLE_CUE_TAIL_PADDING_MS = 80
SUBTITLE_MIN_DURATION_FOR_TTS_ASR = 2.5
SUBTITLE_MIN_CHUNKS_FOR_TTS_ASR = 2
SUBTITLE_MIN_CUE_DURATION = 0.25
SUBTITLE_DUB_MIN_CUE_DURATION = SUBTITLE_DUB_MIN_CUE_DURATION_MS / 1000.0
SUBTITLE_MAX_CUE_DURATION = SUBTITLE_MAX_CUE_DURATION_MS / 1000.0
SUBTITLE_MIN_GAP = SUBTITLE_MIN_GAP_MS / 1000.0
SUBTITLE_PAUSE_SPLIT_THRESHOLD = SUBTITLE_PAUSE_SPLIT_THRESHOLD_MS / 1000.0
SUBTITLE_CUE_TAIL_PADDING = SUBTITLE_CUE_TAIL_PADDING_MS / 1000.0
SUBTITLE_SPEECH_DETECT_NOISE_DB = -40.0
SUBTITLE_SPEECH_DETECT_MIN_SILENCE_SEC = 0.25

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…。！？;])\s*")
_COMMA_SPLIT_RE = re.compile(r"\s*,\s*")


def subtitle_layout_from_settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve user-configurable subtitle layout limits (3.4) with legacy defaults."""
    s = settings or {}

    def _int(key: str, default: int) -> int:
        try:
            return int(s.get(key, default) or default)
        except (TypeError, ValueError):
            return default

    max_line = _int("subtitle_max_chars_per_line", SUBTITLE_MAX_CHARS_PER_LINE)
    max_lines = _int("subtitle_max_lines_per_cue", s.get("subtitle_max_lines") or SUBTITLE_MAX_LINES)
    return {
        "max_chars_per_cue": max(8, max_line * max(1, max_lines)),
        "min_cue_duration": _int("subtitle_min_cue_duration_ms", SUBTITLE_DUB_MIN_CUE_DURATION_MS) / 1000.0,
        "max_cue_duration": _int("subtitle_max_cue_duration_ms", SUBTITLE_MAX_CUE_DURATION_MS) / 1000.0,
        "min_gap": _int("subtitle_inter_cue_gap_ms", SUBTITLE_MIN_GAP_MS) / 1000.0,
    }


def _is_space_delimited_language(language: str | None) -> bool:
    """Thai (and other scriptio-continua languages) are not split on whitespace."""
    return str(language or "").lower() not in {"th", "tha", "thai"}


def _join_display_words(texts: list[str], language: str | None) -> str:
    parts = [part for part in (t.strip() for t in texts) if part]
    if not parts:
        return ""
    if _is_space_delimited_language(language):
        return " ".join(parts)
    return "".join(parts)


def _thai_word_tokens(text: str) -> list[str]:
    """Tokenize Thai text into words, preferring pythainlp when available."""
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    try:
        from pythainlp.tokenize import word_tokenize  # type: ignore

        tokens = [tok for tok in word_tokenize(cleaned, keep_whitespace=False) if tok.strip()]
        if tokens:
            return tokens
    except Exception:
        pass
    # Fallback: split on whitespace if present, else return the whole run so callers can
    # length-split by characters without breaking mid-cluster where possible.
    if " " in cleaned:
        return [tok for tok in cleaned.split() if tok]
    return [cleaned]


def segment_subtitle_start(segment: dict) -> float:
    placement = segment.get("placement_start")
    if placement is not None:
        return float(placement)
    start = segment.get("start")
    if start is not None:
        return float(start)
    return 0.0


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
        effective = effective_clip_duration(
            entry["clip_duration"],
            entry.get("max_duration"),
            allow_hard_clip=False,
        )
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


def _hard_split_oversized(words: list[str], max_chars: int) -> list[str]:
    """Character-split any single token longer than max_chars (last-resort hard break)."""
    limit = max(1, max_chars)
    expanded: list[str] = []
    for word in words:
        if len(word) > limit:
            expanded.extend(word[i : i + limit] for i in range(0, len(word), limit))
        else:
            expanded.append(word)
    return expanded


def _split_word_chunks(text: str, max_chars: int, *, language: str | None = None) -> list[str]:
    if _is_space_delimited_language(language):
        words = text.split()
        joiner = " "
    else:
        words = _thai_word_tokens(text)
        joiner = ""
    words = _hard_split_oversized(words, max_chars)
    if not words:
        return [text.strip()] if text.strip() else []
    chunks: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current}{joiner}{word}" if current else word
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = word
    if current:
        chunks.append(current)
    return chunks


def split_for_subtitle_display(
    text: str,
    *,
    max_chars: int = SUBTITLE_MAX_CHARS_PER_CUE,
    language: str | None = None,
) -> list[str]:
    """Split translation into chunks shown one at a time on screen."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return []

    sentences = split_translation_sentences(cleaned)
    chunks: list[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            chunks.append(sentence)
            continue
        comma_parts = _split_on_commas(sentence)
        if len(comma_parts) > 1 and all(len(part) <= max_chars for part in comma_parts):
            chunks.extend(comma_parts)
            continue
        chunks.extend(_split_word_chunks(sentence, max_chars, language=language))

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
    from .tts_provenance import resolve_voiced_tts_path

    del job_dir
    return resolve_voiced_tts_path(segment)


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


def transcribe_tts_clip_details_for_subtitles(
    wav_path: Path,
    *,
    vendor_dir: Path,
    settings: dict[str, Any],
    language: str,
    ffmpeg_path: Path,
    cache_dir: Path,
    transcribe_fn: Callable[..., Any],
) -> dict[str, Any]:
    """Transcribe a TTS clip once and return units plus raw segment metadata."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = _load_cached_units(cache_dir, wav_path)
    wav_duration = _wav_duration(wav_path)
    if cached is not None:
        return {
            "aligned_units": _clip_aligned_units(cached, max_duration=wav_duration),
            "segments": [],
            "from_forced_aligner": True,
        }

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
    segments: list[dict[str, Any]] = []
    from_forced = False
    if isinstance(result, dict):
        forced_units = list(result.get("aligned_units") or [])
        segments = list(result.get("segments") or [])
        if forced_units:
            units = forced_units
            from_forced = True
        elif segments:
            units = [
                {
                    "text": str(segment.get("text") or ""),
                    "start": float(segment.get("start", 0.0) or 0.0),
                    "end": float(segment.get("end", 0.0) or 0.0),
                }
                for segment in segments
                if str(segment.get("text") or "").strip()
            ]
    clip_duration = trim_duration or wav_duration
    units = _clip_aligned_units(units, max_duration=clip_duration)
    _store_cached_units(cache_dir, wav_path, units)
    return {
        "aligned_units": units,
        "segments": segments,
        "from_forced_aligner": from_forced,
    }


def _word_display_text(word: dict[str, Any]) -> str:
    return str(word.get("text") or "").strip()


def _max_chars_per_cue(settings: dict[str, Any] | None) -> int:
    if settings is None:
        return SUBTITLE_MAX_CHARS_PER_LINE * SUBTITLE_MAX_LINES
    return int(subtitle_layout_from_settings(settings)["max_chars_per_cue"])


def _split_words_by_pause(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not words:
        return []
    groups: list[list[dict[str, Any]]] = [[words[0]]]
    for previous, current in zip(words, words[1:], strict=False):
        gap = float(current.get("absolute_start", current.get("start", 0.0))) - float(
            previous.get("absolute_end", previous.get("end", 0.0))
        )
        if gap >= SUBTITLE_PAUSE_SPLIT_THRESHOLD:
            groups.append([current])
        else:
            groups[-1].append(current)
    return groups


def _split_word_group_by_punctuation(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for word in words:
        current.append(word)
        text = _word_display_text(word)
        if text and re.search(r"[.!?…。！？;]$", text):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups or [words]


def _split_word_group_by_length(
    words: list[dict[str, Any]],
    *,
    max_chars: int,
    language: str | None = None,
) -> list[list[dict[str, Any]]]:
    joiner_width = 1 if _is_space_delimited_language(language) else 0
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_len = 0
    for word in words:
        text = _word_display_text(word)
        projected = current_len + (joiner_width if current else 0) + len(text)
        if current and projected > max_chars:
            chunks.append(current)
            current = [word]
            current_len = len(text)
        else:
            current.append(word)
            current_len = projected
    if current:
        chunks.append(current)
    return chunks


def _words_to_cue(
    words: list[dict[str, Any]],
    *,
    tail_padding: float,
    max_cue_duration: float = SUBTITLE_MAX_CUE_DURATION,
    min_cue_duration: float = SUBTITLE_DUB_MIN_CUE_DURATION,
    language: str | None = None,
) -> dict[str, Any] | None:
    if not words:
        return None
    text = _join_display_words([_word_display_text(word) for word in words], language)
    if not text:
        return None
    start = float(words[0].get("absolute_start", words[0].get("start", 0.0)))
    end = float(words[-1].get("absolute_end", words[-1].get("end", start)))
    end = min(end + tail_padding, start + max_cue_duration)
    end = max(start + min_cue_duration, end)
    return {"start": round(start, 3), "end": round(end, 3), "text": text}


def _merge_short_cues(
    cues: list[dict[str, Any]],
    *,
    min_cue_duration: float = SUBTITLE_DUB_MIN_CUE_DURATION,
    max_cue_duration: float = SUBTITLE_MAX_CUE_DURATION,
    max_chars: int | None = None,
    language: str | None = None,
) -> list[dict[str, Any]]:
    if len(cues) <= 1:
        return cues
    char_limit = max_chars if max_chars is not None else _max_chars_per_cue(None)
    joiner = " " if _is_space_delimited_language(language) else ""
    merged: list[dict[str, Any]] = []
    buffer: dict[str, Any] | None = None
    for cue in cues:
        duration = float(cue["end"]) - float(cue["start"])
        if buffer is None:
            if duration < min_cue_duration:
                buffer = dict(cue)
            else:
                merged.append(cue)
            continue
        combined_text = f"{buffer['text']}{joiner}{cue['text']}".strip()
        combined_end = float(cue["end"])
        combined_duration = combined_end - float(buffer["start"])
        if combined_duration <= max_cue_duration and len(combined_text) <= char_limit:
            buffer = {
                "start": buffer["start"],
                "end": round(combined_end, 3),
                "text": combined_text,
            }
            if combined_duration >= min_cue_duration:
                merged.append(buffer)
                buffer = None
        else:
            merged.append(buffer)
            buffer = cue if duration < min_cue_duration else None
            if buffer is None:
                merged.append(cue)
    if buffer is not None:
        merged.append(buffer)
    return merged


def _merge_orphan_word_cues(
    cues: list[dict[str, Any]],
    *,
    max_chars: int,
    language: str | None = None,
) -> list[dict[str, Any]]:
    """Merge single-word cues into neighbors when they flash too briefly on screen."""
    if len(cues) < 2:
        return cues

    joiner = " " if _is_space_delimited_language(language) else ""

    def _word_count(text: str) -> int:
        return len([part for part in re.split(r"\s+", text.strip()) if part])

    merged: list[dict[str, Any]] = []
    for cue in cues:
        text = str(cue.get("text") or "").strip()
        if not text:
            continue
        if merged and _word_count(text) <= 1:
            previous = merged[-1]
            combined = f"{previous['text']}{joiner}{text}".strip()
            if len(combined) <= max_chars:
                merged[-1] = {
                    "start": previous["start"],
                    "end": round(max(float(previous["end"]), float(cue["end"])), 3),
                    "text": combined,
                }
                continue
        merged.append(dict(cue))

    resolved: list[dict[str, Any]] = []
    for cue in reversed(merged):
        text = str(cue.get("text") or "").strip()
        if resolved and _word_count(text) <= 1:
            nxt = resolved[-1]
            combined = f"{text}{joiner}{nxt['text']}".strip()
            if len(combined) <= max_chars:
                resolved[-1] = {
                    "start": round(min(float(cue["start"]), float(nxt["start"])), 3),
                    "end": nxt["end"],
                    "text": combined,
                }
                continue
        resolved.append(dict(cue))
    resolved.reverse()
    return resolved


def resolve_ass_quantized_cues(cues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prevent ASS centisecond rounding from creating overlapping cues."""
    if not cues:
        return []
    ordered = sorted(cues, key=lambda item: (float(item["start"]), float(item["end"])))
    resolved: list[dict[str, Any]] = []
    for cue in ordered:
        orig_start_cs = max(0, int(round(float(cue["start"]) * 100)))
        orig_end_cs = max(orig_start_cs + 1, int(round(float(cue["end"]) * 100)))
        min_duration_cs = max(1, orig_end_cs - orig_start_cs)
        start_cs = orig_start_cs
        end_cs = orig_end_cs
        if resolved:
            prev_end_cs = max(0, int(round(float(resolved[-1]["end"]) * 100)))
            if start_cs < prev_end_cs:
                start_cs = prev_end_cs
            end_cs = max(end_cs, start_cs + min_duration_cs)
        resolved.append(
            {
                "start": round(start_cs / 100.0, 3),
                "end": round(end_cs / 100.0, 3),
                "text": str(cue["text"]),
            }
        )
    return resolved


def build_cues_from_dub_words(
    segment: dict[str, Any],
    *,
    settings: dict[str, Any] | None = None,
    next_segment_start: float | None = None,
    language: str | None = None,
) -> list[dict[str, Any]]:
    """Build subtitle cues from final dub word timestamps."""
    dub_words = segment.get("dub_words") or []
    if not dub_words:
        return []
    layout = subtitle_layout_from_settings(settings)
    max_chars = int(layout["max_chars_per_cue"])
    min_cue = float(layout["min_cue_duration"])
    max_cue = float(layout["max_cue_duration"])
    min_gap = float(layout["min_gap"])
    tail_padding = SUBTITLE_CUE_TAIL_PADDING
    grouped: list[dict[str, Any]] = []
    # Prefer sentence/clause punctuation over pause gaps so cues end on ".!?" instead of mid-phrase.
    for punct_group in _split_word_group_by_punctuation(list(dub_words)):
        for pause_group in _split_words_by_pause(punct_group):
            for chunk in _split_word_group_by_length(pause_group, max_chars=max_chars, language=language):
                cue = _words_to_cue(
                    chunk,
                    tail_padding=tail_padding,
                    max_cue_duration=max_cue,
                    min_cue_duration=min_cue,
                    language=language,
                )
                if cue is not None:
                    grouped.append(cue)
    grouped = _merge_short_cues(
        grouped,
        min_cue_duration=min_cue,
        max_cue_duration=max_cue,
        max_chars=max_chars,
        language=language,
    )
    grouped = _merge_orphan_word_cues(grouped, max_chars=max_chars, language=language)
    if next_segment_start is not None:
        cap = float(next_segment_start) - min_gap
        for cue in grouped:
            cue["end"] = min(float(cue["end"]), cap)
            if float(cue["end"]) <= float(cue["start"]):
                cue["end"] = min(cap, float(cue["start"]) + min_cue)
    segment["subtitle_cue_count"] = len(grouped)
    return grouped


def normalize_spoken_for_conservation(text: str) -> str:
    """Normalize for ASS/spoken equality: drop whitespace/punct/case only.

    Digits and letters (including meaning-bearing marks) are preserved.
    """
    import unicodedata

    out: list[str] = []
    for ch in str(text or "").casefold():
        if ch.isspace():
            continue
        category = unicodedata.category(ch)
        if category.startswith("P") or category.startswith("S"):
            continue
        out.append(ch)
    return "".join(out)


def assert_cues_conserve_spoken(
    cues: list[dict[str, Any]],
    spoken_text: str,
    *,
    segment_index: int | None = None,
) -> None:
    """Fail closed when cue text does not conserve approved spoken text."""
    from .errors import AppError
    from .models import ErrorInfo

    spoken = str(spoken_text or "").strip()
    if not spoken:
        return
    if not cues:
        raise AppError(
            409,
            ErrorInfo(
                code="SUBTITLE_CONTENT_CONSERVATION_FAILED",
                message=("Empty subtitle cues for voiced text"
                         + (f" (segment {segment_index})." if segment_index is not None else ".")),
                action="Rebuild subtitles from approved spoken text.",
                detail=f"segment_index={segment_index},empty_cues=1",
            ),
        )
    for cue in cues:
        if not str(cue.get("text") or "").strip():
            raise AppError(
                409,
                ErrorInfo(
                    code="SUBTITLE_CONTENT_CONSERVATION_FAILED",
                    message=("Empty voiced subtitle cue"
                             + (f" (segment {segment_index})." if segment_index is not None else ".")),
                    action="Rebuild subtitles from approved spoken text.",
                    detail=f"segment_index={segment_index},empty_cue=1",
                ),
            )
    joined = "".join(str(cue.get("text") or "") for cue in cues)
    if normalize_spoken_for_conservation(joined) != normalize_spoken_for_conservation(spoken):
        raise AppError(
            409,
            ErrorInfo(
                code="SUBTITLE_CONTENT_CONSERVATION_FAILED",
                message=("Subtitle cue text diverged from approved spoken text"
                         + (f" (segment {segment_index})." if segment_index is not None else ".")),
                action="Do not fall back to ASR lexical text; rebuild cues from spoken_text.",
                detail=f"segment_index={segment_index},cue_count={len(cues)}",
            ),
        )


def _rebase_cue_texts_to_spoken(
    cues: list[dict[str, Any]],
    spoken_text: str,
    *,
    language: str | None = None,
    settings: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Keep cue timing windows; force display text from approved spoken_text only."""
    spoken = str(spoken_text or "").strip()
    if not cues or not spoken:
        return cues
    layout = subtitle_layout_from_settings(settings)
    chunks = split_for_subtitle_display(
        spoken,
        max_chars=int(layout["max_chars_per_cue"]),
        language=language,
    )
    if not chunks:
        return cues
    if len(chunks) == len(cues):
        for cue, chunk in zip(cues, chunks):
            cue["text"] = chunk
        return cues
    # Unequal count: rebuild texts on existing time spans proportionally by character weight.
    total_chars = sum(max(1, len(c)) for c in chunks) or 1
    t0 = float(cues[0]["start"])
    t1 = float(cues[-1]["end"])
    span = max(0.05, t1 - t0)
    rebuilt: list[dict[str, Any]] = []
    cursor = t0
    for i, chunk in enumerate(chunks):
        weight = max(1, len(chunk)) / total_chars
        dur = span * weight
        end = t1 if i + 1 == len(chunks) else min(t1, cursor + dur)
        if end <= cursor:
            end = min(t1, cursor + 0.12)
        rebuilt.append({"start": round(cursor, 3), "end": round(end, 3), "text": chunk})
        cursor = end
    return rebuilt


def build_segment_subtitle_cues(
    segment: dict[str, Any],
    *,
    job_dir: Path | None,
    settings: dict[str, Any] | None,
    vendor_dir: Path | None,
    ffmpeg_path: Path | None,
    transcribe_fn: Callable[..., Any] | None,
    tts_asr_align: bool,
    next_segment_start: float | None = None,
    target_language: str | None = None,
) -> list[dict[str, Any]]:
    # Content lineage must match spoken audio (cluster compact/re-TTS updates).
    translation = str(
        segment.get("tts_spoken_text")
        or segment.get("translation")
        or segment.get("target_text")
        or ""
    ).strip()
    if not translation:
        return []

    from .final_dub_alignment import refresh_segment_dub_word_timestamps, segment_has_usable_dub_words

    refresh_segment_dub_word_timestamps(segment)
    window_start = segment_subtitle_start(segment)
    window_end = segment_subtitle_end(segment)
    if segment_has_usable_dub_words(segment):
        cues = build_cues_from_dub_words(
            segment,
            settings=settings,
            next_segment_start=next_segment_start,
            language=target_language,
        )
        cues = _rebase_cue_texts_to_spoken(
            cues,
            translation,
            language=target_language,
            settings=settings,
        )
        cues = _ensure_cues_cover_speech(
            cues,
            speech_start=window_start,
            speech_end=window_end,
            next_segment_start=next_segment_start,
        )
        assert_cues_conserve_spoken(
            cues,
            translation,
            segment_index=int(segment.get("index", 0) or 0),
        )
        return cues

    layout = subtitle_layout_from_settings(settings)
    chunks = split_for_subtitle_display(
        translation,
        max_chars=int(layout["max_chars_per_cue"]),
        language=target_language,
    )
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
                    cues = _ensure_cues_cover_speech(
                        normalized,
                        speech_start=speech_start,
                        speech_end=speech_end,
                        next_segment_start=next_segment_start,
                    )
                    assert_cues_conserve_spoken(
                        cues,
                        translation,
                        segment_index=int(segment.get("index", 0) or 0),
                    )
                    return cues
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

    cues = allocate_proportional_cues(chunks, speech_start, speech_end)
    cues = _ensure_cues_cover_speech(
        cues,
        speech_start=speech_start,
        speech_end=speech_end,
        next_segment_start=next_segment_start,
    )
    assert_cues_conserve_spoken(
        cues,
        translation,
        segment_index=int(segment.get("index", 0) or 0),
    )
    return cues


def _ensure_cues_cover_speech(
    cues: list[dict[str, Any]],
    *,
    speech_start: float,
    speech_end: float,
    next_segment_start: float | None = None,
    early_slack_sec: float = 0.08,
    min_gap_sec: float = 0.05,
) -> list[dict[str, Any]]:
    """Keep cues near speech; never leave the spoken window without active text.

    ChatGPT TL: may lead audio by 50–100ms and linger slightly past the tail,
    but must not start after spoken words are already clear, and must not vanish
    while the clause is still playing.
    """
    if not cues:
        return cues
    limit = float(speech_end)
    if next_segment_start is not None:
        limit = min(limit, float(next_segment_start) - min_gap_sec)
    first = cues[0]
    first_start = float(first.get("start") or 0.0)
    # Pull early start back toward speech onset (allow small lead).
    target_start = max(0.0, float(speech_start) - early_slack_sec)
    if first_start > float(speech_start) + 0.12:
        first["start"] = round(target_start, 3)
    else:
        first["start"] = round(min(first_start, float(speech_start)), 3)
        first["start"] = round(max(float(first["start"]), target_start), 3)
    # Extend last cue through remaining speech.
    last = cues[-1]
    last["end"] = round(max(float(last.get("end") or 0.0), limit), 3)
    if float(last["end"]) <= float(last.get("start") or 0.0):
        last["end"] = round(float(last["start"]) + SUBTITLE_MIN_CUE_DURATION, 3)
    return cues


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
    from .final_dub_alignment import refresh_all_segment_dub_word_timestamps

    target_language: str | None = None
    if settings is not None:
        try:
            from .dubbing_languages import dub_language_from_settings

            target_language = dub_language_from_settings(settings)
        except Exception:
            target_language = None

    refresh_all_segment_dub_word_timestamps(segments)
    annotate_subtitle_playback_windows(segments)
    ordered = sorted(segments, key=lambda item: segment_subtitle_start(item))
    cues: list[dict[str, Any]] = []
    for index, segment in enumerate(ordered):
        next_start = None
        if index + 1 < len(ordered):
            next_start = segment_subtitle_start(ordered[index + 1])
        cues.extend(
            build_segment_subtitle_cues(
                segment,
                job_dir=job_dir,
                settings=settings,
                vendor_dir=vendor_dir,
                ffmpeg_path=ffmpeg_path,
                transcribe_fn=transcribe_fn,
                tts_asr_align=tts_asr_align,
                next_segment_start=next_start,
                target_language=target_language,
            )
        )
    return resolve_ass_quantized_cues(resolve_overlapping_cues(cues))
