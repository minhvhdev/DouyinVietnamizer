"""Final dub alignment: map repaired TTS audio timestamps onto canonical target text."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

DUB_ALIGNMENT_CACHE_VERSION = 2
DUB_ALIGNMENT_SCHEMA_VERSION = 1
TOKEN_NORMALIZATION_VERSION = 1
SENTENCE_END_RE = re.compile(r"[.!?…。！？;]$")
PUNCT_ATTACH_RE = re.compile(r"^(.+?)([,.!?…。！？;:\"'»«]+)$")
VIETNAMESE_VOWEL_RE = re.compile(
    r"[aeiouyăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
    flags=re.IGNORECASE,
)

# Qwen forced aligner is bundled with Qwen3-ASR; word timestamps are best-effort for Vietnamese.
QWEN_FORCED_ALIGNER_LANGUAGES = frozenset({"Chinese", "English", "Vietnamese", "Japanese", "Korean"})


@dataclass(frozen=True)
class TargetToken:
    text: str
    norm: str


@dataclass(frozen=True)
class AsrToken:
    text: str
    norm: str
    start: float
    end: float


@dataclass(frozen=True)
class AlignmentPair:
    target_index: int | None
    asr_start_index: int | None
    asr_end_index: int | None
    operation: str


@dataclass(frozen=True)
class AsrBackendInfo:
    method: str
    has_forced_aligner_units: bool
    has_word_timestamps: bool
    has_segment_timestamps: bool
    unit_count: int


def segment_target_text(segment: dict[str, Any]) -> str:
    return str(
        segment.get("tts_spoken_text")
        or segment.get("target_text")
        or segment.get("translation")
        or ""
    ).strip()


def segment_placement_start(segment: dict[str, Any]) -> float:
    placement = segment.get("placement_start")
    if placement is not None:
        try:
            value = float(placement)
            if math.isfinite(value) and value >= 0:
                return value
        except (TypeError, ValueError):
            pass
    start = segment.get("start")
    if start is not None:
        try:
            value = float(start)
            if math.isfinite(value) and value >= 0:
                return value
        except (TypeError, ValueError):
            pass
    return 0.0


def _dub_word_absolute_times(
    word: dict[str, Any],
    *,
    placement_start: float,
) -> tuple[float, float] | None:
    if word.get("absolute_start") is not None and word.get("absolute_end") is not None:
        try:
            start = float(word["absolute_start"])
            end = float(word["absolute_end"])
        except (TypeError, ValueError):
            return None
    else:
        try:
            rel_start = float(word.get("start", 0.0))
            rel_end = float(word.get("end", rel_start))
        except (TypeError, ValueError):
            return None
        start = placement_start + rel_start
        end = placement_start + rel_end
    if not math.isfinite(start) or not math.isfinite(end):
        return None
    if end < start:
        return None
    return start, end


def filter_valid_dub_words(
    words: list[dict[str, Any]],
    segment: dict[str, Any],
    *,
    outside_margin_sec: float = 0.15,
) -> list[dict[str, Any]]:
    """Return dub_words safe for subtitle cues without mutating the input list.

    dub_words contract on segments:
    - ``start``/``end`` are relative to clip audio (0 = clip start).
    - ``absolute_start``/``absolute_end`` are playback-timeline seconds after placement.
    - Cache files store relative timestamps only; placement rebasing happens via
      ``apply_placement_to_dub_words`` — consumers must not rebase again.
    Words outside the playback interval by more than ``outside_margin_sec`` are dropped
  (tolerates small ASR/float jitter at boundaries).
    """
    from .timing_placement import segment_playback_interval

    placement_start = segment_placement_start(segment)
    play_start, play_end = segment_playback_interval(segment)
    valid: list[dict[str, Any]] = []
    previous_end = -1.0
    for word in words:
        if not str(word.get("text") or "").strip():
            continue
        times = _dub_word_absolute_times(word, placement_start=placement_start)
        if times is None:
            continue
        abs_start, abs_end = times
        if abs_end < play_start - outside_margin_sec or abs_start > play_end + outside_margin_sec:
            continue
        if abs_start < previous_end - 0.001:
            continue
        previous_end = abs_end
        valid.append(word)
    return valid

def supports_forced_alignment(language: str) -> bool:
    return language in QWEN_FORCED_ALIGNER_LANGUAGES


def supports_word_timestamps(language: str) -> bool:
    return language in QWEN_FORCED_ALIGNER_LANGUAGES


def supports_character_timestamps(language: str) -> bool:
    return language in QWEN_FORCED_ALIGNER_LANGUAGES


def normalize_alignment_token(text: str) -> str:
    """Normalize token text for fuzzy alignment comparisons."""
    normalized = unicodedata.normalize("NFC", text or "")
    normalized = normalized.lower().strip()
    normalized = re.sub(r"[^\w]+", "", normalized, flags=re.UNICODE)
    return normalized


def tokenize_target_text(text: str) -> list[TargetToken]:
    """Split target text into display tokens (Model B: punctuation attached to token)."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    cleaned = cleaned.replace("—", " — ").replace("–", " – ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return []
    raw_parts = cleaned.split()
    tokens: list[TargetToken] = []
    for part in raw_parts:
        if part in {"—", "–", "-"} and tokens:
            tokens[-1] = TargetToken(text=f"{tokens[-1].text}{part}", norm=tokens[-1].norm)
            continue
        match = PUNCT_ATTACH_RE.match(part)
        if not match:
            tokens.append(TargetToken(text=part, norm=normalize_alignment_token(part)))
            continue
        core, punct = match.group(1), match.group(2)
        if core:
            display = f"{core}{punct}" if punct else core
            tokens.append(TargetToken(text=display, norm=normalize_alignment_token(core)))
        elif punct and tokens:
            prev = tokens[-1]
            tokens[-1] = TargetToken(text=f"{prev.text}{punct}", norm=prev.norm)
        elif punct:
            tokens.append(TargetToken(text=punct, norm=""))
    return [token for token in tokens if token.text.strip()]


def reconstruct_target_text_from_dub_words(words: list[dict[str, Any]]) -> str:
    return " ".join(str(word.get("text") or "").strip() for word in words if str(word.get("text") or "").strip())


def tokenize_asr_units(units: list[dict[str, Any]]) -> list[AsrToken]:
    tokens: list[AsrToken] = []
    for unit in units:
        text = str(unit.get("text") or "").strip()
        if not text:
            continue
        try:
            start = max(0.0, float(unit.get("start", 0.0) or 0.0))
            end = max(start, float(unit.get("end", start) or start))
        except (TypeError, ValueError):
            continue
        if end <= start:
            end = start + 0.01
        for part in text.split():
            norm = normalize_alignment_token(part)
            if not norm:
                continue
            tokens.append(AsrToken(text=part, norm=norm, start=start, end=end))
    return _expand_asr_token_timings(tokens)


def _expand_asr_token_timings(tokens: list[AsrToken]) -> list[AsrToken]:
    if not tokens:
        return []
    expanded: list[AsrToken] = []
    for index, token in enumerate(tokens):
        start = token.start if index == 0 else max(token.start, expanded[-1].end)
        end = max(start + 0.01, token.end)
        expanded.append(AsrToken(text=token.text, norm=token.norm, start=start, end=end))
    return expanded


def align_target_tokens_to_asr_tokens(
    target_tokens: list[TargetToken],
    asr_tokens: list[AsrToken],
) -> list[AlignmentPair]:
    """Sequence-align target tokens to ASR tokens using dynamic programming."""
    n = len(target_tokens)
    m = len(asr_tokens)
    gap = 1
    match = 0
    mismatch = 1

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt: list[list[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * gap
        bt[i][0] = "insert"
    for j in range(1, m + 1):
        dp[0][j] = j * gap
        bt[0][j] = "delete"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            t_norm = target_tokens[i - 1].norm
            a_norm = asr_tokens[j - 1].norm
            sub_cost = match if t_norm and t_norm == a_norm else mismatch
            choices = (
                (dp[i - 1][j - 1] + sub_cost, "match" if sub_cost == match else "replace"),
                (dp[i - 1][j] + gap, "insert"),
                (dp[i][j - 1] + gap, "delete"),
            )
            best_cost, best_op = min(choices, key=lambda item: item[0])
            dp[i][j] = best_cost
            bt[i][j] = best_op

    raw_pairs: list[tuple[int | None, int | None, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = bt[i][j]
        if op in {"match", "replace"}:
            raw_pairs.append((i - 1, j - 1, op))
            i -= 1
            j -= 1
        elif op == "insert":
            raw_pairs.append((i - 1, None, "insert"))
            i -= 1
        else:
            raw_pairs.append((None, j - 1, "delete"))
            j -= 1
    raw_pairs.reverse()

    grouped: dict[int, dict[str, Any]] = {}
    for target_index, asr_index, operation in raw_pairs:
        if target_index is None:
            continue
        entry = grouped.setdefault(target_index, {"asr_indices": [], "operation": operation})
        if asr_index is not None:
            entry["asr_indices"].append(asr_index)
        if operation == "match":
            entry["operation"] = "match"
        elif entry["operation"] != "match" and operation == "replace":
            entry["operation"] = "replace"

    merged: list[AlignmentPair] = []
    for target_index in range(n):
        entry = grouped.get(target_index)
        if entry is None:
            merged.append(
                AlignmentPair(
                    target_index=target_index,
                    asr_start_index=None,
                    asr_end_index=None,
                    operation="insert",
                )
            )
            continue
        indices = entry["asr_indices"]
        merged.append(
            AlignmentPair(
                target_index=target_index,
                asr_start_index=indices[0] if indices else None,
                asr_end_index=indices[-1] if indices else None,
                operation=str(entry["operation"]),
            )
        )
    return merged


def _token_weight(token: str) -> float:
    core = re.sub(r"[^\w]", "", token, flags=re.UNICODE)
    if not core:
        return 0.35
    vowels = len(VIETNAMESE_VOWEL_RE.findall(core))
    if vowels:
        return max(1.0, float(vowels))
    return max(1.0, len(core) * 0.45)


def interpolate_token_timestamps(
    target_tokens: list[TargetToken],
    *,
    duration: float,
    min_token_duration: float = 0.05,
) -> list[tuple[float, float]]:
    if not target_tokens or duration <= 0:
        return []
    weights = [_token_weight(token.text) for token in target_tokens]
    total = sum(weights) or float(len(target_tokens))
    cursor = 0.0
    spans: list[tuple[float, float]] = []
    for index, weight in enumerate(weights):
        portion = duration * weight / total
        end = duration if index == len(weights) - 1 else min(duration, cursor + max(min_token_duration, portion))
        spans.append((cursor, end))
        cursor = end
    if spans:
        spans[-1] = (spans[-1][0], duration)
    return spans


def assign_timestamps_to_target_tokens(
    target_tokens: list[TargetToken],
    asr_tokens: list[AsrToken],
    pairs: list[AlignmentPair],
    *,
    duration: float,
) -> list[dict[str, Any]]:
    """Map ASR timestamps onto canonical target tokens with weighted fallback."""
    fallback_spans = interpolate_token_timestamps(target_tokens, duration=duration)
    words: list[dict[str, Any]] = []
    for index, token in enumerate(target_tokens):
        pair = next((item for item in pairs if item.target_index == index), None)
        start: float | None = None
        end: float | None = None
        alignment = "interpolated"
        confidence = 0.5
        if pair is not None and pair.asr_start_index is not None and asr_tokens:
            start_idx = pair.asr_start_index
            end_idx = pair.asr_end_index if pair.asr_end_index is not None else start_idx
            start = asr_tokens[start_idx].start
            end = asr_tokens[end_idx].end
            alignment = "exact" if pair.operation == "match" else "replace"
            confidence = 0.97 if pair.operation == "match" else 0.82
        if start is None or end is None:
            fb_start, fb_end = fallback_spans[index]
            start, end = fb_start, fb_end
            alignment = "interpolated"
            confidence = 0.5
        words.append(
            {
                "text": token.text,
                "start": round(start, 3),
                "end": round(end, 3),
                "confidence": round(confidence, 3),
                "alignment": alignment,
            }
        )
    return words


def validate_word_timeline(
    words: list[dict[str, Any]],
    *,
    max_duration: float,
    min_gap: float = 0.001,
) -> list[dict[str, Any]]:
    if not words:
        return []
    validated: list[dict[str, Any]] = []
    cursor = 0.0
    cap = max(0.01, float(max_duration))
    for word in words:
        text = str(word.get("text") or "")
        try:
            start = max(0.0, float(word.get("start", 0.0) or 0.0))
            end = max(start, float(word.get("end", start) or start))
        except (TypeError, ValueError):
            continue
        start = max(cursor, min(start, cap))
        end = max(start + min_gap, min(end, cap))
        validated.append({**word, "text": text, "start": round(start, 3), "end": round(end, 3)})
        cursor = end
    if validated:
        validated[-1]["end"] = round(min(cap, max(validated[-1]["end"], validated[-1]["start"] + min_gap)), 3)
    return validated


def strip_absolute_from_dub_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    relative: list[dict[str, Any]] = []
    for word in words:
        cleaned = {key: value for key, value in word.items() if key not in {"absolute_start", "absolute_end"}}
        relative.append(cleaned)
    return relative


def apply_placement_to_dub_words(
    words: list[dict[str, Any]],
    *,
    placement_start: float,
) -> list[dict[str, Any]]:
    base = float(placement_start)
    enriched: list[dict[str, Any]] = []
    for word in words:
        start = float(word["start"])
        end = float(word["end"])
        enriched.append(
            {
                **word,
                "absolute_start": round(base + start, 3),
                "absolute_end": round(base + end, 3),
            }
        )
    return enriched


def refresh_segment_dub_word_timestamps(segment: dict[str, Any]) -> None:
    words = segment.get("dub_words") or []
    if not words:
        return
    relative = strip_absolute_from_dub_words(words)
    segment["dub_words"] = apply_placement_to_dub_words(relative, placement_start=segment_placement_start(segment))


def refresh_all_segment_dub_word_timestamps(segments: list[dict[str, Any]]) -> None:
    for segment in segments:
        refresh_segment_dub_word_timestamps(segment)


def segment_has_usable_dub_words(segment: dict[str, Any]) -> bool:
    words = segment.get("dub_words") or []
    if not words:
        return False
    status = str(segment.get("dub_alignment_status") or "")
    if status in {"failed", "skipped"}:
        return False
    return len(filter_valid_dub_words(list(words), segment)) > 0


def text_similarity(target_text: str, asr_text: str) -> float:
    target_norm = normalize_alignment_token(target_text.replace(" ", ""))
    asr_norm = normalize_alignment_token(asr_text.replace(" ", ""))
    if not target_norm and not asr_norm:
        return 1.0
    if not target_norm or not asr_norm:
        return 0.0
    n, m = len(target_norm), len(asr_norm)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if target_norm[i - 1] == asr_norm[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    distance = dp[n][m]
    return round(max(0.0, 1.0 - distance / max(n, m)), 4)


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def compute_audio_content_hash(path: Path, *, chunk_size: int = 65536) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_alignment_cache_identity(
    *,
    audio_path: Path,
    target_text: str,
    target_language: str,
    asr_model: str,
    aligner_model: str,
) -> str:
    parts = [
        str(DUB_ALIGNMENT_CACHE_VERSION),
        str(TOKEN_NORMALIZATION_VERSION),
        compute_audio_content_hash(audio_path),
        target_text,
        target_language,
        asr_model,
        aligner_model,
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]


def _cache_file_path(cache_dir: Path, wav_path: Path, cache_identity: str) -> Path:
    return cache_dir / f"{wav_path.stem}_{cache_identity}.json"


def _load_alignment_cache(cache_path: Path, *, expected_identity: str) -> dict[str, Any] | None:
    if not cache_path.is_file():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if payload.get("cache_version") != DUB_ALIGNMENT_CACHE_VERSION:
        return None
    if payload.get("cache_identity") != expected_identity:
        return None
    words = payload.get("dub_words")
    if words is not None and not isinstance(words, list):
        return None
    return payload


def _store_alignment_cache(cache_path: Path, payload: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def wav_has_detectable_speech(path: Path, *, min_rms: float = 0.002) -> bool:
    """Lightweight RMS check; returns False for near-silent clips."""
    try:
        with wave.open(str(path), "rb") as handle:
            sample_width = handle.getsampwidth()
            frame_count = handle.getnframes()
            if frame_count <= 0:
                return False
            raw = handle.readframes(min(frame_count, handle.getframerate() * 10))
    except (OSError, wave.Error):
        return False
    if sample_width != 2 or not raw:
        return False
    total = 0.0
    count = len(raw) // 2
    for index in range(0, len(raw) - 1, 2):
        sample = int.from_bytes(raw[index : index + 2], "little", signed=True)
        total += sample * sample
    if count <= 0:
        return False
    rms = math.sqrt(total / count) / 32768.0
    return rms >= min_rms


def classify_qwen_asr_backend(
    *,
    aligned_units: list[dict[str, Any]],
    asr_segments: list[dict[str, Any]],
    language: str,
) -> AsrBackendInfo:
    unit_count = len(aligned_units)
    has_forced = unit_count > 0 and any(str(unit.get("text") or "").strip() for unit in aligned_units)
    has_segments = bool(asr_segments)
    has_word = False
    has_char = False
    if has_forced:
        tokenized = tokenize_asr_units(aligned_units)
        if tokenized:
            avg_token_len = sum(len(token.norm) for token in tokenized) / len(tokenized)
            has_word = avg_token_len >= 2.0 or len(tokenized) >= max(2, len(asr_segments) * 2)
            has_char = not has_word
    if has_forced and supports_forced_alignment(language):
        method = "qwen_forced_aligner_words" if has_word else "qwen_forced_aligner_chars"
    elif has_segments:
        method = "qwen_asr_segment_mapping"
    else:
        method = "weighted_interpolation"
    return AsrBackendInfo(
        method=method,
        has_forced_aligner_units=has_forced,
        has_word_timestamps=has_word,
        has_segment_timestamps=has_segments,
        unit_count=unit_count,
    )


def resolve_final_alignment_method(
    backend: AsrBackendInfo,
    *,
    interpolated_count: int,
    total_tokens: int,
    fallback_reason: str | None = None,
) -> tuple[str, str, float]:
    if total_tokens <= 0:
        return "skipped", "none", 0.0
    if interpolated_count >= total_tokens:
        method = "weighted_interpolation"
        if fallback_reason:
            method = f"{method}:{fallback_reason}"
        return "fallback_interpolated", method, 0.4
    if interpolated_count > 0:
        method = f"{backend.method}_partial"
        return "fallback_interpolated", method, 0.75
    return "aligned", backend.method, 0.95


def resolve_segment_audio_path(job_dir: Path, segment: dict[str, Any]) -> tuple[Path | None, bool]:
    from .tts_provenance import resolve_voiced_tts_path

    del job_dir  # provenance must come from the segment, not index guessing
    path = resolve_voiced_tts_path(segment)
    if path is None:
        return None, False
    # True means "non-canonical raw/provenance path" for telemetry only.
    claimed = Path(str(segment.get("tts_path") or ""))
    fallback = bool(segment.get("tts_path")) is False or (
        claimed.is_file() and claimed.resolve() != path.resolve()
    )
    return path, fallback


def _apply_cached_alignment_to_segment(
    segment: dict[str, Any],
    cached: dict[str, Any],
    *,
    placement_start: float,
) -> None:
    for key, value in cached.items():
        if key in {"cache_version", "cache_identity", "dub_words"}:
            continue
        segment[key] = value
    relative_words = strip_absolute_from_dub_words(list(cached.get("dub_words") or []))
    segment["dub_words"] = apply_placement_to_dub_words(relative_words, placement_start=placement_start)


def _transcribe_final_dub_units(
    audio_path: Path,
    *,
    segment_index: object,
    vendor_dir: Path,
    settings: dict[str, Any],
    language: str,
    ffmpeg_path: Path,
    cache_dir: Path,
    transcribe_fn: Callable[..., Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], AsrBackendInfo]:
    from .subtitle_timing import transcribe_tts_clip_details_for_subtitles

    details = transcribe_tts_clip_details_for_subtitles(
        audio_path,
        vendor_dir=vendor_dir,
        settings=settings,
        language=language,
        ffmpeg_path=ffmpeg_path,
        cache_dir=cache_dir,
        transcribe_fn=transcribe_fn,
    )
    units = list(details.get("aligned_units") or [])
    asr_segments = list(details.get("segments") or [])
    backend = classify_qwen_asr_backend(aligned_units=units, asr_segments=asr_segments, language=language)
    if details.get("from_forced_aligner"):
        backend = AsrBackendInfo(
            method=backend.method if backend.has_forced_aligner_units else "qwen_forced_aligner_words",
            has_forced_aligner_units=True,
            has_word_timestamps=backend.has_word_timestamps,
            has_segment_timestamps=backend.has_segment_timestamps,
            unit_count=backend.unit_count,
        )
    elif asr_segments and backend.method == "weighted_interpolation":
        backend = AsrBackendInfo(
            method="qwen_asr_segment_mapping",
            has_forced_aligner_units=False,
            has_word_timestamps=False,
            has_segment_timestamps=True,
            unit_count=len(units),
        )
    logger.info(
        "Final dub ASR backend segment=%s method=%s units=%s language=%s forced=%s",
        segment_index,
        backend.method,
        backend.unit_count,
        language,
        details.get("from_forced_aligner"),
    )
    return units, asr_segments, backend


def align_segment_final_dub(
    segment: dict[str, Any],
    *,
    job_dir: Path,
    cache_dir: Path,
    transcribe_fn: Callable[..., Any] | None,
    vendor_dir: Path,
    settings: dict[str, Any],
    ffmpeg_path: Path,
    language: str,
    allow_model: bool = True,
) -> dict[str, Any]:
    """Align one segment's repaired dub audio to canonical target text."""
    target_text = segment_target_text(segment)
    if not target_text:
        segment["dub_alignment_status"] = "skipped"
        segment["dub_alignment_method"] = "none"
        segment["dub_words"] = []
        return {"status": "skipped", "cache_hit": False, "model_called": False}

    audio_path, used_raw_fallback = resolve_segment_audio_path(job_dir, segment)
    if audio_path is None:
        segment["dub_alignment_status"] = "failed"
        segment["dub_alignment_method"] = "none"
        segment["dub_alignment_error"] = "missing_audio"
        segment["dub_words"] = []
        return {"status": "failed", "cache_hit": False, "model_called": False}

    placement_start = segment_placement_start(segment)
    repaired_duration = float(segment.get("repaired_duration") or _wav_duration(audio_path))
    asr_model = str(settings.get("qwen3_asr_model", "") or "")
    aligner_model = str(settings.get("qwen3_aligner_model", "") or "")
    cache_identity = build_alignment_cache_identity(
        audio_path=audio_path,
        target_text=target_text,
        target_language=language,
        asr_model=asr_model,
        aligner_model=aligner_model,
    )
    cache_path = _cache_file_path(cache_dir, audio_path, cache_identity)
    cached = _load_alignment_cache(cache_path, expected_identity=cache_identity)
    if cached is not None:
        _apply_cached_alignment_to_segment(segment, cached, placement_start=placement_start)
        return {
            "status": str(segment.get("dub_alignment_status") or "aligned"),
            "cache_hit": True,
            "text_similarity": float(segment.get("dub_text_similarity") or 0.0),
            "model_called": False,
        }

    if not allow_model or transcribe_fn is None:
        segment["dub_alignment_status"] = "failed"
        segment["dub_alignment_method"] = "none"
        segment["dub_alignment_error"] = "model_unavailable"
        segment["dub_words"] = []
        return {"status": "failed", "cache_hit": False, "model_called": False}

    model_called = True
    asr_units: list[dict[str, Any]] = []
    backend = AsrBackendInfo(
        method="weighted_interpolation",
        has_forced_aligner_units=False,
        has_word_timestamps=False,
        has_segment_timestamps=False,
        unit_count=0,
    )
    try:
        asr_units, _segments, backend = _transcribe_final_dub_units(
            audio_path,
            segment_index=segment.get("index"),
            vendor_dir=vendor_dir,
            settings=settings,
            language=language,
            ffmpeg_path=ffmpeg_path,
            cache_dir=cache_dir,
            transcribe_fn=transcribe_fn,
        )
        logger.info(
            "Final dub alignment segment=%s backend=%s units=%s",
            segment.get("index"),
            backend.method,
            backend.unit_count,
        )
    except Exception as exc:
        logger.warning("Final dub ASR failed for segment %s: %s", segment.get("index"), exc)
        asr_units = []

    target_tokens = tokenize_target_text(target_text)
    asr_tokens = tokenize_asr_units(asr_units)
    dub_asr_text = " ".join(token.text for token in asr_tokens)
    similarity = text_similarity(target_text, dub_asr_text)

    if not target_tokens:
        segment["dub_alignment_status"] = "skipped"
        segment["dub_alignment_method"] = "none"
        segment["dub_words"] = []
        return {"status": "skipped", "cache_hit": False, "model_called": model_called}

    fallback_reason: str | None = None
    if not asr_tokens:
        spans = interpolate_token_timestamps(target_tokens, duration=repaired_duration)
        words = [
            {
                "text": token.text,
                "start": round(start, 3),
                "end": round(end, 3),
                "confidence": 0.4,
                "alignment": "interpolated",
            }
            for token, (start, end) in zip(target_tokens, spans, strict=True)
        ]
        from .omnivoice_diagnostics import diagnostics_enabled, log_event, probe_wav_path

        speech_ok = wav_has_detectable_speech(audio_path)
        if diagnostics_enabled():
            probe = probe_wav_path(audio_path)
            log_event(
                "final_alignment_input",
                {
                    "stage": "final_alignment_input",
                    "speech_detected": speech_ok,
                    "probe": probe,
                    "expected_duration": repaired_duration,
                    "actual_duration": probe.get("duration_sec"),
                },
            )
            if not speech_ok:
                log_event(
                    "final_alignment_no_speech",
                    {
                        "stage": "final_alignment",
                        "probe": probe,
                        "expected_duration": repaired_duration,
                        "actual_duration": probe.get("duration_sec"),
                    },
                )
        if speech_ok:
            status = "fallback_interpolated"
            fallback_reason = "asr_empty_with_detected_audio"
            method = f"weighted_interpolation:{fallback_reason}"
            confidence = 0.4
        else:
            status = "no_speech"
            method = "weighted_interpolation:silent_audio"
            confidence = 0.35
    else:
        pairs = align_target_tokens_to_asr_tokens(target_tokens, asr_tokens)
        words = assign_timestamps_to_target_tokens(
            target_tokens,
            asr_tokens,
            pairs,
            duration=repaired_duration,
        )
        interpolated_count = sum(1 for word in words if word.get("alignment") == "interpolated")
        status, method, confidence = resolve_final_alignment_method(
            backend,
            interpolated_count=interpolated_count,
            total_tokens=len(words),
        )

    words = validate_word_timeline(words, max_duration=repaired_duration)
    relative_words = strip_absolute_from_dub_words(words)
    segment["dub_alignment_status"] = status
    segment["dub_alignment_method"] = method
    segment["dub_alignment_confidence"] = round(confidence, 3)
    segment["dub_alignment_audio_path"] = str(audio_path.relative_to(job_dir)).replace("\\", "/")
    segment["dub_alignment_text"] = target_text
    segment["dub_asr_text"] = dub_asr_text
    segment["dub_text_similarity"] = similarity
    segment["dub_words"] = apply_placement_to_dub_words(relative_words, placement_start=placement_start)
    segment["dub_alignment_used_raw_fallback"] = used_raw_fallback

    payload = {
        "cache_version": DUB_ALIGNMENT_CACHE_VERSION,
        "cache_identity": cache_identity,
        "dub_alignment_status": status,
        "dub_alignment_method": method,
        "dub_alignment_confidence": segment["dub_alignment_confidence"],
        "dub_alignment_audio_path": segment["dub_alignment_audio_path"],
        "dub_alignment_text": target_text,
        "dub_asr_text": dub_asr_text,
        "dub_text_similarity": similarity,
        "dub_words": relative_words,
        "dub_alignment_used_raw_fallback": used_raw_fallback,
    }
    _store_alignment_cache(cache_path, payload)
    return {
        "status": status,
        "cache_hit": False,
        "text_similarity": similarity,
        "model_called": model_called,
    }


def align_job_segments_final_dub(
    segments: list[dict[str, Any]],
    *,
    job_dir: Path,
    cache_dir: Path,
    transcribe_fn: Callable[..., Any] | None,
    vendor_dir: Path,
    settings: dict[str, Any],
    ffmpeg_path: Path,
    language: str,
) -> dict[str, Any]:
    """Align all segments; only invokes model when at least one cache miss needs it."""
    cache_hits = 0
    cache_misses = 0
    model_calls = 0
    results: list[dict[str, Any]] = []

    pending: list[dict[str, Any]] = []
    for segment in segments:
        target_text = segment_target_text(segment)
        if not target_text:
            segment["dub_alignment_status"] = "skipped"
            segment["dub_alignment_method"] = "none"
            segment["dub_words"] = []
            continue
        audio_path, _ = resolve_segment_audio_path(job_dir, segment)
        if audio_path is None:
            pending.append(segment)
            continue
        cache_identity = build_alignment_cache_identity(
            audio_path=audio_path,
            target_text=target_text,
            target_language=language,
            asr_model=str(settings.get("qwen3_asr_model", "") or ""),
            aligner_model=str(settings.get("qwen3_aligner_model", "") or ""),
        )
        cache_path = _cache_file_path(cache_dir, audio_path, cache_identity)
        if _load_alignment_cache(cache_path, expected_identity=cache_identity) is None:
            pending.append(segment)

    allow_model = bool(pending) and transcribe_fn is not None
    if allow_model:
        logger.info("Final dub alignment will invoke Qwen for %s/%s segments", len(pending), len(segments))
    elif pending:
        logger.warning("Final dub alignment has %s cache misses but no model available", len(pending))

    for segment in segments:
        result = align_segment_final_dub(
            segment,
            job_dir=job_dir,
            cache_dir=cache_dir,
            transcribe_fn=transcribe_fn,
            vendor_dir=vendor_dir,
            settings=settings,
            ffmpeg_path=ffmpeg_path,
            language=language,
            allow_model=allow_model,
        )
        results.append(result)
        if result.get("cache_hit"):
            cache_hits += 1
        else:
            cache_misses += 1
        if result.get("model_called"):
            model_calls += 1

    return {
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "model_calls": model_calls,
        "results": results,
    }


def validate_dub_words_timeline(
    words: list[dict[str, Any]],
    *,
    placement_start: float,
    max_duration: float,
) -> dict[str, Any]:
    warnings: list[str] = []
    if not words:
        return {"relative_timeline_valid": False, "absolute_timeline_valid": False, "warnings": ["no_words"]}
    relative_valid = True
    absolute_valid = True
    previous_end = -1.0
    for word in words:
        start = float(word.get("start", 0.0))
        end = float(word.get("end", start))
        if end < start or start < previous_end - 0.001:
            relative_valid = False
            warnings.append("non_monotonic_relative")
        if end > max_duration + 0.001:
            relative_valid = False
            warnings.append("relative_exceeds_duration")
        previous_end = end
        abs_start = float(word.get("absolute_start", placement_start + start))
        abs_end = float(word.get("absolute_end", placement_start + end))
        if abs(abs_start - (placement_start + start)) > 0.01:
            absolute_valid = False
            warnings.append("absolute_start_mismatch")
        if abs(abs_end - (placement_start + end)) > 0.01:
            absolute_valid = False
            warnings.append("absolute_end_mismatch")
    return {
        "relative_timeline_valid": relative_valid,
        "absolute_timeline_valid": absolute_valid,
        "warnings": warnings,
    }


def _word_display_text(word: dict[str, Any]) -> str:
    return str(word.get("text") or "").strip()


def compute_subtitle_qc_metrics(
    segments: list[dict[str, Any]],
    cues: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure subtitle sync quality against dub word boundaries."""
    drifts: list[float] = []
    overlap_count = 0
    out_of_bounds = 0
    unaligned_word_count = 0

    for segment in segments:
        dub_words = segment.get("dub_words") or []
        unaligned_word_count += sum(
            1 for word in dub_words if word.get("alignment") == "interpolated"
        )

    ordered_cues = sorted(cues, key=lambda item: (float(item["start"]), float(item["end"])))
    for index, cue in enumerate(ordered_cues):
        if index > 0:
            prev_end = float(ordered_cues[index - 1]["end"])
            if float(cue["start"]) < prev_end - 0.001:
                overlap_count += 1
        cue_text = str(cue.get("text") or "")
        matching_words = []
        for segment in segments:
            for word in segment.get("dub_words") or []:
                if _word_display_text(word) and _word_display_text(word) in cue_text:
                    matching_words.append(word)
        if matching_words:
            first = matching_words[0]
            last = matching_words[-1]
            drifts.append(abs(float(cue["start"]) - float(first.get("absolute_start", first["start"]))) * 1000)
            drifts.append(abs(float(cue["end"]) - float(last.get("absolute_end", last["end"]))) * 1000)

    segment_windows = [
        (
            segment_placement_start(segment),
            segment_placement_start(segment)
            + float(segment.get("repaired_duration") or segment.get("subtitle_playback_duration") or 0.0),
        )
        for segment in segments
    ]
    for cue in ordered_cues:
        start = float(cue["start"])
        end = float(cue["end"])
        if not any(window[0] - 0.05 <= start and end <= window[1] + 0.15 for window in segment_windows):
            out_of_bounds += 1

    return {
        "subtitle_word_drift_ms": round(sum(drifts) / len(drifts), 2) if drifts else None,
        "subtitle_cue_overlap_count": overlap_count,
        "subtitle_out_of_bounds_count": out_of_bounds,
        "unaligned_word_count": unaligned_word_count,
    }


def summarize_alignment_results(segments: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    aligned_count = 0
    fallback_count = 0
    failed_count = 0
    similarities: list[float] = []
    per_segment: list[dict[str, Any]] = []

    for segment in segments:
        status = str(segment.get("dub_alignment_status") or "skipped")
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == "aligned":
            aligned_count += 1
        elif status in {"fallback_interpolated", "no_speech"}:
            fallback_count += 1
        elif status == "failed":
            failed_count += 1
        if segment.get("dub_text_similarity") is not None:
            similarities.append(float(segment["dub_text_similarity"]))
        target_tokens = tokenize_target_text(segment_target_text(segment))
        dub_words = segment.get("dub_words") or []
        per_segment.append(
            {
                "segment_index": segment.get("index"),
                "alignment_status": status,
                "target_token_count": len(target_tokens),
                "aligned_token_count": sum(
                    1 for word in dub_words if word.get("alignment") in {"exact", "replace"}
                ),
                "text_similarity": segment.get("dub_text_similarity"),
                "fallback_used": status in {"fallback_interpolated", "no_speech"},
                "subtitle_cue_count": segment.get("subtitle_cue_count"),
            }
        )

    return {
        "dub_alignment_status_counts": status_counts,
        "dub_alignment_failure_count": failed_count,
        "dub_alignment_fallback_count": fallback_count,
        "dub_text_similarity": round(sum(similarities) / len(similarities), 4) if similarities else None,
        "average_text_similarity": round(sum(similarities) / len(similarities), 4) if similarities else None,
        "aligned_count": aligned_count,
        "per_segment": per_segment,
    }
