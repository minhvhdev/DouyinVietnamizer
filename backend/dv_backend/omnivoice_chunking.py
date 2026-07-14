"""Semantic text chunking for OmniVoice long-segment synthesis."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from .adapters.tts import sanitize_tts_text, split_omnivoice_tts_text

CHUNK_CACHE_SCHEMA_VERSION = 3

_DECIMAL_RE = re.compile(r"\d+\.\d+")
_MODEL_TOKEN_RE = re.compile(r"\b(?:RTX|GTX|RX)\s*\d{3,5}\b", re.IGNORECASE)


def omnivoice_chunk_settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    settings = settings or {}

    def _int(key: str, default: int) -> int:
        try:
            return int(settings.get(key, default) or default)
        except (TypeError, ValueError):
            return default

    def _bool(key: str, default: bool) -> bool:
        value = settings.get(key, default)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() not in {"0", "false", "no", "off"}

    target = _int("omnivoice_chunk_target_chars", 180)
    hard_max = _int("omnivoice_chunk_max_chars", 220)
    min_chars = _int("omnivoice_chunk_min_chars", 40)
    if hard_max < min_chars:
        hard_max = min_chars + 20
    if target > hard_max:
        target = hard_max
    return {
        "external_chunking_enabled": _bool("omnivoice_external_chunking_enabled", False),
        "fallback_full_segment_enabled": _bool("omnivoice_chunk_fidelity_fallback_full_segment", True),
        "retry_on_fidelity_failure": _bool("omnivoice_chunk_retry_on_fidelity_failure", False),
        "target_chars": target,
        "max_chars": hard_max,
        "min_chars": min_chars,
        "long_text_threshold": _int("omnivoice_long_text_threshold", 240),
        "very_long_threshold": _int("omnivoice_very_long_text_threshold", 500),
        "pause_comma_ms": _int("omnivoice_pause_comma_ms", 140),
        "pause_sentence_ms": _int("omnivoice_pause_sentence_ms", 260),
        "pause_hard_ms": _int("omnivoice_pause_hard_ms", 50),
        "max_retries": _int("omnivoice_chunk_max_retries", 2),
        "retry_max_chars": (
            _int("omnivoice_chunk_retry_max_chars_1", 220),
            _int("omnivoice_chunk_retry_max_chars_2", 140),
            _int("omnivoice_chunk_retry_max_chars_3", 90),
        ),
        "fidelity_threshold": float(settings.get("omnivoice_fidelity_good_threshold", 0.85) or 0.85),
        "fidelity_review_threshold": float(settings.get("omnivoice_fidelity_review_threshold", 0.70) or 0.70),
        "fidelity_critical_threshold": float(settings.get("omnivoice_fidelity_critical_threshold", 0.55) or 0.55),
        "fidelity_check_min_chars": _int("omnivoice_fidelity_check_min_chars", 240),
        "fidelity_enabled": _bool("omnivoice_fidelity_check_enabled", True),
    }


def normalize_text_for_compare(text: str) -> str:
    cleaned = sanitize_tts_text(text or "")
    cleaned = unicodedata.normalize("NFKC", cleaned).casefold()
    cleaned = re.sub(r"[^\w\s]", " ", cleaned, flags=re.UNICODE)
    return re.sub(r"\s+", " ", cleaned).strip()


def validate_chunk_reconstruction(original: str, chunks: list[str]) -> None:
    if not chunks:
        raise ValueError("Chunk list is empty.")
    if any(not chunk.strip() for chunk in chunks):
        raise ValueError("Empty chunk detected.")
    cleaned = re.sub(r"\s+", " ", sanitize_tts_text(original).strip())
    cursor = 0
    for chunk in chunks:
        position = cleaned.find(chunk, cursor)
        if position < 0:
            joined_norm = normalize_text_for_compare("".join(chunks))
            original_norm = normalize_text_for_compare(cleaned)
            if joined_norm != original_norm:
                raise ValueError(
                    f"Chunk reconstruction mismatch: original={len(original_norm)} joined={len(joined_norm)}"
                )
            return
        cursor = position + len(chunk)
    if cursor != len(cleaned):
        joined_norm = normalize_text_for_compare("".join(chunks))
        original_norm = normalize_text_for_compare(cleaned)
        if joined_norm != original_norm:
            raise ValueError(
                f"Chunk reconstruction incomplete: consumed={cursor} expected={len(cleaned)}"
            )


def _protect_decimal_spans(text: str) -> tuple[str, dict[str, str]]:
    placeholders: dict[str, str] = {}

    def _repl_decimal(match: re.Match[str]) -> str:
        key = f"__DEC{len(placeholders)}__"
        placeholders[key] = match.group(0)
        return key

    def _repl_model(match: re.Match[str]) -> str:
        key = f"__MDL{len(placeholders)}__"
        placeholders[key] = match.group(0)
        return key

    protected = _DECIMAL_RE.sub(_repl_decimal, text)
    protected = _MODEL_TOKEN_RE.sub(_repl_model, protected)
    return protected, placeholders


def _restore_placeholders(text: str, placeholders: dict[str, str]) -> str:
    restored = text
    for key, value in placeholders.items():
        restored = restored.replace(key, value)
    return restored


def split_omnivoice_text_semantic(
    text: str,
    *,
    max_chars: int,
    target_chars: int | None = None,
) -> list[dict[str, Any]]:
    """Split text into semantic chunks with metadata."""
    del target_chars  # reserved for future target-aware packing
    cleaned = re.sub(r"\s+", " ", sanitize_tts_text(text).strip())
    if not cleaned:
        return []

    protected, placeholders = _protect_decimal_spans(cleaned)
    raw_chunks = split_omnivoice_tts_text(protected, max_chars=max_chars)
    restored = [_restore_placeholders(chunk, placeholders) for chunk in raw_chunks]
    validate_chunk_reconstruction(cleaned, restored)

    result: list[dict[str, Any]] = []
    cursor = 0
    for index, chunk in enumerate(restored):
        needle = chunk.strip()
        start = cleaned.find(needle, cursor)
        if start < 0:
            start = cursor
        if index + 1 < len(restored):
            next_needle = restored[index + 1].strip()
            next_start = cleaned.find(next_needle, start + len(needle))
            end = next_start if next_start >= start else start + len(needle)
        else:
            end = len(cleaned)
        exact = cleaned[start:end]
        cursor = end
        split_kind = "sentence"
        trimmed = exact.rstrip()
        boundary = trimmed[-1:] if trimmed else ""
        if boundary in ".!?…。！？":
            split_kind = "sentence"
        elif boundary in ",，;；:：":
            split_kind = "comma"
        elif len(exact) >= max_chars - 1:
            split_kind = "hard"
        else:
            split_kind = "word"
        result.append(
            {
                "chunk_index": index,
                "text": exact,
                "text_start": start,
                "text_end": end,
                "split_kind": split_kind,
            }
        )
    validate_chunk_reconstruction(cleaned, [item["text"] for item in result])
    return result


def chunking_required(text: str, settings: dict[str, Any] | None) -> bool:
    cfg = omnivoice_chunk_settings(settings)
    if not cfg["external_chunking_enabled"]:
        return False
    cleaned = re.sub(r"\s+", " ", sanitize_tts_text(text).strip())
    return len(cleaned) > cfg["max_chars"]


def pause_ms_for_chunk(chunk_text: str, split_kind: str, cfg: dict[str, Any]) -> int:
    trailing = chunk_text.rstrip()[-1:] if chunk_text.rstrip() else ""
    if trailing in ".!?…。！？":
        return int(cfg["pause_sentence_ms"])
    if trailing in ",，;；:：":
        return int(cfg["pause_comma_ms"])
    if split_kind == "hard":
        return int(cfg["pause_hard_ms"])
    if split_kind == "word":
        return int(cfg["pause_hard_ms"])
    return 0


def segment_text_diagnostics(text: str, settings: dict[str, Any] | None) -> dict[str, Any]:
    cfg = omnivoice_chunk_settings(settings)
    length = len(re.sub(r"\s+", " ", sanitize_tts_text(text).strip()))
    flags: list[str] = []
    if length > cfg["long_text_threshold"]:
        flags.append("long_text_segment")
    if length > cfg["very_long_threshold"]:
        flags.append("very_long_text_segment")
    if chunking_required(text, settings):
        flags.append("tts_chunking_required")
    sentence_count = len(re.findall(r"[.!?…。！？]", text or ""))
    if sentence_count >= 2:
        flags.append("multi_sentence_segment")
    return {
        "text_length": length,
        "segment_diagnostics": flags,
    }
