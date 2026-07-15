"""Clause-aware content fidelity for OmniVoice clone diagnostics and mitigations."""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

ClauseStrategy = Literal["d1", "d2", "balanced", "clauses"]

_CLAUSE_SPLIT_RE = re.compile(r"(?<=[?!.!;；。！？\n])\s*")
_PUNCT_RE = re.compile(r"[?!.!;；。！？,:，、…]+")
_WHITESPACE_RUN_RE = re.compile(r"\s{2,}")


def normalize_content_compare_text(text: str) -> str:
    """Normalize text for ASR/content comparison without dropping Vietnamese letters."""
    value = unicodedata.normalize("NFC", str(text or ""))
    value = value.casefold()
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", value)
    value = _PUNCT_RE.sub(" ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def compact_content_compare_text(text: str) -> str:
    """Vietnamese ASR often omits spaces; compact form avoids false negatives."""
    return "".join(normalize_content_compare_text(text).split())


def _sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _phrase_key(phrase: str) -> str:
    return normalize_content_compare_text(phrase)


def split_content_clauses(text: str) -> list[str]:
    """Split on ?, ., !, ;, newline while keeping punctuation with the clause."""
    cleaned = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        return []
    parts = _CLAUSE_SPLIT_RE.split(cleaned)
    clauses: list[str] = []
    for part in parts:
        piece = part.strip()
        if piece:
            clauses.append(piece)
    return _merge_punctuation_only_clauses(clauses) or [cleaned]


def _merge_punctuation_only_clauses(clauses: list[str]) -> list[str]:
    """Attach punctuation-only fragments to neighboring clauses for TTS chunking."""
    merged: list[str] = []
    pending_prefix = ""
    for clause in clauses:
        stripped = clause.strip()
        if not stripped:
            continue
        if re.fullmatch(r"[\s?!.!;；。！？,:，、…]+", stripped):
            if merged:
                merged[-1] = f"{merged[-1]}{stripped}"
            else:
                pending_prefix += stripped
            continue
        if pending_prefix:
            merged.append(f"{pending_prefix}{stripped}")
            pending_prefix = ""
        else:
            merged.append(stripped)
    if pending_prefix:
        if merged:
            merged[-1] = f"{merged[-1]}{pending_prefix}"
        else:
            merged.append(pending_prefix)
    return merged


def plan_clone_semantic_chunks(
    text: str,
    *,
    strategy: ClauseStrategy = "balanced",
    min_merge_chars: int = 12,
) -> list[str]:
    """Plan clone-aware semantic chunks without dropping meaningful characters."""
    clauses = split_content_clauses(text)
    if not clauses:
        return []
    if strategy == "clauses":
        return list(clauses)
    if strategy == "d1":
        if len(clauses) <= 1:
            return list(clauses)
        mid = max(1, len(clauses) // 2)
        return [
            " ".join(clauses[:mid]).strip(),
            " ".join(clauses[mid:]).strip(),
        ]
    if strategy == "d2":
        if len(clauses) <= 2:
            return list(clauses)
        # Prefer short opener, middle pair, trailing clause(s).
        first = clauses[0]
        last = clauses[-1]
        middle = " ".join(clauses[1:-1]).strip()
        return [first, middle, last] if middle else [first, last]

    # balanced: keep punctuation splits, merge very short clauses into neighbors.
    merged: list[str] = []
    for clause in clauses:
        if merged and len(normalize_content_compare_text(clause)) < min_merge_chars:
            merged[-1] = f"{merged[-1]} {clause}".strip()
        else:
            merged.append(clause)
    # If still many tiny pieces, pair adjacent clauses (D1-like).
    if len(merged) > 3:
        paired: list[str] = []
        index = 0
        while index < len(merged):
            if index + 1 < len(merged):
                paired.append(f"{merged[index]} {merged[index + 1]}".strip())
                index += 2
            else:
                paired.append(merged[index])
                index += 1
        return paired
    return merged


def _edit_distance(left: list[str], right: list[str]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    prev = list(range(len(right) + 1))
    for i, left_token in enumerate(left, start=1):
        curr = [i]
        for j, right_token in enumerate(right, start=1):
            cost = 0 if left_token == right_token else 1
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1]


def _ordered_token_coverage(expected: str, heard: str) -> float:
    expected_tokens = normalize_content_compare_text(expected).split()
    heard_tokens = normalize_content_compare_text(heard).split()
    if not expected_tokens:
        return 1.0
    # Prefer spaced matching when ASR keeps spaces.
    if heard_tokens:
        cursor = 0
        matched = 0
        for token in expected_tokens:
            while cursor < len(heard_tokens) and heard_tokens[cursor] != token:
                cursor += 1
            if cursor >= len(heard_tokens):
                break
            matched += 1
            cursor += 1
        spaced = matched / len(expected_tokens)
        if spaced > 0:
            return spaced
    # Fallback: contiguous compact scan (common for Vietnamese ASR).
    expected_compact = compact_content_compare_text(expected)
    heard_compact = compact_content_compare_text(heard)
    if not expected_compact:
        return 1.0
    if not heard_compact:
        return 0.0
    cursor = 0
    matched_chars = 0
    for token in expected_tokens:
        compact_token = "".join(token.split())
        if not compact_token:
            continue
        pos = heard_compact.find(compact_token, cursor)
        if pos < 0:
            break
        matched_chars += len(compact_token)
        cursor = pos + len(compact_token)
    return matched_chars / max(1, len(expected_compact))


def _ordered_clauses_ok(expected_clauses: list[str], heard_norm: str, heard_compact: str) -> bool:
    cursor_spaced = 0
    cursor_compact = 0
    for clause in expected_clauses:
        piece = normalize_content_compare_text(clause)
        compact = compact_content_compare_text(clause)
        if not piece:
            continue
        pos = heard_norm.find(piece, cursor_spaced)
        if pos >= 0:
            cursor_spaced = pos + len(piece)
            cursor_compact = heard_compact.find(compact, cursor_compact)
            if cursor_compact >= 0:
                cursor_compact += len(compact)
            continue
        pos_c = heard_compact.find(compact, cursor_compact)
        if pos_c < 0:
            return False
        cursor_compact = pos_c + len(compact)
    return True


def evaluate_content_fidelity(
    expected_text: str,
    recognized_text: str,
    critical_phrases: list[str] | None = None,
) -> dict[str, Any]:
    expected_clauses = split_content_clauses(expected_text)
    recognized_clauses = split_content_clauses(recognized_text)
    expected_norm = normalize_content_compare_text(expected_text)
    heard_norm = normalize_content_compare_text(recognized_text)
    heard_compact = compact_content_compare_text(recognized_text)

    missing_clauses: list[str] = []
    for clause in expected_clauses:
        clause_norm = normalize_content_compare_text(clause)
        clause_compact = compact_content_compare_text(clause)
        if not clause_norm:
            continue
        if clause_norm in heard_norm or clause_compact in heard_compact:
            continue
        tokens = clause_norm.split()
        if tokens and all(
            token in heard_norm or "".join(token.split()) in heard_compact for token in tokens
        ):
            continue
        missing_clauses.append(clause_norm)

    phrases = critical_phrases or []
    critical: dict[str, bool] = {}
    for phrase in phrases:
        key = _phrase_key(phrase)
        compact = compact_content_compare_text(phrase)
        critical[key] = bool(key) and (key in heard_norm or compact in heard_compact)

    expected_chars = list(compact_content_compare_text(expected_text))
    heard_chars = list(heard_compact)
    expected_tokens = expected_norm.split()
    heard_tokens = heard_norm.split()
    cer = (
        _edit_distance(expected_chars, heard_chars) / max(1, len(expected_chars))
        if expected_chars
        else 0.0
    )
    # If ASR is spaceless, approximate WER with character CER of token join.
    if len(heard_tokens) <= 1 and expected_tokens:
        wer = cer
    else:
        wer = (
            _edit_distance(expected_tokens, heard_tokens) / max(1, len(expected_tokens))
            if expected_tokens
            else 0.0
        )

    return {
        "ordered_token_coverage": round(_ordered_token_coverage(expected_text, recognized_text), 4),
        "cer": round(cer, 4),
        "wer": round(wer, 4),
        "expected_clauses": [normalize_content_compare_text(c) for c in expected_clauses],
        "recognized_clauses": [normalize_content_compare_text(c) for c in recognized_clauses],
        "missing_clauses": missing_clauses,
        "critical_phrases": critical,
        "ordered_clause_ok": _ordered_clauses_ok(expected_clauses, heard_norm, heard_compact),
        "missing_any_clause": bool(missing_clauses),
    }


def describe_target_text_for_generate(text: str, *, mode: str) -> dict[str, Any]:
    value = str(text or "")
    nfc = unicodedata.normalize("NFC", value)
    collapsed = re.sub(r"\s+", " ", nfc).strip()
    whitespace_runs = [match.group(0) for match in _WHITESPACE_RUN_RE.finditer(value)]
    punctuation_sequence = "".join(ch for ch in value if ch in "?!.!;；。！？,:，、…")
    critical = "tôi là minh"
    return {
        "mode": mode,
        "target_text_length": len(value),
        "target_text_sha256": _sha256_hex(value),
        "normalized_target_text_sha256": _sha256_hex(collapsed),
        "unicode_normalization": "NFC",
        "whitespace_runs": len(whitespace_runs),
        "whitespace_run_lengths": [len(run) for run in whitespace_runs],
        "punctuation_sequence": punctuation_sequence,
        "contains_critical_span": critical in normalize_content_compare_text(value),
        "clause_count": len(split_content_clauses(value)),
    }


def normalize_target_text_for_synthesis(text: str) -> str:
    """Safe production normalization applied before auto/instruct/clone branching."""
    value = unicodedata.normalize("NFC", str(text or ""))
    value = "".join(ch for ch in value if not (ord(ch) < 32 and ch not in "\t\n\r"))
    value = re.sub(r"[ \t\f\v]+", " ", value)
    value = re.sub(r" *\n+ *", "\n", value)
    return value.strip()


def resolve_clone_chunk_strategy(settings: dict[str, Any] | None = None) -> str:
    """Resolve clone clause strategy. Env wins: DV_OMNIVOICE_CLONE_CHUNK_STRATEGY."""
    env = str(os.environ.get("DV_OMNIVOICE_CLONE_CHUNK_STRATEGY", "") or "").strip().lower()
    if env in {"clauses", "off"}:
        return env
    configured = str((settings or {}).get("omnivoice_clone_chunk_strategy", "clauses") or "clauses")
    configured = configured.strip().lower()
    if configured in {"clauses", "off"}:
        return configured
    return "clauses"


def split_omnivoice_clone_clauses(text: str) -> list[str]:
    """Canonical clone clause splitter with reconstruction invariant."""
    cleaned = normalize_target_text_for_synthesis(text)
    if not cleaned:
        raise ValueError("Cannot split empty OmniVoice clone text.")
    chunks = split_content_clauses(cleaned)
    if not chunks:
        raise ValueError("OmniVoice clone clause split produced no chunks.")
    joined = " ".join(chunks)
    if normalize_content_compare_text(joined) != normalize_content_compare_text(cleaned):
        raise ValueError(
            "OmniVoice clone clause reconstruction invariant failed "
            f"(source_hash={_sha256_hex(cleaned)[:12]} joined_hash={_sha256_hex(joined)[:12]})."
        )
    return chunks


def clone_content_chunking_required(
    text: str,
    *,
    is_clone: bool,
    settings: dict[str, Any] | None = None,
) -> bool:
    """Enable clause chunking only for OmniVoice clone with multiple clauses."""
    if not is_clone:
        return False
    if resolve_clone_chunk_strategy(settings) == "off":
        return False
    return len(split_content_clauses(text)) >= 2


def is_clone_voice_path(voice: str | None) -> bool:
    value = str(voice or "").strip()
    if not value:
        return False
    lower = value.lower()
    return lower.endswith(".wav") or lower.endswith(".flac") or lower.endswith(".mp3")


def synthesize_clone_content_preserving(
    *,
    text: str,
    output_path: Path,
    synthesize_fn: Callable[[str, Path], None],
    validate_chunk_fn: Callable[[Path], None] | None = None,
    pause_ms: int = 0,
    strategy: ClauseStrategy = "clauses",
) -> dict[str, Any]:
    """Generate clone audio clause-by-clause and concatenate in order.

    Caller must reuse the same clone conditioning across chunks (worker prompt cache).
    Fails the whole request if any chunk fails validation.
    """
    from .errors import AppError
    from .models import ErrorInfo
    from .omnivoice_chunk_synthesis import _validate_chunk_wav
    from .omnivoice_wav_concat import concat_omnivoice_chunks
    from .tts_speech_analysis import measure_speech_envelope

    cleaned = normalize_target_text_for_synthesis(text)
    if strategy == "clauses":
        chunks = split_omnivoice_clone_clauses(cleaned)
    else:
        chunks = plan_clone_semantic_chunks(cleaned, strategy=strategy)
        joined = " ".join(chunks)
        if normalize_content_compare_text(joined) != normalize_content_compare_text(cleaned):
            raise AppError(
                502,
                ErrorInfo(
                    code="OMNIVOICE_CLONE_CLAUSE_SPLIT_INVALID",
                    message="Clone clause reconstruction invariant failed.",
                    action="Retry synthesis or disable clone clause chunking.",
                    retryable=True,
                ),
            )
    if len(chunks) <= 1:
        synthesize_fn(cleaned, output_path)
        validator = validate_chunk_fn or _validate_chunk_wav
        validator(output_path)
        return {
            "tts_clone_content_chunking_used": False,
            "tts_clone_content_chunk_count": 1,
            "tts_clone_content_chunks": chunks or [cleaned],
            "clone_chunk_strategy": strategy,
            "chunk_count": 1,
            "chunk_lengths": [len(cleaned)],
        }

    chunk_dir = output_path.parent / f"{output_path.stem}.clone_clauses"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths: list[Path] = []
    chunk_durations: list[float] = []
    validator = validate_chunk_fn or _validate_chunk_wav
    for index, chunk_text in enumerate(chunks):
        if not str(chunk_text).strip():
            raise AppError(
                502,
                ErrorInfo(
                    code="OMNIVOICE_CLONE_CLAUSE_EMPTY",
                    message="Clone clause chunk is empty.",
                    action="Retry TTS for this segment.",
                    detail=json.dumps(
                        {
                            "chunk_index": index,
                            "chunk_count": len(chunks),
                            "chunk_text_hash": _sha256_hex(chunk_text)[:16],
                            "mode": "clone",
                            "failure_reason": "empty_chunk",
                        },
                        ensure_ascii=True,
                    ),
                    retryable=True,
                ),
            )
        chunk_path = chunk_dir / f"chunk_{index:03d}.wav"
        try:
            synthesize_fn(chunk_text, chunk_path)
            validator(chunk_path)
        except AppError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AppError(
                502,
                ErrorInfo(
                    code="OMNIVOICE_CLONE_CLAUSE_FAILED",
                    message="Clone clause synthesis failed.",
                    action="Retry TTS for this segment.",
                    detail=json.dumps(
                        {
                            "chunk_index": index,
                            "chunk_count": len(chunks),
                            "chunk_text_hash": _sha256_hex(chunk_text)[:16],
                            "mode": "clone",
                            "failure_reason": str(exc)[:240],
                        },
                        ensure_ascii=True,
                    ),
                    retryable=True,
                ),
            ) from exc
        envelope = measure_speech_envelope(chunk_path)
        chunk_durations.append(float(envelope.raw_wav_duration or 0.0))
        chunk_paths.append(chunk_path)

    pauses = [max(0, int(pause_ms))] * max(0, len(chunk_paths) - 1)
    concat_omnivoice_chunks(chunk_paths, pause_ms_list=pauses, output_path=output_path)
    validator(output_path)
    final_envelope = measure_speech_envelope(output_path)
    sum_chunk = sum(chunk_durations)
    final_duration = float(final_envelope.raw_wav_duration or 0.0)
    return {
        "tts_clone_content_chunking_used": True,
        "tts_clone_content_chunk_count": len(chunks),
        "tts_clone_content_chunks": chunks,
        "tts_clone_content_strategy": strategy,
        "clone_chunk_strategy": strategy,
        "chunk_count": len(chunks),
        "chunk_lengths": [len(chunk) for chunk in chunks],
        "chunk_duration_sec": [round(value, 4) for value in chunk_durations],
        "final_duration_sec": round(final_duration, 4),
        "sum_chunk_duration_sec": round(sum_chunk, 4),
        "duration_delta_sec": round(final_duration - sum_chunk, 4),
    }
