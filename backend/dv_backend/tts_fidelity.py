"""TTS content fidelity checks (text similarity, optional ASR)."""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

from .adapters.tts import sanitize_tts_text
from .omnivoice_chunking import omnivoice_chunk_settings


def normalize_fidelity_text(text: str) -> str:
    cleaned = sanitize_tts_text(text or "")
    cleaned = unicodedata.normalize("NFKC", cleaned).casefold()
    cleaned = re.sub(r"[^\w\s]", " ", cleaned, flags=re.UNICODE)
    return re.sub(r"\s+", " ", cleaned).strip()


def compact_fidelity_text(text: str) -> str:
    return "".join(normalize_fidelity_text(text).split())


def compact_text_similarity(expected: str, heard: str) -> float:
    left = compact_fidelity_text(expected)
    right = compact_fidelity_text(heard)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def text_similarity(expected: str, heard: str) -> float:
    left = normalize_fidelity_text(expected)
    right = normalize_fidelity_text(heard)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def content_coverage(expected: str, heard: str) -> float:
    expected_tokens = normalize_fidelity_text(expected).split()
    heard_norm = normalize_fidelity_text(heard)
    if not expected_tokens:
        return 1.0
    if not heard_norm:
        return 0.0
    matched = sum(1 for token in expected_tokens if token in heard_norm)
    return matched / len(expected_tokens)


def fidelity_status_from_scores(
    similarity: float,
    *,
    cfg: dict[str, Any],
) -> str:
    if similarity >= float(cfg["fidelity_threshold"]):
        return "good"
    if similarity >= float(cfg["fidelity_review_threshold"]):
        return "review"
    if similarity >= float(cfg["fidelity_critical_threshold"]):
        return "poor"
    return "failed"


def should_run_fidelity_check(
    *,
    text: str,
    chunk_count: int,
    settings: dict[str, Any] | None,
    speech_duration: float | None = None,
    raw_duration: float | None = None,
) -> bool:
    cfg = omnivoice_chunk_settings(settings)
    if not cfg["fidelity_enabled"]:
        return False
    if bool((settings or {}).get("omnivoice_fidelity_check_all_segments", False)):
        return True
    cleaned = re.sub(r"\s+", " ", sanitize_tts_text(text).strip())
    if chunk_count > 1:
        return True
    if len(cleaned) >= int(cfg["fidelity_check_min_chars"]):
        return True
    if speech_duration is not None and raw_duration is not None:
        if raw_duration > 0 and speech_duration / raw_duration < 0.12 and len(cleaned) >= 80:
            return True
    return False


def max_contiguous_deletion_chars(expected: str, heard: str) -> int:
    """Longest deleted run in compact form (insertions in heard ignored)."""
    left = compact_fidelity_text(expected)
    right = compact_fidelity_text(heard)
    if not left:
        return 0
    matcher = SequenceMatcher(None, left, right)
    longest = 0
    for _tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if _tag == "delete":
            longest = max(longest, i2 - i1)
    return longest


def evaluate_tts_fidelity(
    *,
    expected_text: str,
    heard_text: str,
    settings: dict[str, Any] | None,
) -> dict[str, Any]:
    cfg = omnivoice_chunk_settings(settings)
    spaced_similarity = text_similarity(expected_text, heard_text)
    compact_similarity = compact_text_similarity(expected_text, heard_text)
    coverage = content_coverage(expected_text, heard_text)
    # Vietnamese ASR can return text without word spacing. Use the compact score for status
    # when it shows high literal coverage, while preserving both raw scores for diagnostics.
    effective_similarity = max(spaced_similarity, compact_similarity)
    deletion_span = max_contiguous_deletion_chars(expected_text, heard_text)
    status = fidelity_status_from_scores(effective_similarity, cfg=cfg)
    warnings: list[str] = []
    if status in {"poor", "failed"}:
        warnings.append("tts_fidelity_low")
    if deletion_span >= 12 and coverage < 0.98:
        if status == "good":
            status = "failed"
        warnings.append("tts_contiguous_deletion_span")
    if coverage < float(cfg["fidelity_critical_threshold"]):
        warnings.append("tts_content_coverage_low")
    return {
        "tts_text_similarity": round(effective_similarity, 4),
        "tts_spaced_text_similarity": round(spaced_similarity, 4),
        "tts_compact_text_similarity": round(compact_similarity, 4),
        "tts_content_coverage": round(coverage, 4),
        "tts_max_contiguous_deletion": deletion_span,
        "tts_fidelity_status": status,
        "tts_fidelity_warnings": warnings,
        "tts_asr_text": heard_text if heard_text else None,
    }


TranscribeFn = Callable[[Path], str]


def transcribe_wav_to_text(
    wav_path: Path,
    *,
    vendor_dir: Path,
    language: str = "Vietnamese",
) -> str:
    from .adapters.asr import transcribe_audio

    segments = transcribe_audio(
        wav_path,
        vendor_dir=vendor_dir,
        language=language,
        include_alignment=False,
    )
    if isinstance(segments, dict):
        segments = segments.get("segments") or []
    return " ".join(str(item.get("text") or "").strip() for item in segments).strip()


def run_segment_fidelity_check(
    *,
    wav_path: Path,
    expected_text: str,
    settings: dict[str, Any] | None,
    chunk_count: int,
    speech_duration: float | None,
    raw_duration: float | None,
    transcribe_fn: TranscribeFn | None = None,
    vendor_dir: Path | None = None,
) -> dict[str, Any]:
    if not should_run_fidelity_check(
        text=expected_text,
        chunk_count=chunk_count,
        settings=settings,
        speech_duration=speech_duration,
        raw_duration=raw_duration,
    ):
        return {
            "tts_text_similarity": None,
            "tts_content_coverage": None,
            "tts_fidelity_status": "not_checked",
            "tts_fidelity_warnings": [],
            "tts_asr_text": None,
        }
    if transcribe_fn is not None:
        heard = transcribe_fn(wav_path)
    elif vendor_dir is not None and wav_path.is_file():
        try:
            heard = transcribe_wav_to_text(wav_path, vendor_dir=vendor_dir)
        except Exception:
            return {
                "tts_text_similarity": None,
                "tts_content_coverage": None,
                "tts_fidelity_status": "not_checked",
                "tts_fidelity_warnings": ["tts_fidelity_asr_unavailable"],
                "tts_asr_text": None,
            }
    else:
        return {
            "tts_text_similarity": None,
            "tts_content_coverage": None,
            "tts_fidelity_status": "not_checked",
            "tts_fidelity_warnings": [],
            "tts_asr_text": None,
        }
    return evaluate_tts_fidelity(
        expected_text=expected_text,
        heard_text=heard,
        settings=settings,
    )
