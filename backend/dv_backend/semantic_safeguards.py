"""Deterministic semantic safeguards for translation candidate selection."""

from __future__ import annotations

import re
from typing import Any

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?%?")
_PERCENT = re.compile(r"\d+(?:[.,]\d+)?%")
_MODEL_ID = re.compile(r"\b(?:RTX|GTX|RX|USB|AI|Wi-?Fi)\s*[-\w]*\d+[\w-]*\b", re.IGNORECASE)
_ALNUM_MODEL = re.compile(r"\b[A-Z]{2,}\d{2,}[A-Z0-9-]*\b")
_PROPER_NOUN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
_NEGATION_VI = re.compile(r"\b(không|chưa|chẳng|đừng|đừng có|chớ|đừng nên)\b", re.IGNORECASE)
_NEGATION_ZH = re.compile(r"(不|没|未|别|勿|无)")
_COMPARISON = re.compile(
    r"\b(hơn|kém|lớn nhất|nhỏ nhất|nhanh hơn|chậm hơn|cao hơn|thấp hơn|tốt hơn|mạnh hơn)\b",
    re.IGNORECASE,
)
_QUANTIFIER = re.compile(r"\b(tất cả|mọi|một số|chỉ|duy nhất|hầu hết)\b", re.IGNORECASE)
_QUESTION = re.compile(r"[?？]|^(?:ai|gì|sao|tại sao|bao nhiêu|ở đâu)\b", re.IGNORECASE)


def _count(pattern: re.Pattern[str], text: str) -> int:
    return len(pattern.findall(text or ""))


def evaluate_semantic_safeguards(
    candidate_text: str,
    *,
    source_text: str,
    reference_text: str | None = None,
) -> dict[str, Any]:
    """Return semantic score, penalties, and rejection reasons."""
    candidate = (candidate_text or "").strip()
    source = (source_text or "").strip()
    reference = (reference_text or source).strip()
    penalties: list[str] = []
    critical = False

    if not candidate:
        return {
            "semantic_score": 0.0,
            "penalties": ["rejected_empty"],
            "critical_violation": True,
            "rejection_reasons": ["rejected_empty"],
        }

    cand_nums = _count(_NUMBER, candidate)
    ref_nums = max(_count(_NUMBER, reference), _count(_NUMBER, source))
    if ref_nums > 0 and cand_nums < ref_nums:
        penalties.append("rejected_missing_number")
        critical = True

    cand_pct = _count(_PERCENT, candidate)
    ref_pct = max(_count(_PERCENT, reference), _count(_PERCENT, source))
    if ref_pct > 0 and cand_pct < ref_pct:
        penalties.append("rejected_missing_percentage")
        critical = True

    ref_neg = bool(_NEGATION_VI.search(reference) or _NEGATION_ZH.search(source))
    cand_neg = bool(_NEGATION_VI.search(candidate))
    if ref_neg and not cand_neg:
        penalties.append("rejected_missing_negation")
        critical = True

    ref_models = set(_MODEL_ID.findall(reference)) | set(_ALNUM_MODEL.findall(reference))
    cand_models = set(_MODEL_ID.findall(candidate)) | set(_ALNUM_MODEL.findall(candidate))
    if ref_models and not ref_models.issubset(cand_models):
        penalties.append("rejected_missing_entity")
        critical = True

    ref_entities = set(_PROPER_NOUN.findall(reference))
    cand_entities = set(_PROPER_NOUN.findall(candidate))
    if ref_entities and len(cand_entities) < len(ref_entities):
        missing = ref_entities - cand_entities
        if missing:
            penalties.append("rejected_missing_entity")
            if len(missing) >= max(1, len(ref_entities) // 2):
                critical = True

    ref_cmp = _count(_COMPARISON, reference)
    cand_cmp = _count(_COMPARISON, candidate)
    if ref_cmp > 0 and cand_cmp < ref_cmp:
        penalties.append("rejected_missing_comparison")

    ref_q = _count(_QUANTIFIER, reference)
    cand_q = _count(_QUANTIFIER, candidate)
    if ref_q > 0 and cand_q < ref_q:
        penalties.append("rejected_missing_quantifier")

    if len(candidate) < max(3, len(reference) * 0.3):
        penalties.append("rejected_too_short")

    words = re.findall(r"\w+", candidate, flags=re.UNICODE)
    if len(words) >= 2:
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        if unique_ratio < 0.6:
            penalties.append("rejected_duplicate")

    penalty_weight = min(1.0, 0.2 * len(penalties) + (0.5 if critical else 0.0))
    semantic_score = max(0.0, 1.0 - penalty_weight)

    return {
        "semantic_score": round(semantic_score, 4),
        "penalties": penalties,
        "critical_violation": critical,
        "rejection_reasons": list(penalties),
    }


def candidate_passes_semantic_guards(
    candidate_text: str,
    *,
    source_text: str,
    reference_text: str | None = None,
) -> bool:
    result = evaluate_semantic_safeguards(
        candidate_text,
        source_text=source_text,
        reference_text=reference_text,
    )
    return not result["critical_violation"]
