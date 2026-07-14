"""Typography sanitation for spoken/subtitle text (no semantic rewrite)."""

from __future__ import annotations

import re
from typing import Any


_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
_MULTI_SPACE = re.compile(r"\s{2,}")
_LEADING_ELLIPSIS = re.compile(r"^(?:\.\.\.|…|\. \. \.)\s*")
_SPACED_ELLIPSIS = re.compile(r"(?:\.\s*){2,}\.")


def sanitize_spoken_text(text: str, *, strip_leading_ellipsis: bool = False) -> str:
    """Fix orphan punctuation only. Keep intentional '...' pauses intact."""
    cleaned = (text or "").replace("…", "...")
    if strip_leading_ellipsis:
        cleaned = _LEADING_ELLIPSIS.sub("", cleaned)
    # Normalize spaced ellipsis ". . ." → "..." before any dot collapsing.
    cleaned = _SPACED_ELLIPSIS.sub("...", cleaned)
    cleaned = cleaned.replace("...", "\0ELLIPSIS\0")
    cleaned = cleaned.replace(" . ", ". ")
    cleaned = cleaned.replace("..", ".")
    cleaned = cleaned.replace("\0ELLIPSIS\0", "...")
    cleaned = _SPACE_BEFORE_PUNCT.sub(r"\1", cleaned)
    cleaned = _MULTI_SPACE.sub(" ", cleaned).strip()
    return cleaned


def punctuation_artifact_issues(text: str) -> list[str]:
    raw = text or ""
    issues: list[str] = []
    if " . " in raw:
        issues.append("space_dot_space")
    if ". ." in raw:
        issues.append("dot_space_dot")
    without_ellipsis = raw.replace("...", "")
    if ".." in without_ellipsis:
        issues.append("double_dot")
    if re.search(r"\s+[,.;:!?]", raw):
        issues.append("space_before_punctuation")
    stripped = raw.strip()
    if stripped and re.fullmatch(r"[.!?…,\s]+", stripped):
        issues.append("punctuation_only")
    words = [w for w in re.split(r"\s+", stripped) if w and not re.fullmatch(r"[.!?…]+", w)]
    if 0 < len(words) <= 2 and stripped.startswith(("...", "…", ". . .")):
        issues.append("orphan_short_continuation")
    return issues


def validate_segments_text_sanitation(segments: list[dict[str, Any]]) -> dict[str, Any]:
    bad: list[dict[str, Any]] = []
    for segment in segments:
        text = str(
            segment.get("tts_spoken_text")
            or segment.get("translation")
            or segment.get("target_text")
            or ""
        ).strip()
        if not text:
            continue
        issues = punctuation_artifact_issues(text)
        if issues:
            bad.append({"index": int(segment.get("index", -1)), "issues": issues, "text": text[:80]})
    return {"passed": not bad, "blocking": bad, "count": len(bad)}
