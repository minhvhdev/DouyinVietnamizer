"""TTS provenance validation and clause-seam detection."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spoken_text(segment: dict[str, Any]) -> str:
    return str(
        segment.get("tts_spoken_text")
        or segment.get("translation")
        or segment.get("target_text")
        or ""
    ).strip()


def resolve_voiced_tts_path(segment: dict[str, Any]) -> Path | None:
    """Resolve voiced audio from explicit provenance only (no index guessing)."""
    for key in ("tts_path", "tts_raw_path"):
        candidate = segment.get(key)
        if not candidate:
            continue
        path = Path(str(candidate))
        if path.is_file():
            return path
    return None


def validate_segment_tts_provenance(segment: dict[str, Any]) -> list[str]:
    """Return blocking issues for a voiced segment's audio provenance."""
    text = spoken_text(segment)
    if not text:
        if segment.get("timing_status") == "SILENCE" or segment.get("no_speech"):
            return []
        # Empty spoken text with no silence mark — still allow if no expected speech.
        return []

    issues: list[str] = []
    path = resolve_voiced_tts_path(segment)
    if path is None:
        issues.append("missing_tts_path")
        return issues
    if not path.is_file():
        issues.append("tts_path_missing_file")
        return issues

    claimed = segment.get("tts_path")
    if claimed and Path(str(claimed)).resolve() != path.resolve():
        issues.append("resolver_path_mismatch")

    expected = segment.get("tts_sha256")
    if expected:
        actual = sha256_file(path)
        if actual != expected:
            issues.append("tts_sha256_mismatch")
    return issues


def validate_segments_tts_provenance(segments: list[dict[str, Any]]) -> dict[str, Any]:
    bad: list[dict[str, Any]] = []
    for segment in segments:
        issues = validate_segment_tts_provenance(segment)
        if issues:
            bad.append({"index": int(segment.get("index", -1)), "issues": issues, "text": spoken_text(segment)[:80]})
    return {"passed": not bad, "blocking": bad, "count": len(bad)}


_ELLIPSIS_END = re.compile(r"(?:\.\.\.|…|\. \. \.)\s*$")
_TERMINAL = re.compile(r"[.!?…。！？]\s*$")


def is_incomplete_clause_end(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if _ELLIPSIS_END.search(cleaned):
        return True
    if cleaned.endswith((",", ";", ":", "-", "—")):
        return True
    if not _TERMINAL.search(cleaned):
        return True
    return False


def is_continuation_start(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if cleaned.startswith(("...", "…", ". . .")):
        return True
    # Single mid-sentence word/token starters like "chân." after verb complement.
    first = cleaned.split()[0] if cleaned.split() else cleaned
    if first[:1].islower():
        return True
    if len(first) <= 6 and first.rstrip(".!?…").islower():
        return True
    return False


def chinese_token_seam(left_cn: str, right_cn: str) -> bool:
    left = (left_cn or "").strip()
    right = (right_cn or "").strip()
    if not left or not right:
        return False
    # Same leading char duplicated across boundary (腿|腿打断).
    if right.startswith(left[-1:]) and len(left[-1:]) == 1:
        return True
    if left.endswith(right[:1]) and len(right) > 1:
        return True
    return False


def detect_clause_seams(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(segments, key=lambda item: int(item.get("index", 0) or 0))
    seams: list[dict[str, Any]] = []
    for left, right in zip(ordered, ordered[1:], strict=False):
        left_text = spoken_text(left)
        right_text = spoken_text(right)
        reasons: list[str] = []
        if is_incomplete_clause_end(left_text) and is_continuation_start(right_text):
            reasons.append("vi_ellipsis_continuation")
        if chinese_token_seam(str(left.get("text") or ""), str(right.get("text") or "")):
            reasons.append("cn_token_continuation")
        # Special case from evidence: gãy... | chân.
        if "gãy" in left_text and right_text.lower().startswith("chân"):
            reasons.append("vi_verb_complement_split")
        if "còn lại" in left_text and "trăm" in right_text and right_text.startswith(("...", "…", ". . .")):
            reasons.append("vi_quantity_split")
        if not reasons:
            continue
        seams.append(
            {
                "left_index": int(left.get("index", 0) or 0),
                "right_index": int(right.get("index", 0) or 0),
                "reasons": reasons,
                "left_text": left_text[:80],
                "right_text": right_text[:80],
            }
        )
    return seams


_HARD_REASON_KEYS = {
    "vi_verb_complement_split",
    "vi_quantity_split",
}


def _looks_like_entity_split(left_text: str, right_text: str) -> bool:
    stem = _ELLIPSIS_END.sub("", (left_text or "").strip()).strip()
    if not stem or not right_text:
        return False
    last = stem.split()[-1].rstrip(",;:") if stem.split() else ""
    first = right_text.strip().split()[0].rstrip(".!?…") if right_text.strip().split() else ""
    if not last or not first:
        return False
    # Title/name broken across units: "Đại. . ." | "nhân ..."
    if last[:1].isupper() and len(last) <= 6 and first[:1].islower() and len(first) <= 8:
        return True
    return False


def _both_sides_complete_clauses(left_text: str, right_text: str) -> bool:
    left = (left_text or "").strip()
    right = (right_text or "").strip()
    if not left or not right:
        return False
    if _ELLIPSIS_END.search(left) or is_continuation_start(right):
        return False
    if not _TERMINAL.search(left):
        return False
    first = right[0]
    return first.isupper() or first in "“\"'«"


def classify_clause_seam(
    seam: dict[str, Any],
    *,
    left_text: str | None = None,
    right_text: str | None = None,
) -> str:
    """Return 'hard' or 'soft' for a detected seam (ChatGPT TL P0.2)."""
    reasons = set(seam.get("reasons") or [])
    left = left_text if left_text is not None else str(seam.get("left_text") or "")
    right = right_text if right_text is not None else str(seam.get("right_text") or "")
    if reasons & _HARD_REASON_KEYS:
        return "hard"
    if _looks_like_entity_split(left, right):
        return "hard"
    if "vi_ellipsis_continuation" in reasons:
        # Upstream chunking residue that users hear as broken sentences.
        return "hard"
    if reasons == {"cn_token_continuation"} and _both_sides_complete_clauses(left, right):
        # Shared CN edge char across ASR cut with two complete VI clauses.
        return "soft"
    if "cn_token_continuation" in reasons and not _both_sides_complete_clauses(left, right):
        return "hard"
    return "soft"


def classify_clause_seams(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_index = {int(item.get("index", -1)): item for item in segments}
    classified: list[dict[str, Any]] = []
    for seam in detect_clause_seams(segments):
        left = by_index.get(int(seam["left_index"]), {})
        right = by_index.get(int(seam["right_index"]), {})
        item = dict(seam)
        item["severity"] = classify_clause_seam(
            seam,
            left_text=spoken_text(left),
            right_text=spoken_text(right),
        )
        item["action"] = "heal_merge" if item["severity"] == "hard" else "keep"
        classified.append(item)
    return classified


def hard_seam_clusters(classified: list[dict[str, Any]]) -> list[list[int]]:
    """Merge adjacent hard seam edges into contiguous index clusters."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for seam in classified:
        if seam.get("severity") != "hard":
            continue
        union(int(seam["left_index"]), int(seam["right_index"]))
    groups: dict[int, list[int]] = {}
    for idx in list(parent):
        groups.setdefault(find(idx), []).append(idx)
    return [sorted(values) for values in groups.values()]


def join_clause_texts(parts: list[str]) -> str:
    from dv_backend.text_sanitation import sanitize_spoken_text

    cleaned: list[str] = []
    for part in parts:
        text = sanitize_spoken_text(part, strip_leading_ellipsis=True)
        if text:
            cleaned.append(text)
    if not cleaned:
        return ""
    merged = cleaned[0]
    for part in cleaned[1:]:
        # Continuation after ellipsis/chunk: drop "..." and glue (Đại + nhân, Quý + tới).
        if merged.endswith(("...", "…")) and part and part[0].islower():
            stem = merged[:-3].rstrip(".… ").rstrip()
            last = stem.split()[-1] if stem.split() else ""
            first = part.split()[0].rstrip(".!?…")
            if last and last.lower() == first.lower():
                head = " ".join(stem.split()[:-1]).strip()
                merged = f"{head} {part}".strip() if head else part
            else:
                merged = f"{stem} {part}".strip()
            continue
        entity = re.search(r"^(.*?)(\b[\wÀ-ỹ]{1,6})\.$", merged)
        if entity and part and part[0].islower():
            stem, last = entity.group(1).rstrip(), entity.group(2)
            first = part.split()[0].rstrip(".!?…")
            if last.lower() == first.lower():
                merged = f"{stem} {part}".strip() if stem else part
            else:
                merged = f"{stem} {last} {part}".strip() if stem else f"{last} {part}"
            continue
        if merged.endswith(("-", "—")):
            merged = f"{merged.rstrip('-—')}{part}"
        elif merged.endswith((".", "!", "?")) and part and part[0].islower():
            merged = f"{merged[:-1].rstrip()} {part}"
        else:
            merged = f"{merged.rstrip()} {part.lstrip()}"
    return sanitize_spoken_text(merged, strip_leading_ellipsis=False)
