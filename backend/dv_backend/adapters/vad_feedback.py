"""ASR feedback helpers for filtering likely VAD false positives."""

from __future__ import annotations


def is_likely_vad_false_positive(
    text: str,
    prev_text: str | None,
    *,
    prev_end: float | None = None,
    current_start: float | None = None,
    max_duplicate_gap_sec: float = 0.35,
) -> bool:
    """Return True when ASR output suggests the upstream VAD region was noise."""
    cleaned = text.strip()
    if not cleaned:
        return True
    if prev_text is not None and cleaned == prev_text.strip():
        if prev_end is None or current_start is None:
            return True
        gap = float(current_start) - float(prev_end)
        if -0.05 <= gap <= max_duplicate_gap_sec:
            return True
        return False
    return False


def filter_asr_false_positives(
    segments: list[dict],
    *,
    enabled: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Drop segments that look like VAD false positives; return kept + rejected."""
    if not enabled or not segments:
        return list(segments), []

    kept: list[dict] = []
    rejected: list[dict] = []
    prev_text: str | None = None
    prev_end: float | None = None
    for segment in sorted(segments, key=lambda item: float(item.get("start", 0.0) or 0.0)):
        text = str(segment.get("text") or "")
        start = float(segment.get("start", 0.0) or 0.0)
        if is_likely_vad_false_positive(
            text,
            prev_text,
            prev_end=prev_end,
            current_start=start,
        ):
            rejected.append(
                {
                    **segment,
                    "vad_false_positive_reason": (
                        "empty_asr" if not text.strip() else "duplicate_asr"
                    ),
                }
            )
            continue
        kept.append(segment)
        prev_text = text.strip()
        prev_end = float(segment.get("end", start) or start)
    return kept, rejected
