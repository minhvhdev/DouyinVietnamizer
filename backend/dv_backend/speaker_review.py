"""Speaker review trigger logic."""

from __future__ import annotations

from .diarization_models import (
    AttributedSegment,
    DiarizationDiagnostics,
    SpeakerAssignmentConfig,
)


def should_require_speaker_review(
    segments: list[AttributedSegment],
    diagnostics: DiarizationDiagnostics,
    config: SpeakerAssignmentConfig,
    *,
    min_speakers: int,
    max_speakers: int,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    total_speech = sum(max(0.0, seg.end - seg.start) for seg in segments)
    if total_speech <= 0:
        return False, reasons

    low_conf_duration = sum(
        max(0.0, seg.end - seg.start)
        for seg in segments
        if seg.speaker_confidence < config.review_confidence_threshold
    )
    low_conf_ratio = low_conf_duration / total_speech
    if low_conf_ratio > 0.25:
        reasons.append(f"low_confidence_ratio={low_conf_ratio:.2f}")

    if diagnostics.overlap_ratio > 0.15:
        reasons.append(f"overlap_ratio={diagnostics.overlap_ratio:.2f}")

    if diagnostics.backend_comparison and diagnostics.backend_comparison.agreement_ratio < 0.6:
        reasons.append(
            f"backend_disagreement={diagnostics.backend_comparison.agreement_ratio:.2f}"
        )

    speaker_ids = {seg.speaker_id for seg in segments if seg.speaker_id}
    if len(speaker_ids) > max_speakers:
        reasons.append(f"speaker_count_above_max={len(speaker_ids)}")
    if len(speaker_ids) < min_speakers and total_speech > 5.0:
        reasons.append("speaker_count_below_min")

    fragmentation = len(segments) / max(1, len(speaker_ids))
    if fragmentation > 25:
        reasons.append(f"fragmentation={fragmentation:.1f}")

    ambiguous_duration = sum(
        max(0.0, seg.end - seg.start)
        for seg in segments
        if "no_speaker_match" in seg.flags or "boundary_ambiguous" in seg.flags
    )
    if ambiguous_duration / total_speech > 0.2:
        reasons.append("ambiguous_assignment")

    return bool(reasons), reasons
