"""Speaker attribution from aligned units and diarization timelines."""

from __future__ import annotations

from dataclasses import dataclass

from .diarization_models import (
    AttributedSegment,
    AttributedTranscript,
    AttributedUnit,
    DiarizationTimeline,
    DiarizationTurn,
    OverlapRegion,
    SpeakerAssignmentConfig,
)

MIN_UNIT_DURATION = 1e-4


@dataclass(frozen=True)
class _OverlapStats:
    coverage_by_speaker: dict[str, float]
    overlap_ratio: float
    exclusive_speaker: str | None


def unit_duration(start: float, end: float) -> float:
    return max(MIN_UNIT_DURATION, end - start)


def temporal_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def speaker_coverage_for_unit(
    unit_start: float,
    unit_end: float,
    turns: list[DiarizationTurn],
) -> dict[str, float]:
    duration = unit_duration(unit_start, unit_end)
    coverage: dict[str, float] = {}
    for turn in turns:
        overlap = temporal_overlap(unit_start, unit_end, turn.start, turn.end)
        if overlap <= 0:
            continue
        coverage[turn.speaker_id] = coverage.get(turn.speaker_id, 0.0) + overlap
    return {speaker: amount / duration for speaker, amount in coverage.items()}


def exclusive_speaker_for_unit(
    unit_start: float,
    unit_end: float,
    exclusive_turns: list[DiarizationTurn],
) -> str | None:
    coverage = speaker_coverage_for_unit(unit_start, unit_end, exclusive_turns)
    if not coverage:
        return None
    return max(coverage.items(), key=lambda item: item[1])[0]


def compute_overlap_stats(
    unit_start: float,
    unit_end: float,
    regular_turns: list[DiarizationTurn],
    exclusive_turns: list[DiarizationTurn],
) -> _OverlapStats:
    duration = unit_duration(unit_start, unit_end)
    regular_coverage = speaker_coverage_for_unit(unit_start, unit_end, regular_turns)
    exclusive_speaker = exclusive_speaker_for_unit(unit_start, unit_end, exclusive_turns)

    overlap_duration = 0.0
    if len(regular_coverage) >= 2:
        sorted_turns = sorted(
            (
                turn
                for turn in regular_turns
                if temporal_overlap(unit_start, unit_end, turn.start, turn.end) > 0
            ),
            key=lambda turn: turn.start,
        )
        for index in range(len(sorted_turns) - 1):
            left = sorted_turns[index]
            right = sorted_turns[index + 1]
            if left.speaker_id == right.speaker_id:
                continue
            overlap_duration += temporal_overlap(
                unit_start,
                unit_end,
                max(left.start, right.start),
                min(left.end, right.end),
            )

    overlap_ratio = min(1.0, overlap_duration / duration)
    return _OverlapStats(
        coverage_by_speaker=regular_coverage,
        overlap_ratio=overlap_ratio,
        exclusive_speaker=exclusive_speaker,
    )


def coverage_margin(coverage: dict[str, float]) -> tuple[float, float, str | None]:
    if not coverage:
        return 0.0, 0.0, None
    ranked = sorted(coverage.items(), key=lambda item: item[1], reverse=True)
    best_speaker, best_cov = ranked[0]
    second_cov = ranked[1][1] if len(ranked) > 1 else 0.0
    return best_cov, best_cov - second_cov, best_speaker


def compute_unit_confidence(
    *,
    coverage: float,
    margin: float,
    overlap_ratio: float,
    config: SpeakerAssignmentConfig,
) -> float:
    overlap_penalty = min(1.0, overlap_ratio / max(config.overlap_flag_threshold, 1e-6))
    raw = (0.55 * coverage) + (0.35 * margin) - (0.25 * overlap_penalty)
    return max(0.0, min(1.0, raw))


def attribute_unit(
    text: str,
    start: float,
    end: float,
    regular: DiarizationTimeline,
    exclusive: DiarizationTimeline,
    config: SpeakerAssignmentConfig,
    *,
    backend_disagreement: bool = False,
) -> AttributedUnit:
    stats = compute_overlap_stats(start, end, regular.turns, exclusive.turns)
    best_cov, margin, best_regular = coverage_margin(stats.coverage_by_speaker)
    speaker_id = stats.exclusive_speaker or best_regular
    flags: list[str] = []

    if stats.overlap_ratio >= config.overlap_flag_threshold:
        flags.append("overlap_speech")
    if best_cov < config.min_coverage:
        flags.append("low_confidence")
    if margin < config.min_margin:
        flags.append("boundary_ambiguous")
    if speaker_id is None:
        flags.append("no_speaker_match")
    if backend_disagreement:
        flags.append("backend_disagreement")

    confidence = compute_unit_confidence(
        coverage=best_cov,
        margin=margin,
        overlap_ratio=stats.overlap_ratio,
        config=config,
    )
    if confidence < config.review_confidence_threshold:
        if "low_confidence" not in flags:
            flags.append("low_confidence")

    return AttributedUnit(
        text=text,
        start=round(start, 3),
        end=round(end, 3),
        speaker_id=speaker_id,
        speaker_coverage=round(best_cov, 4),
        speaker_margin=round(margin, 4),
        overlap_ratio=round(stats.overlap_ratio, 4),
        speaker_confidence=round(confidence, 4),
        flags=flags,
    )


def merge_attributed_units(
    units: list[AttributedUnit],
    config: SpeakerAssignmentConfig,
) -> list[AttributedSegment]:
    if not units:
        return []

    segments: list[AttributedSegment] = []
    current_units: list[AttributedUnit] = [units[0]]

    def flush() -> None:
        if not current_units:
            return
        text = "".join(unit.text for unit in current_units)
        flags = sorted({flag for unit in current_units for flag in unit.flags})
        avg_conf = sum(unit.speaker_confidence for unit in current_units) / len(current_units)
        avg_cov = sum(unit.speaker_coverage for unit in current_units) / len(current_units)
        avg_margin = sum(unit.speaker_margin for unit in current_units) / len(current_units)
        avg_overlap = sum(unit.overlap_ratio for unit in current_units) / len(current_units)
        segments.append(
            AttributedSegment(
                index=len(segments),
                start=current_units[0].start,
                end=current_units[-1].end,
                text=text,
                speaker_id=current_units[0].speaker_id,
                speaker_coverage=round(avg_cov, 4),
                speaker_margin=round(avg_margin, 4),
                overlap_ratio=round(avg_overlap, 4),
                speaker_confidence=round(avg_conf, 4),
                flags=flags,
                unit_count=len(current_units),
            )
        )
        current_units.clear()

    boundary_flags = {"overlap_speech", "boundary_ambiguous", "backend_disagreement", "no_speaker_match"}

    for unit in units[1:]:
        prev = current_units[-1]
        gap = unit.start - prev.end
        same_speaker = unit.speaker_id == prev.speaker_id and unit.speaker_id is not None
        ambiguous_boundary = bool(boundary_flags.intersection(prev.flags) or boundary_flags.intersection(unit.flags))
        if same_speaker and gap <= config.merge_gap_sec and not ambiguous_boundary:
            current_units.append(unit)
        else:
            flush()
            current_units = [unit]
    flush()
    return segments


def attribute_speakers(
    aligned_units: list[dict],
    regular: DiarizationTimeline,
    exclusive: DiarizationTimeline,
    config: SpeakerAssignmentConfig,
    *,
    backend_disagreement_map: dict[int, bool] | None = None,
) -> AttributedTranscript:
    attributed_units: list[AttributedUnit] = []
    for index, raw in enumerate(aligned_units):
        text = str(raw.get("text") or "")
        if not text.strip():
            continue
        start = float(raw.get("start", 0.0))
        end = float(raw.get("end", start))
        if end <= start:
            end = start + MIN_UNIT_DURATION
        attributed_units.append(
            attribute_unit(
                text,
                start,
                end,
                regular,
                exclusive,
                config,
                backend_disagreement=bool((backend_disagreement_map or {}).get(index, False)),
            )
        )
    segments = merge_attributed_units(attributed_units, config)
    return AttributedTranscript(units=attributed_units, segments=segments)


def detect_overlap_regions(
    regular: DiarizationTimeline,
    *,
    min_overlap_sec: float = 0.05,
) -> list[OverlapRegion]:
    turns = sorted(regular.turns, key=lambda turn: turn.start)
    regions: list[OverlapRegion] = []
    for index in range(len(turns)):
        for other in turns[index + 1 :]:
            if other.start >= turns[index].end:
                break
            overlap_start = max(turns[index].start, other.start)
            overlap_end = min(turns[index].end, other.end)
            if overlap_end - overlap_start < min_overlap_sec:
                continue
            duration = overlap_end - overlap_start
            regions.append(
                OverlapRegion(
                    start=round(overlap_start, 3),
                    end=round(overlap_end, 3),
                    speakers=[turns[index].speaker_id, other.speaker_id],
                    overlap_ratio=1.0,
                )
            )
    return regions


def map_speakers_by_overlap(
    primary: DiarizationTimeline,
    secondary: DiarizationTimeline,
) -> dict[str, str]:
    mapping: dict[str, tuple[str, float]] = {}
    for sec_turn in secondary.turns:
        best_primary = ""
        best_overlap = 0.0
        for pri_turn in primary.turns:
            overlap = temporal_overlap(sec_turn.start, sec_turn.end, pri_turn.start, pri_turn.end)
            if overlap > best_overlap:
                best_overlap = overlap
                best_primary = pri_turn.speaker_id
        if best_primary:
            current = mapping.get(best_primary)
            if current is None or best_overlap > current[1]:
                mapping[best_primary] = (sec_turn.speaker_id, best_overlap)
    return {primary_id: sec_id for primary_id, (sec_id, _) in mapping.items()}


def compare_timelines(
    primary: DiarizationTimeline,
    secondary: DiarizationTimeline,
) -> tuple[float, float]:
    """Return (agreement_ratio, disagreement_duration_sec) on a 0.1s grid."""
    if not primary.turns:
        return 0.0, 0.0
    start = min(turn.start for turn in primary.turns + secondary.turns)
    end = max(turn.end for turn in primary.turns + secondary.turns)
    if end <= start:
        return 1.0, 0.0

    step = 0.1
    agree = 0.0
    disagree = 0.0
    mapping = map_speakers_by_overlap(primary, secondary)
    t = start
    while t < end:
        t_next = min(end, t + step)
        pri = _dominant_speaker(primary.turns, t, t_next)
        sec_raw = _dominant_speaker(secondary.turns, t, t_next)
        sec = mapping.get(pri or "", sec_raw) if pri else sec_raw
        if pri and sec:
            if pri == sec:
                agree += t_next - t
            else:
                disagree += t_next - t
        t = t_next
    total = agree + disagree
    ratio = agree / total if total > 0 else 1.0
    return ratio, disagree


def _dominant_speaker(turns: list[DiarizationTurn], start: float, end: float) -> str | None:
    coverage = speaker_coverage_for_unit(start, end, turns)
    if not coverage:
        return None
    return max(coverage.items(), key=lambda item: item[1])[0]
