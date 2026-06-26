"""Scoped FunASR/CampPlus second pass for low-confidence diarization windows."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .diarization_models import (
    AttributedSegment,
    AttributedTranscript,
    AttributedUnit,
    DiarizationOptions,
    DiarizationTimeline,
    SpeakerAssignmentConfig,
)
from .speaker_attribution import (
    attribute_unit,
    map_speakers_by_overlap,
    merge_attributed_units,
    temporal_overlap,
)

logger = logging.getLogger(__name__)

WINDOW_PADDING_SEC = 0.25
WINDOW_MERGE_GAP_SEC = 0.5


@dataclass(frozen=True)
class SecondPassWindow:
    start: float
    end: float
    reason: str


@dataclass(frozen=True)
class SecondPassWindowResult:
    window: SecondPassWindow
    timeline: DiarizationTimeline
    speaker_mapping: dict[str, str]


def derive_second_pass_windows(
    segments: list[AttributedSegment],
    assignment: SpeakerAssignmentConfig,
    *,
    padding_sec: float = WINDOW_PADDING_SEC,
    merge_gap_sec: float = WINDOW_MERGE_GAP_SEC,
) -> list[SecondPassWindow]:
    raw: list[SecondPassWindow] = []
    for segment in segments:
        ambiguous = (
            segment.speaker_confidence < assignment.review_confidence_threshold
            or "overlap_speech" in segment.flags
            or "no_speaker_match" in segment.flags
            or "backend_disagreement" in segment.flags
            or "boundary_ambiguous" in segment.flags
        )
        if not ambiguous:
            continue
        raw.append(
            SecondPassWindow(
                start=max(0.0, segment.start - padding_sec),
                end=segment.end + padding_sec,
                reason="low_confidence_or_ambiguous",
            )
        )
    if not raw:
        return []
    raw.sort(key=lambda item: item.start)
    merged: list[SecondPassWindow] = [raw[0]]
    for window in raw[1:]:
        prev = merged[-1]
        if window.start <= prev.end + merge_gap_sec:
            merged[-1] = SecondPassWindow(
                start=prev.start,
                end=max(prev.end, window.end),
                reason=prev.reason,
            )
        else:
            merged.append(window)
    return merged


def _extract_window_audio(
    audio_path: Path,
    window: SecondPassWindow,
    output_path: Path,
    ffmpeg_path: Path,
    run_ffmpeg: Callable[[list[str]], None] | None = None,
) -> None:
    duration = max(0.05, window.end - window.start)
    cmd = [
        str(ffmpeg_path),
        "-y",
        "-ss",
        f"{window.start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    if run_ffmpeg is not None:
        run_ffmpeg(cmd)
        return
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def run_scoped_second_pass(
    *,
    audio_path: Path,
    ffmpeg_path: Path,
    windows: list[SecondPassWindow],
    options: DiarizationOptions,
    primary_timeline: DiarizationTimeline,
    run_window_diarization: Callable[[Path, DiarizationOptions], DiarizationTimeline],
    run_ffmpeg: Callable[[list[str]], None] | None = None,
) -> list[SecondPassWindowResult]:
    results: list[SecondPassWindowResult] = []
    for window in windows:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            clip_path = Path(tmp.name)
        try:
            _extract_window_audio(
                audio_path,
                window,
                clip_path,
                ffmpeg_path,
                run_ffmpeg=run_ffmpeg,
            )
            timeline = run_window_diarization(clip_path, options)
            for turn in timeline.turns:
                turn.start = round(turn.start + window.start, 3)
                turn.end = round(turn.end + window.start, 3)
            mapping = map_speakers_by_overlap(primary_timeline, timeline)
            results.append(
                SecondPassWindowResult(
                    window=window,
                    timeline=timeline,
                    speaker_mapping=mapping,
                )
            )
        finally:
            if clip_path.is_file():
                clip_path.unlink(missing_ok=True)
    return results


def _unit_in_window(unit: AttributedUnit, window: SecondPassWindow) -> bool:
    return temporal_overlap(unit.start, unit.end, window.start, window.end) > 0


def _fallback_score(coverage: float, margin: float, overlap_ratio: float, confidence: float) -> float:
    return (0.45 * coverage) + (0.25 * margin) + (0.30 * confidence) - (0.20 * overlap_ratio)


def apply_second_pass_to_units(
    attributed: AttributedTranscript,
    primary_regular: DiarizationTimeline,
    primary_exclusive: DiarizationTimeline,
    window_results: list[SecondPassWindowResult],
    assignment: SpeakerAssignmentConfig,
) -> tuple[AttributedTranscript, list[dict[str, Any]]]:
    if not window_results:
        return attributed, []

    units = [unit.model_copy(deep=True) for unit in attributed.units]
    diagnostics: list[dict[str, Any]] = []
    for result in window_results:
        fallback_regular = result.timeline
        fallback_exclusive = DiarizationTimeline(
            backend=fallback_regular.backend,
            model=fallback_regular.model,
            device=fallback_regular.device,
            turns=fallback_regular.turns,
            metadata={"derived": True, "scoped_second_pass": True},
        )
        applied = 0
        for index, unit in enumerate(units):
            if not _unit_in_window(unit, result.window):
                continue
            original_score = _fallback_score(
                unit.speaker_coverage,
                unit.speaker_margin,
                unit.overlap_ratio,
                unit.speaker_confidence,
            )
            candidate = attribute_unit(
                unit.text,
                unit.start,
                unit.end,
                fallback_regular,
                fallback_exclusive,
                assignment,
                backend_disagreement=True,
            )
            candidate_score = _fallback_score(
                candidate.speaker_coverage,
                candidate.speaker_margin,
                candidate.overlap_ratio,
                candidate.speaker_confidence,
            )
            should_apply = (
                unit.speaker_id is None
                or "no_speaker_match" in unit.flags
                or unit.speaker_coverage < assignment.min_coverage
                or candidate_score >= original_score + 0.08
            )
            if not should_apply or candidate.speaker_id is None:
                continue
            mapped = result.speaker_mapping.get(candidate.speaker_id, candidate.speaker_id)
            units[index] = candidate.model_copy(
                update={
                    "speaker_id": mapped,
                    "flags": sorted(set(candidate.flags + ["second_pass_applied"])),
                }
            )
            applied += 1
        diagnostics.append(
            {
                "window_start": result.window.start,
                "window_end": result.window.end,
                "reason": result.window.reason,
                "backend": fallback_regular.backend,
                "units_updated": applied,
                "speaker_mapping": result.speaker_mapping,
            }
        )

    segments = merge_attributed_units(units, assignment)
    return AttributedTranscript(units=units, segments=segments), diagnostics
