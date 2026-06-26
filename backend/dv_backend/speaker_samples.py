"""Representative speaker sample selection and WAV extraction."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable

from .diarization_models import AttributedSegment, SpeakerProfile

logger = logging.getLogger(__name__)

MIN_SAMPLE_DURATION_SEC = 2.0
MAX_SAMPLE_DURATION_SEC = 8.0
MAX_SAMPLES_PER_SPEAKER = 2
MIN_GAP_BETWEEN_SAMPLES_SEC = 1.0

SPEAKER_ID_SAFE = re.compile(r"[^A-Za-z0-9_\-]+")


def sanitize_speaker_id(speaker_id: str) -> str:
    cleaned = SPEAKER_ID_SAFE.sub("_", speaker_id.strip())
    return cleaned or "SPK_unknown"


def clip_interval(start: float, end: float, *, max_duration: float = MAX_SAMPLE_DURATION_SEC) -> tuple[float, float]:
    duration = max(0.0, end - start)
    if duration <= max_duration:
        return start, end
    center = (start + end) / 2.0
    half = max_duration / 2.0
    return max(0.0, center - half), center + half


def select_sample_candidates(
    segments: list[AttributedSegment],
    *,
    speaker_id: str,
    review_confidence_threshold: float,
    demucs_used: bool = False,
    overlap_ratio_limit: float = 0.05,
) -> list[dict[str, Any]]:
    blocked_flags = {"overlap_speech", "low_confidence", "no_speaker_match", "backend_disagreement"}
    if demucs_used:
        blocked_flags.add("boundary_ambiguous")

    ranked = sorted(
        [segment for segment in segments if segment.speaker_id == speaker_id],
        key=lambda item: (item.speaker_confidence, item.end - item.start),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    for segment in ranked:
        duration = segment.end - segment.start
        if duration < MIN_SAMPLE_DURATION_SEC:
            continue
        if segment.speaker_confidence < review_confidence_threshold:
            continue
        if segment.overlap_ratio > overlap_ratio_limit:
            continue
        if blocked_flags.intersection(segment.flags):
            continue
        start, end = clip_interval(segment.start, segment.end)
        if end - start < MIN_SAMPLE_DURATION_SEC:
            continue
        if any(abs(start - item["start"]) < MIN_GAP_BETWEEN_SAMPLES_SEC for item in selected):
            continue
        selected.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "duration_sec": round(end - start, 3),
                "text": segment.text[:80],
                "confidence": segment.speaker_confidence,
            }
        )
        if len(selected) >= MAX_SAMPLES_PER_SPEAKER:
            break
    return selected


def extract_speaker_sample_wav(
    *,
    audio_path: Path,
    sample: dict[str, Any],
    output_path: Path,
    ffmpeg_path: Path,
    run_ffmpeg: Callable[[list[str]], None] | None = None,
) -> None:
    start = float(sample["start"])
    end = float(sample["end"])
    duration = max(0.05, end - start)
    cmd = [
        str(ffmpeg_path),
        "-y",
        "-ss",
        f"{start:.3f}",
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
    import subprocess

    subprocess.run(cmd, check=True, capture_output=True, text=True)


def generate_speaker_sample_files(
    *,
    job_dir: Path,
    audio_path: Path,
    ffmpeg_path: Path,
    profiles: list[SpeakerProfile],
    segments: list[AttributedSegment],
    review_confidence_threshold: float,
    demucs_used: bool = False,
    run_ffmpeg: Callable[[list[str]], None] | None = None,
) -> list[SpeakerProfile]:
    samples_root = job_dir / "artifacts" / "diarization" / "speaker_samples"
    samples_root.mkdir(parents=True, exist_ok=True)

    updated_profiles: list[SpeakerProfile] = []
    for profile in profiles:
        candidates = select_sample_candidates(
            segments,
            speaker_id=profile.speaker_id,
            review_confidence_threshold=review_confidence_threshold,
            demucs_used=demucs_used,
        )
        serialized_samples: list[dict[str, Any]] = []
        safe_id = sanitize_speaker_id(profile.speaker_id)
        for index, candidate in enumerate(candidates, start=1):
            filename = f"{safe_id}_{index:02d}.wav"
            relative_path = f"artifacts/diarization/speaker_samples/{filename}"
            output_path = samples_root / filename
            try:
                extract_speaker_sample_wav(
                    audio_path=audio_path,
                    sample=candidate,
                    output_path=output_path,
                    ffmpeg_path=ffmpeg_path,
                    run_ffmpeg=run_ffmpeg,
                )
            except Exception as error:
                logger.warning(
                    "Failed to extract speaker sample for %s: %s",
                    profile.speaker_id,
                    error,
                )
                continue
            serialized_samples.append(
                {
                    **candidate,
                    "artifact_path": relative_path,
                    "playback_url": f"/api/jobs/{job_dir.name}/diarization/samples/{filename}",
                }
            )
        updated_profiles.append(
            profile.model_copy(update={"representative_samples": serialized_samples})
        )
    return updated_profiles
