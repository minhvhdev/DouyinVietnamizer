"""Speaker profile aggregation and TTS voice mapping."""

from __future__ import annotations

from pathlib import Path

from .adapters.tts import VIENEU_PRESET_VOICES
from .diarization_models import AttributedSegment, AttributedUnit, SpeakerAssignmentConfig, SpeakerProfile


def build_speaker_profiles(
    segments: list[AttributedSegment],
    units: list[AttributedUnit],
    config: SpeakerAssignmentConfig,
    *,
    speaker_voices: dict[str, str] | None = None,
    manual_overrides: dict[str, str] | None = None,
    default_voice: str = "Xuân Vĩnh",
    fallback_voice: str | None = None,
) -> list[SpeakerProfile]:
    speaker_voices = speaker_voices or {}
    manual_overrides = manual_overrides or {}
    fallback = fallback_voice or default_voice
    grouped: dict[str, list[AttributedSegment]] = {}
    for segment in segments:
        if segment.speaker_id is None:
            continue
        grouped.setdefault(segment.speaker_id, []).append(segment)

    profiles: list[SpeakerProfile] = []
    preset_cycle = list(VIENEU_PRESET_VOICES)
    auto_index = 0

    for speaker_id in sorted(grouped.keys()):
        speaker_segments = grouped[speaker_id]
        total_speech = sum(max(0.0, seg.end - seg.start) for seg in speaker_segments)
        confidences = [seg.speaker_confidence for seg in speaker_segments]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        below = total_speech < config.profile_min_seconds
        flags: list[str] = []
        if below:
            flags.append("below_profile_threshold")
        if any("overlap_speech" in seg.flags for seg in speaker_segments):
            flags.append("overlap_speech")
        if any("low_confidence" in seg.flags for seg in speaker_segments):
            flags.append("low_confidence")

        manual_voice = manual_overrides.get(speaker_id)
        mapped_voice = speaker_voices.get(speaker_id)
        if manual_voice:
            tts_voice = manual_voice
            source = "manual"
            manual = True
        elif mapped_voice:
            tts_voice = mapped_voice
            source = "settings"
            manual = False
        elif below:
            tts_voice = fallback
            source = "fallback"
            manual = False
        else:
            tts_voice = preset_cycle[auto_index % len(preset_cycle)]
            auto_index += 1
            source = "auto"
            manual = False

        samples = _representative_samples(speaker_id, units, speaker_segments)
        profiles.append(
            SpeakerProfile(
                speaker_id=speaker_id,
                first_seen=min(seg.start for seg in speaker_segments),
                last_seen=max(seg.end for seg in speaker_segments),
                total_speech_sec=round(total_speech, 3),
                turn_count=len(speaker_segments),
                confidence=round(avg_conf, 4),
                representative_samples=samples,
                tts_voice=tts_voice,
                tts_voice_source=source,
                manual_override=manual,
                below_profile_threshold=below,
                flags=flags,
            )
        )
    return profiles


def _representative_samples(
    speaker_id: str,
    units: list[AttributedUnit],
    segments: list[AttributedSegment],
    *,
    max_samples: int = 3,
    min_duration: float = 0.4,
) -> list[dict]:
    samples: list[dict] = []
    for segment in sorted(segments, key=lambda item: item.speaker_confidence, reverse=True):
        if segment.speaker_id != speaker_id:
            continue
        if segment.end - segment.start < min_duration:
            continue
        if "overlap_speech" in segment.flags or "low_confidence" in segment.flags:
            continue
        samples.append(
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text[:80],
                "confidence": segment.speaker_confidence,
            }
        )
        if len(samples) >= max_samples:
            break
    return samples


def remap_segments_to_voice_slots(
    segments: list[AttributedSegment],
    profiles: list[SpeakerProfile],
    *,
    max_slots: int = 10,
) -> list[dict]:
    """Map diarization speaker IDs to numeric voice slots for TTS compatibility."""
    ranked = sorted(profiles, key=lambda profile: profile.total_speech_sec, reverse=True)
    slot_map: dict[str, str] = {}
    minor_slot = str(max_slots - 1)
    for index, profile in enumerate(ranked):
        if index < max_slots - 1:
            slot_map[profile.speaker_id] = str(index)
        else:
            slot_map[profile.speaker_id] = minor_slot

    output: list[dict] = []
    for segment in segments:
        speaker_slot = slot_map.get(segment.speaker_id or "", "0")
        output.append(
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "speaker_id": speaker_slot,
                "speaker_confidence": segment.speaker_confidence,
                "diarization_speaker_id": segment.speaker_id,
                "flags": segment.flags,
            }
        )
    return output


def speaker_voice_map_from_profiles(profiles: list[SpeakerProfile]) -> dict[str, str]:
    return {
        profile.speaker_id: profile.tts_voice or "Xuân Vĩnh"
        for profile in profiles
        if profile.tts_voice
    }
