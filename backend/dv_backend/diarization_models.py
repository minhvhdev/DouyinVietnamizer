"""Project-owned diarization and speaker-attribution data models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

DiarizationBackendId = Literal[
    "auto",
    "pyannote_community_1",
    "funasr_campp",
]
DiarizationDemucsMode = Literal["off", "fallback_only", "always_for_testing"]


class DiarizationTurn(BaseModel):
    speaker_id: str
    start: float
    end: float
    confidence: float | None = None


class DiarizationTimeline(BaseModel):
    backend: str
    model: str
    device: str
    turns: list[DiarizationTurn] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DiarizationResult(BaseModel):
    regular: DiarizationTimeline
    exclusive: DiarizationTimeline
    runtime_ms: int = 0


class DiarizationOptions(BaseModel):
    min_speakers: int = 1
    max_speakers: int = 6
    device: str = "cuda:0"
    model_cache_dir: str | None = None
    hf_token: str | None = None


class AttributedUnit(BaseModel):
    text: str
    start: float
    end: float
    speaker_id: str | None = None
    speaker_coverage: float = 0.0
    speaker_margin: float = 0.0
    overlap_ratio: float = 0.0
    speaker_confidence: float = 0.0
    flags: list[str] = Field(default_factory=list)


class AttributedSegment(BaseModel):
    index: int
    start: float
    end: float
    text: str
    speaker_id: str | None = None
    speaker_coverage: float = 0.0
    speaker_margin: float = 0.0
    overlap_ratio: float = 0.0
    speaker_confidence: float = 0.0
    flags: list[str] = Field(default_factory=list)
    unit_count: int = 0


class AttributedTranscript(BaseModel):
    units: list[AttributedUnit] = Field(default_factory=list)
    segments: list[AttributedSegment] = Field(default_factory=list)


class OverlapRegion(BaseModel):
    start: float
    end: float
    speakers: list[str] = Field(default_factory=list)
    overlap_ratio: float = 0.0


class SpeakerProfile(BaseModel):
    speaker_id: str
    first_seen: float = 0.0
    last_seen: float = 0.0
    total_speech_sec: float = 0.0
    turn_count: int = 0
    confidence: float = 0.0
    representative_samples: list[dict[str, Any]] = Field(default_factory=list)
    tts_voice: str | None = None
    tts_voice_source: str = "auto"
    manual_override: bool = False
    below_profile_threshold: bool = False
    flags: list[str] = Field(default_factory=list)


class BackendComparison(BaseModel):
    primary_backend: str
    secondary_backend: str | None = None
    agreement_ratio: float = 0.0
    disagreement_duration_sec: float = 0.0
    speaker_mapping: dict[str, str] = Field(default_factory=dict)


class DiarizationDiagnostics(BaseModel):
    backend_used: str
    fallback_backend: str | None = None
    fallback_reason: str | None = None
    demucs_used: bool = False
    demucs_mode: str = "fallback_only"
    device: str = "cpu"
    model: str = ""
    resolved_model_path: str | None = None
    offline_local_load: bool = False
    backend_version: str | None = None
    runtime_ms: int = 0
    speaker_count: int = 0
    overlap_regions: list[OverlapRegion] = Field(default_factory=list)
    overlap_duration_sec: float = 0.0
    overlap_ratio: float = 0.0
    low_confidence_duration_sec: float = 0.0
    low_confidence_ratio: float = 0.0
    fragmentation_segment_count: int = 0
    coverage_mean: float = 0.0
    margin_mean: float = 0.0
    backend_comparison: BackendComparison | None = None
    second_pass_windows: list[dict[str, Any]] = Field(default_factory=list)
    review_required: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    manual_review_completed: bool = False
    warnings: list[str] = Field(default_factory=list)


class SpeakerAssignmentConfig(BaseModel):
    min_coverage: float = 0.75
    min_margin: float = 0.20
    overlap_flag_threshold: float = 0.25
    review_confidence_threshold: float = 0.65
    merge_gap_sec: float = 0.35
    profile_min_seconds: float = 3.0
