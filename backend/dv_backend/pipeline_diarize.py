"""Dedicated diarization pipeline step."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from .adapters.asr import reset_model_cache
from .adapters.audio_analysis import audio_has_prominent_bgm
from .adapters.diarization.service import FunASRCampPlusBackend, run_diarization_with_fallback
from .adapters.separation import separate_vocals
from .checkpoint_compat import (
    ASR_ALIGNMENT_SCHEMA_VERSION,
    DIARIZE_CHECKPOINT_SCHEMA_VERSION,
    asr_checkpoint_fingerprint,
    diarization_settings_fingerprint,
    validate_asr_for_diarization,
)
from .checkpoints import load_checkpoint, save_checkpoint
from .config import AppConfig
from .database import Database
from .diarization_artifacts import write_diarization_artifacts
from .diarization_models import (
    BackendComparison,
    DiarizationDiagnostics,
    DiarizationOptions,
    SpeakerAssignmentConfig,
)
from .diarization_second_pass import (
    apply_second_pass_to_units,
    derive_second_pass_windows,
    run_scoped_second_pass,
)
from .errors import AppError
from .gpu_lease import gpu_lease
from .models import ErrorInfo
from .speaker_attribution import attribute_speakers, compare_timelines, detect_overlap_regions
from .speaker_profiles import build_speaker_profiles, remap_segments_to_voice_slots
from .speaker_review import should_require_speaker_review
from .speaker_samples import generate_speaker_sample_files

logger = logging.getLogger(__name__)


def _load_settings(database: Database) -> dict:
    rows = database.connection.execute("SELECT key, value FROM settings").fetchall()
    return {row["key"]: json.loads(row["value"]) for row in rows}


def _assignment_config(settings: dict) -> SpeakerAssignmentConfig:
    return SpeakerAssignmentConfig(
        min_coverage=float(settings.get("speaker_assignment_min_coverage", 0.75)),
        min_margin=float(settings.get("speaker_assignment_min_margin", 0.20)),
        overlap_flag_threshold=float(settings.get("speaker_overlap_flag_threshold", 0.25)),
        review_confidence_threshold=float(settings.get("speaker_review_confidence_threshold", 0.65)),
        merge_gap_sec=float(settings.get("speaker_merge_gap_sec", 0.35)),
        profile_min_seconds=float(settings.get("speaker_profile_min_seconds", 3.0)),
    )


def _legacy_checkpoint_from_asr(asr_cp: dict, settings: dict, *, job_id: str) -> dict:
    segments = asr_cp.get("segments", [])
    assignment = _assignment_config(settings)
    from .diarization_models import AttributedSegment, AttributedTranscript, AttributedUnit

    units = [
        AttributedUnit(
            text=str(segment.get("text") or ""),
            start=float(segment.get("start", 0.0)),
            end=float(segment.get("end", 0.0)),
            speaker_id=str(segment.get("speaker_id")) if segment.get("speaker_id") is not None else None,
            speaker_confidence=float(segment.get("speaker_confidence", 0.5)),
            flags=["legacy_asr_diarization"],
        )
        for segment in segments
    ]
    attr_segments = [
        AttributedSegment(
            index=index,
            start=float(segment.get("start", 0.0)),
            end=float(segment.get("end", 0.0)),
            text=str(segment.get("text") or ""),
            speaker_id=str(segment.get("speaker_id")) if segment.get("speaker_id") is not None else None,
            speaker_confidence=float(segment.get("speaker_confidence", 0.5)),
            flags=["legacy_asr_diarization"],
            unit_count=1,
        )
        for index, segment in enumerate(segments)
    ]
    profiles = build_speaker_profiles(
        attr_segments,
        units,
        assignment,
        speaker_voices=settings.get("speaker_voices") or {},
        manual_overrides=asr_cp.get("speaker_manual_overrides") or {},
        default_voice=str(settings.get("vieneu_voice", "Xuân Vĩnh") or "Xuân Vĩnh"),
        fallback_voice=settings.get("speaker_fallback_voice"),
    )
    remapped = remap_segments_to_voice_slots(attr_segments, profiles)
    return {
        "schema_version": DIARIZE_CHECKPOINT_SCHEMA_VERSION,
        "step_name": "diarize",
        "job_id": job_id,
        "legacy": True,
        "asr_fingerprint": asr_checkpoint_fingerprint(asr_cp),
        "settings_fingerprint": diarization_settings_fingerprint(settings),
        "alignment_schema_version": int(asr_cp.get("schema_version") or 1),
        "segments": remapped,
        "speaker_profiles": [profile.model_dump() for profile in profiles],
        "speaker_manual_overrides": asr_cp.get("speaker_manual_overrides") or {},
        "review_required": False,
        "review_reasons": [],
        "manual_review_completed": False,
    }


def _check_cancelled(runner, job_id: str) -> None:
    if runner and runner.is_cancelled(job_id):
        raise AppError(
            409,
            ErrorInfo(
                code="JOB_CANCELLED",
                message="The job was cancelled by the user.",
                action="Create a new job if you still want to process this video.",
            ),
        )


def diarize_step(job_id: str, config: AppConfig, database: Database, runner) -> dict:
    settings = _load_settings(database)
    speaker_enabled = bool(settings.get("speaker_diarization", False))

    if not speaker_enabled:
        checkpoint_data = {
            "schema_version": DIARIZE_CHECKPOINT_SCHEMA_VERSION,
            "job_id": job_id,
            "step_name": "diarize",
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "skipped": True,
            "segments": [],
            "speaker_profiles": [],
            "review_required": False,
            "manual_review_completed": False,
        }
        save_checkpoint(config.data_dir, job_id, "diarize", checkpoint_data)
        return checkpoint_data

    asr_cp = load_checkpoint(config.data_dir, job_id, "asr")
    if not asr_cp:
        raise AppError(
            400,
            ErrorInfo(
                code="MISSING_ASR_CHECKPOINT",
                message="ASR checkpoint is missing.",
                action="Resume the ASR step.",
            ),
        )

    _check_cancelled(runner, job_id)

    aligned_units = validate_asr_for_diarization(asr_cp, speaker_diarization_enabled=True)
    if (
        not aligned_units
        and asr_cp.get("segments")
        and any(segment.get("speaker_id") is not None for segment in asr_cp.get("segments", []))
    ):
        legacy = _legacy_checkpoint_from_asr(asr_cp, settings, job_id=job_id)
        legacy["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save_checkpoint(config.data_dir, job_id, "diarize", legacy)
        return legacy

    job_dir = config.data_dir / "jobs" / job_id
    audio_16k = job_dir / "artifacts" / "audio_16k.wav"
    if not audio_16k.is_file():
        raise AppError(
            400,
            ErrorInfo(
                code="MISSING_AUDIO_FILE",
                message="Audio file for diarization is missing.",
                action="Resume extract_audio step.",
            ),
        )

    from .pipeline import resolve_tool_path, run_subprocess_with_cancel

    assignment = _assignment_config(settings)
    project_root = Path(__file__).resolve().parents[2]
    vendor_dir = Path(os.environ.get("DV_VENDOR_DIR", project_root / "vendor"))
    pyannote_cache = vendor_dir / "pyannote"
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    options = DiarizationOptions(
        min_speakers=int(settings.get("diarization_min_speakers", 1)),
        max_speakers=int(settings.get("diarization_max_speakers", 6)),
        device=str(settings.get("qwen3_device", "cuda:0") or "cuda:0"),
        model_cache_dir=str(pyannote_cache.resolve()),
        hf_token=hf_token,
    )

    primary_backend = str(settings.get("diarization_backend", "pyannote_community_1"))
    fallback_backend = str(settings.get("diarization_fallback_backend", "funasr_campp"))

    started = time.perf_counter()
    result, backend_used, used_fallback, fallback_reason = run_diarization_with_fallback(
        audio_16k,
        options,
        primary_backend=primary_backend,
        fallback_backend=fallback_backend
        if settings.get("diarization_ensemble_enabled", True) and fallback_backend not in {"", "none"}
        else None,
        job_id=job_id,
    )
    _check_cancelled(runner, job_id)
    reset_model_cache()

    demucs_mode = str(settings.get("diarization_demucs_mode", "fallback_only"))
    secondary_result = None
    demucs_used = False
    if demucs_mode != "off" and audio_has_prominent_bgm(audio_16k):
        overlap_regions_pre = detect_overlap_regions(result.regular)
        total = max(
            0.001,
            sum(max(0.0, turn.end - turn.start) for turn in result.regular.turns),
        )
        overlap_duration = sum(region.end - region.start for region in overlap_regions_pre)
        should_run_demucs = demucs_mode == "always_for_testing" or overlap_duration / total > 0.12
        if should_run_demucs:
            original_48k = job_dir / "artifacts" / "original_48k.wav"
            vocals_wav = job_dir / "artifacts" / "vocals.wav"
            bgm_wav = job_dir / "artifacts" / "bgm.wav"
            ffmpeg_path = resolve_tool_path(config, "ffmpeg")
            if original_48k.is_file():
                reset_model_cache()
                separate_vocals(
                    original_48k,
                    vocals_out=vocals_wav,
                    bgm_out=bgm_wav,
                    ffmpeg_path=ffmpeg_path,
                    device=str(settings.get("vieneu_device", "cuda") or "cuda"),
                    job_id=job_id,
                    runner=runner,
                )
                reset_model_cache()
                demucs_used = True
                vocals_16k = job_dir / "artifacts" / "vocals_16k.wav"
                if not vocals_16k.is_file() or vocals_16k.stat().st_mtime < vocals_wav.stat().st_mtime:
                    cmd = [
                        str(ffmpeg_path),
                        "-y",
                        "-i",
                        str(vocals_wav),
                        "-ac",
                        "1",
                        "-acodec",
                        "pcm_s16le",
                        "-ar",
                        "16000",
                        str(vocals_16k),
                    ]
                    run_subprocess_with_cancel(cmd, job_id, runner)
                secondary_result, _, _, _ = run_diarization_with_fallback(
                    vocals_16k,
                    options,
                    primary_backend=backend_used,
                    fallback_backend=None,
                    job_id=job_id,
                )
                reset_model_cache()

    attributed = attribute_speakers(
        aligned_units,
        result.regular,
        result.exclusive,
        assignment,
    )

    second_pass_diagnostics: list[dict] = []
    if bool(settings.get("diarization_second_pass_on_low_confidence", True)):
        windows = derive_second_pass_windows(attributed.segments, assignment)
        if windows:
            funasr_backend = FunASRCampPlusBackend()

            def _run_window(path: Path, opts: DiarizationOptions):
                with gpu_lease(f"job-{job_id}:diarize-second-pass"):
                    timeline = funasr_backend.diarize(path, opts).regular
                return timeline

            window_results = run_scoped_second_pass(
                audio_path=audio_16k,
                ffmpeg_path=resolve_tool_path(config, "ffmpeg"),
                windows=windows,
                options=options,
                primary_timeline=result.regular,
                run_window_diarization=_run_window,
            )
            attributed, second_pass_diagnostics = apply_second_pass_to_units(
                attributed,
                result.regular,
                result.exclusive,
                window_results,
                assignment,
            )

    comparison: BackendComparison | None = None
    if secondary_result is not None:
        agreement, disagree_duration = compare_timelines(result.regular, secondary_result.regular)
        from .speaker_attribution import map_speakers_by_overlap

        comparison = BackendComparison(
            primary_backend=backend_used,
            secondary_backend=f"{backend_used}+demucs",
            agreement_ratio=round(agreement, 4),
            disagreement_duration_sec=round(disagree_duration, 3),
            speaker_mapping=map_speakers_by_overlap(result.regular, secondary_result.regular),
        )

    overlap_regions = detect_overlap_regions(result.regular)
    total_speech = sum(max(0.0, seg.end - seg.start) for seg in attributed.segments)
    low_conf_duration = sum(
        max(0.0, seg.end - seg.start)
        for seg in attributed.segments
        if seg.speaker_confidence < assignment.review_confidence_threshold
    )
    overlap_duration = sum(region.end - region.start for region in overlap_regions)

    existing_diarize = load_checkpoint(config.data_dir, job_id, "diarize") or {}
    manual_overrides = {
        **(existing_diarize.get("speaker_manual_overrides") or {}),
        **(asr_cp.get("speaker_manual_overrides") or {}),
    }

    profiles = build_speaker_profiles(
        attributed.segments,
        attributed.units,
        assignment,
        speaker_voices=settings.get("speaker_voices") or {},
        manual_overrides=manual_overrides,
        default_voice=str(settings.get("vieneu_voice", "Xuân Vĩnh") or "Xuân Vĩnh"),
        fallback_voice=settings.get("speaker_fallback_voice"),
    )
    profiles = generate_speaker_sample_files(
        job_dir=job_dir,
        audio_path=audio_16k,
        ffmpeg_path=resolve_tool_path(config, "ffmpeg"),
        profiles=profiles,
        segments=attributed.segments,
        review_confidence_threshold=assignment.review_confidence_threshold,
        demucs_used=demucs_used,
    )
    remapped_segments = remap_segments_to_voice_slots(attributed.segments, profiles)

    resolved_model_path = result.regular.metadata.get("resolved_model_path")
    diagnostics = DiarizationDiagnostics(
        backend_used=backend_used,
        fallback_backend=used_fallback,
        fallback_reason=fallback_reason,
        demucs_used=demucs_used,
        demucs_mode=demucs_mode,
        device=result.regular.device,
        model=result.regular.model,
        resolved_model_path=str(resolved_model_path) if resolved_model_path else None,
        offline_local_load=bool(result.regular.metadata.get("offline_local_load")),
        backend_version=result.regular.metadata.get("backend_version"),
        runtime_ms=result.runtime_ms,
        speaker_count=len({profile.speaker_id for profile in profiles}),
        overlap_regions=overlap_regions,
        overlap_duration_sec=round(overlap_duration, 3),
        overlap_ratio=round(overlap_duration / max(total_speech, 0.001), 4),
        low_confidence_duration_sec=round(low_conf_duration, 3),
        low_confidence_ratio=round(low_conf_duration / max(total_speech, 0.001), 4),
        fragmentation_segment_count=len(attributed.segments),
        coverage_mean=round(
            sum(seg.speaker_coverage for seg in attributed.segments) / max(len(attributed.segments), 1),
            4,
        ),
        margin_mean=round(
            sum(seg.speaker_margin for seg in attributed.segments) / max(len(attributed.segments), 1),
            4,
        ),
        backend_comparison=comparison,
        second_pass_windows=second_pass_diagnostics,
    )

    review_required, review_reasons = should_require_speaker_review(
        attributed.segments,
        diagnostics,
        assignment,
        min_speakers=int(settings.get("diarization_min_speakers", 1)),
        max_speakers=int(settings.get("diarization_max_speakers", 6)),
    )
    diagnostics.review_required = review_required
    diagnostics.review_reasons = review_reasons

    write_diarization_artifacts(
        job_dir,
        regular=result.regular,
        exclusive=result.exclusive,
        attributed=attributed,
        overlap_regions=overlap_regions,
        speaker_profiles=profiles,
        diagnostics=diagnostics,
    )

    checkpoint_data = {
        "schema_version": DIARIZE_CHECKPOINT_SCHEMA_VERSION,
        "job_id": job_id,
        "step_name": "diarize",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "asr_fingerprint": asr_checkpoint_fingerprint(asr_cp),
        "settings_fingerprint": diarization_settings_fingerprint(settings),
        "alignment_schema_version": ASR_ALIGNMENT_SCHEMA_VERSION,
        "segments": remapped_segments,
        "speaker_profiles": [profile.model_dump() for profile in profiles],
        "speaker_manual_overrides": manual_overrides,
        "diagnostics": diagnostics.model_dump(),
        "review_required": review_required,
        "review_reasons": review_reasons,
        "manual_review_completed": False,
    }
    save_checkpoint(config.data_dir, job_id, "diarize", checkpoint_data)
    logger.info(
        "Diarization completed for job %s backend=%s speakers=%s review=%s offline=%s runtime_ms=%s",
        job_id,
        backend_used,
        diagnostics.speaker_count,
        review_required,
        diagnostics.offline_local_load,
        round((time.perf_counter() - started) * 1000),
    )
    return checkpoint_data
