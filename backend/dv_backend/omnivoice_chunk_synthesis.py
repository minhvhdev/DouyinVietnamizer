"""Orchestrate chunked OmniVoice synthesis, cache, concat, and fidelity."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from .errors import AppError
from .models import ErrorInfo
from .omnivoice_chunking import (
    CHUNK_CACHE_SCHEMA_VERSION,
    chunking_required,
    omnivoice_chunk_settings,
    pause_ms_for_chunk,
    segment_text_diagnostics,
    split_omnivoice_text_semantic,
    validate_chunk_reconstruction,
)
from .omnivoice_wav_concat import concat_omnivoice_chunks
from .tts_cache import build_tts_cache_identity, cache_key_from_identity, read_tts_sidecar, write_tts_sidecar
from .tts_fidelity import run_segment_fidelity_check
from .tts_speech_analysis import attach_speech_metrics, measure_speech_envelope

SynthesizeFn = Callable[[str, Path], None]


def chunk_manifest_path(chunks_dir: Path) -> Path:
    return chunks_dir / "manifest.json"


def chunk_wav_path(chunks_dir: Path, chunk_index: int) -> Path:
    return chunks_dir / f"chunk_{chunk_index:03d}.wav"


def build_chunk_cache_identity(
    settings: dict[str, Any],
    *,
    chunk_text: str,
    language: str,
    segment_index: int,
    segment_text: str,
    chunk_index: int,
) -> dict[str, Any]:
    identity = build_tts_cache_identity(settings, text=chunk_text, language=language)
    identity["schema_version"] = CHUNK_CACHE_SCHEMA_VERSION
    identity["chunk_index"] = chunk_index
    identity["segment_index"] = segment_index
    identity["segment_text_hash"] = hashlib.sha256(segment_text.encode("utf-8")).hexdigest()
    identity["chunk_text_hash"] = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
    identity["chunk_stable_id"] = hashlib.sha256(
        f"{segment_index}|{identity['segment_text_hash']}|{chunk_index}|{identity['chunk_text_hash']}".encode(
            "utf-8"
        )
    ).hexdigest()[:24]
    return identity


def chunk_cache_valid(wav_path: Path, expected_identity: dict[str, Any]) -> bool:
    if not wav_path.is_file() or wav_path.stat().st_size <= 44:
        return False
    sidecar = read_tts_sidecar(wav_path)
    if not sidecar:
        return False
    if sidecar.get("schema_version") != CHUNK_CACHE_SCHEMA_VERSION:
        return False
    expected_key = cache_key_from_identity(expected_identity)
    if sidecar.get("cache_key") != expected_key:
        return False
    return sidecar.get("chunk_text_hash") == expected_identity.get("chunk_text_hash")


def _validate_chunk_wav(path: Path, *, min_speech_sec: float = 0.05) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 44:
        raise AppError(
            502,
            ErrorInfo(
                code="OMNIVOICE_CHUNK_INVALID",
                message="Chunk synthesis produced a missing or empty WAV.",
                action="Retry TTS for this segment.",
                retryable=True,
            ),
        )
    envelope = measure_speech_envelope(path)
    if envelope.speech_duration <= min_speech_sec:
        raise AppError(
            502,
            ErrorInfo(
                code="OMNIVOICE_CHUNK_SILENT",
                message="Chunk synthesis produced silent or unusable audio.",
                action="Retry with smaller chunks.",
                retryable=True,
            ),
        )
    return {
        "raw_duration": envelope.raw_wav_duration,
        "speech_duration": envelope.speech_duration,
        "leading_silence": envelope.leading_silence,
        "trailing_silence": envelope.trailing_silence,
        "measurement_confidence": envelope.measurement_confidence,
    }


def _write_manifest(chunks_dir: Path, payload: dict[str, Any]) -> None:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_manifest_path(chunks_dir).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def synthesize_omnivoice_with_chunking(
    *,
    text: str,
    output_path: Path,
    synthesize_fn: SynthesizeFn,
    settings: dict[str, Any],
    segment: dict[str, Any] | None = None,
    language: str = "vi",
    transcribe_fn: Callable[[Path], str] | None = None,
    vendor_dir: Path | None = None,
) -> dict[str, Any]:
    """Synthesize narration; chunk long text, concat, validate, optional fidelity."""
    segment = segment if segment is not None else {}
    cfg = omnivoice_chunk_settings(settings)
    diagnostics = segment_text_diagnostics(text, settings)
    segment_index = int(segment.get("index", 0))
    chunks_dir = output_path.parent / "chunks" / str(segment_index)

    retry_limits = list(cfg["retry_max_chars"])
    max_attempts = 1 + int(cfg["max_retries"])
    attempt = 0
    last_error: Exception | None = None
    chunk_retry_count = 0
    chunk_cache_hits = 0
    chunk_cache_misses = 0

    def _similarity(value: dict[str, Any]) -> float:
        similarity = value.get("tts_text_similarity")
        return float(similarity) if isinstance(similarity, (int, float)) else -1.0

    while attempt < max_attempts:
        max_chars = retry_limits[min(attempt, len(retry_limits) - 1)]
        try:
            chunk_specs = split_omnivoice_text_semantic(text, max_chars=max_chars, target_chars=cfg["target_chars"])
            validate_chunk_reconstruction(text, [item["text"] for item in chunk_specs])
        except ValueError as exc:
            raise AppError(
                422,
                ErrorInfo(
                    code="OMNIVOICE_CHUNK_SPLIT_FAILED",
                    message=str(exc),
                    action="Verify translation text for this segment.",
                ),
            ) from exc

        chunk_records: list[dict[str, Any]] = []
        chunk_paths: list[Path] = []
        pause_ms_list: list[int] = []
        trailing_ms: list[int] = []
        failed = False

        for spec in chunk_specs:
            chunk_index = int(spec["chunk_index"])
            chunk_text = str(spec["text"])
            chunk_path = chunk_wav_path(chunks_dir, chunk_index)
            identity = build_chunk_cache_identity(
                settings,
                chunk_text=chunk_text,
                language=language,
                segment_index=segment_index,
                segment_text=text,
                chunk_index=chunk_index,
            )
            cache_key = cache_key_from_identity(identity)
            metrics: dict[str, Any] = {}
            status = "accepted"
            retry_count = 0

            if chunk_cache_valid(chunk_path, identity):
                metrics = _validate_chunk_wav(chunk_path)
                chunk_cache_hits += 1
            else:
                chunk_cache_misses += 1
                if chunk_path.is_file():
                    chunk_path.unlink()
                synthesize_fn(chunk_text, chunk_path)
                metrics = _validate_chunk_wav(chunk_path)
                write_tts_sidecar(
                    chunk_path,
                    identity,
                    extra={
                        "chunk_index": chunk_index,
                        "segment_index": segment_index,
                        "chunk_text_preview": chunk_text[:120],
                    },
                )

            chunk_paths.append(chunk_path)
            pause_ms_list.append(pause_ms_for_chunk(chunk_text, str(spec["split_kind"]), cfg))
            trailing_ms.append(int(float(metrics.get("trailing_silence", 0.0)) * 1000))

            chunk_records.append(
                {
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "text_start": spec["text_start"],
                    "text_end": spec["text_end"],
                    "split_kind": spec["split_kind"],
                    "cache_key": cache_key,
                    "raw_audio_path": str(chunk_path),
                    "raw_duration": metrics["raw_duration"],
                    "speech_duration": metrics["speech_duration"],
                    "status": status,
                    "retry_count": retry_count,
                }
            )

        if failed:
            attempt += 1
            chunk_retry_count += 1
            continue

        timeline = concat_omnivoice_chunks(
            chunk_paths,
            pause_ms_list=pause_ms_list,
            output_path=output_path,
            trailing_silence_ms=trailing_ms,
        )
        spoken_aggregate = "".join(str(record["text"]) for record in chunk_records)
        if spoken_aggregate:
            output_path.with_suffix(".spoken.txt").write_text(spoken_aggregate, encoding="utf-8")
        for record, slot in zip(chunk_records, timeline):
            record["audio_start"] = slot["audio_start"]
            record["audio_end"] = slot["audio_end"]
            record["concat_pause_ms"] = slot["concat_pause_ms"]

        envelope = measure_speech_envelope(output_path)
        fidelity = run_segment_fidelity_check(
            wav_path=output_path,
            expected_text=text,
            settings=settings,
            chunk_count=len(chunk_records),
            speech_duration=envelope.speech_duration,
            raw_duration=envelope.raw_wav_duration,
            transcribe_fn=transcribe_fn,
            vendor_dir=vendor_dir,
        )

        low_fidelity = fidelity.get("tts_fidelity_status") in {"poor", "failed"}
        if low_fidelity and cfg["fallback_full_segment_enabled"] and len(chunk_specs) > 1:
            import shutil

            chunked_backup = output_path.with_suffix(".chunked_candidate.wav")
            shutil.copy2(output_path, chunked_backup)
            try:
                synthesize_fn(text, output_path)
                full_envelope = measure_speech_envelope(output_path)
                full_fidelity = run_segment_fidelity_check(
                    wav_path=output_path,
                    expected_text=text,
                    settings=settings,
                    chunk_count=1,
                    speech_duration=full_envelope.speech_duration,
                    raw_duration=full_envelope.raw_wav_duration,
                    transcribe_fn=transcribe_fn,
                    vendor_dir=vendor_dir,
                )
                full_status = str(full_fidelity.get("tts_fidelity_status") or "not_checked")
                use_full_segment = full_status not in {"poor", "failed"} or _similarity(full_fidelity) >= _similarity(fidelity)
                if use_full_segment:
                    manifest = {
                        "schema_version": CHUNK_CACHE_SCHEMA_VERSION,
                        "segment_index": segment_index,
                        "canonical_text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "chunk_count": len(chunk_records),
                        "attempt": attempt,
                        "max_chars_used": max_chars,
                        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "chunks": chunk_records,
                        "fidelity": fidelity,
                        "full_segment_fallback_fidelity": full_fidelity,
                        "selected_path": "full_segment_fallback",
                    }
                    _write_manifest(chunks_dir, manifest)
                    result = {
                        "tts_chunking_used": False,
                        "tts_chunk_count": 1,
                        "tts_chunks": [],
                        "tts_external_chunking_attempted": True,
                        "tts_external_chunking_fallback_used": True,
                        "tts_external_chunking_chunk_count": len(chunk_records),
                        "tts_external_chunking_fidelity_status": fidelity.get("tts_fidelity_status"),
                        "tts_external_chunking_text_similarity": fidelity.get("tts_text_similarity"),
                        "tts_chunk_retry_count": chunk_retry_count,
                        "tts_chunk_cache_hits": chunk_cache_hits,
                        "tts_chunk_cache_misses": chunk_cache_misses,
                        **diagnostics,
                        **full_fidelity,
                    }
                    if segment is not None:
                        segment.update(result)
                        attach_speech_metrics(segment, full_envelope)
                    return result
                shutil.copy2(chunked_backup, output_path)
            except Exception as exc:
                last_error = exc
                if chunked_backup.is_file():
                    shutil.copy2(chunked_backup, output_path)
            finally:
                chunked_backup.unlink(missing_ok=True)

        if (
            low_fidelity
            and cfg["retry_on_fidelity_failure"]
            and attempt + 1 < max_attempts
            and len(chunk_specs) > 1
        ):
            attempt += 1
            chunk_retry_count += 1
            last_error = None
            continue

        manifest = {
            "schema_version": CHUNK_CACHE_SCHEMA_VERSION,
            "segment_index": segment_index,
            "canonical_text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "chunk_count": len(chunk_records),
            "attempt": attempt,
            "max_chars_used": max_chars,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "chunks": chunk_records,
            "fidelity": fidelity,
            "selected_path": "external_chunking",
        }
        _write_manifest(chunks_dir, manifest)

        result = {
            "tts_chunking_used": len(chunk_records) > 1,
            "tts_chunk_count": len(chunk_records),
            "tts_chunks": chunk_records,
            "tts_external_chunking_attempted": True,
            "tts_external_chunking_fallback_used": False,
            "tts_chunk_retry_count": chunk_retry_count,
            "tts_chunk_cache_hits": chunk_cache_hits,
            "tts_chunk_cache_misses": chunk_cache_misses,
            **diagnostics,
            **fidelity,
        }
        if segment is not None:
            segment.update(result)
            attach_speech_metrics(segment, envelope)
        return result

    if last_error is not None:
        raise last_error
    raise AppError(
        502,
        ErrorInfo(
            code="OMNIVOICE_CHUNK_RETRY_EXHAUSTED",
            message="Chunked TTS failed after retries.",
            action="Review segment translation and retry TTS.",
            retryable=True,
        ),
    )


def synthesize_short_or_chunked(
    *,
    text: str,
    output_path: Path,
    synthesize_fn: SynthesizeFn,
    settings: dict[str, Any],
    segment: dict[str, Any] | None = None,
    language: str = "vi",
    transcribe_fn: Callable[[Path], str] | None = None,
    vendor_dir: Path | None = None,
) -> dict[str, Any]:
    if chunking_required(text, settings):
        return synthesize_omnivoice_with_chunking(
            text=text,
            output_path=output_path,
            synthesize_fn=synthesize_fn,
            settings=settings,
            segment=segment,
            language=language,
            transcribe_fn=transcribe_fn,
            vendor_dir=vendor_dir,
        )

    diagnostics = segment_text_diagnostics(text, settings)
    # 2.6: retry synthesis when the ASR-back fidelity check fails for a single-shot segment.
    # OmniVoice sampling is stochastic, so re-synthesizing can recover dropped words. We keep
    # the attempt with the best measured fidelity and never exceed the configured retry budget.
    try:
        max_fidelity_retries = max(0, int(settings.get("tts_fidelity_retry_max_attempts", 1) or 0))
    except (TypeError, ValueError):
        max_fidelity_retries = 1

    import shutil

    best_similarity = -1.0
    best_backup: Path | None = None
    best_envelope = None
    best_fidelity: dict[str, Any] | None = None
    envelope = None
    fidelity: dict[str, Any] = {}
    retry_used = 0
    for attempt in range(max_fidelity_retries + 1):
        if attempt > 0:
            retry_used += 1
        synthesize_fn(text, output_path)
        envelope = measure_speech_envelope(output_path)
        fidelity = run_segment_fidelity_check(
            wav_path=output_path,
            expected_text=text,
            settings=settings,
            chunk_count=1,
            speech_duration=envelope.speech_duration,
            raw_duration=envelope.raw_wav_duration,
            transcribe_fn=transcribe_fn,
            vendor_dir=vendor_dir,
        )
        status = str(fidelity.get("tts_fidelity_status") or "not_checked")
        similarity = fidelity.get("tts_text_similarity")
        similarity = float(similarity) if isinstance(similarity, (int, float)) else -1.0
        # Remember the best attempt so a worse retry never discards a better take.
        if best_backup is None or similarity > best_similarity:
            best_similarity = similarity
            best_backup = output_path.with_suffix(".bestfidelity.wav")
            shutil.copy2(output_path, best_backup)
            best_envelope = envelope
            best_fidelity = fidelity
        if status not in {"poor", "failed"} or attempt >= max_fidelity_retries:
            break

    if best_backup is not None and best_backup.is_file():
        if best_fidelity is not None and best_fidelity is not fidelity:
            shutil.copy2(best_backup, output_path)
            envelope = best_envelope
            fidelity = best_fidelity
        best_backup.unlink(missing_ok=True)

    result = {
        "tts_chunking_used": False,
        "tts_chunk_count": 1,
        "tts_chunks": [],
        "tts_chunk_retry_count": retry_used,
        "tts_chunk_cache_hits": 0,
        "tts_chunk_cache_misses": 0,
        **diagnostics,
        **fidelity,
    }
    if segment is not None:
        segment.update(result)
        attach_speech_metrics(segment, envelope)
    return result
