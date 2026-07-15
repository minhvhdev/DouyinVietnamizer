"""Orchestrate chunked OmniVoice synthesis, cache, concat, and fidelity."""
from __future__ import annotations

import hashlib
import json
import logging
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
    smaller_retry_max_chars,
    split_omnivoice_text_semantic,
    validate_chunk_reconstruction,
)
from .omnivoice_wav_concat import concat_omnivoice_chunks
from .tts_cache import (
    build_tts_cache_identity,
    cache_key_from_identity,
    fidelity_status_cacheable,
    read_tts_sidecar,
    write_tts_sidecar,
)
from .tts_fidelity import run_segment_fidelity_check
from .tts_speech_analysis import attach_speech_metrics, measure_speech_envelope

SynthesizeFn = Callable[[str, Path], None]

logger = logging.getLogger(__name__)


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
    if not fidelity_status_cacheable(sidecar.get("tts_fidelity_status")):
        return False
    expected_key = cache_key_from_identity(expected_identity)
    if sidecar.get("cache_key") != expected_key:
        return False
    return sidecar.get("chunk_text_hash") == expected_identity.get("chunk_text_hash")


def _validate_chunk_wav(path: Path, *, min_speech_sec: float = 0.05) -> dict[str, Any]:
    from .omnivoice_diagnostics import diagnostics_enabled, log_event, probe_wav_path

    if diagnostics_enabled():
        log_event("chunk_validation_input", {"probe": probe_wav_path(path), "path_name": path.name})
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
        if diagnostics_enabled():
            log_event(
                "chunk_validation_silent",
                {
                    "stage": "chunk_validation",
                    "probe": probe_wav_path(path),
                    "speech_duration": envelope.speech_duration,
                    "raw_duration": envelope.raw_wav_duration,
                },
            )
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


def _preview(text: str, limit: int = 80) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "…"


def _similarity(value: dict[str, Any]) -> float:
    similarity = value.get("tts_text_similarity")
    return float(similarity) if isinstance(similarity, (int, float)) else -1.0


def _is_low_fidelity(fidelity: dict[str, Any]) -> bool:
    return str(fidelity.get("tts_fidelity_status") or "") in {"poor", "failed"}


def _fidelity_reason(fidelity: dict[str, Any]) -> str:
    warnings = fidelity.get("tts_fidelity_warnings") or []
    if isinstance(warnings, list) and warnings:
        return ",".join(str(item) for item in warnings)
    status = str(fidelity.get("tts_fidelity_status") or "unknown")
    similarity = fidelity.get("tts_text_similarity")
    if isinstance(similarity, (int, float)):
        return f"{status};similarity={similarity:.3f}"
    return status


def _log_final_chunk_fidelity_failure(
    *,
    segment_index: int,
    chunk_index: int,
    attempt: int,
    max_chars: int,
    fidelity: dict[str, Any],
    preview: str,
) -> None:
    """Emit structured diagnostics without dumping full chunk text at INFO."""
    logger.error(
        "omnivoice fidelity final failure: seg=%s chunk_index=%s attempt=%s max_chars=%s "
        "status=%s reason=%s deletion_span=%s similarity=%s preview=%r",
        segment_index,
        chunk_index,
        attempt,
        max_chars,
        fidelity.get("tts_fidelity_status"),
        _fidelity_reason(fidelity),
        fidelity.get("tts_max_contiguous_deletion"),
        fidelity.get("tts_text_similarity"),
        preview,
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
    initial_max_chars: int | None = None,
) -> dict[str, Any]:
    """Synthesize narration; chunk long text, concat, validate, optional fidelity."""
    segment = segment if segment is not None else {}
    cfg = omnivoice_chunk_settings(settings)
    diagnostics = segment_text_diagnostics(text, settings)
    segment_index = int(segment.get("index", 0))
    chunks_dir = output_path.parent / "chunks" / str(segment_index)

    max_retries = max(0, int(cfg["max_retries"]))
    retry_budget = {"left": max_retries, "used": 0}
    chunk_cache_hits = 0
    chunk_cache_misses = 0
    next_leaf_index = 0

    def _allocate_index() -> int:
        nonlocal next_leaf_index
        value = next_leaf_index
        next_leaf_index += 1
        return value

    def _synth_leaf(
        chunk_text: str,
        *,
        split_kind: str,
        max_chars_used: int,
        force_resynth: bool = False,
    ) -> dict[str, Any]:
        nonlocal chunk_cache_hits, chunk_cache_misses
        chunk_index = _allocate_index()
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
        used_cache = False
        if not force_resynth and chunk_cache_valid(chunk_path, identity):
            metrics = _validate_chunk_wav(chunk_path)
            chunk_cache_hits += 1
            used_cache = True
        else:
            chunk_cache_misses += 1
            if chunk_path.is_file():
                chunk_path.unlink()
            if len(chunk_text) > max_chars_used:
                raise AppError(
                    422,
                    ErrorInfo(
                        code="OMNIVOICE_CHUNK_TOO_LONG",
                        message=f"Chunk length {len(chunk_text)} exceeds max_chars={max_chars_used}.",
                        action="Retry with smaller chunk limits.",
                    ),
                )
            logger.debug(
                "omnivoice synthesize leaf: seg=%s idx=%s chars=%d max=%d preview=%r",
                segment_index,
                chunk_index,
                len(chunk_text),
                max_chars_used,
                _preview(chunk_text),
            )
            synthesize_fn(chunk_text, chunk_path)
            metrics = _validate_chunk_wav(chunk_path)

        fidelity = run_segment_fidelity_check(
            wav_path=chunk_path,
            expected_text=chunk_text,
            settings=settings,
            chunk_count=1,
            speech_duration=metrics["speech_duration"],
            raw_duration=metrics["raw_duration"],
            transcribe_fn=transcribe_fn,
            vendor_dir=vendor_dir,
        )
        # Always write status; wav_cache_valid / chunk_cache_valid refuse poor|failed.
        write_tts_sidecar(
            chunk_path,
            identity,
            extra={
                "chunk_index": chunk_index,
                "segment_index": segment_index,
                "chunk_text_preview": chunk_text[:120],
                "tts_fidelity_status": fidelity.get("tts_fidelity_status"),
                "tts_text_similarity": fidelity.get("tts_text_similarity"),
                "tts_max_contiguous_deletion": fidelity.get("tts_max_contiguous_deletion"),
                "max_chars_used": max_chars_used,
            },
        )
        return {
            "chunk_index": chunk_index,
            "text": chunk_text,
            "split_kind": split_kind,
            "cache_key": cache_key,
            "raw_audio_path": str(chunk_path),
            "raw_duration": metrics["raw_duration"],
            "speech_duration": metrics["speech_duration"],
            "status": "accepted",
            "retry_count": 0,
            "used_cache": used_cache,
            "fidelity": fidelity,
            "trailing_silence": metrics.get("trailing_silence", 0.0),
            "path": chunk_path,
            "max_chars_used": max_chars_used,
        }

    def _synthesize_tree(chunk_text: str, *, max_chars: int, depth: int = 0) -> list[dict[str, Any]]:
        try:
            specs = split_omnivoice_text_semantic(
                chunk_text,
                max_chars=max_chars,
                target_chars=min(cfg["target_chars"], max_chars),
            )
            validate_chunk_reconstruction(chunk_text, [item["text"] for item in specs])
        except ValueError as exc:
            raise AppError(
                422,
                ErrorInfo(
                    code="OMNIVOICE_CHUNK_SPLIT_FAILED",
                    message=str(exc),
                    action="Verify translation text for this segment.",
                ),
            ) from exc

        if not specs:
            return []

        logger.info(
            "omnivoice chunk decision: seg=%s depth=%d text_len=%d max_chars=%d chunks=%d retries_left=%d",
            segment_index,
            depth,
            len(chunk_text),
            max_chars,
            len(specs),
            retry_budget["left"],
        )

        leaves: list[dict[str, Any]] = []
        for spec in specs:
            piece = str(spec["text"])
            split_kind = str(spec["split_kind"])
            record = _synth_leaf(piece, split_kind=split_kind, max_chars_used=max_chars)
            low = _is_low_fidelity(record["fidelity"])
            if not low or not cfg["retry_on_fidelity_failure"] or retry_budget["left"] <= 0:
                if low:
                    _log_final_chunk_fidelity_failure(
                        segment_index=segment_index,
                        chunk_index=int(record["chunk_index"]),
                        attempt=retry_budget["used"],
                        max_chars=max_chars,
                        fidelity=record["fidelity"],
                        preview=_preview(piece),
                    )
                leaves.append(record)
                continue

            next_max = smaller_retry_max_chars(max_chars, cfg)
            can_split_smaller = False
            if next_max is not None and next_max < len(piece):
                try:
                    sub_specs = split_omnivoice_text_semantic(piece, max_chars=next_max)
                    can_split_smaller = len(sub_specs) > 1
                except ValueError:
                    can_split_smaller = False

            if can_split_smaller and next_max is not None:
                retry_budget["left"] -= 1
                retry_budget["used"] += 1
                logger.info(
                    "omnivoice fidelity fallback: seg=%s failed_chunk_chars=%d -> max_chars=%d "
                    "(retries_left=%d) preview=%r",
                    segment_index,
                    len(piece),
                    next_max,
                    retry_budget["left"],
                    _preview(piece),
                )
                # Drop the failed leaf wav from the timeline; keep siblings untouched.
                sub_leaves = _synthesize_tree(piece, max_chars=next_max, depth=depth + 1)
                for sub in sub_leaves:
                    sub["retry_count"] = int(sub.get("retry_count") or 0) + 1
                leaves.extend(sub_leaves)
                continue

            # Still one chunk at this size: at most one same-input retry per budget unit.
            retry_budget["left"] -= 1
            retry_budget["used"] += 1
            logger.info(
                "omnivoice same-input chunk retry: seg=%s chars=%d retries_left=%d preview=%r",
                segment_index,
                len(piece),
                retry_budget["left"],
                _preview(piece),
            )
            retried = _synth_leaf(
                piece,
                split_kind=split_kind,
                max_chars_used=max_chars,
                force_resynth=True,
            )
            retried["retry_count"] = int(record.get("retry_count") or 0) + 1
            if _similarity(retried["fidelity"]) >= _similarity(record["fidelity"]):
                chosen = retried
            else:
                chosen = record
            if _is_low_fidelity(chosen["fidelity"]):
                _log_final_chunk_fidelity_failure(
                    segment_index=segment_index,
                    chunk_index=int(chosen["chunk_index"]),
                    attempt=retry_budget["used"],
                    max_chars=max_chars,
                    fidelity=chosen["fidelity"],
                    preview=_preview(piece),
                )
            leaves.append(chosen)
        return leaves

    start_max = int(initial_max_chars) if initial_max_chars is not None else int(cfg["max_chars"])
    start_max = max(int(cfg["min_chars"]), start_max)
    chunk_records = _synthesize_tree(text, max_chars=start_max)
    if not chunk_records:
        raise AppError(
            422,
            ErrorInfo(
                code="OMNIVOICE_CHUNK_SPLIT_FAILED",
                message="Chunk list is empty.",
                action="Verify translation text for this segment.",
            ),
        )

    # Re-index leaves sequentially for stable manifest / concat paths already written.
    for index, record in enumerate(chunk_records):
        record["chunk_index"] = index
        record["text_start"] = None
        record["text_end"] = None

    chunk_paths = [Path(record["path"]) for record in chunk_records]
    pause_ms_list = [
        pause_ms_for_chunk(str(record["text"]), str(record["split_kind"]), cfg) for record in chunk_records
    ]
    trailing_ms = [int(float(record.get("trailing_silence", 0.0)) * 1000) for record in chunk_records]

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
        record.pop("path", None)
        record.pop("trailing_silence", None)
        record.pop("used_cache", None)
        # Keep lightweight fidelity summary only.
        fidelity = record.pop("fidelity", {}) or {}
        record["tts_fidelity_status"] = fidelity.get("tts_fidelity_status")

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

    low_fidelity = _is_low_fidelity(fidelity)
    if low_fidelity and cfg["fallback_full_segment_enabled"] and len(chunk_records) > 1:
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
            use_full_segment = full_status not in {"poor", "failed"} or _similarity(full_fidelity) >= _similarity(
                fidelity
            )
            if use_full_segment:
                manifest = {
                    "schema_version": CHUNK_CACHE_SCHEMA_VERSION,
                    "segment_index": segment_index,
                    "canonical_text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "chunk_count": len(chunk_records),
                    "attempt": retry_budget["used"],
                    "max_chars_used": start_max,
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
                    "tts_chunk_retry_count": retry_budget["used"],
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
        except Exception:
            if chunked_backup.is_file():
                shutil.copy2(chunked_backup, output_path)
        finally:
            chunked_backup.unlink(missing_ok=True)

    manifest = {
        "schema_version": CHUNK_CACHE_SCHEMA_VERSION,
        "segment_index": segment_index,
        "canonical_text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "chunk_count": len(chunk_records),
        "attempt": retry_budget["used"],
        "max_chars_used": start_max,
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
        "tts_chunk_retry_count": retry_budget["used"],
        "tts_chunk_cache_hits": chunk_cache_hits,
        "tts_chunk_cache_misses": chunk_cache_misses,
        **diagnostics,
        **fidelity,
    }
    if segment is not None:
        segment.update(result)
        attach_speech_metrics(segment, envelope)
    return result


def record_direct_segment_result(
    *,
    text: str,
    output_path: Path,
    settings: dict[str, Any],
    segment: dict[str, Any] | None = None,
    language: str = "vi",
    transcribe_fn: Callable[[Path], str] | None = None,
    vendor_dir: Path | None = None,
    chunk_retry_count: int = 0,
) -> dict[str, Any]:
    """Measure speech envelope and fidelity for a single-shot worker WAV."""
    _ = language
    diagnostics = segment_text_diagnostics(text, settings)
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
    result = {
        "tts_chunking_used": False,
        "tts_chunk_count": 1,
        "tts_chunks": [],
        "tts_chunk_retry_count": chunk_retry_count,
        "tts_chunk_cache_hits": 0,
        "tts_chunk_cache_misses": 0,
        **diagnostics,
        **fidelity,
    }
    if segment is not None:
        segment.update(result)
        attach_speech_metrics(segment, envelope)
    return result


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
    cfg = omnivoice_chunk_settings(settings)
    cleaned = " ".join((text or "").split()).strip()
    logger.info(
        "omnivoice routing: text_len=%d max_chars=%d external=%s chunking_required=%s",
        len(cleaned),
        cfg["max_chars"],
        cfg["external_chunking_enabled"],
        chunking_required(text, settings),
    )
    logger.debug("omnivoice routing preview=%r", _preview(cleaned))

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
    smaller_fallback_tried = False

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
        if best_backup is None or similarity > best_similarity:
            best_similarity = similarity
            best_backup = output_path.with_suffix(".bestfidelity.wav")
            shutil.copy2(output_path, best_backup)
            best_envelope = envelope
            best_fidelity = fidelity

        if status not in {"poor", "failed"}:
            break

        # Short-text fidelity fail: try fallback splitter with a smaller max before same-input retry.
        if (
            cfg["retry_on_fidelity_failure"]
            and cfg["external_chunking_enabled"]
            and not smaller_fallback_tried
        ):
            smaller_fallback_tried = True
            fallback_max = smaller_retry_max_chars(cfg["max_chars"], cfg)
            if fallback_max is None:
                fallback_max = max(cfg["min_chars"], min(cfg["max_chars"] - 1, 140))
            try:
                specs = split_omnivoice_text_semantic(text, max_chars=fallback_max)
            except ValueError:
                specs = []
            if len(specs) > 1:
                logger.info(
                    "omnivoice short-text fidelity -> chunked fallback: text_len=%d max_chars=%d chunks=%d",
                    len(cleaned),
                    fallback_max,
                    len(specs),
                )
                if best_backup is not None:
                    best_backup.unlink(missing_ok=True)
                return synthesize_omnivoice_with_chunking(
                    text=text,
                    output_path=output_path,
                    synthesize_fn=synthesize_fn,
                    settings=settings,
                    segment=segment,
                    language=language,
                    transcribe_fn=transcribe_fn,
                    vendor_dir=vendor_dir,
                    initial_max_chars=fallback_max,
                )
            logger.info(
                "omnivoice short-text still single chunk after fallback max=%d; same-input retry budget=%d",
                fallback_max,
                max_fidelity_retries - attempt,
            )

        if attempt >= max_fidelity_retries:
            break

    if best_backup is not None and best_backup.is_file():
        if best_fidelity is not None and best_fidelity is not fidelity:
            shutil.copy2(best_backup, output_path)
            envelope = best_envelope
            fidelity = best_fidelity
        best_backup.unlink(missing_ok=True)

    if _is_low_fidelity(fidelity):
        _log_final_chunk_fidelity_failure(
            segment_index=int(segment.get("index", 0)) if segment else 0,
            chunk_index=0,
            attempt=retry_used,
            max_chars=int(cfg["max_chars"]),
            fidelity=fidelity,
            preview=_preview(cleaned),
        )

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
