"""Background voice duration calibration runner."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from .adapters.tts import TtsSession
from .database import Database
from .errors import AppError
from .models import ErrorInfo
from .voice_calibration_dataset import (
    CALIBRATION_MODES,
    CalibrationSample,
    load_calibration_dataset,
    select_calibration_samples,
)
from .voice_calibration_samples import (
    ANALYSIS_SCHEMA_VERSION,
    SampleEvaluation,
    aggregate_calibration_profile,
    cache_key_for_sample,
    compute_validation_metrics_for_samples,
    evaluate_calibration_sample,
)
from .voice_calibration_store import (
    create_job_manifest,
    job_dir,
    load_manifest,
    load_progress,
    save_analysis,
    save_manifest,
    save_progress,
    save_report,
    sample_record_by_id,
    upsert_sample_progress,
    utc_now,
)
from .voice_duration_profile import merge_bootstrap_profile, reset_voice_profile
from .voice_identity import (
    generation_config_hash,
    identity_from_settings,
    identity_profile_key,
    settings_for_cloned_voice,
)
from .voice_profile_policy import classify_profile_quality


class VoiceCalibrationRunner:
    def __init__(self, *, data_dir: Path, database: Database, settings_getter: Callable[[], dict[str, Any]]) -> None:
        self.data_dir = data_dir
        self.database = database
        self.settings_getter = settings_getter
        self.lock = threading.Lock()
        self.threads: dict[str, threading.Thread] = {}
        self.cancelled_jobs: set[str] = set()
        self._manifest_locks: dict[str, threading.Lock] = {}

    def _manifest_lock(self, job_id: str) -> threading.Lock:
        with self.lock:
            return self._manifest_locks.setdefault(job_id, threading.Lock())

    def _save_manifest_locked(self, manifest: dict[str, Any]) -> None:
        job_id = str(manifest["job_id"])
        with self._manifest_lock(job_id):
            save_manifest(self.data_dir, manifest)

    def _load_manifest_locked(self, job_id: str) -> dict[str, Any]:
        with self._manifest_lock(job_id):
            return load_manifest(self.data_dir, job_id)

    def is_cancelled(self, job_id: str) -> bool:
        with self.lock:
            return job_id in self.cancelled_jobs

    def cancel_job(self, job_id: str) -> None:
        with self.lock:
            self.cancelled_jobs.add(job_id)
        manifest = self._load_manifest_locked(job_id)
        manifest["status"] = "cancelled"
        self._save_manifest_locked(manifest)
        self._update_voice_calibration_status(manifest["voice_id"], "cancelled", manifest)
        try:
            from .gpu_lease import clear_gpu_lease_state

            clear_gpu_lease_state(reason=f"cancel_voice_calibration:{job_id}")
        except Exception:
            pass

    def _update_voice_calibration_status(
        self,
        voice_id: str,
        status: str,
        manifest: dict[str, Any] | None = None,
    ) -> None:
        fields: dict[str, Any] = {
            "duration_profile_status": status,
        }
        if manifest:
            fields["active_calibration_job_id"] = manifest.get("job_id")
            if status in {"ready", "partial", "failed", "cancelled"}:
                fields["active_calibration_job_id"] = None
        set_clause = ", ".join(f"{key} = ?" for key in fields)
        values = list(fields.values()) + [voice_id]
        with self.database.connection:
            self.database.connection.execute(
                f"UPDATE cloned_voices SET {set_clause} WHERE id = ?",
                values,
            )

    def _get_cloned_voice(self, voice_id: str) -> dict[str, Any]:
        row = self.database.connection.execute(
            "SELECT * FROM cloned_voices WHERE id = ?",
            (voice_id,),
        ).fetchone()
        if not row:
            raise AppError(
                404,
                ErrorInfo(code="VOICE_NOT_FOUND", message="Cloned voice not found.", action="Verify voice ID."),
            )
        return dict(row)

    def _cloned_voice_dir(self, backend: str) -> Path:
        return self.data_dir / "cloned_voices_omnivoice"

    def _resolve_paths(self, row: dict[str, Any]) -> tuple[Path, str]:
        cloned_dir = self._cloned_voice_dir(row.get("backend") or "omnivoice")
        wav_path = cloned_dir / row["wav_filename"]
        if not wav_path.is_file():
            raise AppError(
                404,
                ErrorInfo(
                    code="REFERENCE_AUDIO_MISSING",
                    message="Reference audio file is missing.",
                    action="Re-upload the cloned voice reference audio.",
                ),
            )
        transcript = (row.get("transcript") or "").strip()
        sidecar = wav_path.with_suffix(".txt")
        if not transcript and sidecar.is_file():
            transcript = sidecar.read_text(encoding="utf-8").strip()
        if not transcript:
            raise AppError(
                422,
                ErrorInfo(
                    code="OMNIVOICE_REF_TEXT_REQUIRED",
                    message="Reference text is required for calibration.",
                    action="Add ref_text to the cloned voice.",
                ),
            )
        return wav_path, transcript

    def preflight(self, voice_id: str, mode: str) -> dict[str, Any]:
        row = self._get_cloned_voice(voice_id)
        if (row.get("voice_status") or "ready") != "ready":
            raise AppError(
                409,
                ErrorInfo(code="VOICE_NOT_READY", message="Voice clone is not ready.", action="Finish voice clone first."),
            )
        wav_path, transcript = self._resolve_paths(row)
        mode_key = (mode or "standard").strip().lower()
        if mode_key not in CALIBRATION_MODES:
            raise AppError(
                422,
                ErrorInfo(code="INVALID_CALIBRATION_MODE", message="Invalid calibration mode.", action="Use quick, standard, or full."),
            )
        active = self.database.connection.execute(
            """
            SELECT job_id FROM voice_calibration_jobs
            WHERE voice_id = ? AND status IN ('queued', 'running')
            LIMIT 1
            """,
            (voice_id,),
        ).fetchone()
        if active:
            raise AppError(
                409,
                ErrorInfo(
                    code="CALIBRATION_ALREADY_RUNNING",
                    message="Another calibration job is already active for this voice.",
                    action="Cancel or wait for the current job.",
                ),
            )
        base_settings = self.settings_getter()
        settings = settings_for_cloned_voice(
            base_settings,
            voice_id=voice_id,
            wav_path=wav_path,
            transcript=transcript,
        )
        identity = identity_from_settings(
            settings,
            cloned_voice_id=voice_id,
            reference_audio_path=wav_path,
            reference_text=transcript,
        )
        identity_key = identity_profile_key(identity)
        dataset = load_calibration_dataset()
        samples = select_calibration_samples(dataset, mode_key)
        return {
            "voice_id": voice_id,
            "mode": mode_key,
            "voice_identity_key": identity_key,
            "sample_total": len(samples),
            "dataset_version": dataset.get("version"),
        }

    def start_calibration(self, voice_id: str, mode: str = "standard", *, resume_job_id: str | None = None) -> dict[str, Any]:
        if resume_job_id:
            return self.resume_calibration(voice_id, resume_job_id)
        preflight = self.preflight(voice_id, mode)
        manifest = create_job_manifest(
            data_dir=self.data_dir,
            voice_id=voice_id,
            voice_identity_key=preflight["voice_identity_key"],
            mode=preflight["mode"],
            dataset_version=str(preflight["dataset_version"]),
            sample_total=int(preflight["sample_total"]),
        )
        with self.database.connection:
            self.database.connection.execute(
                """
                INSERT INTO voice_calibration_jobs (
                    job_id, voice_id, voice_identity_key, mode, status, sample_total,
                    sample_completed, sample_accepted, sample_rejected, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0, 0, 0, ?, ?)
                """,
                (
                    manifest["job_id"],
                    voice_id,
                    manifest["voice_identity_key"],
                    manifest["mode"],
                    "queued",
                    manifest["sample_total"],
                    manifest["created_at"],
                    manifest["updated_at"],
                ),
            )
            self.database.connection.execute(
                """
                UPDATE cloned_voices
                SET duration_profile_status = 'queued',
                    active_calibration_job_id = ?,
                    voice_status = COALESCE(voice_status, 'ready')
                WHERE id = ?
                """,
                (manifest["job_id"], voice_id),
            )
        self._launch_thread(manifest["job_id"])
        return self.public_status(manifest)

    def resume_calibration(self, voice_id: str, job_id: str) -> dict[str, Any]:
        manifest = self._load_manifest_locked(job_id)
        if manifest.get("voice_id") != voice_id:
            raise AppError(
                404,
                ErrorInfo(code="CALIBRATION_JOB_NOT_FOUND", message="Calibration job not found.", action="Start a new calibration."),
            )
        if manifest.get("status") in {"running", "queued"}:
            return self.public_status(manifest)
        row = self._get_cloned_voice(voice_id)
        wav_path, transcript = self._resolve_paths(row)
        settings = settings_for_cloned_voice(
            self.settings_getter(),
            voice_id=voice_id,
            wav_path=wav_path,
            transcript=transcript,
        )
        identity = identity_from_settings(
            settings,
            cloned_voice_id=voice_id,
            reference_audio_path=wav_path,
            reference_text=transcript,
        )
        if identity_profile_key(identity) != manifest.get("voice_identity_key"):
            manifest["status"] = "stale"
            self._save_manifest_locked(manifest)
            raise AppError(
                409,
                ErrorInfo(
                    code="CALIBRATION_IDENTITY_CHANGED",
                    message="Voice identity changed; resume is not allowed.",
                    action="Start a new calibration job.",
                ),
            )
        manifest["status"] = "queued"
        self._save_manifest_locked(manifest)
        with self.database.connection:
            self.database.connection.execute(
                """
                UPDATE voice_calibration_jobs
                SET status = 'queued', updated_at = ?
                WHERE job_id = ?
                """,
                (utc_now(), job_id),
            )
            self.database.connection.execute(
                "UPDATE cloned_voices SET duration_profile_status = 'queued', active_calibration_job_id = ? WHERE id = ?",
                (job_id, voice_id),
            )
        with self.lock:
            self.cancelled_jobs.discard(job_id)
        self._launch_thread(job_id)
        return self.public_status(manifest)

    def _launch_thread(self, job_id: str) -> None:
        with self.lock:
            existing = self.threads.get(job_id)
            if existing is not None and existing.is_alive():
                return
            thread = threading.Thread(target=self._run_job, args=(job_id,), daemon=True, name=f"voice-calibration-{job_id[:8]}")
            self.threads[job_id] = thread
            thread.start()

    def get_status(self, voice_id: str) -> dict[str, Any] | None:
        row = self.database.connection.execute(
            """
            SELECT * FROM voice_calibration_jobs
            WHERE voice_id = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (voice_id,),
        ).fetchone()
        if not row:
            voice = self._get_cloned_voice(voice_id)
            return {
                "voice_id": voice_id,
                "status": voice.get("duration_profile_status") or "not_started",
                "duration_profile_quality": voice.get("duration_profile_quality"),
                "duration_profile_sample_count": voice.get("duration_profile_sample_count") or 0,
            }
        manifest = self._load_manifest_locked(row["job_id"])
        return self.public_status(manifest, voice_row=dict(row))

    def public_status(self, manifest: dict[str, Any], voice_row: dict[str, Any] | None = None) -> dict[str, Any]:
        accepted = int(manifest.get("sample_accepted") or 0)
        quality = classify_profile_quality(
            accepted_count=accepted,
            validation_mae_ms=manifest.get("validation_mae_ms"),
            mode=str(manifest.get("mode") or "standard"),
            status=str(manifest.get("status") or "running"),
        )
        return {
            "voice_id": manifest.get("voice_id"),
            "job_id": manifest.get("job_id"),
            "status": manifest.get("status"),
            "mode": manifest.get("mode"),
            "completed": int(manifest.get("sample_completed") or 0),
            "total": int(manifest.get("sample_total") or 0),
            "accepted": int(manifest.get("sample_accepted") or 0),
            "rejected": int(manifest.get("sample_rejected") or 0),
            "estimated_profile_quality": quality,
            "validation_median_error_ms": manifest.get("validation_median_error_ms"),
            "syllables_per_second": manifest.get("aggregate_sps"),
        }

    def _run_job(self, job_id: str) -> None:
        tts_session: TtsSession | None = None
        manifest = self._load_manifest_locked(job_id)
        voice_id = str(manifest["voice_id"])
        try:
            manifest["status"] = "running"
            self._save_manifest_locked(manifest)
            self._update_voice_calibration_status(voice_id, "running", manifest)
            with self.database.connection:
                self.database.connection.execute(
                    "UPDATE voice_calibration_jobs SET status = 'running', updated_at = ? WHERE job_id = ?",
                    (utc_now(), job_id),
                )

            row = self._get_cloned_voice(voice_id)
            wav_path, transcript = self._resolve_paths(row)
            settings = settings_for_cloned_voice(
                self.settings_getter(),
                voice_id=voice_id,
                wav_path=wav_path,
                transcript=transcript,
            )
            identity = identity_from_settings(
                settings,
                cloned_voice_id=voice_id,
                reference_audio_path=wav_path,
                reference_text=transcript,
            )
            identity_key = identity_profile_key(identity)
            gen_hash = generation_config_hash({"speed": 1.0, "clone_mode": "reference"})
            dataset = load_calibration_dataset()
            mode = str(manifest.get("mode") or "standard")
            target_samples = select_calibration_samples(dataset, mode, str(manifest.get("dataset_version")))
            progress = load_progress(self.data_dir, job_id)
            sample_map = {sample.id: sample for sample in target_samples}
            evaluations = self._load_existing_evaluations(job_id, target_samples, progress)
            evaluated_ids = {item.sample_id for item in evaluations}

            tts_session = TtsSession(
                settings,
                data_dir=self.data_dir,
                runner=None,
            )

            for sample in target_samples:
                if self.is_cancelled(job_id):
                    break
                if sample.id in evaluated_ids:
                    continue
                evaluation, cache_hit = self._process_sample(
                    job_id=job_id,
                    sample=sample,
                    tts_session=tts_session,
                    identity_key=identity_key,
                    dataset_version=str(manifest.get("dataset_version")),
                    gen_hash=gen_hash,
                    progress=progress,
                )
                evaluations.append(evaluation)
                manifest["sample_completed"] = len(evaluations)
                if cache_hit:
                    manifest["sample_cache_hits"] = int(manifest.get("sample_cache_hits") or 0) + 1
                manifest["sample_accepted"] = len([item for item in evaluations if item.accepted])
                manifest["sample_rejected"] = len([item for item in evaluations if not item.accepted])
                if not cache_hit:
                    manifest["sample_synthesized"] = int(manifest.get("sample_synthesized") or 0) + 1
                self._save_manifest_locked(manifest)
                self._sync_job_row(manifest)

            if self.is_cancelled(job_id):
                manifest["status"] = "cancelled"
                self._save_manifest_locked(manifest)
                self._update_voice_calibration_status(voice_id, "cancelled", manifest)
                self._sync_job_row(manifest)
                return

            aggregate = aggregate_calibration_profile(evaluations)
            validation = compute_validation_metrics_for_samples(
                target_samples,
                evaluations,
                profile_sps=float(aggregate.get("syllables_per_second") or 0),
            )
            accepted_after = int(aggregate.get("accepted_after_outlier_filter") or 0)
            quality = classify_profile_quality(
                accepted_count=accepted_after,
                validation_mae_ms=validation.get("prediction_mae_ms"),
                mode=mode,
                status="ready" if accepted_after >= 10 else "partial",
            )
            profile_status = "ready" if quality in {"good", "partial"} else ("partial" if accepted_after >= 5 else "failed")
            if accepted_after >= 10 and quality in {"good", "partial", "poor"}:
                profile_payload = merge_bootstrap_profile(
                    self.data_dir,
                    identity=identity,
                    aggregate=aggregate,
                    validation=validation,
                    manifest=manifest,
                    quality=quality,
                )
                self._update_voice_profile_fields(voice_id, profile_payload, profile_status)
            elif accepted_after >= 5:
                profile_payload = merge_bootstrap_profile(
                    self.data_dir,
                    identity=identity,
                    aggregate=aggregate,
                    validation=validation,
                    manifest=manifest,
                    quality="partial",
                )
                self._update_voice_profile_fields(voice_id, profile_payload, "partial")
            else:
                self._update_voice_calibration_status(voice_id, profile_status, manifest)

            manifest.update(
                {
                    "status": profile_status,
                    "aggregate_sps": aggregate.get("syllables_per_second"),
                    "validation_mae_ms": validation.get("prediction_mae_ms"),
                    "validation_median_error_ms": validation.get("prediction_median_error_ms"),
                    "validation_p90_error_ms": validation.get("prediction_p90_error_ms"),
                }
            )
            self._save_manifest_locked(manifest)
            self._write_report(job_id, target_samples, evaluations, aggregate, validation, identity, manifest, quality)
            self._sync_job_row(manifest)
            self._update_voice_calibration_status(voice_id, profile_status, manifest)
        except Exception as exc:
            manifest = self._load_manifest_locked(job_id)
            manifest["status"] = "failed"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            self._save_manifest_locked(manifest)
            self._sync_job_row(manifest)
            self._update_voice_calibration_status(voice_id, "failed", manifest)
            traceback.print_exc()
        finally:
            if tts_session is not None:
                try:
                    tts_session.close()
                except Exception:
                    pass
            try:
                from .gpu_lease import clear_gpu_lease_state

                clear_gpu_lease_state(reason=f"voice_calibration_done:{job_id}")
            except Exception:
                pass

    def _load_existing_evaluations(
        self,
        job_id: str,
        samples: list[CalibrationSample],
        progress: dict[str, Any],
    ) -> list[SampleEvaluation]:
        sample_map = {sample.id: sample for sample in samples}
        evaluations: list[SampleEvaluation] = []
        for record in progress.get("samples") or []:
            sample_id = record.get("sample_id")
            sample = sample_map.get(sample_id)
            if not sample:
                continue
            raw_path = Path(record["raw_audio_path"]) if record.get("raw_audio_path") else job_dir(self.data_dir, job_id) / "raw" / f"{sample_id}.wav"
            if record.get("status") == "accepted":
                evaluation = evaluate_calibration_sample(sample, wav_path=raw_path if raw_path.is_file() else None)
            else:
                evaluation = evaluate_calibration_sample(
                    sample,
                    wav_path=raw_path if raw_path.is_file() else None,
                    tts_failed=record.get("rejection_reason") == "tts_failed",
                    cancelled=record.get("rejection_reason") == "cancelled",
                )
                if record.get("rejection_reason") and not evaluation.rejection_reason:
                    evaluation = SampleEvaluation(
                        sample_id=sample.id,
                        accepted=False,
                        rejection_reason=record.get("rejection_reason"),
                        syllables=evaluation.syllables,
                        observed_sps=evaluation.observed_sps,
                        envelope=evaluation.envelope,
                        clipping_ratio=evaluation.clipping_ratio,
                        analysis=evaluation.analysis,
                    )
            evaluations.append(evaluation)
        return evaluations

    def _process_sample(
        self,
        *,
        job_id: str,
        sample: CalibrationSample,
        tts_session: TtsSession,
        identity_key: str,
        dataset_version: str,
        gen_hash: str,
        progress: dict[str, Any],
    ) -> tuple[SampleEvaluation, bool]:
        existing = sample_record_by_id(progress, sample.id)
        raw_dir = job_dir(self.data_dir, job_id) / "raw"
        raw_path = raw_dir / f"{sample.id}.wav"
        cache_hit = False

        if existing and existing.get("status") in {"accepted", "rejected"} and existing.get("identity_key") == identity_key:
            if raw_path.is_file():
                cache_hit = True
            elif existing.get("raw_audio_path") and Path(existing["raw_audio_path"]).is_file():
                raw_path = Path(existing["raw_audio_path"])

        if not cache_hit:
            try:
                tts_session.synthesize(sample.text, raw_path, segment={"text": sample.text})
            except Exception:
                evaluation = evaluate_calibration_sample(sample, wav_path=None, tts_failed=True)
                self._persist_sample(job_id, sample, evaluation, raw_path=None, identity_key=identity_key, cache_hit=False)
                return evaluation, False

        evaluation = evaluate_calibration_sample(sample, wav_path=raw_path, speed=1.0)
        self._persist_sample(job_id, sample, evaluation, raw_path=raw_path, identity_key=identity_key, cache_hit=cache_hit)
        return evaluation, cache_hit

    def _persist_sample(
        self,
        job_id: str,
        sample: CalibrationSample,
        evaluation: SampleEvaluation,
        *,
        raw_path: Path | None,
        identity_key: str,
        cache_hit: bool,
    ) -> None:
        audio_sha256 = None
        if raw_path and raw_path.is_file():
            digest = hashlib.sha256()
            with raw_path.open("rb") as handle:
                while chunk := handle.read(65536):
                    digest.update(chunk)
            audio_sha256 = digest.hexdigest()
        record = {
            "sample_id": sample.id,
            "status": "accepted" if evaluation.accepted else "rejected",
            "raw_audio_path": str(raw_path) if raw_path else None,
            "audio_sha256": audio_sha256,
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
            "rejection_reason": evaluation.rejection_reason,
            "identity_key": identity_key,
            "cache_hit": cache_hit,
            "category": sample.category,
        }
        upsert_sample_progress(self.data_dir, job_id, record)
        if evaluation.analysis:
            save_analysis(self.data_dir, job_id, sample.id, evaluation.analysis)

    def _sync_job_row(self, manifest: dict[str, Any]) -> None:
        with self.database.connection:
            self.database.connection.execute(
                """
                UPDATE voice_calibration_jobs
                SET status = ?, sample_completed = ?, sample_accepted = ?, sample_rejected = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    manifest.get("status"),
                    int(manifest.get("sample_completed") or 0),
                    int(manifest.get("sample_accepted") or 0),
                    int(manifest.get("sample_rejected") or 0),
                    utc_now(),
                    manifest.get("job_id"),
                ),
            )

    def _update_voice_profile_fields(self, voice_id: str, profile: dict[str, Any], status: str) -> None:
        with self.database.connection:
            self.database.connection.execute(
                """
                UPDATE cloned_voices
                SET duration_profile_status = ?,
                    duration_profile_key = ?,
                    duration_profile_quality = ?,
                    duration_profile_sample_count = ?,
                    last_calibrated_at = ?,
                    active_calibration_job_id = NULL,
                    voice_status = COALESCE(voice_status, 'ready')
                WHERE id = ?
                """,
                (
                    status,
                    profile.get("profile_key"),
                    profile.get("quality"),
                    int(profile.get("sample_count_accepted") or 0),
                    profile.get("updated_at"),
                    voice_id,
                ),
            )

    def _write_report(
        self,
        job_id: str,
        samples: list[CalibrationSample],
        evaluations: list[SampleEvaluation],
        aggregate: dict[str, Any],
        validation: dict[str, Any],
        identity: dict[str, Any],
        manifest: dict[str, Any],
        quality: str,
    ) -> None:
        sample_map = {sample.id: sample for sample in samples}
        per_sample = []
        rejection_counts: dict[str, int] = {}
        for evaluation in evaluations:
            sample = sample_map.get(evaluation.sample_id)
            if evaluation.rejection_reason:
                rejection_counts[evaluation.rejection_reason] = rejection_counts.get(evaluation.rejection_reason, 0) + 1
            per_sample.append(
                {
                    "sample_id": evaluation.sample_id,
                    "text": sample.text if sample else "",
                    "category": sample.category if sample else "",
                    "accepted": evaluation.accepted,
                    "reason": evaluation.rejection_reason,
                    "observed_sps": evaluation.observed_sps,
                    "speech_envelope_duration": evaluation.envelope.speech_duration if evaluation.envelope else None,
                    "syllables": evaluation.syllables,
                }
            )
        report = {
            "job_id": job_id,
            "voice_identity": identity,
            "dataset_version": manifest.get("dataset_version"),
            "mode": manifest.get("mode"),
            "quality": quality,
            "aggregate": aggregate,
            "validation": validation,
            "rejection_reasons": rejection_counts,
            "samples": per_sample,
            "telemetry": {
                "sample_total": manifest.get("sample_total"),
                "sample_synthesized": manifest.get("sample_synthesized"),
                "sample_cache_hits": manifest.get("sample_cache_hits"),
                "sample_accepted": manifest.get("sample_accepted"),
                "sample_rejected": manifest.get("sample_rejected"),
            },
        }
        save_report(self.data_dir, job_id, report)

    def delete_profile(self, voice_id: str) -> dict[str, Any]:
        row = self._get_cloned_voice(voice_id)
        profile_key = row.get("duration_profile_key")
        if profile_key:
            try:
                reset_voice_profile(self.data_dir, profile_key)
            except Exception:
                pass
        with self.database.connection:
            self.database.connection.execute(
                """
                UPDATE cloned_voices
                SET duration_profile_status = 'not_started',
                    duration_profile_key = NULL,
                    duration_profile_quality = NULL,
                    duration_profile_sample_count = 0,
                    last_calibrated_at = NULL,
                    active_calibration_job_id = NULL
                WHERE id = ?
                """,
                (voice_id,),
            )
        return {"status": "reset", "voice_id": voice_id}
