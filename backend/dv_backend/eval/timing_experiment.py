"""Timing A/B experiment orchestration: clone jobs and run baseline vs experiment."""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..checkpoints import PIPELINE_STEPS, checkpoint_path, load_checkpoint, save_checkpoint
from .experiment_comparability import source_artifact_fingerprint, validate_experiment_comparability
from ..jobs import JobService

EXPERIMENT_PREFIX_STEPS = tuple(PIPELINE_STEPS[: PIPELINE_STEPS.index("translate")])

ARTIFACT_NAMES = (
    "original.mp4",
    "original_48k.wav",
    "audio_16k.wav",
    "bgm.wav",
    "vocals.wav",
    "bgm_16k.wav",
    "vocals_16k.wav",
    "normalized.wav",
)

ARTIFACT_DIRS = ("asr_sparse",)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def experiment_dir(data_dir: Path, experiment_id: str) -> Path:
    return data_dir / "experiments" / experiment_id


def load_experiment_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Experiment config must be a JSON object.")
    return payload


def default_phase2_config() -> dict[str, Any]:
    return {
        "name": "phase2_default",
        "fixed_settings": {},
        "baseline_settings": {
            "timing_candidate_translation_enabled": False,
            "voice_duration_profile_enabled": False,
        },
        "experiment_settings": {
            "timing_candidate_translation_enabled": True,
            "timing_translation_candidate_count": 3,
            "timing_max_candidate_tts_attempts": 2,
            "timing_max_tts_attempts": 3,
            "voice_duration_profile_enabled": True,
        },
    }


def capture_settings(database, keys: tuple[str, ...] | None = None) -> dict[str, Any]:
    rows = database.connection.execute("SELECT key, value FROM settings").fetchall()
    settings = {row["key"]: json.loads(row["value"]) for row in rows}
    if keys is None:
        return settings
    return {key: settings[key] for key in keys if key in settings}


def apply_settings(database, updates: dict[str, Any]) -> None:
    from ..pipeline import save_setting

    for key, value in updates.items():
        save_setting(database, key, value)


def clone_job_prefix(
    job_service: JobService,
    source_job_id: str,
    *,
    label: str,
) -> str:
    source = job_service.get(source_job_id)
    norm_cp = load_checkpoint(job_service.data_dir, source_job_id, "normalize_segments")
    if not norm_cp or not norm_cp.get("segments"):
        raise ValueError(f"Source job {source_job_id} missing normalize_segments checkpoint.")

    new_job_id = str(uuid4())
    now = utc_now()
    src_dir = job_service.data_dir / "jobs" / source_job_id
    dst_dir = job_service.data_dir / "jobs" / new_job_id
    dst_artifacts = dst_dir / "artifacts"
    dst_artifacts.mkdir(parents=True, exist_ok=True)
    (dst_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (dst_dir / "output").mkdir(parents=True, exist_ok=True)

    for name in ARTIFACT_NAMES:
        src = src_dir / "artifacts" / name
        if src.is_file():
            shutil.copy2(src, dst_artifacts / name)

    for dirname in ARTIFACT_DIRS:
        src = src_dir / "artifacts" / dirname
        if src.is_dir():
            shutil.copytree(src, dst_artifacts / dirname, dirs_exist_ok=True)

    for step in EXPERIMENT_PREFIX_STEPS:
        cp = load_checkpoint(job_service.data_dir, source_job_id, step)
        if cp is None:
            raise ValueError(f"Source job missing checkpoint: {step}")
        cloned = dict(cp)
        cloned["job_id"] = new_job_id
        cloned["experiment_clone_of"] = source_job_id
        cloned["experiment_label"] = label
        save_checkpoint(job_service.data_dir, new_job_id, step, cloned)

    with job_service.database.connection:
        job_service.database.connection.execute(
            "INSERT INTO jobs (id, source_url, title, status, created_at, updated_at, current_step) "
            "VALUES (?, ?, ?, 'queued', ?, ?, ?)",
            (
                new_job_id,
                source.source_url,
                f"{source.title or 'Job'} [{label}]",
                now,
                now,
                "translate",
            ),
        )
        job_service.database.connection.executemany(
            "INSERT INTO job_steps (job_id, name, position, status, checkpoint_path) VALUES (?, ?, ?, ?, ?)",
            [
                (
                    new_job_id,
                    name,
                    position,
                    "completed" if name in EXPERIMENT_PREFIX_STEPS else "pending",
                    str(checkpoint_path(job_service.data_dir, new_job_id, name)),
                )
                for position, name in enumerate(PIPELINE_STEPS)
            ],
        )
        for step in EXPERIMENT_PREFIX_STEPS:
            job_service.database.connection.execute(
                "UPDATE job_steps SET completed_at = ? WHERE job_id = ? AND name = ?",
                (now, new_job_id, step),
            )

    meta = {
        "cloned_from": source_job_id,
        "label": label,
        "prefix_steps": list(EXPERIMENT_PREFIX_STEPS),
        "created_at": now,
    }
    (dst_dir / "artifacts" / "experiment_clone.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return new_job_id


def wait_for_job(job_service: JobService, job_id: str, *, timeout_sec: float = 6 * 3600, poll_sec: float = 5.0) -> str:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        job = job_service.get(job_id)
        if job.status in {"completed", "failed", "interrupted"}:
            return job.status
        time.sleep(poll_sec)
    raise TimeoutError(f"Timed out waiting for job {job_id}")


def run_experiment_arm(
    *,
    job_service: JobService,
    runner,
    job_id: str,
    settings_snapshot: dict[str, Any],
    database,
) -> str:
    apply_settings(database, settings_snapshot)
    job_service.prepare_job_for_resume(job_id)
    runner.start_job(job_id)
    status = wait_for_job(job_service, job_id)
    if status != "completed":
        job = job_service.get(job_id)
        raise RuntimeError(f"Job {job_id} ended with status={status} error={job.last_error_code}")
    return status


def build_manifest(
    *,
    experiment_id: str,
    source_job_id: str,
    baseline_job_id: str,
    experiment_job_id: str,
    data_dir: Path,
    config: dict[str, Any],
    fixed_settings: dict[str, Any],
    baseline_settings: dict[str, Any],
    experiment_settings: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "source_job_id": source_job_id,
        "baseline_job_id": baseline_job_id,
        "experiment_job_id": experiment_job_id,
        "source_artifact_fingerprint": source_artifact_fingerprint(data_dir, source_job_id),
        "created_at": utc_now(),
        "fixed_settings": fixed_settings,
        "baseline_settings": baseline_settings,
        "experiment_settings": experiment_settings,
        "config_name": config.get("name"),
        "status": status,
    }


def save_manifest(data_dir: Path, experiment_id: str, manifest: dict[str, Any]) -> Path:
    path = experiment_dir(data_dir, experiment_id) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_manifest(data_dir: Path, experiment_id: str) -> dict[str, Any] | None:
    path = experiment_dir(data_dir, experiment_id) / "manifest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def validate_fixed_settings_match(
    baseline_settings: dict[str, Any],
    experiment_settings: dict[str, Any],
    fixed_settings: dict[str, Any],
) -> list[str]:
    mismatches: list[str] = []
    for key, expected in fixed_settings.items():
        if baseline_settings.get(key) != expected or experiment_settings.get(key) != expected:
            mismatches.append(key)
    return mismatches
