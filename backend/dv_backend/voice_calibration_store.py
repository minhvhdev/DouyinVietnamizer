"""Persistent storage for voice calibration jobs."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from .voice_calibration_dataset import DATASET_VERSION


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def calibration_root(data_dir: Path) -> Path:
    return data_dir / "voice_calibrations"


def job_dir(data_dir: Path, job_id: str) -> Path:
    return calibration_root(data_dir) / job_id


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        try:
            os.replace(tmp_name, path)
        except PermissionError:
            if path.exists():
                path.unlink(missing_ok=True)
            os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def create_job_manifest(
    *,
    data_dir: Path,
    voice_id: str,
    voice_identity_key: str,
    mode: str,
    dataset_version: str = DATASET_VERSION,
    sample_total: int = 0,
) -> dict[str, Any]:
    job_id = str(uuid4())
    now = utc_now()
    manifest = {
        "job_type": "voice_calibration",
        "job_id": job_id,
        "voice_id": voice_id,
        "voice_identity_key": voice_identity_key,
        "language": "vi",
        "dataset_version": dataset_version,
        "mode": mode,
        "status": "queued",
        "sample_total": sample_total,
        "sample_completed": 0,
        "sample_accepted": 0,
        "sample_rejected": 0,
        "sample_synthesized": 0,
        "sample_cache_hits": 0,
        "created_at": now,
        "updated_at": now,
    }
    root = job_dir(data_dir, job_id)
    root.mkdir(parents=True, exist_ok=True)
    (root / "raw").mkdir(exist_ok=True)
    (root / "analysis").mkdir(exist_ok=True)
    (root / "checkpoints").mkdir(exist_ok=True)
    _atomic_write(root / "manifest.json", manifest)
    _atomic_write(root / "checkpoints" / "progress.json", {"samples": [], "updated_at": now})
    return manifest


def load_manifest(data_dir: Path, job_id: str) -> dict[str, Any]:
    path = job_dir(data_dir, job_id) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(job_id)
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(data_dir: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = utc_now()
    _atomic_write(job_dir(data_dir, manifest["job_id"]) / "manifest.json", manifest)


def load_progress(data_dir: Path, job_id: str) -> dict[str, Any]:
    path = job_dir(data_dir, job_id) / "checkpoints" / "progress.json"
    if not path.is_file():
        return {"samples": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"samples": [], "recovered_from_corruption": True}
    payload.setdefault("samples", [])
    return payload


def save_progress(data_dir: Path, job_id: str, progress: dict[str, Any]) -> None:
    progress["updated_at"] = utc_now()
    _atomic_write(job_dir(data_dir, job_id) / "checkpoints" / "progress.json", progress)


def upsert_sample_progress(
    data_dir: Path,
    job_id: str,
    sample_record: dict[str, Any],
) -> None:
    progress = load_progress(data_dir, job_id)
    samples = progress.setdefault("samples", [])
    replaced = False
    for index, existing in enumerate(samples):
        if existing.get("sample_id") == sample_record.get("sample_id"):
            samples[index] = sample_record
            replaced = True
            break
    if not replaced:
        samples.append(sample_record)
    save_progress(data_dir, job_id, progress)


def sample_record_by_id(progress: dict[str, Any], sample_id: str) -> dict[str, Any] | None:
    for item in progress.get("samples") or []:
        if item.get("sample_id") == sample_id:
            return item
    return None


def save_analysis(data_dir: Path, job_id: str, sample_id: str, analysis: dict[str, Any]) -> Path:
    path = job_dir(data_dir, job_id) / "analysis" / f"{sample_id}.json"
    _atomic_write(path, analysis)
    return path


def save_report(data_dir: Path, job_id: str, report: dict[str, Any]) -> Path:
    path = job_dir(data_dir, job_id) / "report.json"
    _atomic_write(path, report)
    return path
