import json
from pathlib import Path

import pytest

from dv_backend.checkpoints import PIPELINE_STEPS, checkpoint_path, save_checkpoint
from dv_backend.config import AppConfig
from dv_backend.database import Database
from dv_backend.errors import AppError
from dv_backend.jobs import JobService


@pytest.fixture
def job_service(tmp_path: Path) -> JobService:
    database = Database(tmp_path / "app.db")
    database.migrate()
    return JobService(database, tmp_path)


def _create_completed_job(service: JobService, job_id: str = "job-rerun") -> None:
    config = AppConfig(service.data_dir)
    config.ensure_directories()
    now = "2026-01-01T00:00:00+00:00"
    with service.database.connection:
        service.database.connection.execute(
            "INSERT INTO jobs (id, source_url, status, created_at, updated_at) VALUES (?, ?, 'completed', ?, ?)",
            (job_id, "https://www.bilibili.com/video/BV123", now, now),
        )
        service.database.connection.executemany(
            "INSERT INTO job_steps (job_id, name, position, status, checkpoint_path) VALUES (?, ?, ?, 'completed', ?)",
            [
                (
                    job_id,
                    name,
                    position,
                    str(checkpoint_path(service.data_dir, job_id, name)),
                )
                for position, name in enumerate(PIPELINE_STEPS)
            ],
        )

    for step_name in PIPELINE_STEPS:
        save_checkpoint(
            service.data_dir,
            job_id,
            step_name,
            {"step_name": step_name, "job_id": job_id},
        )


def test_rerun_resets_selected_suffix_steps(job_service: JobService) -> None:
    job_id = "job-rerun"
    _create_completed_job(job_service, job_id)

    keep_steps = list(PIPELINE_STEPS[: PIPELINE_STEPS.index("tts")])
    job = job_service.rerun(job_id, keep_steps)

    assert job.status == "queued"
    for step_name in keep_steps:
        step = next(item for item in job.steps if item.name == step_name)
        assert step.status == "completed"
        assert checkpoint_path(job_service.data_dir, job_id, step_name).is_file()

    for step_name in PIPELINE_STEPS[PIPELINE_STEPS.index("tts") :]:
        step = next(item for item in job.steps if item.name == step_name)
        assert step.status == "pending"
        assert not checkpoint_path(job_service.data_dir, job_id, step_name).is_file()


def test_rerun_rejects_non_prefix_keep_steps(job_service: JobService) -> None:
    job_id = "job-prefix"
    _create_completed_job(job_service, job_id)

    with pytest.raises(AppError) as exc:
        job_service.rerun(job_id, ["translate"])

    assert exc.value.info.code == "INVALID_RERUN_KEEP_PREFIX"


def test_rerun_rejects_when_all_steps_kept(job_service: JobService) -> None:
    job_id = "job-all-kept"
    _create_completed_job(job_service, job_id)

    with pytest.raises(AppError) as exc:
        job_service.rerun(job_id, list(PIPELINE_STEPS))

    assert exc.value.info.code == "RERUN_NOTHING_TO_RESET"


def test_redub_keeps_through_translate(job_service: JobService) -> None:
    job_id = "job-redub"
    _create_completed_job(job_service, job_id)

    job = job_service.redub(job_id)

    translate_index = PIPELINE_STEPS.index("translate")
    for step_name in PIPELINE_STEPS[: translate_index + 1]:
        step = next(item for item in job.steps if item.name == step_name)
        assert step.status == "completed"

    for step_name in PIPELINE_STEPS[translate_index + 1 :]:
        step = next(item for item in job.steps if item.name == step_name)
        assert step.status == "pending"


def test_rerun_from_tts_clears_tts_artifacts(job_service: JobService) -> None:
    job_id = "job-tts-artifacts"
    _create_completed_job(job_service, job_id)
    tts_dir = job_service.data_dir / "jobs" / job_id / "artifacts" / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)
    stale = tts_dir / "tts_0.wav"
    stale.write_bytes(b"RIFF" + b"\x00" * 40)

    keep_steps = list(PIPELINE_STEPS[: PIPELINE_STEPS.index("tts")])
    job_service.rerun(job_id, keep_steps)

    assert not stale.is_file()
    assert not tts_dir.is_dir()
