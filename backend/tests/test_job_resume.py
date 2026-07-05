from pathlib import Path

from dv_backend.checkpoints import PIPELINE_STEPS
from dv_backend.config import AppConfig
from dv_backend.database import Database
from dv_backend.jobs import JobService


def test_reconcile_interrupted_resets_running_steps_to_pending(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    database.migrate()
    service = JobService(database, tmp_path)
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")
    job = service.create_imported(video, original_filename="sample.mp4")

    with database.connection:
        database.connection.execute(
            "UPDATE jobs SET status = 'running', current_step = 'tts' WHERE id = ?",
            (job.id,),
        )
        database.connection.execute(
            "UPDATE job_steps SET status = 'running', started_at = 'now' WHERE job_id = ? AND name = 'tts'",
            (job.id,),
        )

    resumed = service.reconcile_interrupted()
    assert resumed == [job.id]

    hydrated = service.get(job.id)
    assert hydrated.status == "interrupted"
    assert hydrated.last_error_code is None
    tts_step = next(step for step in hydrated.steps if step.name == "tts")
    assert tts_step.status == "pending"


def test_prepare_job_for_resume_clears_failed_steps(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    database.migrate()
    service = JobService(database, tmp_path)
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")
    job = service.create_imported(video, original_filename="sample.mp4")

    with database.connection:
        database.connection.execute(
            "UPDATE jobs SET status = 'interrupted', last_error_code = 'OLD', last_error_message = 'old' WHERE id = ?",
            (job.id,),
        )
        database.connection.execute(
            "UPDATE job_steps SET status = 'completed' WHERE job_id = ? AND name != 'tts'",
            (job.id,),
        )
        database.connection.execute(
            "UPDATE job_steps SET status = 'failed', error_code = 'APP_INTERRUPTED' WHERE job_id = ? AND name = 'tts'",
            (job.id,),
        )

    service.prepare_job_for_resume(job.id)
    hydrated = service.get(job.id)
    assert hydrated.status == "queued"
    assert hydrated.last_error_code is None
    completed = [step.name for step in hydrated.steps if step.status == "completed"]
    assert completed == [name for name in PIPELINE_STEPS if name != "tts"]
    tts_step = next(step for step in hydrated.steps if step.name == "tts")
    assert tts_step.status == "pending"
