from pathlib import Path
from unittest.mock import patch

from dv_backend.checkpoints import PIPELINE_STEPS
from dv_backend.config import AppConfig
from dv_backend.database import Database
from dv_backend.jobs import JobService
from dv_backend.runner import JobRunner


def test_successful_resume_clears_stale_errors(tmp_path: Path) -> None:
    config = AppConfig(tmp_path)
    config.ensure_directories()
    database = Database(config.database_path)
    database.migrate()
    job = JobService(database, tmp_path).create("https://www.douyin.com/video/123")

    with database.connection:
        database.connection.execute(
            """
            UPDATE jobs
            SET status = 'failed', last_error_code = 'OLD_ERROR',
                last_error_message = 'old failure'
            WHERE id = ?
            """,
            (job.id,),
        )
        database.connection.execute(
            "UPDATE job_steps SET status = 'completed' WHERE job_id = ?",
            (job.id,),
        )
        database.connection.execute(
            """
            UPDATE job_steps
            SET status = 'pending', error_code = 'OLD_ERROR',
                error_message = 'old failure'
            WHERE job_id = ? AND name = ?
            """,
            (job.id, PIPELINE_STEPS[-1]),
        )

    with patch("dv_backend.pipeline.qc_step", return_value={}):
        JobRunner(config, database)._run_job(job.id)

    job_row = database.connection.execute(
        "SELECT status, last_error_code, last_error_message FROM jobs WHERE id = ?",
        (job.id,),
    ).fetchone()
    step_row = database.connection.execute(
        "SELECT status, error_code, error_message FROM job_steps WHERE job_id = ? AND name = ?",
        (job.id, PIPELINE_STEPS[-1]),
    ).fetchone()

    assert dict(job_row) == {
        "status": "completed",
        "last_error_code": None,
        "last_error_message": None,
    }
    assert dict(step_row) == {
        "status": "completed",
        "error_code": None,
        "error_message": None,
    }
