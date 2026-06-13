from pathlib import Path

from dv_backend.checkpoints import PIPELINE_STEPS
from dv_backend.database import Database
from dv_backend.jobs import JobService


def test_create_job_creates_all_pipeline_steps(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    database.migrate()
    service = JobService(database, tmp_path)

    job = service.create("https://www.douyin.com/video/123")

    assert job.status == "queued"
    assert [step.name for step in job.steps] == list(PIPELINE_STEPS)
    assert all(step.status == "pending" for step in job.steps)


def test_reconcile_marks_running_job_interrupted(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    database.migrate()
    service = JobService(database, tmp_path)
    job = service.create("https://v.douyin.com/example/")
    database.connection.execute(
        "UPDATE jobs SET status = 'running', current_step = 'resolve' WHERE id = ?",
        (job.id,),
    )
    database.connection.commit()

    service.reconcile_interrupted()

    assert service.get(job.id).status == "interrupted"

