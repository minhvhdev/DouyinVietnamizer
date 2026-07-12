from pathlib import Path

import threading
from unittest.mock import patch



from dv_backend.checkpoints import PIPELINE_STEPS

from dv_backend.config import AppConfig

from dv_backend.database import Database

from dv_backend.jobs import JobService

from dv_backend.runner import JobRunner





def _create_job(service: JobService, data_dir: Path):

    video = data_dir / "sample.mp4"

    video.write_bytes(b"fake-video")

    return service.create_imported(video, original_filename="sample.mp4")





def test_runner_records_step_duration_ms(tmp_path: Path) -> None:

    config = AppConfig(tmp_path)

    config.ensure_directories()

    database = Database(config.database_path)

    database.migrate()

    job = _create_job(JobService(database, tmp_path), tmp_path)



    with database.connection:

        database.connection.execute(

            "UPDATE job_steps SET status = 'completed' WHERE job_id = ?",

            (job.id,),

        )

        database.connection.execute(

            "UPDATE job_steps SET status = 'pending' WHERE job_id = ? AND name = ?",

            (job.id, PIPELINE_STEPS[-1]),

        )



    with patch("dv_backend.pipeline.qc_step", return_value={}):

        JobRunner(config, database)._run_job(job.id)



    step_row = database.connection.execute(

        "SELECT duration_ms FROM job_steps WHERE job_id = ? AND name = ?",

        (job.id, PIPELINE_STEPS[-1]),

    ).fetchone()

    hydrated_step = next(step for step in JobService(database, tmp_path).get(job.id).steps if step.name == PIPELINE_STEPS[-1])



    assert step_row["duration_ms"] is not None

    assert step_row["duration_ms"] >= 0

    assert hydrated_step.duration_ms == step_row["duration_ms"]





def test_runner_clears_previous_failure_on_success(tmp_path: Path) -> None:

    config = AppConfig(tmp_path)

    config.ensure_directories()

    database = Database(config.database_path)

    database.migrate()

    job = _create_job(JobService(database, tmp_path), tmp_path)



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


def test_start_job_queues_when_another_is_active(tmp_path: Path) -> None:
    config = AppConfig(tmp_path)
    config.ensure_directories()
    database = Database(config.database_path)
    database.migrate()
    service = JobService(database, tmp_path)
    job1 = _create_job(service, tmp_path)
    job2 = _create_job(service, tmp_path)
    runner = JobRunner(config, database)

    started = threading.Event()
    release = threading.Event()

    def slow_execute(job_id: str) -> None:
        started.set()
        release.wait(timeout=2)

    with patch.object(runner, "_execute_job", side_effect=slow_execute):
        runner.start_job(job1.id)
        assert started.wait(timeout=1)
        runner.start_job(job2.id)
        assert runner.pending_job_ids == [job2.id]
        release.set()
        runner.threads[job1.id].join(timeout=2)
        runner.threads[job2.id].join(timeout=2)

    assert runner.active_job_id is None
    assert runner.pending_job_ids == []


def test_start_job_after_cancel_queues_until_thread_finishes(tmp_path: Path) -> None:
    config = AppConfig(tmp_path)
    config.ensure_directories()
    database = Database(config.database_path)
    database.migrate()
    service = JobService(database, tmp_path)
    job = _create_job(service, tmp_path)
    runner = JobRunner(config, database)

    execute_count = {"value": 0}
    first_run_started = threading.Event()
    unblock_first_run = threading.Event()
    second_run_started = threading.Event()

    def tracked_execute(job_id: str) -> None:
        execute_count["value"] += 1
        if execute_count["value"] == 1:
            first_run_started.set()
            unblock_first_run.wait(timeout=2)
            return
        second_run_started.set()
        with database.connection:
            database.connection.execute(
                "UPDATE jobs SET status = 'completed', current_step = NULL WHERE id = ?",
                (job_id,),
            )

    with patch.object(runner, "_execute_job", side_effect=tracked_execute):
        runner.start_job(job.id)
        assert first_run_started.wait(timeout=1)

        runner.cancel_job(job.id)
        assert service.get(job.id).status == "interrupted"
        service.prepare_job_for_resume(job.id)
        runner.start_job(job.id)
        assert runner.pending_job_ids == [job.id]

        unblock_first_run.set()
        runner.threads[job.id].join(timeout=2)

    assert second_run_started.wait(timeout=1)
    runner.threads[job.id].join(timeout=2)
    assert execute_count["value"] == 2
    assert runner.pending_job_ids == []
    assert runner.active_job_id is None

