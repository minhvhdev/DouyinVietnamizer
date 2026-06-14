import threading
import time
from datetime import datetime, timezone
import traceback

from .config import AppConfig
from .database import Database
from .errors import AppError
from .checkpoints import PIPELINE_STEPS
from . import pipeline


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobRunner:
    def __init__(self, config: AppConfig, database: Database) -> None:
        self.config = config
        self.database = database
        self.running_processes = {}  # job_id -> subprocess.Popen
        self.cancelled_jobs = set()
        self.threads = {}  # job_id -> Thread
        self.lock = threading.Lock()

    def register_process(self, job_id: str, proc) -> None:
        with self.lock:
            self.running_processes[job_id] = proc

    def unregister_process(self, job_id: str) -> None:
        with self.lock:
            self.running_processes.pop(job_id, None)

    def is_cancelled(self, job_id: str) -> bool:
        with self.lock:
            return job_id in self.cancelled_jobs

    def cancel_job(self, job_id: str) -> None:
        with self.lock:
            self.cancelled_jobs.add(job_id)
            proc = self.running_processes.get(job_id)
            if proc:
                try:
                    proc.kill()
                except Exception:
                    pass
        now = utc_now()
        with self.database.connection:
            self.database.connection.execute(
                "UPDATE jobs SET status = 'failed', last_error_code = 'CANCELLED', last_error_message = 'Job cancelled by user.', updated_at = ? WHERE id = ?",
                (now, job_id)
            )
            self.database.connection.execute(
                "UPDATE job_steps SET status = 'failed', error_code = 'CANCELLED', error_message = 'Cancelled by user.' WHERE job_id = ? AND status = 'running'",
                (job_id,)
            )

    def start_job(self, job_id: str) -> None:
        with self.lock:
            if job_id in self.cancelled_jobs:
                self.cancelled_jobs.remove(job_id)

            if job_id in self.threads and self.threads[job_id].is_alive():
                return

            thread = threading.Thread(target=self._run_job, args=(job_id,), name=f"job-{job_id}")
            self.threads[job_id] = thread
            thread.start()

    def _run_job(self, job_id: str) -> None:
        now = utc_now()
        with self.database.connection:
            self.database.connection.execute(
                "UPDATE jobs SET status = 'running', updated_at = ? WHERE id = ?",
                (now, job_id)
            )

        for step_name in PIPELINE_STEPS:
            if self.is_cancelled(job_id):
                break

            step = self.database.connection.execute(
                "SELECT status FROM job_steps WHERE job_id = ? AND name = ?",
                (job_id, step_name)
            ).fetchone()

            if not step:
                break

            if step["status"] == "completed":
                continue

            now_start = utc_now()
            with self.database.connection:
                self.database.connection.execute(
                    """
                    UPDATE job_steps
                    SET status = 'running', started_at = ?, completed_at = NULL,
                        error_code = NULL, error_message = NULL
                    WHERE job_id = ? AND name = ?
                    """,
                    (now_start, job_id, step_name)
                )
                self.database.connection.execute(
                    "UPDATE jobs SET current_step = ?, updated_at = ? WHERE id = ?",
                    (step_name, now_start, job_id)
                )

            try:
                func = getattr(pipeline, f"{step_name}_step")
                func(job_id, self.config, self.database, self)

                now_end = utc_now()
                with self.database.connection:
                    self.database.connection.execute(
                        """
                        UPDATE job_steps
                        SET status = 'completed', completed_at = ?,
                            error_code = NULL, error_message = NULL
                        WHERE job_id = ? AND name = ?
                        """,
                        (now_end, job_id, step_name)
                    )
            except AppError as e:
                if e.info.code == "NO_VIDEO_SELECTED":
                    now_end = utc_now()
                    with self.database.connection:
                        self.database.connection.execute(
                            "UPDATE job_steps SET status = 'pending', started_at = NULL WHERE job_id = ? AND name = ?",
                            (job_id, step_name)
                        )
                        self.database.connection.execute(
                            "UPDATE jobs SET status = 'waiting_for_selection', updated_at = ? WHERE id = ?",
                            (now_end, job_id)
                        )
                    return

                now_end = utc_now()
                with self.database.connection:
                    self.database.connection.execute(
                        "UPDATE job_steps SET status = 'failed', error_code = ?, error_message = ?, completed_at = ? WHERE job_id = ? AND name = ?",
                        (e.info.code, e.info.message, now_end, job_id, step_name)
                    )
                    self.database.connection.execute(
                        "UPDATE jobs SET status = 'failed', last_error_code = ?, last_error_message = ?, updated_at = ? WHERE id = ?",
                        (e.info.code, e.info.message, now_end, job_id)
                    )
                    self.database.connection.execute(
                        "INSERT INTO events (level, code, message, job_id, created_at) VALUES ('error', ?, ?, ?, ?)",
                        (e.info.code, f"Step '{step_name}' failed: {e.info.message}", job_id, now_end)
                    )
                return
            except Exception as e:
                traceback.print_exc()
                now_end = utc_now()
                err_msg = str(e)
                with self.database.connection:
                    self.database.connection.execute(
                        "UPDATE job_steps SET status = 'failed', error_code = 'UNEXPECTED_ERROR', error_message = ?, completed_at = ? WHERE job_id = ? AND name = ?",
                        (err_msg, now_end, job_id, step_name)
                    )
                    self.database.connection.execute(
                        "UPDATE jobs SET status = 'failed', last_error_code = 'UNEXPECTED_ERROR', last_error_message = ?, updated_at = ? WHERE id = ?",
                        (err_msg, now_end, job_id)
                    )
                    self.database.connection.execute(
                        "INSERT INTO events (level, code, message, job_id, created_at) VALUES ('error', 'UNEXPECTED_ERROR', ?, ?, ?)",
                        (f"Step '{step_name}' crashed: {err_msg}", job_id, now_end)
                    )
                return

        if not self.is_cancelled(job_id):
            now_end = utc_now()
            with self.database.connection:
                self.database.connection.execute(
                    """
                    UPDATE jobs
                    SET status = 'completed', current_step = NULL,
                        last_error_code = NULL, last_error_message = NULL,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (now_end, job_id)
                )
                self.database.connection.execute(
                    "INSERT INTO events (level, code, message, job_id, created_at) VALUES ('info', 'JOB_COMPLETED', 'Job completed successfully.', ?, ?)",
                    (job_id, now_end)
                )
