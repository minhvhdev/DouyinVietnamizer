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


def elapsed_ms_since(started_at: str | None) -> int | None:
    if not started_at:
        return None
    try:
        started = datetime.fromisoformat(started_at)
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        return max(0, round((datetime.now(timezone.utc) - started).total_seconds() * 1000))
    except ValueError:
        return None


class JobRunner:
    def __init__(self, config: AppConfig, database: Database) -> None:
        self.config = config
        self.database = database
        self.running_processes = {}  # job_id -> subprocess.Popen
        self.cancelled_jobs = set()
        self.threads = {}  # job_id -> Thread
        self.active_job_id: str | None = None
        self.pending_job_ids: list[str] = []
        self.lock = threading.Lock()

    def register_process(self, job_id: str, proc) -> None:
        with self.lock:
            self.running_processes[job_id] = proc

    def unregister_process(self, job_id: str) -> None:
        with self.lock:
            self.running_processes.pop(job_id, None)

    def kill_managed_processes(self) -> list[str]:
        with self.lock:
            processes = list(self.running_processes.items())
            self.running_processes.clear()
        killed: list[str] = []
        for job_id, proc in processes:
            try:
                proc.kill()
                killed.append(f"{job_id}:{getattr(proc, 'pid', 'unknown')}")
            except Exception:
                pass
        return killed

    def is_cancelled(self, job_id: str) -> bool:
        with self.lock:
            return job_id in self.cancelled_jobs

    def cancel_job(self, job_id: str) -> None:
        with self.lock:
            self.cancelled_jobs.add(job_id)
            self.pending_job_ids = [queued_id for queued_id in self.pending_job_ids if queued_id != job_id]
            proc = self.running_processes.get(job_id)
            if proc:
                try:
                    proc.kill()
                except Exception:
                    pass
        now = utc_now()
        with self.database.connection:
            self.database.connection.execute(
                """
                UPDATE jobs
                SET status = 'interrupted',
                    last_error_code = NULL,
                    last_error_message = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, job_id),
            )
            self.database.connection.execute(
                """
                UPDATE job_steps
                SET status = 'pending',
                    started_at = NULL,
                    completed_at = NULL,
                    duration_ms = NULL,
                    error_code = NULL,
                    error_message = NULL
                WHERE job_id = ? AND status = 'running'
                """,
                (job_id,),
            )

    def start_job(self, job_id: str) -> None:
        with self.lock:
            if job_id in self.cancelled_jobs:
                self.cancelled_jobs.remove(job_id)

            existing_thread = self.threads.get(job_id)
            if existing_thread is not None and existing_thread.is_alive():
                row = self.database.connection.execute(
                    "SELECT status FROM jobs WHERE id = ?",
                    (job_id,),
                ).fetchone()
                if row and row["status"] == "running":
                    return
                if job_id not in self.pending_job_ids:
                    self.pending_job_ids.append(job_id)
                return

            if job_id in self.pending_job_ids:
                return

            if self.active_job_id is not None:
                self.pending_job_ids.append(job_id)
                return

            self._spawn_job_thread_locked(job_id)

    def _spawn_job_thread_locked(self, job_id: str) -> None:
        self.active_job_id = job_id
        thread = threading.Thread(target=self._run_job, args=(job_id,), name=f"job-{job_id}")
        self.threads[job_id] = thread
        thread.start()

    def _release_and_start_next(self, job_id: str) -> None:
        with self.lock:
            if self.active_job_id == job_id:
                self.active_job_id = None
            while self.pending_job_ids:
                next_job_id = self.pending_job_ids.pop(0)
                if next_job_id in self.cancelled_jobs:
                    continue
                row = self.database.connection.execute(
                    "SELECT status FROM jobs WHERE id = ?",
                    (next_job_id,),
                ).fetchone()
                if row and row["status"] in ("queued", "interrupted"):
                    self._spawn_job_thread_locked(next_job_id)
                    return

    def _run_job(self, job_id: str) -> None:
        try:
            self._execute_job(job_id)
        finally:
            self._release_and_start_next(job_id)

    def _execute_job(self, job_id: str) -> None:
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
            step_started = time.perf_counter()
            with self.database.connection:
                self.database.connection.execute(
                    """
                    UPDATE job_steps
                    SET status = 'running', started_at = ?, completed_at = NULL,
                        duration_ms = NULL, error_code = NULL, error_message = NULL
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
                duration_ms = round((time.perf_counter() - step_started) * 1000)
                with self.database.connection:
                    self.database.connection.execute(
                        """
                        UPDATE job_steps
                        SET status = 'completed', completed_at = ?, duration_ms = ?,
                            error_code = NULL, error_message = NULL
                        WHERE job_id = ? AND name = ?
                        """,
                        (now_end, duration_ms, job_id, step_name)
                    )
            except AppError as e:
                if e.info.code == "NO_VIDEO_SELECTED":
                    now_end = utc_now()
                    with self.database.connection:
                        self.database.connection.execute(
                            "UPDATE job_steps SET status = 'pending', started_at = NULL, completed_at = NULL, duration_ms = NULL WHERE job_id = ? AND name = ?",
                            (job_id, step_name),
                        )
                        self.database.connection.execute(
                            "UPDATE jobs SET status = 'waiting_for_selection', updated_at = ? WHERE id = ?",
                            (now_end, job_id),
                        )
                    return

                if e.info.code == "JOB_CANCELLED" or self.is_cancelled(job_id):
                    return

                now_end = utc_now()
                duration_ms = round((time.perf_counter() - step_started) * 1000)
                with self.database.connection:
                    self.database.connection.execute(
                        "UPDATE job_steps SET status = 'failed', error_code = ?, error_message = ?, completed_at = ?, duration_ms = ? WHERE job_id = ? AND name = ?",
                        (e.info.code, e.info.message, now_end, duration_ms, job_id, step_name)
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
                if self.is_cancelled(job_id):
                    return
                traceback.print_exc()
                now_end = utc_now()
                duration_ms = round((time.perf_counter() - step_started) * 1000)
                err_msg = str(e)
                with self.database.connection:
                    self.database.connection.execute(
                        "UPDATE job_steps SET status = 'failed', error_code = 'UNEXPECTED_ERROR', error_message = ?, completed_at = ?, duration_ms = ? WHERE job_id = ? AND name = ?",
                        (err_msg, now_end, duration_ms, job_id, step_name)
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
            row = self.database.connection.execute(
                "SELECT status FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if not row or row["status"] != "running":
                return
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
