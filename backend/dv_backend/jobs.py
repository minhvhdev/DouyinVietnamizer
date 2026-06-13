from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from .checkpoints import PIPELINE_STEPS, checkpoint_path
from .database import Database
from .errors import AppError
from .models import ErrorInfo, Job, JobStep


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobService:
    def __init__(self, database: Database, data_dir: Path) -> None:
        self.database = database
        self.data_dir = data_dir

    def create(self, source_url: str) -> Job:
        host = (urlparse(source_url).hostname or "").lower()
        if host != "douyin.com" and not host.endswith(".douyin.com"):
            raise AppError(
                422,
                ErrorInfo(
                    code="INVALID_DOUYIN_URL",
                    message="The URL is not a recognized Douyin link.",
                    action="Paste a video, share, or channel URL from douyin.com.",
                ),
            )
        job_id = str(uuid4())
        now = utc_now()
        with self.database.connection:
            self.database.connection.execute(
                "INSERT INTO jobs (id, source_url, status, created_at, updated_at) VALUES (?, ?, 'queued', ?, ?)",
                (job_id, source_url, now, now),
            )
            self.database.connection.executemany(
                "INSERT INTO job_steps (job_id, name, position, status, checkpoint_path) VALUES (?, ?, ?, 'pending', ?)",
                [
                    (job_id, name, position, str(checkpoint_path(self.data_dir, job_id, name)))
                    for position, name in enumerate(PIPELINE_STEPS)
                ],
            )
        return self.get(job_id)

    def list(self) -> list[Job]:
        rows = self.database.connection.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC"
        ).fetchall()
        return [self._hydrate(row) for row in rows]

    def get(self, job_id: str) -> Job:
        row = self.database.connection.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise AppError(
                404,
                ErrorInfo(
                    code="JOB_NOT_FOUND",
                    message="The requested job does not exist.",
                    action="Return to the jobs dashboard and select an available job.",
                ),
            )
        return self._hydrate(row)

    def reconcile_interrupted(self) -> None:
        now = utc_now()
        with self.database.connection:
            self.database.connection.execute(
                "UPDATE jobs SET status = 'interrupted', updated_at = ? WHERE status = 'running'",
                (now,),
            )
            self.database.connection.execute(
                "UPDATE job_steps SET status = 'failed', error_code = 'APP_INTERRUPTED', "
                "error_message = 'The application closed while this step was running.' "
                "WHERE status = 'running'"
            )

    def _hydrate(self, row) -> Job:
        steps = self.database.connection.execute(
            "SELECT name, position, status, checkpoint_path FROM job_steps WHERE job_id = ? ORDER BY position",
            (row["id"],),
        ).fetchall()
        return Job(
            **dict(row),
            steps=[JobStep(**dict(step)) for step in steps],
        )

