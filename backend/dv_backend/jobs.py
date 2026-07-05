from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
from urllib.parse import urlparse
from uuid import uuid4

from .checkpoints import PIPELINE_STEPS, checkpoint_path
from .database import Database
from .errors import AppError
from .models import ErrorInfo, Job, JobStep
from .source_urls import is_supported_source_host, normalize_source_url


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobService:
    SUPPORTED_IMPORT_EXTENSIONS = (
        ".mp4", ".mov", ".m4v", ".mkv", ".webm", ".flv", ".avi", ".ts",
        ".mp3", ".wav", ".m4a", ".ogg", ".opus",
    )
    IMPORTED_SOURCE_PREFIX = "import://"
    SKIP_STEPS_FOR_IMPORT = ("resolve", "download")

    def __init__(self, database: Database, data_dir: Path) -> None:
        self.database = database
        self.data_dir = data_dir
        self._sync_pipeline_steps()

    def _sync_pipeline_steps(self) -> None:
        """Keep persisted jobs aligned with the current pipeline definition."""
        valid_steps = set(PIPELINE_STEPS)
        job_rows = self.database.connection.execute("SELECT id FROM jobs").fetchall()
        with self.database.connection:
            self.database.connection.execute(
                "DELETE FROM job_steps WHERE name NOT IN (%s)"
                % ",".join("?" for _ in PIPELINE_STEPS),
                tuple(PIPELINE_STEPS),
            )
            for job_row in job_rows:
                job_id = job_row["id"]
                existing = {
                    row["name"]
                    for row in self.database.connection.execute(
                        "SELECT name FROM job_steps WHERE job_id = ?", (job_id,)
                    ).fetchall()
                    if row["name"] in valid_steps
                }
                for position, step_name in enumerate(PIPELINE_STEPS):
                    if step_name in existing:
                        self.database.connection.execute(
                            "UPDATE job_steps SET position = ? WHERE job_id = ? AND name = ?",
                            (position, job_id, step_name),
                        )
                    else:
                        self.database.connection.execute(
                            """
                            INSERT INTO job_steps (job_id, name, position, status, checkpoint_path)
                            VALUES (?, ?, ?, 'pending', ?)
                            """,
                            (
                                job_id,
                                step_name,
                                position,
                                str(checkpoint_path(self.data_dir, job_id, step_name)),
                            ),
                        )

    def create(self, source_url: str) -> Job:
        normalized = normalize_source_url(source_url)
        host = (urlparse(normalized).hostname or "").lower()
        if not is_supported_source_host(host):
            raise AppError(
                422,
                ErrorInfo(
                    code="INVALID_SOURCE_URL",
                    message="Liên kết không thuộc Douyin hoặc Bilibili.",
                    action="Dán URL video hoặc link chia sẻ từ douyin.com hoặc bilibili.com.",
                ),
            )
        job_id = str(uuid4())
        now = utc_now()
        with self.database.connection:
            self.database.connection.execute(
                "INSERT INTO jobs (id, source_url, status, created_at, updated_at, current_step) "
                "VALUES (?, ?, 'queued', ?, ?, ?)",
                (job_id, normalized, now, now, PIPELINE_STEPS[0]),
            )
            self.database.connection.executemany(
                "INSERT INTO job_steps (job_id, name, position, status, checkpoint_path) VALUES (?, ?, ?, 'pending', ?)",
                [
                    (job_id, name, position, str(checkpoint_path(self.data_dir, job_id, name)))
                    for position, name in enumerate(PIPELINE_STEPS)
                ],
            )
        return self.get(job_id)

    def create_imported(
        self,
        file_path: Path,
        *,
        original_filename: str,
        title: str | None = None,
    ) -> Job:
        """Create a job from a local video file.

        The uploaded file is copied into the job artifacts directory
        as original.mp4 and the pipeline starts at extract_audio.
        """
        if not file_path.is_file():
            raise AppError(
                400,
                ErrorInfo(
                    code="IMPORT_FILE_MISSING",
                    message="The uploaded file is missing or could not be read.",
                    action="Pick a valid video file and try again.",
                ),
            )

        suffix = file_path.suffix.lower()
        if suffix not in self.SUPPORTED_IMPORT_EXTENSIONS:
            raise AppError(
                415,
                ErrorInfo(
                    code="IMPORT_UNSUPPORTED_FORMAT",
                    message=f"Unsupported file format: {suffix or '(none)'}",
                    action=f"Use one of: {', '.join(self.SUPPORTED_IMPORT_EXTENSIONS)}",
                ),
            )

        job_id = str(uuid4())
        now = utc_now()
        job_dir = self.data_dir / 'jobs' / job_id
        artifacts_dir = job_dir / 'artifacts'
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        stored_path = artifacts_dir / 'original.mp4'
        try:
            shutil.copy2(file_path, stored_path)
        except OSError as exc:
            raise AppError(
                500,
                ErrorInfo(
                    code="IMPORT_FILE_SAVE_FAILED",
                    message="Failed to save the imported file to the job workspace.",
                    action="Check disk space and write permissions, then try again.",
                    detail=str(exc),
                ),
            )

        safe_filename = Path(original_filename or 'imported').name or 'imported'
        source_url = f'{self.IMPORTED_SOURCE_PREFIX}{safe_filename}'
        resolved_title = (title or '').strip() or Path(safe_filename).stem

        skip_set = set(self.SKIP_STEPS_FOR_IMPORT)
        first_active_idx = 0
        for index, name in enumerate(PIPELINE_STEPS):
            if name not in skip_set:
                first_active_idx = index
                break

        with self.database.connection:
            self.database.connection.execute(
                "INSERT INTO jobs (id, source_url, title, status, created_at, updated_at, current_step) "
                "VALUES (?, ?, ?, 'queued', ?, ?, ?)",
                (job_id, source_url, resolved_title, now, now, PIPELINE_STEPS[first_active_idx]),
            )
            self.database.connection.executemany(
                "INSERT INTO job_steps (job_id, name, position, status, checkpoint_path) "
                "VALUES (?, ?, ?, 'pending', ?)",
                [
                    (job_id, name, position, str(checkpoint_path(self.data_dir, job_id, name)))
                    for position, name in enumerate(PIPELINE_STEPS)
                ],
            )
            for step_name in self.SKIP_STEPS_FOR_IMPORT:
                self.database.connection.execute(
                    "UPDATE job_steps SET status = 'completed', completed_at = ? "
                    "WHERE job_id = ? AND name = ?",
                    (now, job_id, step_name),
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

    def reconcile_interrupted(self) -> list[str]:
        """Mark jobs left in `running` as `interrupted` after a backend restart.

        Returns the affected job ids so callers can surface or inspect them.
        """
        now = utc_now()
        running_rows = self.database.connection.execute(
            "SELECT id FROM jobs WHERE status = 'running' ORDER BY updated_at DESC"
        ).fetchall()
        job_ids = [row["id"] for row in running_rows]
        if not job_ids:
            return []

        placeholders = ",".join("?" for _ in job_ids)
        with self.database.connection:
            self.database.connection.execute(
                f"""
                UPDATE jobs
                SET status = 'interrupted',
                    last_error_code = NULL,
                    last_error_message = NULL,
                    updated_at = ?
                WHERE id IN ({placeholders})
                """,
                (now, *job_ids),
            )
            self.database.connection.execute(
                f"""
                UPDATE job_steps
                SET status = 'pending',
                    started_at = NULL,
                    completed_at = NULL,
                    duration_ms = NULL,
                    error_code = NULL,
                    error_message = NULL
                WHERE status = 'running' AND job_id IN ({placeholders})
                """,
                tuple(job_ids),
            )
        return job_ids

    def prepare_job_for_resume(self, job_id: str) -> None:
        now = utc_now()
        with self.database.connection:
            self.database.connection.execute(
                """
                UPDATE jobs
                SET status = 'queued',
                    last_error_code = NULL,
                    last_error_message = NULL,
                    updated_at = ?
                WHERE id = ? AND status IN ('interrupted', 'failed')
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
                WHERE job_id = ? AND status = 'failed'
                """,
                (job_id,),
            )

    def latest_interrupted_job_id(self) -> str | None:
        row = self.database.connection.execute(
            "SELECT id FROM jobs WHERE status = 'interrupted' ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        return row["id"] if row else None

    def rerun(self, job_id: str, keep_steps: list[str]) -> Job:
        job = self.get(job_id)
        if job.status not in {"completed", "failed", "interrupted"}:
            raise AppError(
                409,
                ErrorInfo(
                    code="JOB_NOT_RERUNNABLE",
                    message="Only completed, failed, or stopped jobs can be rerun.",
                    action="Cancel the running job first.",
                ),
            )

        keep_set = set(keep_steps)
        unknown = keep_set.difference(PIPELINE_STEPS)
        if unknown:
            raise AppError(
                422,
                ErrorInfo(
                    code="INVALID_RERUN_STEPS",
                    message="One or more keep_steps values are not valid pipeline steps.",
                    action="Use step names from the pipeline in order.",
                    detail=", ".join(sorted(unknown)),
                ),
            )

        if keep_set:
            max_kept_index = max(PIPELINE_STEPS.index(step_name) for step_name in keep_set)
            expected_prefix = set(PIPELINE_STEPS[: max_kept_index + 1])
            if keep_set != expected_prefix:
                raise AppError(
                    422,
                    ErrorInfo(
                        code="INVALID_RERUN_KEEP_PREFIX",
                        message="Kept steps must form a continuous prefix from the start of the pipeline.",
                        action="Keep earlier steps before keeping later ones.",
                    ),
                )

        first_reset_idx: int | None = None
        for index, step_name in enumerate(PIPELINE_STEPS):
            if step_name not in keep_set:
                first_reset_idx = index
                break

        if first_reset_idx is None:
            raise AppError(
                422,
                ErrorInfo(
                    code="RERUN_NOTHING_TO_RESET",
                    message="At least one pipeline step must be rerun.",
                    action="Uncheck the steps you want to execute again.",
                ),
            )

        reset_steps = PIPELINE_STEPS[first_reset_idx:]
        now = utc_now()
        with self.database.connection:
            self.database.connection.execute(
                """
                UPDATE jobs
                SET status = 'queued', current_step = NULL,
                    last_error_code = NULL, last_error_message = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, job_id),
            )
            for step_name in reset_steps:
                self.database.connection.execute(
                    """
                    UPDATE job_steps
                    SET status = 'pending', started_at = NULL, completed_at = NULL,
                        duration_ms = NULL, error_code = NULL, error_message = NULL
                    WHERE job_id = ? AND name = ?
""",
                    (job_id, step_name),
                )

        for step_name in reset_steps:
            cp_f = checkpoint_path(self.data_dir, job_id, step_name)
            if cp_f.is_file():
                try:
                    cp_f.unlink()
                except OSError:
                    pass

        self._clear_rerun_artifacts(job_id, reset_steps)

        return self.get(job_id)

    def _clear_rerun_artifacts(self, job_id: str, reset_steps: list[str]) -> None:
        """Drop on-disk artifacts for reset steps so reruns do real work."""
        reset = set(reset_steps)
        job_dir = self.data_dir / "jobs" / job_id
        artifacts = job_dir / "artifacts"

        if "tts" in reset:
            tts_dir = artifacts / "tts"
            if tts_dir.is_dir():
                shutil.rmtree(tts_dir, ignore_errors=True)

        if reset.intersection({"mix", "render"}):
            for name in ("narration.wav", "mixed.wav", "normalized.wav"):
                path = artifacts / name
                if path.is_file():
                    try:
                        path.unlink()
                    except OSError:
                        pass

        if "render" in reset:
            output_dir = job_dir / "output"
            for name in ("dubbed.mp4", "vietnamese_narration.wav", "subtitles.ass"):
                path = output_dir / name
                if path.is_file():
                    try:
                        path.unlink()
                    except OSError:
                        pass

    def redub(self, job_id: str) -> Job:
        translate_index = PIPELINE_STEPS.index("translate")
        keep_steps = list(PIPELINE_STEPS[: translate_index + 1])
        return self.rerun(job_id, keep_steps)

    def delete(self, job_id: str) -> None:
        job = self.get(job_id)
        if job.status == "running":
            raise AppError(
                409,
                ErrorInfo(
                    code="JOB_NOT_DELETABLE",
                    message="A running job cannot be deleted.",
                    action="Wait for the job to complete or cancel it first.",
                ),
            )
        
        with self.database.connection:
            self.database.connection.execute("DELETE FROM job_steps WHERE job_id = ?", (job_id,))
            self.database.connection.execute("DELETE FROM events WHERE job_id = ?", (job_id,))
            self.database.connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            
        job_dir = self.data_dir / "jobs" / job_id
        if job_dir.exists():
            try:
                shutil.rmtree(job_dir)
            except OSError:
                pass

    def _hydrate(self, row) -> Job:
        steps = self.database.connection.execute(
            """
            SELECT name, position, status, checkpoint_path, started_at, completed_at, duration_ms
            FROM job_steps
            WHERE job_id = ?
            ORDER BY position
            """,
            (row["id"],),
        ).fetchall()
        valid_steps = set(PIPELINE_STEPS)
        return Job(
            **dict(row),
            steps=[JobStep(**dict(step)) for step in steps if step["name"] in valid_steps],
        )

