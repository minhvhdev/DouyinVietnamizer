from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .config import AppConfig
from .database import Database
from .errors import AppError, app_error_handler
from .jobs import JobService
from .models import ErrorInfo, Job, JobCreate
from .runtime import RuntimeReport, default_runtime_service
from .runner import JobRunner
from .checkpoints import load_checkpoint, save_checkpoint
from .settings import SettingsService


class VideoSelect(BaseModel):
    index: int


def create_app(config: AppConfig | None = None) -> FastAPI:
    config = config or AppConfig.from_env()
    config.ensure_directories()
    database = Database(config.database_path)
    database.migrate()

    settings = SettingsService(database)

    jobs = JobService(database, config.data_dir)
    jobs.reconcile_interrupted()
    runtime = default_runtime_service(config, database)
    if runtime.latest() is None:
        runtime.run()

    runner = JobRunner(config, database)

    app = FastAPI(title="Douyin Vietnamizer Backend")
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^(file://|http://(localhost|127\.0\.0\.1)(:\d+)?)$",
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_exception_handler(AppError, app_error_handler)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/api/capabilities")
    def capabilities() -> dict:
        runtime_status = runtime.latest()
        return {
            "cpu_mode": True,
            "implemented_steps": [
                "resolve", "download", "extract_audio", "vad", "asr",
                "normalize_segments", "translate", "tts", "duration_repair",
                "mix", "render", "qc"
            ],
            "optional_backends": {"whisper_vulkan": True, "qwen3_asr": True},
            "runtime_status": runtime_status.status if runtime_status else "not_run",
        }

    @app.get("/api/runtime/status", response_model=RuntimeReport)
    def runtime_status() -> RuntimeReport:
        return runtime.latest() or runtime.run()

    @app.post("/api/runtime/smoke-test", response_model=RuntimeReport)
    def run_runtime_smoke_test() -> RuntimeReport:
        return runtime.run()

    @app.get("/api/jobs", response_model=list[Job])
    def list_jobs() -> list[Job]:
        return jobs.list()

    @app.post("/api/jobs", status_code=201, response_model=Job)
    def create_job(payload: JobCreate) -> Job:
        job = jobs.create(payload.source_url)
        # Automatically trigger job run
        runner.start_job(job.id)
        return jobs.get(job.id)

    @app.get("/api/jobs/{job_id}", response_model=Job)
    def get_job(job_id: str) -> Job:
        return jobs.get(job_id)

    @app.post("/api/jobs/{job_id}/start")
    def start_job(job_id: str) -> dict:
        runner.start_job(job_id)
        return {"status": "started"}

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict:
        runner.cancel_job(job_id)
        return {"status": "cancelled"}

    @app.post("/api/jobs/{job_id}/select-video")
    def select_video(job_id: str, payload: VideoSelect) -> dict:
        resolve_cp = load_checkpoint(config.data_dir, job_id, "resolve")
        if not resolve_cp or not resolve_cp.get("videos"):
            raise AppError(
                400,
                ErrorInfo(
                    code="RESOLVE_NOT_COMPLETED",
                    message="Resolution is not completed yet or returned no videos.",
                    action="Wait for the resolve step to complete."
                )
            )

        videos = resolve_cp["videos"]
        if payload.index < 0 or payload.index >= len(videos):
            raise AppError(
                400,
                ErrorInfo(
                    code="INVALID_VIDEO_INDEX",
                    message="The selected video index is out of bounds.",
                    action="Select a valid index from the resolved video list."
                )
            )

        selected = videos[payload.index]
        resolve_cp["selected_video"] = selected
        save_checkpoint(config.data_dir, job_id, "resolve", resolve_cp)

        # Reset jobs and step status in database so download can run
        now = datetime.now(timezone.utc).isoformat()
        with database.connection:
            database.connection.execute(
                "UPDATE jobs SET title = ?, status = 'queued', updated_at = ? WHERE id = ?",
                (selected["title"], now, job_id)
            )
            database.connection.execute(
                "UPDATE job_steps SET status = 'pending', started_at = NULL, completed_at = NULL, error_code = NULL, error_message = NULL WHERE job_id = ? AND name = 'download'",
                (job_id,)
            )

        runner.start_job(job_id)
        return {"status": "selected", "video": selected}

    @app.get("/api/jobs/{job_id}/checkpoint/{step_name}")
    def get_checkpoint(job_id: str, step_name: str) -> Any:
        data = load_checkpoint(config.data_dir, job_id, step_name)
        if not data:
            return JSONResponse(status_code=404, content={"message": "Checkpoint not found"})
        return data

    @app.get("/api/jobs/{job_id}/output")
    def get_job_output(job_id: str) -> FileResponse:
        output_file = config.data_dir / "jobs" / job_id / "output" / "dubbed.mp4"
        if not output_file.is_file():
            raise AppError(
                404,
                ErrorInfo(
                    code="OUTPUT_NOT_FOUND",
                    message="The dubbed video file does not exist.",
                    action="Wait for the render step to complete successfully."
                )
            )
        return FileResponse(str(output_file), media_type="video/mp4", filename=f"{job_id}_dubbed.mp4")

    @app.get("/api/outputs")
    def list_outputs() -> list[dict]:
        rows = database.connection.execute(
            "SELECT id, title, source_url, updated_at FROM jobs WHERE status = 'completed' ORDER BY updated_at DESC"
        ).fetchall()

        outputs = []
        for r in rows:
            job_id = r["id"]
            output_file = config.data_dir / "jobs" / job_id / "output" / "dubbed.mp4"
            if output_file.is_file():
                outputs.append({
                    "job_id": job_id,
                    "title": r["title"] or "Untitled Video",
                    "source_url": r["source_url"],
                    "completed_at": r["updated_at"],
                    "file_size": output_file.stat().st_size
                })
        return outputs

    @app.get("/api/settings")
    def get_settings() -> dict:
        return settings.get_all()

    @app.put("/api/settings")
    def update_settings(payload: dict) -> dict:
        try:
            updated = settings.update(payload)
        except ValueError as cause:
            raise AppError(
                422,
                ErrorInfo(
                    code="INVALID_SETTINGS",
                    message=str(cause),
                    action="Choose a supported settings value.",
                ),
            ) from cause
        now = datetime.now(timezone.utc).isoformat()
        with database.connection:
            database.connection.execute(
                "INSERT INTO events (level, code, message, created_at) VALUES ('info', 'SETTINGS_UPDATED', 'Application settings updated.', ?)",
                (now,),
            )
        return updated

    @app.get("/api/events")
    def get_events() -> list[dict]:
        rows = database.connection.execute(
            "SELECT id, level, code, message, job_id, created_at FROM events ORDER BY id DESC LIMIT 100"
        ).fetchall()
        return [dict(row) for row in rows]

    app.state.config = config
    app.state.database = database
    app.state.jobs = jobs
    app.state.runtime = runtime
    app.state.runner = runner
    app.state.settings = settings
    return app
