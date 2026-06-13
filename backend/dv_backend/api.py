from datetime import datetime, timezone
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import AppConfig
from .database import Database
from .errors import AppError, app_error_handler
from .jobs import JobService
from .models import Job, JobCreate


def create_app(config: AppConfig | None = None) -> FastAPI:
    config = config or AppConfig.from_env()
    config.ensure_directories()
    database = Database(config.database_path)
    database.migrate()
    jobs = JobService(database, config.data_dir)
    jobs.reconcile_interrupted()

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
        return {
            "cpu_mode": True,
            "implemented_steps": [],
            "optional_backends": {"whisper_vulkan": False, "qwen3_asr": False},
        }

    @app.get("/api/jobs", response_model=list[Job])
    def list_jobs() -> list[Job]:
        return jobs.list()

    @app.post("/api/jobs", status_code=201, response_model=Job)
    def create_job(payload: JobCreate) -> Job:
        return jobs.create(payload.source_url)

    @app.get("/api/jobs/{job_id}", response_model=Job)
    def get_job(job_id: str) -> Job:
        return jobs.get(job_id)

    def read_settings() -> dict:
        rows = database.connection.execute("SELECT key, value FROM settings").fetchall()
        return {row["key"]: json.loads(row["value"]) for row in rows}

    @app.get("/api/settings")
    def get_settings() -> dict:
        return read_settings()

    @app.put("/api/settings")
    def update_settings(payload: dict) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        with database.connection:
            database.connection.executemany(
                "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                [(key, json.dumps(value), now) for key, value in payload.items()],
            )
            database.connection.execute(
                "INSERT INTO events (level, code, message, created_at) VALUES ('info', 'SETTINGS_UPDATED', 'Application settings updated.', ?)",
                (now,),
            )
        return read_settings()

    @app.get("/api/events")
    def get_events() -> list[dict]:
        rows = database.connection.execute(
            "SELECT id, level, code, message, job_id, created_at FROM events ORDER BY id DESC LIMIT 100"
        ).fetchall()
        return [dict(row) for row in rows]

    app.state.config = config
    app.state.database = database
    app.state.jobs = jobs
    return app
