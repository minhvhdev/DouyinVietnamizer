from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .config import AppConfig
from .database import Database
from .errors import AppError, app_error_handler
from .jobs import JobService
from .models import ErrorInfo, Job, JobCreate, JobRerun
from .runtime import RuntimeReport, default_runtime_service
from .runner import JobRunner
from .checkpoints import load_checkpoint, save_checkpoint
from .settings import SettingsService


class VideoSelect(BaseModel):
    index: int


class BootstrapPayload(BaseModel):
    profile: str


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
            "cpu_mode": False,
            "asr_backend": "qwen3_asr",
            "asr_model": "Qwen/Qwen3-ASR-1.7B",
            "implemented_steps": [
                "resolve", "download", "extract_audio", "vad", "asr",
                "normalize_segments", "translate", "tts", "duration_repair",
                "mix", "render", "qc"
            ],
            "tts_backend": "vieneu",
            "runtime_status": runtime_status.status if runtime_status else "not_run",
        }

    @app.get("/api/runtime/status", response_model=RuntimeReport)
    def runtime_status() -> RuntimeReport:
        return runtime.latest() or runtime.run()

    @app.post("/api/runtime/smoke-test", response_model=RuntimeReport)
    def run_runtime_smoke_test() -> RuntimeReport:
        return runtime.run()

    @app.get("/api/runtime/detect-hardware")
    def detect_hardware() -> dict:
        from .hardware import get_hardware_report
        return get_hardware_report()

    @app.post("/api/runtime/bootstrap-vendor")
    def bootstrap_vendor(payload: BootstrapPayload) -> dict:
        from .bootstrap import BootstrapManager
        import os
        # Path details
        project_root = Path(__file__).resolve().parents[2]
        vendor_dir = Path(os.environ.get("DV_VENDOR_DIR", project_root / "vendor"))
        default_manifest_path = os.environ.get("DV_DEFAULT_MANIFEST")
        default_manifest = Path(default_manifest_path) if default_manifest_path else None
        
        success = BootstrapManager.start_bootstrap(
            profile=payload.profile,
            vendor_dir=vendor_dir,
            default_manifest_path=default_manifest
        )
        return {"status": "started" if success else "already_running"}

    @app.get("/api/runtime/bootstrap-progress")
    def bootstrap_progress() -> dict:
        from .bootstrap import BootstrapManager
        return BootstrapManager.get_status()

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

    @app.post("/api/jobs/{job_id}/rerun")
    def rerun_job(job_id: str, payload: JobRerun) -> dict:
        job = jobs.rerun(job_id, payload.keep_steps)
        runner.start_job(job_id)
        return {"status": "queued", "job": job}

    @app.post("/api/jobs/{job_id}/redub")
    def redub_job(job_id: str) -> dict:
        job = jobs.redub(job_id)
        runner.start_job(job_id)
        return {"status": "queued", "job": job}

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str) -> dict:
        jobs.delete(job_id)
        return {"status": "deleted"}

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

    @app.get("/api/jobs/{job_id}/segments/{index}/wav")
    def get_segment_wav(job_id: str, index: int) -> FileResponse:
        job_dir = config.data_dir / "jobs" / job_id
        tts_dir = job_dir / "artifacts" / "tts"
        
        # Try repaired wav first, then fallback to raw wav
        wav_path = tts_dir / f"tts_repaired_{index}.wav"
        if not wav_path.is_file():
            wav_path = tts_dir / f"tts_{index}.wav"
            
        if not wav_path.is_file():
            raise AppError(
                404,
                ErrorInfo(
                    code="SEGMENT_WAV_NOT_FOUND",
                    message="The WAV file for this segment was not found.",
                    action="Wait for the TTS/duration repair steps to complete."
                )
            )
        return FileResponse(str(wav_path), media_type="audio/wav", filename=f"seg_{index}.wav")

    @app.get("/api/jobs/{job_id}/files")
    def get_job_files(job_id: str) -> list[dict]:
        job_dir = config.data_dir / "jobs" / job_id
        files = []
        
        candidates = [
            {"key": "dubbed_video", "name": "Video lồng tiếng (dubbed.mp4)", "path": job_dir / "output" / "dubbed.mp4", "media_type": "video/mp4", "url": f"/api/jobs/{job_id}/files/dubbed_video"},
            {"key": "original_video", "name": "Video gốc tải về (original.mp4)", "path": job_dir / "artifacts" / "original.mp4", "media_type": "video/mp4", "url": f"/api/jobs/{job_id}/files/original_video"},
            {"key": "bgm", "name": "Nhạc nền tách ra (bgm.wav)", "path": job_dir / "artifacts" / "bgm.wav", "media_type": "audio/wav", "url": f"/api/jobs/{job_id}/files/bgm"},
            {"key": "vocals", "name": "Giọng nói gốc (vocals.wav)", "path": job_dir / "artifacts" / "vocals.wav", "media_type": "audio/wav", "url": f"/api/jobs/{job_id}/files/vocals"},
            {"key": "vietnamese_narration", "name": "Giọng lồng tiếng Việt (vietnamese_narration.wav)", "path": job_dir / "output" / "vietnamese_narration.wav", "media_type": "audio/wav", "url": f"/api/jobs/{job_id}/files/vietnamese_narration"},
            {"key": "subtitles", "name": "Phụ đề (subtitles.ass)", "path": job_dir / "output" / "subtitles.ass", "media_type": "text/plain", "url": f"/api/jobs/{job_id}/files/subtitles"},
        ]
        
        for item in candidates:
            path = item["path"]
            if path.is_file():
                files.append({
                    "key": item["key"],
                    "name": item["name"],
                    "size": path.stat().st_size,
                    "media_type": item["media_type"],
                    "url": item["url"]
                })
        return files

    @app.get("/api/jobs/{job_id}/files/{key}")
    def get_job_file_content(job_id: str, key: str) -> FileResponse:
        job_dir = config.data_dir / "jobs" / job_id
        
        candidates = {
            "dubbed_video": (job_dir / "output" / "dubbed.mp4", "video/mp4", "dubbed.mp4"),
            "original_video": (job_dir / "artifacts" / "original.mp4", "video/mp4", "original.mp4"),
            "bgm": (job_dir / "artifacts" / "bgm.wav", "audio/wav", "bgm.wav"),
            "vocals": (job_dir / "artifacts" / "vocals.wav", "audio/wav", "vocals.wav"),
            "vietnamese_narration": (job_dir / "output" / "vietnamese_narration.wav", "audio/wav", "vietnamese_narration.wav"),
            "subtitles": (job_dir / "output" / "subtitles.ass", "text/plain", "subtitles.ass"),
        }
        
        if key not in candidates:
            raise AppError(
                400,
                ErrorInfo(
                    code="INVALID_FILE_KEY",
                    message="Requested file key is invalid.",
                    action="Select a valid candidate file from the list."
                )
            )
            
        file_path, media_type, filename = candidates[key]
        if not file_path.is_file():
            raise AppError(
                404,
                ErrorInfo(
                    code="FILE_NOT_FOUND",
                    message="The requested file was not found on disk.",
                    action="Check if the corresponding step ran successfully."
                )
            )
            
        return FileResponse(str(file_path), media_type=media_type, filename=filename)

    @app.get("/api/outputs")
    def list_outputs() -> list[dict]:
        rows = database.connection.execute(
            "SELECT id, title, title_vi, source_url, updated_at FROM jobs "
            "WHERE status = 'completed' ORDER BY updated_at DESC"
        ).fetchall()

        outputs = []
        for r in rows:
            job_id = r["id"]
            output_file = config.data_dir / "jobs" / job_id / "output" / "dubbed.mp4"
            if output_file.is_file():
                title_vi = r["title_vi"]
                if not title_vi:
                    translate_cp = load_checkpoint(config.data_dir, job_id, "translate")
                    if translate_cp:
                        title_vi = translate_cp.get("title_vi")
                outputs.append({
                    "job_id": job_id,
                    "title": r["title"] or "Untitled Video",
                    "title_vi": title_vi,
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

    @app.get("/api/cloned-voices")
    def list_cloned_voices() -> list[dict]:
        rows = database.connection.execute(
            "SELECT id, name, wav_filename, created_at FROM cloned_voices ORDER BY created_at DESC"
        ).fetchall()
        
        cloned_dir = config.data_dir / "cloned_voices"
        cloned_dir.mkdir(exist_ok=True)
        
        voices = []
        for r in rows:
            wav_path = cloned_dir / r["wav_filename"]
            if wav_path.is_file():
                voices.append({
                    "id": r["id"],
                    "name": r["name"],
                    "wav_filename": r["wav_filename"],
                    "wav_path": str(wav_path),
                    "created_at": r["created_at"]
                })
        return voices

    @app.get("/api/cloned-voices/{voice_id}/wav")
    def get_cloned_voice_wav(voice_id: str) -> FileResponse:
        cloned_dir = config.data_dir / "cloned_voices"
        wav_path = cloned_dir / f"{voice_id}.wav"
        if not wav_path.is_file():
            raise AppError(
                404,
                ErrorInfo(
                    code="VOICE_NOT_FOUND",
                    message="The cloned voice file was not found on disk.",
                    action="Verify that the voice exists."
                )
            )
        return FileResponse(str(wav_path), media_type="audio/wav", filename=f"{voice_id}.wav")

    @app.post("/api/cloned-voices", status_code=201)
    def create_cloned_voice(name: str = Form(...), file: UploadFile = File(...)) -> dict:
        cloned_dir = config.data_dir / "cloned_voices"
        cloned_dir.mkdir(exist_ok=True)
        
        existing = database.connection.execute(
            "SELECT 1 FROM cloned_voices WHERE name = ?", (name,)
        ).fetchone()
        if existing:
            raise AppError(
                400,
                ErrorInfo(
                    code="VOICE_NAME_EXISTS",
                    message="A cloned voice with this name already exists.",
                    action="Please choose a different name for your voice."
                )
            )
            
        voice_id = str(uuid4())
        wav_filename = f"{voice_id}.wav"
        wav_path = cloned_dir / wav_filename
        
        try:
            with wav_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise AppError(
                500,
                ErrorInfo(
                    code="FILE_SAVE_FAILED",
                    message="Failed to save the uploaded WAV file.",
                    action="Ensure your storage has write permissions and try again.",
                    detail=str(e)
                )
            )
            
        now = datetime.now(timezone.utc).isoformat()
        with database.connection:
            database.connection.execute(
                "INSERT INTO cloned_voices (id, name, wav_filename, created_at) VALUES (?, ?, ?, ?)",
                (voice_id, name, wav_filename, now)
            )
            
        return {
            "id": voice_id,
            "name": name,
            "wav_filename": wav_filename,
            "wav_path": str(wav_path),
            "created_at": now
        }

    @app.delete("/api/cloned-voices/{voice_id}")
    def delete_cloned_voice(voice_id: str) -> dict:
        row = database.connection.execute(
            "SELECT wav_filename FROM cloned_voices WHERE id = ?", (voice_id,)
        ).fetchone()
        if not row:
            raise AppError(
                404,
                ErrorInfo(
                    code="VOICE_NOT_FOUND",
                    message="The requested voice does not exist.",
                    action="Verify that the voice ID is correct."
                )
            )
            
        wav_filename = row["wav_filename"]
        cloned_dir = config.data_dir / "cloned_voices"
        wav_path = cloned_dir / wav_filename
        
        if wav_path.is_file():
            try:
                wav_path.unlink()
            except OSError:
                pass
                
        with database.connection:
            database.connection.execute(
                "DELETE FROM cloned_voices WHERE id = ?", (voice_id,)
            )
            
        return {"status": "deleted"}

    class VoiceTestPayload(BaseModel):
        text: str

    @app.post("/api/cloned-voices/{voice_id}/test")
    def test_cloned_voice(voice_id: str, payload: VoiceTestPayload) -> FileResponse:
        row = database.connection.execute(
            "SELECT wav_filename FROM cloned_voices WHERE id = ?", (voice_id,)
        ).fetchone()
        if not row:
            raise AppError(
                404,
                ErrorInfo(
                    code="VOICE_NOT_FOUND",
                    message="The requested voice does not exist.",
                    action="Verify that the voice ID is correct."
                )
            )
            
        wav_filename = row["wav_filename"]
        cloned_dir = config.data_dir / "cloned_voices"
        wav_path = cloned_dir / wav_filename
        if not wav_path.is_file():
            raise AppError(
                404,
                ErrorInfo(
                    code="VOICE_FILE_NOT_FOUND",
                    message="The cloned voice WAV file does not exist on disk.",
                    action="Re-upload the reference WAV file."
                )
            )
            
        from .adapters.tts import VieNeuTtsAdapter
        raw_settings = settings.get_raw_all()
        vieneu_tts = VieNeuTtsAdapter(
            device=str(raw_settings.get("vieneu_device", "cuda") or "cuda"),
        )
        
        temp_dir = Path(tempfile.gettempdir())
        output_wav = temp_dir / f"test_synthesize_{voice_id}_{uuid4().hex}.wav"
        
        try:
            vieneu_tts.synthesize(
                text=payload.text,
                output_path=output_wav,
                voice=str(wav_path)
            )
        except AppError as e:
            raise e
        except Exception as e:
            raise AppError(
                502,
                ErrorInfo(
                    code="VIENEU_SYNTHESIZE_FAILED",
                    message="Failed to synthesize test audio using VieNeu-TTS.",
                    action="Ensure VieNeu-TTS is installed and configured properly.",
                    detail=str(e)
                )
            )
            
        if not output_wav.is_file() or output_wav.stat().st_size == 0:
            raise AppError(
                500,
                ErrorInfo(
                    code="SYNTHESIZED_EMPTY",
                    message="Synthesized audio is empty.",
                    action="Try another text sentence."
                )
            )
            
        return FileResponse(str(output_wav), media_type="audio/wav", filename="test_output.wav")

    app.state.config = config
    app.state.database = database
    app.state.jobs = jobs
    app.state.runtime = runtime
    app.state.runner = runner
    app.state.settings = settings
    return app
