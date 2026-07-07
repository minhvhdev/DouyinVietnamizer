from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any
import wave
from uuid import uuid4

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .config import AppConfig
from .database import Database
from .error_logging import configure_error_logging
from .errors import AppError, app_error_handler
from .jobs import JobService
from .local_env import load_repo_dotenv
from .models import ErrorInfo, Job, JobCreate, JobRerun
from .runtime import ReleaseVramResult, RuntimeReport, default_runtime_service, release_vram_resources
from .runner import JobRunner
from .checkpoints import PIPELINE_STEPS, load_checkpoint, save_checkpoint
from .settings import SettingsService


VOICE_UPLOAD_MIME_SUFFIXES = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "video/mp2t": ".mp3",
}
VOICE_UPLOAD_SUFFIXES = frozenset({".wav", ".mp3"})


def _voice_upload_suffix(file: UploadFile) -> str:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix in VOICE_UPLOAD_SUFFIXES:
        return suffix
    content_type = (file.content_type or "").split(";", 1)[0].strip().lower()
    if content_type in VOICE_UPLOAD_MIME_SUFFIXES:
        return VOICE_UPLOAD_MIME_SUFFIXES[content_type]
    raise AppError(
        415,
        ErrorInfo(
            code="VOICE_UNSUPPORTED_FORMAT",
            message="Unsupported cloned voice audio format.",
            action="Upload a .wav or .mp3 file.",
        ),
    )


def _read_transcript_sidecar(wav_path: Path) -> str:
    try:
        return wav_path.with_suffix(".txt").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _convert_audio_to_wav(input_path: Path, output_path: Path) -> Path:
    try:
        from .pipeline import resolve_tool_path

        ffmpeg_path = resolve_tool_path(AppConfig.from_env(), "ffmpeg")
    except Exception:
        ffmpeg_path = Path(shutil.which("ffmpeg") or "ffmpeg")
    cmd = [
        str(ffmpeg_path),
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as exc:
        raise AppError(
            422,
            ErrorInfo(
                code="VOICE_AUDIO_CONVERT_FAILED",
                message="Failed to convert uploaded MP3 to WAV.",
                action="Upload a valid short WAV or MP3 voice sample.",
                detail=str(exc),
            ),
        ) from exc
    if completed.returncode != 0 or not output_path.is_file() or output_path.stat().st_size == 0:
        raise AppError(
            422,
            ErrorInfo(
                code="VOICE_AUDIO_CONVERT_FAILED",
                message="Failed to convert uploaded MP3 to WAV.",
                action="Upload a valid short WAV or MP3 voice sample.",
                detail=completed.stderr[-2000:],
            ),
        )
    return output_path


def _transcribe_anchor_for_voice(wav_path: Path) -> str:
    try:
        with wave.open(str(wav_path), "rb"):
            pass
    except Exception:
        return ""
    try:
        from .adapters.asr import reset_model_cache, transcribe_audio

        project_root = Path(__file__).resolve().parents[2]
        vendor_dir = Path(os.environ.get("DV_VENDOR_DIR", project_root / "vendor"))
        result = transcribe_audio(
            wav_path,
            vendor_dir=vendor_dir,
            language="Vietnamese",
        )
        segments = result.get("segments", []) if isinstance(result, dict) else result
        return " ".join(
            str(segment.get("text", "")).strip()
            for segment in segments
            if str(segment.get("text", "")).strip()
        ).strip()
    except Exception:
        return ""
    finally:
        try:
            reset_model_cache()
        except Exception:
            pass


def _synthesize_voice_preview(
    *,
    voice: str | None,
    text: str,
    settings: dict[str, Any],
    output_suffix: str,
    clone: bool = False,
    clone_mode: str | None = None,
    anchor_text: str | None = None,
    backend: str | None = None,
) -> Path:
    from .adapters.tts import (
        VOXCPM_INSTRUCT_PREFIX,
        create_tts_adapter,
        resolve_tts_voice,
        tts_backend_from_settings,
    )

    cleaned_text = (text or "").strip()
    if not cleaned_text:
        raise AppError(
            400,
            ErrorInfo(
                code="PREVIEW_TEXT_EMPTY",
                message="Preview text is required.",
                action="Enter a short sentence to synthesize.",
            ),
        )

    preview_settings = dict(settings)
    if backend:
        preview_settings["tts_backend"] = backend

    resolved_backend = tts_backend_from_settings(preview_settings)
    tts = create_tts_adapter(preview_settings)
    output_wav = Path(tempfile.gettempdir()) / f"voice_preview_{output_suffix}_{uuid4().hex}.wav"
    preview_voice = (voice or "").strip() or resolve_tts_voice(preview_settings)
    if resolved_backend == "voxcpm":
        instruct = str(preview_settings.get("voxcpm_instruct") or "").strip()
        if instruct and not (voice or "").strip().lower().endswith(".wav"):
            preview_voice = f"{VOXCPM_INSTRUCT_PREFIX}{instruct}"
    try:
        synthesize_kwargs = {
            "text": cleaned_text,
            "output_path": output_wav,
            "voice": preview_voice,
        }
        if clone:
            synthesize_kwargs.update(
                clone=True,
                clone_mode=clone_mode,
                anchor_text=anchor_text,
            )
        tts.synthesize(**synthesize_kwargs)
    except AppError:
        raise
    except Exception as exc:
        labels = {
            "voxcpm": "VoxCPM2",
            "edge_tts": "Edge TTS",
            "google_tts": "Google TTS",
            "gemini_tts": "Gemini TTS",
        }
        label = labels.get(resolved_backend, resolved_backend)
        actions = {
            "voxcpm": "Run 'python scripts/setup_voxcpm.py' for VoxCPM2.",
            "edge_tts": "Check your internet connection and Edge TTS voice selection.",
            "google_tts": "Check your Google Cloud TTS API key and voice selection.",
            "gemini_tts": "Verify Gemini API keys in Settings and retry.",
        }
        raise AppError(
            502,
            ErrorInfo(
                code=f"{resolved_backend.upper()}_SYNTHESIZE_FAILED",
                message=f"Failed to synthesize preview audio using {label}.",
                action=actions.get(resolved_backend, "Retry with different settings."),
                detail=str(exc),
            ),
        ) from exc

    if not output_wav.is_file() or output_wav.stat().st_size == 0:
        raise AppError(
            500,
            ErrorInfo(
                code="SYNTHESIZED_EMPTY",
                message="Synthesized audio is empty.",
                action="Try another text sentence.",
            ),
        )
    return output_wav


class VideoSelect(BaseModel):
    index: int


class SegmentSpeakerPayload(BaseModel):
    speaker_id: str


class BootstrapPayload(BaseModel):
    profile: str


def create_app(config: AppConfig | None = None) -> FastAPI:
    load_repo_dotenv()
    config = config or AppConfig.from_env()
    config.ensure_directories()
    configure_error_logging(config)
    database = Database(config.database_path)
    database.migrate()

    settings = SettingsService(database)

    jobs = JobService(database, config.data_dir)
    jobs.reconcile_interrupted()
    runtime = default_runtime_service(config, database)
    if runtime.latest() is None:
        runtime.run()

    runner = JobRunner(config, database)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            release_vram_resources(runner=runner)

    app = FastAPI(title="Douyin Vietnamizer Backend", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^(file://|tauri://localhost|http://(localhost|127\.0\.0\.1|tauri\.localhost)(:\d+)?)$",
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_exception_handler(AppError, app_error_handler)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/api/capabilities")
    def capabilities() -> dict:
        from .adapters.tts import SUPPORTED_TTS_BACKENDS, tts_backend_from_settings

        runtime_status = runtime.latest()
        raw_settings = settings.get_raw_all()
        return {
            "cpu_mode": False,
            "asr_backend": "qwen3_asr",
            "asr_model": "Qwen/Qwen3-ASR-1.7B",
            "implemented_steps": list(PIPELINE_STEPS),
            "tts_backend": tts_backend_from_settings(raw_settings),
            "tts_backends": list(SUPPORTED_TTS_BACKENDS),
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

    @app.post("/api/runtime/release-vram", response_model=ReleaseVramResult)
    def release_vram() -> ReleaseVramResult:
        return release_vram_resources(runner=runner)

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

    @app.post("/api/runtime/bootstrap-pyannote")
    def bootstrap_pyannote() -> dict:
        from .bootstrap import BootstrapManager

        project_root = Path(__file__).resolve().parents[2]
        vendor_dir = Path(os.environ.get("DV_VENDOR_DIR", project_root / "vendor"))
        success = BootstrapManager.start_pyannote_bootstrap(vendor_dir)
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
        runner.start_job(job.id)
        return jobs.get(job.id)

    @app.post("/api/jobs/import", status_code=201, response_model=Job)
    def import_job(
        file: UploadFile = File(...),
        title: str | None = Form(default=None),
    ) -> Job:
        original_filename = file.filename or 'imported'
        safe_filename = Path(original_filename).name or 'imported'
        suffix = Path(safe_filename).suffix.lower()
        if suffix not in JobService.SUPPORTED_IMPORT_EXTENSIONS:
            raise AppError(
                415,
                ErrorInfo(
                    code="IMPORT_UNSUPPORTED_FORMAT",
                    message=f"Unsupported file format: {suffix or '(none)'}",
                    action=f"Use one of: {', '.join(JobService.SUPPORTED_IMPORT_EXTENSIONS)}",
                ),
            )

        tmp_dir = Path(tempfile.gettempdir()) / 'dv_imports'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"{uuid4().hex}{suffix}"
        try:
            with tmp_path.open('wb') as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as exc:
            raise AppError(
                500,
                ErrorInfo(
                    code="IMPORT_FILE_SAVE_FAILED",
                    message="Failed to save the uploaded file.",
                    action="Check disk space and write permissions, then try again.",
                    detail=str(exc),
                ),
            )
        finally:
            try:
                file.file.close()
            except Exception:
                pass

        try:
            job = jobs.create_imported(
                tmp_path,
                original_filename=safe_filename,
                title=title.strip() if title else None,
            )
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

        runner.start_job(job.id)
        return jobs.get(job.id)

    @app.get("/api/jobs/{job_id}", response_model=Job)
    def get_job(job_id: str) -> Job:
        return jobs.get(job_id)

    @app.post("/api/jobs/{job_id}/start")
    def start_job(job_id: str) -> dict:
        jobs.prepare_job_for_resume(job_id)
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
                    message="Bước phân tích liên kết chưa hoàn thành hoặc không có video.",
                    action="Đợi bước «Phân tích liên kết» xong rồi thử lại.",
                ),
            )

        videos = resolve_cp["videos"]
        if payload.index < 0 or payload.index >= len(videos):
            raise AppError(
                400,
                ErrorInfo(
                    code="INVALID_VIDEO_INDEX",
                    message="Chỉ số video không hợp lệ.",
                    action="Chọn một video trong danh sách hiển thị.",
                ),
            )

        selected = videos[payload.index]
        resolve_cp["selected_video"] = selected
        save_checkpoint(config.data_dir, job_id, "resolve", resolve_cp)

        now = datetime.now(timezone.utc).isoformat()
        with database.connection:
            database.connection.execute(
                "UPDATE jobs SET title = ?, status = 'queued', updated_at = ? WHERE id = ?",
                (selected["title"], now, job_id),
            )
            database.connection.execute(
                """
                UPDATE job_steps
                SET status = 'pending', started_at = NULL, completed_at = NULL,
                    duration_ms = NULL, error_code = NULL, error_message = NULL
                WHERE job_id = ? AND name = 'download'
                """,
                (job_id,),
            )

        runner.start_job(job_id)
        return {"status": "selected", "video": selected}

    @app.post("/api/runtime/update-yt-dlp")
    def update_yt_dlp_runtime() -> dict:
        from .pipeline import resolve_tool_path
        from .ytdlp_tools import update_yt_dlp_binary, yt_dlp_version

        yt_dlp_path = resolve_tool_path(config, "yt_dlp")
        result = update_yt_dlp_binary(yt_dlp_path)
        result["path"] = str(yt_dlp_path)
        result["version"] = yt_dlp_version(yt_dlp_path)
        return result

    @app.get("/api/jobs/{job_id}/checkpoint/{step_name}")
    def get_checkpoint(job_id: str, step_name: str) -> Any:
        data = load_checkpoint(config.data_dir, job_id, step_name)
        if not data:
            return JSONResponse(status_code=404, content={"message": "Checkpoint not found"})
        return data

    @app.patch("/api/jobs/{job_id}/segments/{index}/speaker")
    def update_segment_speaker(job_id: str, index: int, payload: SegmentSpeakerPayload) -> dict:
        speaker_id = str(payload.speaker_id).strip()
        if not speaker_id:
            raise AppError(
                422,
                ErrorInfo(
                    code="INVALID_SPEAKER_ID",
                    message="Speaker id is required.",
                    action="Choose a valid speaker id.",
                ),
            )
        updated_segment = None
        for step_name in ("normalize_segments", "translate"):
            checkpoint = load_checkpoint(config.data_dir, job_id, step_name)
            if not checkpoint:
                continue
            segments = checkpoint.get("segments") or []
            for segment in segments:
                if int(segment.get("index", -1)) == index:
                    segment["speaker_id"] = speaker_id
                    segment["speaker_confidence"] = 1.0
                    updated_segment = segment
                    break
            save_checkpoint(config.data_dir, job_id, step_name, checkpoint)
        if updated_segment is None:
            raise AppError(
                404,
                ErrorInfo(
                    code="SEGMENT_NOT_FOUND",
                    message="The requested segment was not found.",
                    action="Wait for segment normalization to finish, then try again.",
                ),
            )
        return dict(updated_segment)

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
        
        wav_path = tts_dir / f"tts_repaired_{index}.wav"
        if not wav_path.is_file():
            wav_path = tts_dir / f"tts_{index}.wav"
        if not wav_path.is_file():
            wav_path = tts_dir / f"tts_raw_{index}.wav"

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

    @app.get("/api/jobs/{job_id}/folder")
    def get_job_folder(job_id: str) -> dict:
        jobs.get(job_id)
        job_dir = config.data_dir / "jobs" / job_id
        resolved = job_dir.resolve()
        return {
            "path": str(resolved),
            "exists": job_dir.exists(),
        }

    @app.get("/api/jobs/{job_id}/files")
    def get_job_files(job_id: str) -> list[dict]:
        job_dir = config.data_dir / "jobs" / job_id
        files = []

        candidates = [
            {"key": "dubbed_video", "name": "Video lồng tiếng (dubbed.mp4)", "path": job_dir / "output" / "dubbed.mp4", "media_type": "video/mp4", "url": f"/api/jobs/{job_id}/files/dubbed_video"},
            {"key": "original_video", "name": "Video gốc tải về (original.mp4)", "path": job_dir / "artifacts" / "original.mp4", "media_type": "video/mp4", "url": f"/api/jobs/{job_id}/files/original_video"},
            {"key": "bgm", "name": "Nhạc nền gốc (bgm.wav)", "path": job_dir / "artifacts" / "bgm.wav", "media_type": "audio/wav", "url": f"/api/jobs/{job_id}/files/bgm"},
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

    class OpenAiModelsPayload(BaseModel):
        base_url: str | None = None
        api_key: str | None = None

    @app.post("/api/translation/openai-models")
    def list_translation_openai_models(payload: OpenAiModelsPayload) -> list[dict[str, str]]:
        from .adapters.openai_compat import list_openai_models, normalize_openai_api_base

        raw_settings = settings.get_raw_all()
        base_url = normalize_openai_api_base(
            str(payload.base_url or raw_settings.get("openai_api_base") or "")
        )
        api_key = str(payload.api_key or raw_settings.get("openai_api_key") or "").strip()
        if not api_key:
            raise AppError(
                400,
                ErrorInfo(
                    code="MISSING_OPENAI_API_KEY",
                    message="No OpenAPI-compatible API key is configured.",
                    action="Enter and save an API key in Settings → Dịch thuật.",
                ),
            )
        return list_openai_models(base_url, api_key)

    @app.get("/api/events")
    def get_events() -> list[dict]:
        rows = database.connection.execute(
            "SELECT id, level, code, message, job_id, created_at FROM events ORDER BY id DESC LIMIT 100"
        ).fetchall()
        return [dict(row) for row in rows]

    class VoicePreviewPayload(BaseModel):
        voice: str | None = None
        text: str
        backend: str | None = None

    class TtsPreviewPayload(BaseModel):
        text: str
        backend: str | None = None
        voice: str | None = None
        settings: dict[str, Any] | None = None

    @app.get("/api/tts/voices")
    def list_tts_voices(backend: str = "edge_tts") -> list[dict]:
        from .adapters.tts import GEMINI_TTS_VOICES, SUPPORTED_TTS_BACKENDS

        resolved = (backend or "").strip().lower()
        if resolved not in SUPPORTED_TTS_BACKENDS:
            raise AppError(
                422,
                ErrorInfo(
                    code="INVALID_TTS_BACKEND",
                    message="The requested TTS backend is not supported.",
                    action="Choose one of: " + ", ".join(SUPPORTED_TTS_BACKENDS),
                ),
            )
        if resolved == "edge_tts":
            from .adapters.edge_tts import list_edge_tts_voices

            return list_edge_tts_voices()
        if resolved == "google_tts":
            from .adapters.google_tts import GOOGLE_TTS_VOICES

            return [
                {
                    "id": voice["id"],
                    "name": voice["name"],
                    "gender": voice.get("gender"),
                    "tier": voice.get("tier"),
                    "kind": "google_cloud",
                }
                for voice in GOOGLE_TTS_VOICES
            ]
        if resolved == "gemini_tts":
            return [{"id": voice["id"], "name": voice["name"], "kind": "gemini"} for voice in GEMINI_TTS_VOICES]
        presets = [
            "Ngọc Lan",
            "Minh Anh",
            "Hoài Nam",
            "Thu Hà",
            "Quang Huy",
            "Mai Phương",
            "Đức Anh",
            "Bảo Trâm",
            "Gia Hân",
            "Tuấn Kiệt",
        ]
        return [{"id": voice, "name": voice, "kind": "preset"} for voice in presets]

    @app.post("/api/tts/preview")
    def preview_tts(payload: TtsPreviewPayload) -> FileResponse:
        raw_settings = settings.get_raw_all()
        merged_settings = {**raw_settings, **(payload.settings or {})}
        backend = (payload.backend or merged_settings.get("tts_backend") or "voxcpm").strip().lower()
        merged_settings["tts_backend"] = backend
        output_wav = _synthesize_voice_preview(
            voice=payload.voice,
            text=payload.text,
            settings=merged_settings,
            output_suffix=backend.replace("_", "-"),
            backend=backend,
        )
        return FileResponse(
            str(output_wav),
            media_type="audio/wav",
            filename=f"preview_{backend}.wav",
        )

    @app.get("/api/voices/presets")
    def list_preset_voices() -> list[dict]:
        return [item for item in list_tts_voices("voxcpm") if item.get("kind") == "preset"]

    @app.post("/api/voices/preview")
    def preview_voice(payload: VoicePreviewPayload) -> FileResponse:
        voice = (payload.voice or "").strip()
        presets = {item["id"] for item in list_preset_voices()}
        if voice and voice not in presets:
            raise AppError(
                422,
                ErrorInfo(
                    code="INVALID_PRESET_VOICE",
                    message="The selected preset voice is not available.",
                    action="Choose a preset voice from the list.",
                ),
            )
        raw_settings = settings.get_raw_all()
        output_wav = _synthesize_voice_preview(
            voice=voice or None,
            text=payload.text,
            settings=raw_settings,
            output_suffix="voxcpm",
            backend=(payload.backend or "voxcpm").strip().lower(),
        )
        return FileResponse(str(output_wav), media_type="audio/wav", filename="preview_voxcpm.wav")

    @app.get("/api/cloned-voices")
    def list_cloned_voices() -> list[dict]:
        rows = database.connection.execute(
            "SELECT id, name, wav_filename, transcript, created_at FROM cloned_voices ORDER BY created_at DESC"
        ).fetchall()

        cloned_dir = config.data_dir / "cloned_voices"
        cloned_dir.mkdir(exist_ok=True)

        voices = []
        for r in rows:
            wav_path = cloned_dir / r["wav_filename"]
            if wav_path.is_file():
                transcript = (r["transcript"] or _read_transcript_sidecar(wav_path)).strip()
                voices.append({
                    "id": r["id"],
                    "name": r["name"],
                    "wav_filename": r["wav_filename"],
                    "wav_path": str(wav_path),
                    "transcript": transcript,
                    "transcribed": bool(transcript),
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
        voice_name = name.strip()
        if not voice_name:
            raise AppError(
                400,
                ErrorInfo(
                    code="VOICE_NAME_EMPTY",
                    message="Voice name is required.",
                    action="Enter a name for this cloned voice."
                )
            )

        existing = database.connection.execute(
            "SELECT 1 FROM cloned_voices WHERE name = ?", (voice_name,)
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

        suffix = _voice_upload_suffix(file)
        voice_id = str(uuid4())
        wav_filename = f"{voice_id}.wav"
        wav_path = cloned_dir / wav_filename
        upload_path = wav_path if suffix == ".wav" else cloned_dir / f"{voice_id}{suffix}"

        try:
            with upload_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            if suffix == ".mp3":
                _convert_audio_to_wav(upload_path, wav_path)
                upload_path.unlink(missing_ok=True)
        except AppError:
            raise
        except Exception as e:
            raise AppError(
                500,
                ErrorInfo(
                    code="FILE_SAVE_FAILED",
                    message="Failed to save the uploaded voice file.",
                    action="Ensure your storage has write permissions and try again.",
                    detail=str(e)
                )
            )
        finally:
            try:
                file.file.close()
            except Exception:
                pass

        transcript = _transcribe_anchor_for_voice(wav_path).strip()
        if transcript:
            wav_path.with_suffix(".txt").write_text(transcript, encoding="utf-8")

        now = datetime.now(timezone.utc).isoformat()
        with database.connection:
            database.connection.execute(
                "INSERT INTO cloned_voices (id, name, wav_filename, transcript, created_at) VALUES (?, ?, ?, ?, ?)",
                (voice_id, voice_name, wav_filename, transcript, now)
            )

        return {
            "id": voice_id,
            "name": voice_name,
            "wav_filename": wav_filename,
            "wav_path": str(wav_path),
            "transcript": transcript,
            "transcribed": bool(transcript),
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
        try:
            wav_path.with_suffix(".txt").unlink(missing_ok=True)
        except OSError:
            pass

        with database.connection:
            database.connection.execute(
                "DELETE FROM cloned_voices WHERE id = ?", (voice_id,)
            )
            
        return {"status": "deleted"}

    class VoiceTestPayload(BaseModel):
        text: str
        mode: str | None = None

    @app.post("/api/cloned-voices/{voice_id}/test")
    def test_cloned_voice(voice_id: str, payload: VoiceTestPayload) -> FileResponse:
        row = database.connection.execute(
            "SELECT wav_filename, transcript FROM cloned_voices WHERE id = ?", (voice_id,)
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

        clone_mode = (payload.mode or "reference").strip().lower()
        if clone_mode not in {"reference", "ultimate"}:
            raise AppError(
                422,
                ErrorInfo(
                    code="INVALID_CLONE_MODE",
                    message="Clone test mode must be reference or ultimate.",
                    action="Choose reference or ultimate clone mode."
                )
            )
        anchor_text = (row["transcript"] or _read_transcript_sidecar(wav_path)).strip()
        raw_settings = settings.get_raw_all()
        try:
            output_wav = _synthesize_voice_preview(
                voice=str(wav_path),
                text=payload.text,
                settings=raw_settings,
                output_suffix=f"clone_{voice_id}",
                clone=True,
                clone_mode=clone_mode,
                anchor_text=anchor_text,
            )
        except AppError as e:
            raise e

        return FileResponse(str(output_wav), media_type="audio/wav", filename="test_output.wav")

    app.state.config = config
    app.state.database = database
    app.state.jobs = jobs
    app.state.runtime = runtime
    app.state.runner = runner
    app.state.settings = settings
    return app
