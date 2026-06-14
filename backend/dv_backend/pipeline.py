import json
import html
import os
import re
import shutil
import subprocess
import time
import urllib.request
import urllib.error
import urllib.parse
import wave
from pathlib import Path

from .config import AppConfig
from .database import Database
from .errors import AppError
from .models import ErrorInfo
from .checkpoints import save_checkpoint, load_checkpoint
from .vendor import VendorManifest, VendorResolver
from .adapters.translation import GoogleFreeTranslator
from .adapters.tts import EdgeTtsAdapter


def yt_dlp_cookie_args(database: Database) -> list[str]:
    row = database.connection.execute(
        "SELECT value FROM settings WHERE key = 'cookies_browser'"
    ).fetchone()
    browser = json.loads(row["value"]) if row else "none"
    if browser == "none":
        return []
    return ["--cookies-from-browser", browser]


def normalize_douyin_url(source_url: str) -> str:
    parsed = urllib.parse.urlparse(source_url)
    modal_id = urllib.parse.parse_qs(parsed.query).get("modal_id", [None])[0]
    if parsed.netloc.endswith("douyin.com") and modal_id and modal_id.isdigit():
        return f"https://www.douyin.com/video/{modal_id}"
    return source_url


def resolve_tool_path(config: AppConfig, tool_id: str) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    vendor_dir = Path(os.environ.get("DV_VENDOR_DIR", project_root / "vendor"))
    manifest_path = Path(os.environ.get("DV_VENDOR_MANIFEST", vendor_dir / "manifest.json"))
    manifest = VendorManifest.load(manifest_path)
    tool = next((t for t in manifest.tools if t.id == tool_id), None)
    if not tool:
        raise AppError(
            500,
            ErrorInfo(
                code="TOOL_NOT_FOUND",
                message=f"Tool {tool_id} not declared in manifest.",
                action="Verify vendor/manifest.json."
            )
        )
    allow_path_tools = os.environ.get("DV_ALLOW_PATH_TOOLS") == "1"
    resolver = VendorResolver(vendor_dir, allow_path_tools=allow_path_tools)
    resolved = resolver.resolve(tool)
    if resolved.path is None:
        raise AppError(
            500,
            ErrorInfo(
                code="TOOL_RESOLUTION_FAILED",
                message=f"Required tool {tool.display_name} could not be resolved.",
                action="Make sure the tool is bundled or available on PATH."
            )
        )
    return resolved.path


def run_subprocess_with_cancel(cmd: list[str], job_id: str, runner, timeout: float = None) -> subprocess.CompletedProcess:
    # Verify that the job is not cancelled
    if runner and runner.is_cancelled(job_id):
        raise AppError(
            400,
            ErrorInfo(
                code="JOB_CANCELLED",
                message="The job was cancelled by the user.",
                action="Create a new job to start over."
            )
        )
    
    started = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    
    if runner:
        runner.register_process(job_id, proc)
        
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        duration_ms = round((time.perf_counter() - started) * 1000)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd, stdout, stderr)
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        raise AppError(
            500,
            ErrorInfo(
                code="PROCESS_TIMEOUT",
                message=f"Process timed out: {' '.join(cmd[:3])}",
                action="Verify execution environment and check for antivirus interference.",
                detail=stderr
            )
        )
    finally:
        if runner:
            runner.unregister_process(job_id)


def get_wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
        return frames / float(rate)


# OpenAI API integration
def call_openai_chat(api_base: str, api_key: str, model: str, messages: list, json_mode: bool = False) -> dict:
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "messages": messages,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
        
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            res_body = response.read().decode("utf-8")
            return json.loads(res_body)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            err_data = json.loads(body)
            err_msg = err_data.get("error", {}).get("message", body)
        except Exception:
            err_msg = body
        raise AppError(
            502,
            ErrorInfo(
                code="OPENAI_CHAT_ERROR",
                message=f"Translation API error ({e.code}).",
                action="Verify your API Key, base URL, and network connection.",
                detail=err_msg
            )
        )
    except Exception as e:
        raise AppError(
            502,
            ErrorInfo(
                code="OPENAI_CHAT_FAILED",
                message="Failed to connect to Translation API.",
                action="Check settings and try again.",
                detail=str(e)
            )
        )


def call_openai_tts(api_base: str, api_key: str, model: str, voice: str, text: str, output_path: Path) -> None:
    url = f"{api_base.rstrip('/')}/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "input": text,
        "voice": voice,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(response.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        raise AppError(
            502,
            ErrorInfo(
                code="OPENAI_TTS_ERROR",
                message=f"TTS API error ({e.code}).",
                action="Verify settings and billing status.",
                detail=body
            )
        )
    except Exception as e:
        raise AppError(
            502,
            ErrorInfo(
                code="OPENAI_TTS_FAILED",
                message="Failed to connect to TTS API.",
                action="Check your internet connection and API config.",
                detail=str(e)
            )
        )


# Steps implementation
def resolve_step(job_id: str, config: AppConfig, database: Database, runner) -> dict:
    row = database.connection.execute("SELECT source_url FROM jobs WHERE id = ?", (job_id,)).fetchone()
    source_url = normalize_douyin_url(row["source_url"])
    
    yt_dlp_path = resolve_tool_path(config, "yt_dlp")
    
    cmd = [
        str(yt_dlp_path),
        *yt_dlp_cookie_args(database),
        "--dump-single-json",
        "--flat-playlist",
        "--playlist-end", "20",
        source_url
    ]
    
    try:
        res = run_subprocess_with_cancel(cmd, job_id, runner, timeout=40)
        data = json.loads(res.stdout)
    except subprocess.CalledProcessError as e:
        raise AppError(
            500,
            ErrorInfo(
                code="YT_DLP_RESOLVE_FAILED",
                message="Failed to resolve Douyin video/channel URL.",
                action="Ensure URL is valid and public.",
                detail=e.stderr or e.stdout
            )
        )
        
    is_playlist = data.get("_type") == "playlist" or "entries" in data
    
    videos = []
    if is_playlist:
        entries = data.get("entries") or []
        for entry in entries:
            if not entry:
                continue
            video_url = entry.get("url") or f"https://www.douyin.com/video/{entry.get('id')}"
            videos.append({
                "id": entry.get("id"),
                "title": entry.get("title") or "Untitled Video",
                "url": video_url,
                "duration": entry.get("duration"),
                "thumbnail": entry.get("thumbnail") or (entry.get("thumbnails")[0].get("url") if entry.get("thumbnails") else None),
            })
    else:
        video_url = data.get("webpage_url") or source_url
        videos.append({
            "id": data.get("id"),
            "title": data.get("title") or data.get("description") or "Untitled Video",
            "url": video_url,
            "duration": data.get("duration"),
            "thumbnail": data.get("thumbnail") or (data.get("thumbnails")[0].get("url") if data.get("thumbnails") else None),
        })
        
    # Auto-select if there is only 1 video
    selected_video = videos[0] if len(videos) == 1 else None
    
    checkpoint_data = {
        "schema_version": 1,
        "job_id": job_id,
        "step_name": "resolve",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "is_playlist": is_playlist and len(videos) > 1,
        "videos": videos,
        "selected_video": selected_video
    }
    
    save_checkpoint(config.data_dir, job_id, "resolve", checkpoint_data)
    
    if selected_video:
        with database.connection:
            database.connection.execute(
                "UPDATE jobs SET title = ?, updated_at = ? WHERE id = ?",
                (selected_video["title"], time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), job_id)
            )
            
    return checkpoint_data


def download_step(job_id: str, config: AppConfig, database: Database, runner) -> dict:
    resolve_cp = load_checkpoint(config.data_dir, job_id, "resolve")
    if not resolve_cp or not resolve_cp.get("selected_video"):
        raise AppError(
            400,
            ErrorInfo(
                code="NO_VIDEO_SELECTED",
                message="No video selected for download.",
                action="Select a video from the list before downloading."
            )
        )
        
    selected = resolve_cp["selected_video"]
    video_url = selected["url"]
    
    yt_dlp_path = resolve_tool_path(config, "yt_dlp")
    ffmpeg_path = resolve_tool_path(config, "ffmpeg")
    ffmpeg_dir = ffmpeg_path.parent
    
    job_dir = config.data_dir / "jobs" / job_id
    artifacts_dir = job_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    output_mp4 = artifacts_dir / "original.mp4"
    
    cmd = [
        str(yt_dlp_path),
        *yt_dlp_cookie_args(database),
        "--ffmpeg-location", str(ffmpeg_dir),
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", str(output_mp4),
        video_url
    ]
    
    try:
        run_subprocess_with_cancel(cmd, job_id, runner)
    except subprocess.CalledProcessError as e:
        raise AppError(
            500,
            ErrorInfo(
                code="DOWNLOAD_FAILED",
                message="Failed to download video with yt-dlp.",
                action="Retry or check if the video has been removed.",
                detail=e.stderr or e.stdout
            )
        )
        
    checkpoint_data = {
        "schema_version": 1,
        "job_id": job_id,
        "step_name": "download",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_path": str(output_mp4)
    }
    save_checkpoint(config.data_dir, job_id, "download", checkpoint_data)
    return checkpoint_data


def extract_audio_step(job_id: str, config: AppConfig, database: Database, runner) -> dict:
    job_dir = config.data_dir / "jobs" / job_id
    artifacts_dir = job_dir / "artifacts"
    original_mp4 = artifacts_dir / "original.mp4"
    
    if not original_mp4.is_file():
        raise AppError(
            400,
            ErrorInfo(
                code="MISSING_VIDEO_FILE",
                message="Downloaded video file is missing.",
                action="Resume download step."
            )
        )
        
    ffmpeg_path = resolve_tool_path(config, "ffmpeg")
    original_48k = artifacts_dir / "original_48k.wav"
    audio_16k = artifacts_dir / "audio_16k.wav"
    
    # Extract 48k WAV
    cmd_48k = [
        str(ffmpeg_path), "-y",
        "-i", str(original_mp4),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "48000",
        str(original_48k)
    ]
    
    # Extract 16k WAV
    cmd_16k = [
        str(ffmpeg_path), "-y",
        "-i", str(original_mp4),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", "16000",
        str(audio_16k)
    ]
    
    try:
        run_subprocess_with_cancel(cmd_48k, job_id, runner)
        run_subprocess_with_cancel(cmd_16k, job_id, runner)
    except subprocess.CalledProcessError as e:
        raise AppError(
            500,
            ErrorInfo(
                code="AUDIO_EXTRACTION_FAILED",
                message="Failed to extract audio from video.",
                action="Ensure FFmpeg runs correctly and original.mp4 is not corrupted.",
                detail=e.stderr or e.stdout
            )
        )
        
    checkpoint_data = {
        "schema_version": 1,
        "job_id": job_id,
        "step_name": "extract_audio",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "original_48k_path": str(original_48k),
        "audio_16k_path": str(audio_16k)
    }
    save_checkpoint(config.data_dir, job_id, "extract_audio", checkpoint_data)
    return checkpoint_data


def vad_step(job_id: str, config: AppConfig, database: Database, runner) -> dict:
    job_dir = config.data_dir / "jobs" / job_id
    audio_16k = job_dir / "artifacts" / "audio_16k.wav"
    
    if not audio_16k.is_file():
        raise AppError(
            400,
            ErrorInfo(
                code="MISSING_AUDIO_FILE",
                message="Audio file for VAD is missing.",
                action="Resume extract_audio step."
            )
        )
        
    ffmpeg_path = resolve_tool_path(config, "ffmpeg")
    
    # Detect silence
    cmd = [
        str(ffmpeg_path),
        "-i", str(audio_16k),
        "-af", "silencedetect=n=-30dB:d=0.5",
        "-f", "null",
        "-"
    ]
    
    try:
        res = run_subprocess_with_cancel(cmd, job_id, runner)
    except subprocess.CalledProcessError as e:
        raise AppError(
            500,
            ErrorInfo(
                code="VAD_DETECTION_FAILED",
                message="Failed to run silence detection on audio.",
                action="Verify FFmpeg is correctly installed.",
                detail=e.stderr or e.stdout
            )
        )
        
    stderr = res.stderr
    
    # Parse total duration
    total_duration = 0.0
    duration_match = re.search(r"Duration:\s*(\d{2}):(\d{2}):(\d{2})\.(\d{2})", stderr)
    if duration_match:
        h, m, s, ms = map(int, duration_match.groups())
        total_duration = h * 3600 + m * 60 + s + ms / 100.0
    else:
        total_duration = get_wav_duration(audio_16k)
        
    # Parse silences
    starts = [float(x) for x in re.findall(r"silence_start:\s*(\d+\.?\d*)", stderr)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*(\d+\.?\d*)", stderr)]
    
    silences = []
    for i in range(min(len(starts), len(ends))):
        silences.append((starts[i], ends[i]))
    if len(starts) > len(ends):
        silences.append((starts[-1], total_duration))
        
    silences.sort()
    
    # Invert silence to get speech regions
    speech_regions = []
    current_time = 0.0
    
    for sil_start, sil_end in silences:
        if sil_start > current_time + 0.1:
            speech_regions.append({
                "start": round(current_time, 2),
                "end": round(sil_start, 2)
            })
        current_time = sil_end
        
    if total_duration > current_time + 0.1:
        speech_regions.append({
            "start": round(current_time, 2),
            "end": round(total_duration, 2)
        })
        
    checkpoint_data = {
        "schema_version": 1,
        "job_id": job_id,
        "step_name": "vad",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_duration": round(total_duration, 2),
        "speech_regions": speech_regions
    }
    
    save_checkpoint(config.data_dir, job_id, "vad", checkpoint_data)
    return checkpoint_data


def asr_step(job_id: str, config: AppConfig, database: Database, runner) -> dict:
    job_dir = config.data_dir / "jobs" / job_id
    audio_16k = job_dir / "artifacts" / "audio_16k.wav"
    
    if not audio_16k.is_file():
        raise AppError(
            400,
            ErrorInfo(
                code="MISSING_AUDIO_FILE",
                message="Audio file for ASR is missing.",
                action="Resume extract_audio step."
            )
        )
        
    rows = database.connection.execute("SELECT key, value FROM settings").fetchall()
    settings = {r["key"]: json.loads(r["value"]) for r in rows}
    
    backend = settings.get("asr_backend", settings.get("whisper_backend", "whisper_cpu"))
    model_path = settings.get("whisper_model_path", "")
    if not model_path:
        project_root = Path(__file__).resolve().parents[2]
        vendor_dir = Path(os.environ.get("DV_VENDOR_DIR", project_root / "vendor"))
        model_path = str(vendor_dir / "whisper.cpp" / "models" / "ggml-base.bin")

    if not model_path or not Path(model_path).is_file():
        raise AppError(
            400,
            ErrorInfo(
                code="MISSING_ASR_MODEL",
                message="Whisper model file not found.",
                action="Go to Settings and select a valid whisper .bin model file."
            )
        )
        
    def run_qwen3() -> dict:
        exe_path = resolve_tool_path(config, "qwen3_asr")
        json_out = audio_16k.with_suffix(".wav.json")
        if json_out.is_file():
            json_out.unlink()
            
        cmd = [
            str(exe_path),
            "-i", str(audio_16k),
            "-o", str(json_out)
        ]
        
        run_subprocess_with_cancel(cmd, job_id, runner)
        
        if not json_out.is_file():
            raise RuntimeError("Qwen3-ASR did not generate JSON output.")
            
        with open(json_out, "r", encoding="utf-8") as f:
            return json.load(f)
            
    def run_whisper(tool_id: str) -> dict:
        exe_path = resolve_tool_path(config, tool_id)
        json_out = audio_16k.with_suffix(".wav.json")
        if json_out.is_file():
            json_out.unlink()
            
        cmd = [
            str(exe_path),
            "-m", str(model_path),
            "-f", str(audio_16k),
            "-l", "zh",
            "-oj"
        ]
        
        run_subprocess_with_cancel(cmd, job_id, runner)
        
        if not json_out.is_file():
            raise RuntimeError("Whisper did not generate JSON output.")
            
        with open(json_out, "r", encoding="utf-8") as f:
            return json.load(f)
            
    segments = []
    res_data = None
    if backend == "qwen3_asr":
        try:
            res_data = run_qwen3()
        except Exception as e:
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            database.connection.execute(
                "INSERT INTO events (level, code, message, job_id, created_at) VALUES ('warning', 'QWEN3_ASR_FAILED', ?, ?, ?)",
                (f"Qwen3-ASR failed: {e}. Retrying on CPU ASR.", job_id, now)
            )
            backend = "whisper_cpu"
            
    if res_data is None:
        if backend == "whisper_vulkan":
            try:
                res_data = run_whisper("whisper_vulkan")
            except Exception as e:
                now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                database.connection.execute(
                    "INSERT INTO events (level, code, message, job_id, created_at) VALUES ('warning', 'ASR_VULKAN_FAILED', ?, ?, ?)",
                    (f"Vulkan ASR failed: {e}. Retrying on CPU ASR.", job_id, now)
                )
                res_data = run_whisper("whisper_cpu")
        else:
            res_data = run_whisper("whisper_cpu")
        
    raw_segments = []
    if "result" in res_data and "transcription" in res_data["result"]:
        raw_segments = res_data["result"]["transcription"]
    elif "transcription" in res_data:
        raw_segments = res_data["transcription"]
    elif "segments" in res_data:
        raw_segments = res_data["segments"]
        
    for raw in raw_segments:
        if "offsets" in raw:
            offsets = raw.get("offsets", {})
            start = offsets.get("from", 0) / 1000.0
            end = offsets.get("to", 0) / 1000.0
        else:
            start = float(raw.get("start", 0))
            end = float(raw.get("end", 0))
        segments.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "text": raw.get("text", "").strip()
        })

    segments = [segment for segment in segments if segment["text"]]
    if not segments:
        raise AppError(
            422,
            ErrorInfo(
                code="EMPTY_ASR_TRANSCRIPTION",
                message="ASR completed without detecting any spoken text.",
                action="Verify the source audio and ASR model, then resume the ASR step."
            )
        )
        
    checkpoint_data = {
        "schema_version": 1,
        "job_id": job_id,
        "step_name": "asr",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "segments": segments
    }
    save_checkpoint(config.data_dir, job_id, "asr", checkpoint_data)
    return checkpoint_data


def normalize_segments_step(job_id: str, config: AppConfig, database: Database, runner) -> dict:
    asr_cp = load_checkpoint(config.data_dir, job_id, "asr")
    vad_cp = load_checkpoint(config.data_dir, job_id, "vad")
    
    if not asr_cp or not vad_cp:
        raise AppError(
            400,
            ErrorInfo(
                code="MISSING_UPSTREAM_CHECKPOINTS",
                message="ASR or VAD checkpoints are missing.",
                action="Verify earlier steps are completed."
            )
        )
        
    raw_segments = asr_cp.get("segments", [])
    total_duration = vad_cp.get("total_duration", 0.0)
    
    segments = []
    for seg in raw_segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        segments.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": text
        })
        
    segments.sort(key=lambda x: x["start"])
    
    for i in range(len(segments) - 1):
        curr = segments[i]
        nxt = segments[i+1]
        if curr["end"] > nxt["start"]:
            curr["end"] = nxt["start"]
            
    normalized = []
    for i in range(len(segments)):
        curr = segments[i]
        orig_dur = curr["end"] - curr["start"]
        
        if orig_dur <= 0.05:
            orig_dur = 0.5
            curr["end"] = curr["start"] + orig_dur
            
        if i < len(segments) - 1:
            budget = segments[i+1]["start"] - curr["start"]
        else:
            budget = total_duration - curr["start"]
            
        if budget < orig_dur:
            budget = orig_dur
            
        normalized.append({
            "index": i,
            "start": round(curr["start"], 2),
            "end": round(curr["end"], 2),
            "text": curr["text"],
            "original_duration": round(orig_dur, 2),
            "duration_budget": round(budget, 2),
            "translation": None,
            "tts_duration": None
        })
        
    checkpoint_data = {
        "schema_version": 1,
        "job_id": job_id,
        "step_name": "normalize_segments",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "segments": normalized
    }
    save_checkpoint(config.data_dir, job_id, "normalize_segments", checkpoint_data)
    return checkpoint_data


def translate_step(job_id: str, config: AppConfig, database: Database, runner) -> dict:
    norm_cp = load_checkpoint(config.data_dir, job_id, "normalize_segments")
    if not norm_cp:
        raise AppError(
            400,
            ErrorInfo(
                code="MISSING_SEGMENTS",
                message="Normalized segments checkpoint is missing.",
                action="Resume normalize_segments step."
            )
        )
        
    segments = norm_cp.get("segments", [])
    if not segments:
        checkpoint_data = {
            "schema_version": 1,
            "job_id": job_id,
            "step_name": "translate",
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "segments": []
        }
        save_checkpoint(config.data_dir, job_id, "translate", checkpoint_data)
        return checkpoint_data
        
    rows = database.connection.execute("SELECT key, value FROM settings").fetchall()
    settings = {r["key"]: json.loads(r["value"]) for r in rows}
    
    if settings.get("translation_backend", "google_free") != "google_free":
        raise AppError(
            400,
            ErrorInfo(
                code="UNSUPPORTED_TRANSLATION_BACKEND",
                message="The selected translation backend is not available.",
                action="Choose Google Translate Free in Settings."
            )
        )

    translated = GoogleFreeTranslator().translate(
        [segment["text"] for segment in segments],
        source=settings.get("translation_source_language", "zh-CN"),
        target=settings.get("translation_target_language", "vi"),
    )
    for segment, translation in zip(segments, translated, strict=True):
        segment["translation"] = translation
        
    checkpoint_data = {
        "schema_version": 1,
        "job_id": job_id,
        "step_name": "translate",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "segments": segments
    }
    save_checkpoint(config.data_dir, job_id, "translate", checkpoint_data)
    return checkpoint_data


def tts_step(job_id: str, config: AppConfig, database: Database, runner) -> dict:
    trans_cp = load_checkpoint(config.data_dir, job_id, "translate")
    if not trans_cp:
        raise AppError(
            400,
            ErrorInfo(
                code="MISSING_TRANSLATIONS",
                message="Translations checkpoint is missing.",
                action="Resume translate step."
            )
        )
        
    segments = trans_cp.get("segments", [])
    job_dir = config.data_dir / "jobs" / job_id
    artifacts_dir = job_dir / "artifacts"
    tts_dir = artifacts_dir / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)
    
    rows = database.connection.execute("SELECT key, value FROM settings").fetchall()
    settings = {r["key"]: json.loads(r["value"]) for r in rows}
    
    tts_backend = settings.get("tts_backend", "edge")
    ffmpeg_path = resolve_tool_path(config, "ffmpeg")
    
    for s in segments:
        idx = s["index"]
        text = s["translation"]
        
        raw_tts = tts_dir / f"tts_raw_{idx}.wav"
        if raw_tts.is_file():
            raw_tts.unlink()
            
        final_tts = tts_dir / f"tts_{idx}.wav"
        if final_tts.is_file():
            final_tts.unlink()
            
        if tts_backend == "edge":
            mp3_path = tts_dir / f"tts_{idx}.mp3"
            if mp3_path.is_file():
                mp3_path.unlink()
            EdgeTtsAdapter().synthesize(
                text,
                mp3_path,
                voice=settings.get("edge_tts_voice", "vi-VN-HoaiMyNeural"),
                rate=settings.get("edge_tts_rate", "+0%"),
                pitch=settings.get("edge_tts_pitch", "+0Hz"),
                volume=settings.get("edge_tts_volume", "+0%"),
            )

            cmd_conv = [
                str(ffmpeg_path), "-y",
                "-i", str(mp3_path),
                "-ar", "48000",
                "-ac", "2",
                "-c:a", "pcm_s16le",
                str(final_tts)
            ]
            run_subprocess_with_cancel(cmd_conv, job_id, runner)
            if mp3_path.is_file():
                mp3_path.unlink()
        elif tts_backend == "api":
            api_base = settings.get("tts_api_base", "https://api.openai.com/v1")
            api_key = settings.get("tts_api_key", "")
            voice = settings.get("tts_voice", "alloy")
            
            if not api_key:
                raise AppError(
                    400,
                    ErrorInfo(
                        code="MISSING_TTS_API_KEY",
                        message="TTS API key is missing.",
                        action="Go to Settings and enter your TTS/OpenAI API key."
                    )
                )
            
            mp3_path = tts_dir / f"tts_{idx}.mp3"
            if mp3_path.is_file():
                mp3_path.unlink()
            call_openai_tts(api_base, api_key, "tts-1", voice, text, mp3_path)
            
            cmd_conv = [
                str(ffmpeg_path), "-y",
                "-i", str(mp3_path),
                "-ar", "48000",
                "-ac", "2",
                "-c:a", "pcm_s16le",
                str(final_tts)
            ]
            run_subprocess_with_cancel(cmd_conv, job_id, runner)
            if mp3_path.is_file():
                mp3_path.unlink()
                
        else:
            piper_path = resolve_tool_path(config, "piper")
            piper_model = settings.get("piper_model_path", "")
            
            if not piper_model or not Path(piper_model).is_file():
                raise AppError(
                    400,
                    ErrorInfo(
                        code="MISSING_PIPER_MODEL",
                        message="Piper ONNX model not found.",
                        action="Go to Settings and select a valid Piper .onnx model."
                    )
                )
                
            cmd = [
                str(piper_path),
                "--model", str(piper_model),
                "--output_file", str(raw_tts)
            ]
            
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            
            if runner:
                runner.register_process(job_id, proc)
                
            try:
                stdout, stderr = proc.communicate(input=text, timeout=30)
                if proc.returncode != 0:
                    raise AppError(
                        500,
                        ErrorInfo(
                            code="PIPER_FAILED",
                            message="Piper process failed to generate speech.",
                            action="Check that Piper model matches configuration.",
                            detail=stderr
                        )
                    )
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                raise AppError(
                    500,
                    ErrorInfo(
                        code="PIPER_TIMEOUT",
                        message="Piper TTS speech generation timed out.",
                        action="Choose a shorter segment or check system resources."
                    )
                )
            finally:
                if runner:
                    runner.unregister_process(job_id)
                    
            cmd_conv = [
                str(ffmpeg_path), "-y",
                "-i", str(raw_tts),
                "-ar", "48000",
                "-ac", "2",
                "-c:a", "pcm_s16le",
                str(final_tts)
            ]
            run_subprocess_with_cancel(cmd_conv, job_id, runner)
            if raw_tts.is_file():
                raw_tts.unlink()
                
        s["tts_duration"] = round(get_wav_duration(final_tts), 2)
        
    checkpoint_data = {
        "schema_version": 1,
        "job_id": job_id,
        "step_name": "tts",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "segments": segments
    }
    save_checkpoint(config.data_dir, job_id, "tts", checkpoint_data)
    return checkpoint_data


def duration_repair_step(job_id: str, config: AppConfig, database: Database, runner) -> dict:
    tts_cp = load_checkpoint(config.data_dir, job_id, "tts")
    if not tts_cp:
        raise AppError(
            400,
            ErrorInfo(
                code="MISSING_TTS_SEGMENTS",
                message="TTS checkpoint is missing.",
                action="Resume tts step."
            )
        )
        
    segments = tts_cp.get("segments", [])
    job_dir = config.data_dir / "jobs" / job_id
    artifacts_dir = job_dir / "artifacts"
    tts_dir = artifacts_dir / "tts"
    
    rows = database.connection.execute("SELECT key, value FROM settings").fetchall()
    settings = {r["key"]: json.loads(r["value"]) for r in rows}
    
    ffmpeg_path = resolve_tool_path(config, "ffmpeg")
    
    for s in segments:
        idx = s["index"]
        budget = s["duration_budget"]
        tts_dur = s["tts_duration"] or 0.0
        
        orig_file = tts_dir / f"tts_{idx}.wav"
        repaired_file = tts_dir / f"tts_repaired_{idx}.wav"
        if repaired_file.is_file():
            repaired_file.unlink()
            
        if tts_dur <= budget + 0.1:
            import shutil
            shutil.copy(orig_file, repaired_file)
            s["repaired_method"] = "none"
            s["repaired_duration"] = tts_dur
            continue
            
        shortened = False
        api_key = settings.get("openai_api_key", "")
        if api_key:
            api_base = settings.get("openai_api_base", "https://api.openai.com/v1")
            model = settings.get("openai_model", "gpt-4o-mini")
            
            system_prompt = (
                "You are an expert video dubbing editor.\n"
                "The translated Vietnamese subtitle is too long and cannot fit in the budget.\n"
                "You must shorten the subtitle significantly while preserving the core meaning.\n"
                "Return only the shortened Vietnamese translation, nothing else. Do not wrap in quotes."
            )
            user_prompt = f"Original Vietnamese translation: '{s['translation']}'\nTarget duration budget: {budget}s\nCurrent speech duration: {tts_dur}s"
            
            try:
                res = call_openai_chat(api_base, api_key, model, [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ])
                new_translation = res["choices"][0]["message"]["content"].strip().strip("\"'")
                
                temp_wav = tts_dir / f"tts_temp_{idx}.wav"
                if temp_wav.is_file():
                    temp_wav.unlink()
                    
                tts_backend = settings.get("tts_backend", "api")
                if tts_backend == "api":
                    voice = settings.get("tts_voice", "alloy")
                    mp3_path = tts_dir / f"tts_temp_{idx}.mp3"
                    call_openai_tts(api_base, api_key, "tts-1", voice, new_translation, mp3_path)
                    
                    cmd_conv = [
                        str(ffmpeg_path), "-y",
                        "-i", str(mp3_path),
                        "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
                        str(temp_wav)
                    ]
                    run_subprocess_with_cancel(cmd_conv, job_id, runner)
                    if mp3_path.is_file():
                        mp3_path.unlink()
                else:
                    piper_path = resolve_tool_path(config, "piper")
                    piper_model = settings.get("piper_model_path", "")
                    
                    raw_temp = tts_dir / f"tts_temp_raw_{idx}.wav"
                    cmd = [str(piper_path), "--model", str(piper_model), "--output_file", str(raw_temp)]
                    
                    proc = subprocess.Popen(
                        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, encoding="utf-8", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                    )
                    if runner:
                        runner.register_process(job_id, proc)
                    try:
                        proc.communicate(input=new_translation, timeout=30)
                    finally:
                        if runner:
                            runner.unregister_process(job_id)
                            
                    cmd_conv = [
                        str(ffmpeg_path), "-y",
                        "-i", str(raw_temp),
                        "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
                        str(temp_wav)
                    ]
                    run_subprocess_with_cancel(cmd_conv, job_id, runner)
                    if raw_temp.is_file():
                        raw_temp.unlink()
                        
                new_dur = get_wav_duration(temp_wav)
                if new_dur <= budget + 0.1:
                    os.replace(temp_wav, repaired_file)
                    s["translation"] = new_translation
                    s["repaired_method"] = "llm_shorten"
                    s["repaired_duration"] = round(new_dur, 2)
                    shortened = True
                else:
                    if temp_wav.is_file():
                        temp_wav.unlink()
            except Exception:
                pass
                
        if not shortened:
            speed_factor = tts_dur / budget
            speed_factor = min(1.4, max(1.0, speed_factor))
            
            cmd = [
                str(ffmpeg_path), "-y",
                "-i", str(orig_file),
                "-filter:a", f"atempo={speed_factor}",
                str(repaired_file)
            ]
            run_subprocess_with_cancel(cmd, job_id, runner)
            
            s["repaired_method"] = f"time_stretch_{round(speed_factor, 2)}x"
            s["repaired_duration"] = round(get_wav_duration(repaired_file), 2)
            
    checkpoint_data = {
        "schema_version": 1,
        "job_id": job_id,
        "step_name": "duration_repair",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "segments": segments
    }
    save_checkpoint(config.data_dir, job_id, "duration_repair", checkpoint_data)
    return checkpoint_data


def mix_step(job_id: str, config: AppConfig, database: Database, runner) -> dict:
    repair_cp = load_checkpoint(config.data_dir, job_id, "duration_repair")
    audio_cp = load_checkpoint(config.data_dir, job_id, "extract_audio")
    
    if not repair_cp or not audio_cp:
        raise AppError(
            400,
            ErrorInfo(
                code="MISSING_REPAIR_OR_AUDIO",
                message="Duration repair or original audio checkpoints are missing.",
                action="Verify upstream steps."
            )
        )
        
    original_48k = Path(audio_cp["original_48k_path"])
    segments = repair_cp.get("segments", [])
    
    job_dir = config.data_dir / "jobs" / job_id
    artifacts_dir = job_dir / "artifacts"
    tts_dir = artifacts_dir / "tts"
    
    narration_wav = artifacts_dir / "narration.wav"
    mixed_wav = artifacts_dir / "mixed.wav"
    
    with wave.open(str(original_48k), "rb") as orig:
        total_frames = orig.getnframes()
        
    frame_rate = 48000
    audio_data = bytearray(total_frames * 4)
    
    for seg in segments:
        idx = seg["index"]
        start_time = seg["start"]
        seg_path = tts_dir / f"tts_repaired_{idx}.wav"
        if not seg_path.is_file():
            seg_path = tts_dir / f"tts_{idx}.wav"
        if not seg_path.is_file():
            continue
            
        with wave.open(str(seg_path), "rb") as f_seg:
            seg_frames = f_seg.readframes(f_seg.getnframes())
            
        start_frame = int(start_time * frame_rate)
        start_byte = start_frame * 4
        
        if start_byte < len(audio_data):
            write_len = min(len(seg_frames), len(audio_data) - start_byte)
            audio_data[start_byte:start_byte+write_len] = seg_frames[:write_len]
            
    with wave.open(str(narration_wav), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(frame_rate)
        out.writeframes(audio_data)

    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    vietnamese_narration = output_dir / "vietnamese_narration.wav"
    shutil.copyfile(narration_wav, vietnamese_narration)
        
    ffmpeg_path = resolve_tool_path(config, "ffmpeg")
    
    cmd_mix = [
        str(ffmpeg_path), "-y",
        "-i", str(original_48k),
        "-i", str(narration_wav),
        "-filter_complex",
        (
            "[0:a]volume=0.35[bg];[1:a]volume=1.5[fg];"
            "[bg][fg]sidechaincompress=threshold=0.02:ratio=8:"
            "attack=20:release=400[ducked];"
            "[ducked][fg]amix=inputs=2:duration=first:dropout_transition=0[mixed]"
        ),
        "-map", "[mixed]",
        str(mixed_wav)
    ]
    
    try:
        run_subprocess_with_cancel(cmd_mix, job_id, runner)
    except subprocess.CalledProcessError as e:
        raise AppError(
            500,
            ErrorInfo(
                code="MIXING_FAILED",
                message="Failed to mix original background with narration.",
                action="Verify FFmpeg audio filters.",
                detail=e.stderr or e.stdout
            )
        )
        
    checkpoint_data = {
        "schema_version": 1,
        "job_id": job_id,
        "step_name": "mix",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "narration_wav_path": str(narration_wav),
        "mixed_wav_path": str(mixed_wav),
        "vietnamese_narration_path": str(vietnamese_narration)
    }
    save_checkpoint(config.data_dir, job_id, "mix", checkpoint_data)
    return checkpoint_data


def render_step(job_id: str, config: AppConfig, database: Database, runner) -> dict:
    mix_cp = load_checkpoint(config.data_dir, job_id, "mix")
    download_cp = load_checkpoint(config.data_dir, job_id, "download")
    
    if not mix_cp or not download_cp:
        raise AppError(
            400,
            ErrorInfo(
                code="MISSING_UPSTREAM_CHECKPOINTS",
                message="Mix or Download checkpoints are missing.",
                action="Verify upstream steps."
            )
        )
        
    original_mp4 = Path(download_cp["output_path"])
    mixed_wav = Path(mix_cp["mixed_wav_path"])
    
    job_dir = config.data_dir / "jobs" / job_id
    artifacts_dir = job_dir / "artifacts"
    normalized_wav = artifacts_dir / "normalized.wav"
    
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    final_mp4 = output_dir / "dubbed.mp4"
    
    ffmpeg_path = resolve_tool_path(config, "ffmpeg")
    
    cmd_norm = [
        str(ffmpeg_path), "-y",
        "-i", str(mixed_wav),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", "48000",
        "-c:a", "pcm_s16le",
        str(normalized_wav)
    ]
    
    cmd_render = [
        str(ffmpeg_path), "-y",
        "-i", str(original_mp4),
        "-i", str(normalized_wav),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(final_mp4)
    ]
    
    try:
        run_subprocess_with_cancel(cmd_norm, job_id, runner)
        run_subprocess_with_cancel(cmd_render, job_id, runner)
    except subprocess.CalledProcessError as e:
        raise AppError(
            500,
            ErrorInfo(
                code="RENDER_FAILED",
                message="Failed to render final MP4 video file.",
                action="Ensure original video format is compatible.",
                detail=e.stderr or e.stdout
            )
        )
        
    checkpoint_data = {
        "schema_version": 1,
        "job_id": job_id,
        "step_name": "render",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_path": str(final_mp4)
    }
    save_checkpoint(config.data_dir, job_id, "render", checkpoint_data)
    return checkpoint_data


def qc_step(job_id: str, config: AppConfig, database: Database, runner) -> dict:
    repair_cp = load_checkpoint(config.data_dir, job_id, "duration_repair")
    norm_cp = load_checkpoint(config.data_dir, job_id, "normalize_segments")
    render_cp = load_checkpoint(config.data_dir, job_id, "render")
    
    if not repair_cp or not norm_cp or not render_cp:
        raise AppError(
            400,
            ErrorInfo(
                code="MISSING_UPSTREAM_CHECKPOINTS",
                message="Duration repair, Normalized segments, or Render checkpoints are missing.",
                action="Verify upstream steps."
            )
        )
        
    segments = repair_cp.get("segments", [])
    output_path = render_cp["output_path"]
    
    total_segments = len(segments)
    repaired_segments = 0
    shortened_segments = 0
    stretched_segments = 0
    warnings = []
    
    for s in segments:
        method = s.get("repaired_method", "none")
        if method != "none":
            repaired_segments += 1
            if "llm_shorten" in method:
                shortened_segments += 1
            elif "time_stretch" in method:
                stretched_segments += 1
            warnings.append({
                "segment_index": s.get("index"),
                "method": method,
                "duration_budget": s.get("duration_budget"),
                "repaired_duration": s.get("repaired_duration"),
            })
                
    checkpoint_data = {
        "schema_version": 1,
        "job_id": job_id,
        "step_name": "qc",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_segments": total_segments,
        "repaired_count": repaired_segments,
        "shortened_count": shortened_segments,
        "stretched_count": stretched_segments,
        "warnings": warnings,
        "output_video_path": output_path
    }
    
    save_checkpoint(config.data_dir, job_id, "qc", checkpoint_data)
    
    artifacts_qc = Path(config.data_dir) / "jobs" / job_id / "artifacts" / "qc_report.json"
    artifacts_qc.parent.mkdir(parents=True, exist_ok=True)
    with open(artifacts_qc, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

    warning_rows = "".join(
        "<tr>"
        f"<td>{warning.get('segment_index')}</td>"
        f"<td>{html.escape(str(warning.get('method')))}</td>"
        f"<td>{warning.get('duration_budget')}</td>"
        f"<td>{warning.get('repaired_duration')}</td>"
        "</tr>"
        for warning in warnings
    ) or "<tr><td colspan='4'>No timing warnings</td></tr>"
    artifacts_html = artifacts_qc.with_suffix(".html")
    artifacts_html.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Douyin Vietnamizer QC Report</title></head><body>"
        "<h1>Douyin Vietnamizer QC Report</h1>"
        f"<p>Total segments: {total_segments}</p>"
        f"<p>Output: {html.escape(str(output_path))}</p>"
        "<table><thead><tr><th>Segment</th><th>Repair</th><th>Budget</th>"
        f"<th>Result</th></tr></thead><tbody>{warning_rows}</tbody></table>"
        "</body></html>",
        encoding="utf-8",
    )
        
    return checkpoint_data
