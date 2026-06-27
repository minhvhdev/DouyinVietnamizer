import json
import logging
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
from .checkpoints import load_checkpoint, save_checkpoint
from .checkpoint_compat import ASR_ALIGNMENT_SCHEMA_VERSION
from .vendor import VendorManifest, VendorResolver
from .adapters.translation import GoogleFreeTranslator
from .adapters.tts import VOXCPM_INSTRUCT_PREFIX, create_tts_adapter
from .adapters.asr import reset_model_cache, transcribe_audio
from .adapters.separation import MIX_MODE_SEPARATE, separate_vocals
from .adapters.subtitles import ffmpeg_subtitles_filter, probe_video_dimensions, write_ass_file
from .adapters.gemini import GeminiKeyPool, GeminiTranslator
from .source_urls import (
    fallback_playlist_video_url,
    is_douyin_user_profile_url,
    normalize_source_url,
)

logger = logging.getLogger(__name__)

DEFAULT_EXACT_TIMING_TOLERANCE_MS = 40
DEFAULT_EXACT_TIMING_ENABLED = True
DEFAULT_EXACT_TIMING_MAX_STRETCH = 1.8


def yt_dlp_cookie_args(database: Database) -> list[str]:
    row = database.connection.execute(
        "SELECT value FROM settings WHERE key = 'cookies_browser'"
    ).fetchone()
    browser = json.loads(row["value"]) if row else "none"
    if browser == "none":
        return []
    return ["--cookies-from-browser", browser]


def normalize_douyin_url(source_url: str) -> str:
    return normalize_source_url(source_url)


def save_setting(database: Database, key: str, value) -> None:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with database.connection:
        database.connection.execute(
            """
            INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, json.dumps(value), now),
        )


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
    allow_path_tools = os.environ.get("DV_ALLOW_PATH_TOOLS", "1") == "1"
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
    source_url = normalize_source_url(row["source_url"])

    if is_douyin_user_profile_url(source_url):
        raise AppError(
            422,
            ErrorInfo(
                code="DOUYIN_USER_URL_NOT_SUPPORTED",
                message="Douyin user profile URLs cannot be listed with yt-dlp.",
                action=(
                    "Use a single video link (douyin.com/video/ID) or a short share link. "
                    "Channel/user listing is not supported yet."
                ),
            ),
        )
    
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
                message="Failed to resolve video URL.",
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
            video_url = fallback_playlist_video_url(entry, source_url)
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

    if runner and runner.is_cancelled(job_id):
        raise AppError(
            409,
            ErrorInfo(
                code="JOB_CANCELLED",
                message="The job was cancelled by the user.",
                action="Create a new job if you still want to process this video."
            )
        )
        
    rows = database.connection.execute("SELECT key, value FROM settings").fetchall()
    settings = {r["key"]: json.loads(r["value"]) for r in rows}

    project_root = Path(__file__).resolve().parents[2]
    vendor_dir = Path(os.environ.get("DV_VENDOR_DIR", project_root / "vendor"))

    asr_result = transcribe_audio(
        audio_16k,
        vendor_dir=vendor_dir,
        asr_model=str(settings.get("qwen3_asr_model", "") or ""),
        aligner_model=str(settings.get("qwen3_aligner_model", "") or ""),
        device=str(settings.get("qwen3_device", "cuda:0") or "cuda:0"),
        language="Chinese",
        speaker_diarization=False,
        include_alignment=True,
    )

    if isinstance(asr_result, dict):
        segments = asr_result.get("segments", [])
        aligned_units = asr_result.get("aligned_units", [])
    else:
        segments = asr_result
        aligned_units = []

    if not segments:
        raise AppError(
            422,
            ErrorInfo(
                code="EMPTY_ASR_TRANSCRIPTION",
                message="ASR completed without detecting any spoken text.",
                action="Verify the source audio and Qwen3-ASR model, then resume the ASR step."
            )
        )
        
    checkpoint_data = {
        "schema_version": ASR_ALIGNMENT_SCHEMA_VERSION,
        "job_id": job_id,
        "step_name": "asr",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "segments": segments,
        "aligned_units": aligned_units,
        "alignment_required_for_diarization": False,
    }
    save_checkpoint(config.data_dir, job_id, "asr", checkpoint_data)
    reset_model_cache()
    return checkpoint_data


def _split_long_asr_segments_with_vad(
    raw_segments: list[dict],
    speech_regions: list[dict],
    *,
    max_segment_seconds: float = 20.0,
) -> list[dict]:
    if not raw_segments or not speech_regions:
        return raw_segments

    split_segments: list[dict] = []
    for segment in raw_segments:
        text = str(segment.get("text") or "").strip()
        start = float(segment.get("start", 0.0) or 0.0)
        end = float(segment.get("end", start) or start)
        if not text or end - start <= max_segment_seconds:
            split_segments.append(segment)
            continue

        overlapping_regions = [
            {
                "start": max(start, float(region.get("start", 0.0) or 0.0)),
                "end": min(end, float(region.get("end", 0.0) or 0.0)),
            }
            for region in speech_regions
            if float(region.get("end", 0.0) or 0.0) > start
            and float(region.get("start", 0.0) or 0.0) < end
        ]
        overlapping_regions = [region for region in overlapping_regions if region["end"] > region["start"]]
        if len(overlapping_regions) < 2:
            split_segments.append(segment)
            continue

        total_region_duration = sum(region["end"] - region["start"] for region in overlapping_regions)
        cursor = 0
        for index, region in enumerate(overlapping_regions):
            if index == len(overlapping_regions) - 1:
                chunk_text = text[cursor:].strip()
            else:
                ratio = (region["end"] - region["start"]) / total_region_duration
                next_cursor = max(cursor + 1, min(len(text), round(cursor + len(text[cursor:]) * ratio)))
                chunk_text = text[cursor:next_cursor].strip()
                cursor = next_cursor
            if not chunk_text:
                continue
            split_segment = dict(segment)
            split_segment.update(
                {
                    "start": round(region["start"], 2),
                    "end": round(region["end"], 2),
                    "text": chunk_text,
                }
            )
            split_segments.append(split_segment)

    return split_segments


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
        
    raw_segments = _split_long_asr_segments_with_vad(
        asr_cp.get("segments", []),
        vad_cp.get("speech_regions", []),
    )
    total_duration = vad_cp.get("total_duration", 0.0)
    
    segments = []
    for seg in raw_segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        segments.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": text,
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
            "tts_duration": None,
            **({"speaker_id": curr["speaker_id"]} if curr.get("speaker_id") is not None else {}),
            **(
                {"speaker_confidence": curr["speaker_confidence"]}
                if curr.get("speaker_confidence") is not None
                else {}
            ),
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


def _translate_texts(
    settings: dict,
    database: Database,
    texts: list[str],
    *,
    source_lang: str,
    target_lang: str,
) -> list[str]:
    if not texts:
        return []

    translation_backend = settings.get("translation_backend", "google_free")
    if translation_backend == "gemini":
        key_pool = GeminiKeyPool(
            settings.get("gemini_api_keys", []),
            cursor=int(settings.get("gemini_key_cursor", 0)),
        )
        translator = GeminiTranslator(
            key_pool,
            model=settings.get("gemini_translation_model", "gemini-2.5-flash"),
        )
        translated = translator.translate(texts, source=source_lang, target=target_lang)
        save_setting(database, "gemini_key_cursor", translator.key_pool.cursor)
        return translated

    return GoogleFreeTranslator().translate(
        texts,
        source=source_lang,
        target=target_lang,
    )


def _translate_job_title(
    job_id: str,
    database: Database,
    settings: dict,
    *,
    source_lang: str,
    target_lang: str,
) -> str | None:
    row = database.connection.execute(
        "SELECT title FROM jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    original_title = (row["title"] or "").strip() if row else ""
    if not original_title:
        return None

    title_vi = _translate_texts(
        settings,
        database,
        [original_title],
        source_lang=source_lang,
        target_lang=target_lang,
    )[0]
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with database.connection:
        database.connection.execute(
            "UPDATE jobs SET title_vi = ?, updated_at = ? WHERE id = ?",
            (title_vi, now, job_id),
        )
    return title_vi


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
    rows = database.connection.execute("SELECT key, value FROM settings").fetchall()
    settings = {r["key"]: json.loads(r["value"]) for r in rows}

    translation_backend = settings.get("translation_backend", "google_free")
    if translation_backend not in {"google_free", "gemini"}:
        raise AppError(
            400,
            ErrorInfo(
                code="UNSUPPORTED_TRANSLATION_BACKEND",
                message="The selected translation backend is not available.",
                action="Choose Google Translate Free or Gemini in Settings."
            )
        )

    source_lang = settings.get("translation_source_language", "zh-CN")
    target_lang = settings.get("translation_target_language", "vi")
    title_vi = _translate_job_title(
        job_id,
        database,
        settings,
        source_lang=source_lang,
        target_lang=target_lang,
    )

    if not segments:
        checkpoint_data = {
            "schema_version": 1,
            "job_id": job_id,
            "step_name": "translate",
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "title_vi": title_vi,
            "segments": []
        }
        save_checkpoint(config.data_dir, job_id, "translate", checkpoint_data)
        return checkpoint_data

    texts = [segment["text"] for segment in segments]
    translated = _translate_texts(
        settings,
        database,
        texts,
        source_lang=source_lang,
        target_lang=target_lang,
    )
    for segment, translation in zip(segments, translated, strict=True):
        segment["translation"] = translation

    checkpoint_data = {
        "schema_version": 1,
        "job_id": job_id,
        "step_name": "translate",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "title_vi": title_vi,
        "segments": segments
    }
    save_checkpoint(config.data_dir, job_id, "translate", checkpoint_data)
    return checkpoint_data


def _default_tts_voice(settings: dict) -> str:
    instruct = str(settings.get("voxcpm_instruct") or "").strip()
    if instruct:
        return f"{VOXCPM_INSTRUCT_PREFIX}{instruct}"
    ref_audio = str(settings.get("voxcpm_ref_audio") or "").strip()
    if ref_audio:
        return ref_audio
    return "auto"


def _synthesize_segment_tts(
    settings: dict,
    *,
    text: str,
    output_path: Path,
    segment: dict,
    config: AppConfig,
    runner,
) -> None:
    voice = _default_tts_voice(settings)
    ref_text = str(segment.get("text") or "").strip() or None
    last_error: AppError | None = None
    for attempt in range(2):
        if attempt:
            reset_model_cache()
            try:
                import gc
                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        try:
            create_tts_adapter(settings, data_dir=config.data_dir, runner=runner).synthesize(
                text,
                output_path,
                voice=voice,
                ref_text=ref_text,
            )
            return
        except AppError as error:
            last_error = error
            if not error.info.retryable or attempt:
                raise
    if last_error is not None:
        raise last_error


def _convert_tts_to_final_wav(
    ffmpeg_path: Path,
    source_path: Path,
    final_tts: Path,
    job_id: str,
    runner,
) -> None:
    cmd_conv = [
        str(ffmpeg_path), "-y",
        "-i", str(source_path),
        "-ar", "48000",
        "-ac", "2",
        "-c:a", "pcm_s16le",
        str(final_tts),
    ]
    run_subprocess_with_cancel(cmd_conv, job_id, runner)


def _build_atempo_chain(speed_factor: float) -> str:
    factor = max(0.1, float(speed_factor))
    filters: list[str] = []
    while factor > 2.0:
        filters.append("atempo=2.0")
        factor /= 2.0
    while factor < 0.5:
        filters.append("atempo=0.5")
        factor *= 2.0
    filters.append(f"atempo={factor:.5f}")
    return ",".join(filters)


def _run_ffmpeg_audio_filter(
    ffmpeg_path: Path,
    input_path: Path,
    output_path: Path,
    *,
    filter_expr: str,
    job_id: str,
    runner,
) -> None:
    cmd = [
        str(ffmpeg_path), "-y",
        "-i", str(input_path),
        "-filter:a", filter_expr,
        str(output_path),
    ]
    run_subprocess_with_cancel(cmd, job_id, runner)


def _normalize_exact_timing_settings(settings: dict) -> tuple[bool, float, float]:
    enabled = bool(settings.get("exact_timing_enabled", DEFAULT_EXACT_TIMING_ENABLED))
    tolerance_ms = settings.get("exact_timing_tolerance_ms", DEFAULT_EXACT_TIMING_TOLERANCE_MS)
    max_stretch = settings.get("exact_timing_max_stretch", DEFAULT_EXACT_TIMING_MAX_STRETCH)
    try:
        tolerance_ms = float(tolerance_ms)
    except (TypeError, ValueError):
        tolerance_ms = float(DEFAULT_EXACT_TIMING_TOLERANCE_MS)
    try:
        max_stretch = float(max_stretch)
    except (TypeError, ValueError):
        max_stretch = float(DEFAULT_EXACT_TIMING_MAX_STRETCH)
    tolerance_sec = max(0.0, tolerance_ms / 1000.0)
    max_stretch = max(1.0, min(3.0, max_stretch))
    return enabled, tolerance_sec, max_stretch


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
    ffmpeg_path = resolve_tool_path(config, "ffmpeg")
    reset_model_cache()
    
    for s in segments:
        idx = s["index"]
        text = s["translation"]
        
        raw_tts = tts_dir / f"tts_raw_{idx}.wav"
        if raw_tts.is_file():
            raw_tts.unlink()
            
        final_tts = tts_dir / f"tts_{idx}.wav"
        if final_tts.is_file():
            final_tts.unlink()

        _synthesize_segment_tts(
            settings,
            text=text,
            output_path=raw_tts,
            segment=s,
            config=config,
            runner=runner,
        )
        _convert_tts_to_final_wav(ffmpeg_path, raw_tts, final_tts, job_id, runner)
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
    exact_enabled, tolerance_sec, max_stretch = _normalize_exact_timing_settings(settings)
    
    ffmpeg_path = resolve_tool_path(config, "ffmpeg")
    
    for s in segments:
        idx = s["index"]
        budget = s["duration_budget"]
        tts_dur = s["tts_duration"] or 0.0
        
        orig_file = tts_dir / f"tts_{idx}.wav"
        repaired_file = tts_dir / f"tts_repaired_{idx}.wav"
        if repaired_file.is_file():
            repaired_file.unlink()
            
        if not exact_enabled and tts_dur <= budget + 0.1:
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

                raw_temp = tts_dir / f"tts_temp_raw_{idx}.wav"
                tts_adapter = create_tts_adapter(settings, data_dir=config.data_dir, runner=runner)
                ref_text = str(s.get("text") or "").strip() or None
                tts_adapter.synthesize(
                    new_translation,
                    raw_temp,
                    voice=_default_tts_voice(settings),
                    ref_text=ref_text,
                )
                _convert_tts_to_final_wav(ffmpeg_path, raw_temp, temp_wav, job_id, runner)
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
                
        input_for_fit = repaired_file if shortened and repaired_file.is_file() else orig_file
        current_duration = get_wav_duration(input_for_fit)
        fit_methods: list[str] = []

        if budget > 0 and abs(current_duration - budget) > tolerance_sec:
            speed_factor = current_duration / budget
            speed_factor = min(max_stretch, max(1.0 / max_stretch, speed_factor))
            stretched_file = tts_dir / f"tts_stretch_{idx}.wav"
            if stretched_file.is_file():
                stretched_file.unlink()
            _run_ffmpeg_audio_filter(
                ffmpeg_path,
                input_for_fit,
                stretched_file,
                filter_expr=_build_atempo_chain(speed_factor),
                job_id=job_id,
                runner=runner,
            )
            input_for_fit = stretched_file
            current_duration = get_wav_duration(stretched_file)
            fit_methods.append(f"time_stretch_{round(speed_factor, 2)}x")

        if budget > 0 and abs(current_duration - budget) > tolerance_sec:
            exact_file = tts_dir / f"tts_exact_{idx}.wav"
            if exact_file.is_file():
                exact_file.unlink()
            target_dur = max(0.05, float(budget))
            exact_filter = f"apad=pad_dur={target_dur + 0.2:.3f},atrim=0:{target_dur:.3f}"
            _run_ffmpeg_audio_filter(
                ffmpeg_path,
                input_for_fit,
                exact_file,
                filter_expr=exact_filter,
                job_id=job_id,
                runner=runner,
            )
            input_for_fit = exact_file
            current_duration = get_wav_duration(exact_file)
            fit_methods.append("exact_trim_pad")

        if repaired_file.is_file():
            repaired_file.unlink()
        shutil.copy(input_for_fit, repaired_file)
        s["repaired_duration"] = round(get_wav_duration(repaired_file), 2)
        if not fit_methods:
            s["repaired_method"] = "none" if not shortened else "llm_shorten"
        else:
            prefix = "llm_shorten+" if shortened else ""
            s["repaired_method"] = prefix + "+".join(fit_methods)
            
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
    rows = database.connection.execute("SELECT key, value FROM settings").fetchall()
    settings = {r["key"]: json.loads(r["value"]) for r in rows}
    mix_mode = str(settings.get("mix_mode", "duck") or "duck")

    vocals_wav = artifacts_dir / "vocals.wav"
    bgm_wav = artifacts_dir / "bgm.wav"
    background_wav = original_48k

    if mix_mode == MIX_MODE_SEPARATE:
        reset_model_cache()
        if not bgm_wav.is_file() or not vocals_wav.is_file():
            separate_vocals(
            original_48k,
            vocals_out=vocals_wav,
            bgm_out=bgm_wav,
            ffmpeg_path=ffmpeg_path,
            device=str(settings.get("voxcpm_device", "cuda:0") or "cuda:0"),
            job_id=job_id,
            runner=runner,
        )
        background_wav = bgm_wav
        filter_graph = (
            "[0:a]volume=1.0[bg];[1:a]volume=1.3[fg];"
            "[bg][fg]amix=inputs=2:duration=first:dropout_transition=0[mixed]"
        )
    else:
        filter_graph = (
            "[0:a]volume=0.35[bg];[1:a]volume=1.5[fg];"
            "[fg]asplit=2[fg1][fg2];"
            "[bg][fg1]sidechaincompress=threshold=0.02:ratio=8:"
            "attack=20:release=400[ducked];"
            "[ducked][fg2]amix=inputs=2:duration=first:dropout_transition=0[mixed]"
        )

    cmd_mix = [
        str(ffmpeg_path), "-y",
        "-i", str(background_wav),
        "-i", str(narration_wav),
        "-filter_complex",
        filter_graph,
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
        "mix_mode": mix_mode,
        "narration_wav_path": str(narration_wav),
        "mixed_wav_path": str(mixed_wav),
        "vietnamese_narration_path": str(vietnamese_narration)
    }
    save_checkpoint(config.data_dir, job_id, "mix", checkpoint_data)
    return checkpoint_data


def render_step(job_id: str, config: AppConfig, database: Database, runner) -> dict:
    mix_cp = load_checkpoint(config.data_dir, job_id, "mix")
    download_cp = load_checkpoint(config.data_dir, job_id, "download")
    repair_cp = load_checkpoint(config.data_dir, job_id, "duration_repair")
    
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
    ass_path = output_dir / "subtitles.ass"
    
    ffmpeg_path = resolve_tool_path(config, "ffmpeg")
    rows = database.connection.execute("SELECT key, value FROM settings").fetchall()
    settings = {r["key"]: json.loads(r["value"]) for r in rows}

    video_filters: list[str] = []
    if settings.get("subtitles_enabled", True) and repair_cp:
        segments = [
            segment
            for segment in repair_cp.get("segments", [])
            if str(segment.get("translation") or "").strip()
        ]
        if segments:
            width, height = probe_video_dimensions(ffmpeg_path, original_mp4)
            write_ass_file(
                ass_path,
                segments,
                settings,
                play_res_x=width,
                play_res_y=height,
            )
            video_filters.append(ffmpeg_subtitles_filter(ass_path))
    
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
    ]
    if video_filters:
        cmd_render.extend(["-vf", ",".join(video_filters)])
    cmd_render.extend([
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", "superfast",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(final_mp4),
    ])
    
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
        "output_path": str(final_mp4),
        "subtitles_enabled": bool(settings.get("subtitles_enabled", True)),
        "subtitles_path": str(ass_path) if ass_path.is_file() else None,
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
        "schema_version": 2,
        "job_id": job_id,
        "step_name": "qc",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_segments": total_segments,
        "repaired_count": repaired_segments,
        "shortened_count": shortened_segments,
        "stretched_count": stretched_segments,
        "warnings": warnings,
        "output_video_path": output_path,
        "diarization": diarization_qc,
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
    diar = checkpoint_data.get("diarization") or {}
    diar_rows = ""
    if diar:
        diar_rows = (
            "<h2>Diarization</h2>"
            f"<p>Backend: {html.escape(str(diar.get('backend')))} | "
            f"Speakers: {diar.get('speaker_count')} | "
            f"Low confidence ratio: {diar.get('low_confidence_ratio')} | "
            f"Overlap ratio: {diar.get('overlap_ratio')} | "
            f"Demucs fallback: {diar.get('demucs_used')}</p>"
        )
    artifacts_html = artifacts_qc.with_suffix(".html")
    artifacts_html.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Douyin Vietnamizer QC Report</title></head><body>"
        "<h1>Douyin Vietnamizer QC Report</h1>"
        f"<p>Total segments: {total_segments}</p>"
        f"<p>Output: {html.escape(str(output_path))}</p>"
        f"{diar_rows}"
        "<table><thead><tr><th>Segment</th><th>Repair</th><th>Budget</th>"
        f"<th>Result</th></tr></thead><tbody>{warning_rows}</tbody></table>"
        "</body></html>",
        encoding="utf-8",
    )
        
    return checkpoint_data
