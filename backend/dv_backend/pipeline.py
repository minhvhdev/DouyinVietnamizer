import array
import json
import logging
import html
import os
import re
import shutil
import subprocess
import sys
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
from .vendor import VendorManifest, VendorResolver
from .adapters.translation import GoogleFreeTranslator
from .adapters.tts import VOXCPM_INSTRUCT_PREFIX, TtsSession, create_tts_adapter
from .adapters.asr import configure_gpu_manager, reset_model_cache, transcribe_audio
from .gpu_manager import global_gpu_manager
from .audio_probe import get_audio_duration
from .duration_safety import classify_stretch, tail_has_speech
from .segmentation import split_long_segments_with_alignment
from .sparse_asr import (
    build_sparse_chunks,
    build_stitched_timeline,
    map_stitched_segments_to_source,
    merge_overlapping_segments,
    rebase_sparse_segments,
    should_use_sparse_asr,
    stitched_timeline_duration,
)
from .telemetry import TelemetrySink
from .translation_duration import annotate_translation_duration, build_translation_timing_guidance, duration_prompt_suffix
from .adapters.subtitles import ffmpeg_subtitles_filter, probe_video_dimensions, write_ass_file
from .adapters.gemini import (
    GeminiKeyPool,
    GeminiTranslator,
    classify_gemini_failure,
    default_request,
    response_text,
)
from .source_urls import (
    ensure_bilibili_part_url,
    fallback_playlist_video_url,
    is_bilibili_host,
    is_douyin_user_profile_url,
    normalize_source_url,
    source_platform_label,
)
from .ytdlp_tools import (
    COOKIE_BROWSER_FALLBACK_ORDER,
    classify_yt_dlp_failure,
    yt_dlp_cookie_args_for_browser,
)

logger = logging.getLogger(__name__)

ASR_ALIGNMENT_SCHEMA_VERSION = 2
DEFAULT_EXACT_TIMING_TOLERANCE_MS = 40
DEFAULT_EXACT_TIMING_ENABLED = True
DEFAULT_EXACT_TIMING_MAX_STRETCH = 1.2


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
        # Portable runtime keeps tools under <runtime>/tools while manifest
        # executable entries remain relative ("ffmpeg/ffmpeg.exe", "yt-dlp/yt-dlp.exe").
        # If DV_VENDOR_DIR points at runtime root, retry against runtime/tools.
        tools_vendor_dir = vendor_dir / "tools"
        if tools_vendor_dir.is_dir():
            resolved = VendorResolver(tools_vendor_dir, allow_path_tools=allow_path_tools).resolve(tool)
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


def original_video_path(config: AppConfig, job_id: str) -> Path:
    path = config.data_dir / "jobs" / job_id / "artifacts" / "original.mp4"
    if not path.is_file():
        raise AppError(
            400,
            ErrorInfo(
                code="MISSING_VIDEO_FILE",
                message="The imported video file is missing.",
                action="Re-import the video file and start again.",
            ),
        )
    return path


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


def run_yt_dlp_with_browser_fallback(
    yt_dlp_path: Path,
    args: list[str],
    job_id: str,
    runner,
    *,
    timeout: float | None = None,
) -> tuple[subprocess.CompletedProcess, str, list[str]]:
    """Run yt-dlp using browser cookies, falling back Chrome → Edge → Firefox → Brave."""
    last_exc: subprocess.CalledProcessError | None = None
    browsers_tried: list[str] = []
    for browser in COOKIE_BROWSER_FALLBACK_ORDER:
        browsers_tried.append(browser)
        cmd = [str(yt_dlp_path), *yt_dlp_cookie_args_for_browser(browser), *args]
        try:
            result = run_subprocess_with_cancel(cmd, job_id, runner, timeout=timeout)
            return result, browser, browsers_tried
        except subprocess.CalledProcessError as exc:
            last_exc = exc
    assert last_exc is not None
    last_exc.browsers_tried = browsers_tried  # type: ignore[attr-defined]
    raise last_exc


def ffprobe_sibling_for(ffmpeg_path: Path) -> Path:
    name = "ffprobe.exe" if ffmpeg_path.suffix.lower() == ".exe" else "ffprobe"
    try:
        return ffmpeg_path.with_name(name)
    except ValueError:
        return Path(name)


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


def resolve_step(job_id: str, config: AppConfig, database: Database, runner) -> dict:
    row = database.connection.execute("SELECT source_url FROM jobs WHERE id = ?", (job_id,)).fetchone()
    source_url = normalize_source_url(row["source_url"])
    platform_label = source_platform_label(source_url)

    if is_douyin_user_profile_url(source_url):
        raise AppError(
            422,
            ErrorInfo(
                code="DOUYIN_USER_URL_NOT_SUPPORTED",
                message="Liên kết trang cá nhân Douyin không được hỗ trợ.",
                action=(
                    "Dùng liên kết video đơn (douyin.com/video/ID) hoặc link chia sẻ ngắn. "
                    "Liệt kê cả kênh/user chưa hỗ trợ."
                ),
            ),
        )

    yt_dlp_path = resolve_tool_path(config, "yt_dlp")
    yt_args = [
        "--dump-single-json",
        "--flat-playlist",
        "--playlist-end",
        "20",
        source_url,
    ]

    browsers_tried: list[str] = []
    try:
        res, _browser, browsers_tried = run_yt_dlp_with_browser_fallback(
            yt_dlp_path,
            yt_args,
            job_id,
            runner,
            timeout=60,
        )
        data = json.loads(res.stdout)
    except subprocess.CalledProcessError as exc:
        raise AppError(
            500,
            classify_yt_dlp_failure(
                operation="phân tích liên kết",
                stderr=exc.stderr or "",
                stdout=exc.stdout or "",
                browsers_attempted=getattr(exc, "browsers_tried", list(COOKIE_BROWSER_FALLBACK_ORDER)),
                source_label=platform_label,
            ),
        ) from exc
    except json.JSONDecodeError as exc:
        raise AppError(
            500,
            ErrorInfo(
                code="YT_DLP_RESOLVE_PARSE_FAILED",
                message=f"yt-dlp trả về dữ liệu không hợp lệ khi phân tích {platform_label}.",
                action="Thử cập nhật yt-dlp hoặc kiểm tra cookie trình duyệt.",
                detail=str(exc),
            ),
        ) from exc

    is_playlist = data.get("_type") == "playlist" or "entries" in data
    videos: list[dict] = []
    if is_playlist:
        for idx, entry in enumerate(data.get("entries") or []):
            if not entry:
                continue
            page_index = idx + 1
            video_url = fallback_playlist_video_url(entry, source_url, page_index=page_index)
            part_title = entry.get("title")
            if not part_title and is_bilibili_host(urllib.parse.urlparse(source_url).netloc):
                part_title = f"Phần {page_index}"
            videos.append(
                {
                    "id": entry.get("id") or f"p{page_index}",
                    "title": part_title or "Untitled Video",
                    "url": video_url,
                    "page": page_index,
                    "duration": entry.get("duration"),
                    "thumbnail": entry.get("thumbnail")
                    or (
                        entry.get("thumbnails")[0].get("url")
                        if entry.get("thumbnails")
                        else None
                    ),
                }
            )
    else:
        video_url = data.get("webpage_url") or source_url
        videos.append(
            {
                "id": data.get("id"),
                "title": data.get("title") or data.get("description") or "Untitled Video",
                "url": video_url,
                "duration": data.get("duration"),
                "thumbnail": data.get("thumbnail")
                or (
                    data.get("thumbnails")[0].get("url")
                    if data.get("thumbnails")
                    else None
                ),
            }
        )

    if not videos:
        raise AppError(
            404,
            ErrorInfo(
                code="YT_DLP_NO_VIDEOS",
                message=f"Không tìm thấy video nào từ liên kết {platform_label}.",
                action="Kiểm tra URL, đăng nhập trình duyệt, hoặc thử cập nhật yt-dlp.",
            ),
        )

    selected_video = videos[0] if len(videos) == 1 else None
    checkpoint_data = {
        "schema_version": 1,
        "job_id": job_id,
        "step_name": "resolve",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "is_playlist": is_playlist and len(videos) > 1,
        "videos": videos,
        "selected_video": selected_video,
        "source_platform": platform_label,
    }
    save_checkpoint(config.data_dir, job_id, "resolve", checkpoint_data)

    if selected_video:
        with database.connection:
            database.connection.execute(
                "UPDATE jobs SET title = ?, updated_at = ? WHERE id = ?",
                (selected_video["title"], time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), job_id),
            )

    return checkpoint_data


def download_step(job_id: str, config: AppConfig, database: Database, runner) -> dict:
    resolve_cp = load_checkpoint(config.data_dir, job_id, "resolve")
    if not resolve_cp or not resolve_cp.get("selected_video"):
        raise AppError(
            400,
            ErrorInfo(
                code="NO_VIDEO_SELECTED",
                message="Chưa chọn video để tải.",
                action="Chọn một video trong danh sách playlist rồi thử lại.",
            ),
        )

    selected = resolve_cp["selected_video"]
    video_url = selected["url"]
    row = database.connection.execute("SELECT source_url FROM jobs WHERE id = ?", (job_id,)).fetchone()
    source_url = row["source_url"] if row else video_url
    if is_bilibili_host(urllib.parse.urlparse(video_url).netloc):
        video_url = ensure_bilibili_part_url(
            video_url,
            source_url,
            entry=selected,
            page_index=selected.get("page"),
        )
    platform_label = resolve_cp.get("source_platform") or source_platform_label(video_url)

    yt_dlp_path = resolve_tool_path(config, "yt_dlp")
    ffmpeg_path = resolve_tool_path(config, "ffmpeg")
    ffmpeg_dir = ffmpeg_path.parent

    job_dir = config.data_dir / "jobs" / job_id
    artifacts_dir = job_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_mp4 = artifacts_dir / "original.mp4"

    yt_args = [
        "--no-playlist",
        "--ffmpeg-location",
        str(ffmpeg_dir),
        "-f",
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        str(output_mp4),
        video_url,
    ]

    try:
        run_yt_dlp_with_browser_fallback(
            yt_dlp_path,
            yt_args,
            job_id,
            runner,
            timeout=900,
        )
    except subprocess.CalledProcessError as exc:
        raise AppError(
            500,
            classify_yt_dlp_failure(
                operation="tải video",
                stderr=exc.stderr or "",
                stdout=exc.stdout or "",
                browsers_attempted=getattr(exc, "browsers_tried", list(COOKIE_BROWSER_FALLBACK_ORDER)),
                source_label=platform_label,
            ),
        ) from exc

    if not output_mp4.is_file():
        raise AppError(
            500,
            ErrorInfo(
                code="DOWNLOAD_OUTPUT_MISSING",
                message="yt-dlp báo thành công nhưng không tạo file video.",
                action="Thử cập nhật yt-dlp hoặc chạy lại bước tải.",
            ),
        )

    checkpoint_data = {
        "schema_version": 1,
        "job_id": job_id,
        "step_name": "download",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_path": str(output_mp4),
        "video_url": video_url,
    }
    save_checkpoint(config.data_dir, job_id, "download", checkpoint_data)
    return checkpoint_data


# Steps implementation
def _load_settings(database: Database) -> dict[str, object]:
    rows = database.connection.execute("SELECT key, value FROM settings").fetchall()
    return {row["key"]: json.loads(row["value"]) for row in rows}


def _normalize_mix_mode(value: object | None) -> str:
    mode = str(value or "background_only").strip().lower()
    if mode == "separate":
        return "background_only"
    return mode or "background_only"


def _separate_background_stems(
    job_id: str,
    artifacts_dir: Path,
    source_wav: Path,
    ffmpeg_path: Path,
    runner,
) -> tuple[Path, Path]:
    demucs_root = artifacts_dir / "demucs"
    stem_dir = demucs_root / "htdemucs" / source_wav.stem
    bgm_wav = artifacts_dir / "bgm.wav"
    vocals_wav = artifacts_dir / "vocals.wav"

    if demucs_root.exists():
        shutil.rmtree(demucs_root, ignore_errors=True)

    cmd_separate = [
        sys.executable,
        "-m",
        "demucs.separate",
        "--two-stems=vocals",
        "-n",
        "htdemucs",
        "-o",
        str(demucs_root),
        str(source_wav),
    ]
    try:
        run_subprocess_with_cancel(cmd_separate, job_id, runner)
        separated_bgm = stem_dir / "no_vocals.wav"
        separated_vocals = stem_dir / "vocals.wav"
        if not separated_bgm.is_file() or not separated_vocals.is_file():
            raise AppError(
                500,
                ErrorInfo(
                    code="STEM_OUTPUTS_MISSING",
                    message="Demucs did not produce the expected background and vocals files.",
                    action="Retry the job or verify the input audio is valid.",
                ),
            )

        for input_path, output_path in ((separated_bgm, bgm_wav), (separated_vocals, vocals_wav)):
            cmd_convert = [
                str(ffmpeg_path),
                "-y",
                "-i",
                str(input_path),
                "-acodec",
                "pcm_s16le",
                "-ac",
                "2",
                "-ar",
                "48000",
                str(output_path),
            ]
            run_subprocess_with_cancel(cmd_convert, job_id, runner)
    finally:
        shutil.rmtree(demucs_root, ignore_errors=True)

    return bgm_wav, vocals_wav


def extract_audio_step(job_id: str, config: AppConfig, database: Database, runner) -> dict:
    job_dir = config.data_dir / "jobs" / job_id
    artifacts_dir = job_dir / "artifacts"
    original_mp4 = original_video_path(config, job_id)

    ffmpeg_path = resolve_tool_path(config, "ffmpeg")
    original_48k = artifacts_dir / "original_48k.wav"
    audio_16k = artifacts_dir / "audio_16k.wav"
    settings = _load_settings(database)
    requested_mix_mode = _normalize_mix_mode(settings.get("mix_mode"))

    cmd_48k = [
        str(ffmpeg_path), "-y",
        "-i", str(original_mp4),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "48000",
        str(original_48k)
    ]

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

    bgm_path: str | None = None
    vocals_path: str | None = None
    if requested_mix_mode != "duck":
        try:
            bgm_wav, vocals_wav = _separate_background_stems(
                job_id,
                artifacts_dir,
                original_48k,
                ffmpeg_path,
                runner,
            )
            bgm_path = str(bgm_wav)
            vocals_path = str(vocals_wav)
        except subprocess.CalledProcessError as e:
            raise AppError(
                500,
                ErrorInfo(
                    code="STEM_SEPARATION_FAILED",
                    message="Failed to isolate background audio from the original video.",
                    action="Verify Demucs can run and that the input audio contains a valid soundtrack.",
                    detail=e.stderr or e.stdout,
                )
            )

    checkpoint_data = {
        "schema_version": 2,
        "job_id": job_id,
        "step_name": "extract_audio",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "original_48k_path": str(original_48k),
        "audio_16k_path": str(audio_16k),
        "bgm_path": bgm_path,
        "vocals_path": vocals_path,
        "requested_mix_mode": requested_mix_mode,
    }
    save_checkpoint(config.data_dir, job_id, "extract_audio", checkpoint_data)
    return checkpoint_data


def _release_asr_gpu_models(settings: dict | None = None) -> None:
    """Unload Qwen/FunASR weights after ASR so TTS can use VRAM on low-memory GPUs."""
    try:
        reset_model_cache()
    except Exception:
        logger.debug("ASR model cache reset failed", exc_info=True)
    try:
        device = str((settings or {}).get("qwen3_device", "cuda:0") or "cuda:0")
        global_gpu_manager().evict("asr", device, reason="asr_step_complete")
    except Exception:
        logger.debug("ASR GPU lease eviction failed", exc_info=True)


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
    telemetry = TelemetrySink(config.data_dir, job_id)
    started = time.perf_counter()
    total_duration = get_audio_duration(audio_16k, ffprobe_path=ffprobe_sibling_for(ffmpeg_path))

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
        telemetry.record("vad", {
            "status": "failed",
            "wall_time_ms": round((time.perf_counter() - started) * 1000),
            "audio_duration_sec": total_duration,
            "model_config": "ffmpeg_silencedetect",
            "retry_count": 0,
        })
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
        
    speech_duration = sum(region["end"] - region["start"] for region in speech_regions)
    speech_ratio = round(speech_duration / total_duration, 4) if total_duration > 0 else 0.0
    telemetry.record("vad", {
        "status": "ok",
        "wall_time_ms": round((time.perf_counter() - started) * 1000),
        "audio_duration_sec": total_duration,
        "model_config": "ffmpeg_silencedetect:n=-30dB:d=0.5",
        "retry_count": 0,
        "speech_region_count": len(speech_regions),
        "vad_speech_ratio": speech_ratio,
    })

    checkpoint_data = {
        "schema_version": 1,
        "job_id": job_id,
        "step_name": "vad",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_duration": round(total_duration, 2),
        "speech_regions": speech_regions,
        "vad_speech_ratio": speech_ratio,
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
    configure_gpu_manager(settings)
    global_manager = global_gpu_manager()

    project_root = Path(__file__).resolve().parents[2]
    vendor_dir = Path(os.environ.get("DV_VENDOR_DIR", project_root / "vendor"))

    telemetry = TelemetrySink(config.data_dir, job_id)
    started = time.perf_counter()
    try:
        alignment_mode = str(settings.get("asr_alignment_mode", "accurate") or "accurate").strip().lower()
        if alignment_mode not in {"fast", "balanced", "accurate"}:
            alignment_mode = "accurate"
        vad_cp = load_checkpoint(config.data_dir, job_id, "vad") or {}
        speech_regions = vad_cp.get("speech_regions", [])
        include_alignment = alignment_mode == "accurate"
        alignment_requested_reason = "accurate_default"
        if alignment_mode == "balanced":
            long_regions = [
                region
                for region in speech_regions
                if float(region.get("end", 0.0) or 0.0) - float(region.get("start", 0.0) or 0.0) > 20.0
            ]
            include_alignment = bool(long_regions)
            alignment_requested_reason = "balanced_long_vad_region" if long_regions else "balanced_skip_alignment"

        sparse_enabled = bool(settings.get("sparse_asr_enabled", False))
        sparse_decision = should_use_sparse_asr(
            speech_regions,
            total_duration=float(vad_cp.get("total_duration") or 0.0),
            min_silence_ratio=float(settings.get("sparse_asr_min_silence_ratio", 0.35) or 0.35),
        ) if sparse_enabled else None
        dense_or_sparse_mode = "sparse" if sparse_decision and sparse_decision.use_sparse else "dense"

        segments = []
        aligned_units = []
        sparse_asr_fallback_reason = None
        sparse_chunk_count = 0
        stitched_duration_sec = 0.0
        asr_kwargs = {
            "vendor_dir": vendor_dir,
            "asr_model": str(settings.get("qwen3_asr_model", "") or ""),
            "aligner_model": str(settings.get("qwen3_aligner_model", "") or ""),
            "device": str(settings.get("qwen3_device", "cuda:0") or "cuda:0"),
            "language": "Chinese",
            "speaker_diarization": False,
            "include_alignment": include_alignment,
        }
        if dense_or_sparse_mode == "sparse":
            try:
                ffmpeg_path = resolve_tool_path(config, "ffmpeg")
                chunks = build_sparse_chunks(
                    speech_regions,
                    total_duration=float(vad_cp.get("total_duration") or 0.0),
                    merge_gap_sec=0.25,
                    padding_sec=float(settings.get("sparse_asr_padding_ms", 200) or 200) / 1000.0,
                    max_chunk_sec=float(settings.get("sparse_asr_chunk_sec", 25) or 25),
                )
                if not chunks:
                    dense_or_sparse_mode = "dense"
                    sparse_asr_fallback_reason = "no_sparse_chunks"
                else:
                    timeline = build_stitched_timeline(chunks)
                    sparse_chunk_count = len(timeline)
                    stitched_duration_sec = stitched_timeline_duration(timeline)
                    sparse_dir = job_dir / "artifacts" / "asr_sparse"
                    sparse_dir.mkdir(parents=True, exist_ok=True)
                    stitched_path = sparse_dir / "stitched.wav"

                    filter_parts: list[str] = []
                    concat_inputs: list[str] = []
                    for index, span in enumerate(timeline):
                        filter_parts.append(
                            f"[0:a]atrim=start={span['source_start']}:"
                            f"end={span['source_end']},"
                            f"asetpts=PTS-STARTPTS[s{index}]"
                        )
                        concat_inputs.append(f"[s{index}]")
                    filter_parts.append(
                        f"{''.join(concat_inputs)}concat=n={sparse_chunk_count}:v=0:a=1[outa]"
                    )

                    cmd = [
                        str(ffmpeg_path), "-y",
                        "-i", str(audio_16k),
                        "-filter_complex", ";".join(filter_parts),
                        "-map", "[outa]",
                        "-acodec", "pcm_s16le",
                        "-ac", "1",
                        "-ar", "16000",
                        str(stitched_path),
                    ]
                    run_subprocess_with_cancel(cmd, job_id, runner)

                    stitched_result = transcribe_audio(stitched_path, **asr_kwargs)
                    if isinstance(stitched_result, dict):
                        raw_segments = stitched_result.get("segments", [])
                        raw_units = stitched_result.get("aligned_units", [])
                    else:
                        raw_segments = list(stitched_result)
                        raw_units = []

                    segments = merge_overlapping_segments(
                        map_stitched_segments_to_source(timeline, raw_segments)
                    )
                    aligned_units = merge_overlapping_segments(
                        map_stitched_segments_to_source(timeline, raw_units)
                    ) if raw_units else []
                    if not segments:
                        dense_or_sparse_mode = "dense"
                        sparse_asr_fallback_reason = "empty_sparse_result"
            except AppError:
                raise
            except Exception as error:
                logger.info("Sparse ASR fallback for job %s: %s", job_id, error)
                import traceback
                sparse_asr_fallback_reason = f"{type(error).__name__}"
                dense_or_sparse_mode = "dense"
                segments = []
                aligned_units = []

        if dense_or_sparse_mode == "dense":
            if sparse_asr_fallback_reason is None:
                sparse_asr_fallback_reason = sparse_decision.reason if sparse_decision else None
            stitched_duration_sec = float(vad_cp.get("total_duration") or 0.0)
            asr_result = transcribe_audio(audio_16k, **asr_kwargs)
            if isinstance(asr_result, dict):
                segments = asr_result.get("segments", [])
                aligned_units = asr_result.get("aligned_units", [])
            else:
                segments = asr_result
                aligned_units = []

        if not segments:
            telemetry.record("asr", {
                "status": "failed",
                "wall_time_ms": round((time.perf_counter() - started) * 1000),
                "audio_duration_sec": float(vad_cp.get("total_duration") or 0.0),
                "model_config": str(settings.get("qwen3_asr_model", "")),
                "retry_count": 0,
                "dense_or_sparse_mode": dense_or_sparse_mode,
            })
            raise AppError(
                422,
                ErrorInfo(
                    code="EMPTY_ASR_TRANSCRIPTION",
                    message="ASR completed without detecting any spoken text.",
                    action="Verify the source audio and Qwen3-ASR model, then resume the ASR step."
                )
            )

        alignment_status = "available" if aligned_units else ("skipped" if not include_alignment else "unavailable")
        last_lease = global_manager.lease_history[-1] if global_manager.lease_history else {}
        telemetry.record("asr", {
            "status": "ok",
            "wall_time_ms": round((time.perf_counter() - started) * 1000),
            "audio_duration_sec": float(vad_cp.get("total_duration") or 0.0),
            "model_config": str(settings.get("qwen3_asr_model", "")),
            "retry_count": 0,
            "segment_count": len(segments),
            "aligned_unit_count": len(aligned_units),
            "dense_or_sparse_mode": dense_or_sparse_mode,
            "alignment_mode": alignment_mode,
            "alignment_status": alignment_status,
            "gpu_queue_wait_ms": int(last_lease.get("queue_wait_ms") or 0),
            "model_load_ms": int(last_lease.get("load_ms") or 0),
            "cold_start": bool(last_lease.get("cold_start")),
            "vram_before_mb": last_lease.get("vram_before_mb"),
            "vram_after_mb": last_lease.get("vram_after_mb"),
        })

        checkpoint_data = {
            "schema_version": ASR_ALIGNMENT_SCHEMA_VERSION,
            "job_id": job_id,
            "step_name": "asr",
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "segments": segments,
            "aligned_units": aligned_units,
            "alignment_required_for_diarization": False,
            "alignment_mode": alignment_mode,
            "alignment_status": alignment_status,
            "alignment_requested_reason": alignment_requested_reason,
            "alignment_coverage": round(len(aligned_units) / max(1, len(segments)), 4),
            "dense_or_sparse_mode": dense_or_sparse_mode,
            "sparse_asr_fallback_reason": sparse_asr_fallback_reason,
            "sparse_chunk_count": sparse_chunk_count,
            "stitched_duration_sec": stitched_duration_sec,
        }
        save_checkpoint(config.data_dir, job_id, "asr", checkpoint_data)
        return checkpoint_data
    finally:
        _release_asr_gpu_models(settings)


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
        
    raw_segments = split_long_segments_with_alignment(
        asr_cp.get("segments", []),
        vad_cp.get("speech_regions", []),
        asr_cp.get("aligned_units", []) or [],
    )
    raw_segments = _split_long_asr_segments_with_vad(
        raw_segments,
        vad_cp.get("speech_regions", []),
    )
    total_duration = vad_cp.get("total_duration", 0.0)
    
    segments = []
    for seg in raw_segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        entry = {
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": text,
        }
        for key in ("split_method", "original_segment_id", "split_confidence", "split_reason"):
            if seg.get(key) is not None:
                entry[key] = seg[key]
        segments.append(entry)
        
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
            **({"split_method": curr["split_method"]} if curr.get("split_method") is not None else {}),
            **({"original_segment_id": curr["original_segment_id"]} if curr.get("original_segment_id") is not None else {}),
            **({"split_confidence": curr["split_confidence"]} if curr.get("split_confidence") is not None else {}),
            **({"split_reason": curr["split_reason"]} if curr.get("split_reason") is not None else {}),
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
    duration_budgets: list[float] | None = None,
    timing_guidance: list[dict] | None = None,
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
        translated = translator.translate(
            texts,
            source=source_lang,
            target=target_lang,
            duration_budgets=duration_budgets,
            timing_guidance=timing_guidance,
        )
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


def _aligned_units_for_segment(segment: dict, aligned_units: list[dict]) -> list[dict]:
    if not aligned_units:
        return []

    start = float(segment.get("start", 0.0) or 0.0)
    end = float(segment.get("end", start) or start)
    overlaps: list[dict] = []
    for unit in aligned_units:
        try:
            unit_start = float(unit.get("start", 0.0) or 0.0)
            unit_end = float(unit.get("end", unit_start) or unit_start)
        except (TypeError, ValueError):
            continue
        if unit_end <= start or unit_start >= end:
            continue
        overlaps.append(unit)
    return overlaps


def translate_step(job_id: str, config: AppConfig, database: Database, runner) -> dict:
    norm_cp = load_checkpoint(config.data_dir, job_id, "normalize_segments")
    asr_cp = load_checkpoint(config.data_dir, job_id, "asr") or {}
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
    duration_budgets = [float(segment.get("duration_budget") or 0.0) for segment in segments]
    aligned_units = asr_cp.get("aligned_units", []) or []
    timing_guidance = [
        build_translation_timing_guidance(
            segment,
            aligned_units=_aligned_units_for_segment(segment, aligned_units),
        )
        for segment in segments
    ]
    for segment, guidance in zip(segments, timing_guidance, strict=True):
        segment.update(guidance)
    translated = _translate_texts(
        settings,
        database,
        texts,
        source_lang=source_lang,
        target_lang=target_lang,
        duration_budgets=duration_budgets,
        timing_guidance=timing_guidance,
    )
    if len(translated) != len(segments) or any(not str(item).strip() for item in translated):
        raise AppError(
            502,
            ErrorInfo(
                code="TRANSLATION_COUNT_MISMATCH",
                message="Translation backend returned incomplete segment translations.",
                action="Retry translation or switch translation backend.",
                retryable=True,
            )
        )
    for segment, translation in zip(segments, translated, strict=True):
        segment["translation"] = translation
        segment.update(annotate_translation_duration(segment))

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


def _anchor_transcript_for(settings: dict) -> str | None:
    ref_audio = str(settings.get("voxcpm_ref_audio") or "").strip()
    if not ref_audio:
        return None
    sidecar = Path(ref_audio).with_suffix(".txt")
    if not sidecar.is_file():
        return None
    try:
        transcript = sidecar.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return transcript or None


def _resolve_clone_mode(settings: dict) -> str:
    raw = settings.get("voxcpm_clone_mode")
    candidate = str(raw).strip().lower() if raw else "reference"
    if candidate in ("reference", "ultimate"):
        return candidate
    return "reference"


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
    ref_audio = str(settings.get("voxcpm_ref_audio") or "").strip()
    clone = bool(ref_audio)
    clone_mode = _resolve_clone_mode(settings) if clone else "reference"
    anchor_text = _anchor_transcript_for(settings) if clone else None
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
                clone=clone,
                clone_mode=clone_mode,
                anchor_text=anchor_text,
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


def _wav_tail_has_speech(path: Path, *, tail_ms: int = 200) -> bool:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            frame_count = min(handle.getnframes(), max(1, int(rate * tail_ms / 1000.0)))
            handle.setpos(max(0, handle.getnframes() - frame_count))
            frames = handle.readframes(frame_count)
        samples = array.array("h")
        samples.frombytes(frames)
        if channels > 1:
            mono = [float(samples[index]) / 32768.0 for index in range(0, len(samples), channels)]
        else:
            mono = [float(sample) / 32768.0 for sample in samples]
        return tail_has_speech(mono, sample_rate=rate, tail_ms=tail_ms)
    except Exception:
        return False


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
    max_stretch = max(1.0, min(DEFAULT_EXACT_TIMING_MAX_STRETCH, max_stretch))
    return enabled, tolerance_sec, max_stretch


def _estimate_word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _centered_silence_pad_filter(current_duration: float, target_dur: float) -> str:
    gap = max(0.0, target_dur - current_duration)
    lead_sec = gap / 2.0
    tail_sec = max(0.0, gap - lead_sec)
    lead_ms = max(0, round(lead_sec * 1000))
    return (
        f"adelay={lead_ms}:all=1,"
        f"apad=pad_dur={tail_sec + 0.2:.3f},"
        f"atrim=0:{target_dur:.3f}"
    )


def _shorten_translation_with_gemini(
    settings: dict,
    database: Database,
    *,
    text: str,
    budget: float,
    current_duration: float,
) -> tuple[str | None, int]:
    current_words = _estimate_word_count(text)
    if current_words < 2 or budget <= 0 or current_duration <= budget:
        return None, current_words

    target_ratio = max(0.1, min(1.0, float(budget) / float(current_duration)))
    target_words = max(1, round(current_words * target_ratio))
    if target_words >= current_words:
        return None, target_words

    key_pool = GeminiKeyPool(
        settings.get("gemini_api_keys", []),
        cursor=int(settings.get("gemini_key_cursor", 0)),
    )
    if not key_pool.keys:
        return None, target_words

    model = str(settings.get("gemini_translation_model", "gemini-2.5-flash") or "gemini-2.5-flash")
    overrun_pct = max(0.0, ((current_duration / budget) - 1.0) * 100.0)
    prompt = (
        "Rewrite this Vietnamese dubbing line so it stays natural but fits the target timing.\n"
        f"Current line: {text}\n"
        f"Current duration: {current_duration:.2f}s\n"
        f"Target duration budget: {budget:.2f}s\n"
        f"Current word count: approximately {current_words}\n"
        f"Target word count: approximately {target_words} (timing ratio {target_ratio:.3f})\n"
        f"Current line overruns the timing by {overrun_pct:.1f}%.\n"
        "Remove filler words and redundant phrasing first. Preserve names, numbers, core meaning, and causal relationships. "
        "Return only the rewritten Vietnamese line with no quotes, notes, or formatting."
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2},
    }

    last_error: Exception | None = None
    saw_model_unavailable = False
    saw_model_not_found = False
    for index, api_key in key_pool.ordered_keys():
        try:
            shortened = response_text(default_request(api_key, model, payload)).strip().strip("\"'")
            if shortened:
                key_pool.mark_success(index)
                save_setting(database, "gemini_key_cursor", key_pool.cursor)
                return shortened, target_words
            raise ValueError("Gemini returned an empty shortening result.")
        except Exception as cause:
            last_error = cause
            code, _, _ = classify_gemini_failure(cause)
            if code == "GEMINI_MODEL_UNAVAILABLE":
                saw_model_unavailable = True
            if code == "GEMINI_MODEL_NOT_FOUND":
                saw_model_not_found = True

    if saw_model_unavailable and not saw_model_not_found:
        code, message, action = classify_gemini_failure(
            last_error or RuntimeError("Gemini model unavailable")
        )
    elif saw_model_not_found:
        code, message, action = classify_gemini_failure(
            last_error or RuntimeError("Gemini model not found")
        )
    else:
        code, message, action = classify_gemini_failure(
            last_error or RuntimeError("Gemini request failed")
        )

    raise AppError(
        502,
        ErrorInfo(
            code=code,
            message=message,
            action=action,
            detail=str(last_error),
            retryable=True,
        ),
    )


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
    _release_asr_gpu_models(settings)
    telemetry = TelemetrySink(config.data_dir, job_id)
    step_started = time.perf_counter()
    session_started = time.perf_counter()
    from .tts_conversion import TtsConversionResult, conversion_strategy_from_settings, describe as describe_conversion

    strategy = conversion_strategy_from_settings(settings)
    ffmpeg_path = resolve_tool_path(config, "ffmpeg") if strategy == "per_segment" else None
    conversion_process_count = 0
    conversion_input_count = 0
    conversion_wall_time_ms = 0
    micro_batch_enabled = str(settings.get("tts_micro_batch_enabled", True)).lower() not in {"0", "false", "no"}
    try:
        micro_batch_size = int(settings.get("voxcpm_batch_size", 4) or 4)
    except (TypeError, ValueError):
        micro_batch_size = 4
    micro_batch_size = max(1, micro_batch_size)

    with TtsSession(settings, data_dir=config.data_dir, runner=runner, adapter_factory=create_tts_adapter) as session:
        session_create_ms = round((time.perf_counter() - session_started) * 1000)
        pending: list[dict] = []

        def finish_segment(entry: dict, *, synthesize_ms: int, batch_size: int, batch_wall_time_ms: int) -> None:
            nonlocal conversion_process_count, conversion_input_count, conversion_wall_time_ms
            s = entry["segment"]
            idx = s["index"]
            raw_tts = entry["raw_tts"]
            final_tts = entry["final_tts"]
            conversion_ms = 0
            if strategy == "per_segment":
                conversion_started = time.perf_counter()
                _convert_tts_to_final_wav(ffmpeg_path, raw_tts, final_tts, job_id, runner)
                conversion_ms = round((time.perf_counter() - conversion_started) * 1000)
                conversion_process_count += 1
                conversion_input_count += 1
                conversion_wall_time_ms += conversion_ms
                s["tts_duration"] = round(get_wav_duration(final_tts), 2)
            else:
                # lazy_mix: keep raw; duration is read from the raw header.
                s["tts_duration"] = round(get_wav_duration(raw_tts), 2)

            s["tts_raw_path"] = str(raw_tts)
            s["tts_path"] = str(final_tts) if strategy == "per_segment" else None
            s["tts_session_reused"] = True
            telemetry.record("tts_segment", {
                "wall_time_ms": synthesize_ms + conversion_ms,
                "audio_duration_sec": s["tts_duration"],
                "tts_session_create_ms": session_create_ms,
                "synthesize_ms": synthesize_ms,
                "conversion_ms": conversion_ms,
                "output_write_ms": conversion_ms,
                "segment_index": idx,
                "retry_count": 0,
                "cache_hit": None,
                "cache_miss": None,
                "model_config": str(settings.get("voxcpm_model", "")),
                "raw_tts_format": "wav_pcm16le_native",
                "tts_micro_batch_enabled": micro_batch_enabled,
                "tts_micro_batch_size": batch_size,
                "tts_batch_wall_time_ms": batch_wall_time_ms,
            })

        def flush_pending() -> None:
            if not pending:
                return
            batch = list(pending)
            pending.clear()
            synth_started = time.perf_counter()
            session.synthesize_batch([
                {"text": entry["text"], "output_path": entry["raw_tts"], "segment": entry["segment"]}
                for entry in batch
            ])
            synthesize_ms = round((time.perf_counter() - synth_started) * 1000)
            per_segment_synthesize_ms = round(synthesize_ms / max(1, len(batch)))
            for entry in batch:
                finish_segment(
                    entry,
                    synthesize_ms=per_segment_synthesize_ms,
                    batch_size=len(batch),
                    batch_wall_time_ms=synthesize_ms,
                )

        for s in segments:
            idx = s["index"]
            text = s["translation"]

            raw_tts = tts_dir / f"tts_raw_{idx}.wav"
            final_tts = tts_dir / f"tts_{idx}.wav"

            if strategy == "per_segment" and final_tts.is_file() and final_tts.stat().st_size > 44:
                s["tts_duration"] = round(get_wav_duration(final_tts), 2)
                s["tts_raw_path"] = str(raw_tts) if raw_tts.is_file() else str(final_tts)
                s["tts_path"] = str(final_tts)
                s["tts_session_reused"] = True
                continue

            if strategy != "per_segment" and raw_tts.is_file() and raw_tts.stat().st_size > 44:
                s["tts_duration"] = round(get_wav_duration(raw_tts), 2)
                s["tts_raw_path"] = str(raw_tts)
                s["tts_path"] = None
                s["tts_session_reused"] = True
                continue

            if raw_tts.is_file():
                raw_tts.unlink()

            if final_tts.is_file():
                final_tts.unlink()

            entry = {"segment": s, "text": text, "raw_tts": raw_tts, "final_tts": final_tts}
            if micro_batch_enabled:
                pending.append(entry)
                if len(pending) >= micro_batch_size:
                    flush_pending()
                continue

            synth_started = time.perf_counter()
            session.synthesize(text, raw_tts, segment=s)
            synthesize_ms = round((time.perf_counter() - synth_started) * 1000)
            finish_segment(entry, synthesize_ms=synthesize_ms, batch_size=1, batch_wall_time_ms=synthesize_ms)

        flush_pending()

    conversion_result = TtsConversionResult(
        strategy=strategy,
        fallback_reason=None,
        process_count=conversion_process_count,
        wall_time_ms=conversion_wall_time_ms,
        inputs=conversion_input_count if strategy == "per_segment" else len(segments),
    )

    telemetry.record("tts", {
        "wall_time_ms": round((time.perf_counter() - step_started) * 1000),
        "segment_count": len(segments),
        "tts_session_create_ms": 0,
        "retry_count": 0,
        "model_config": str(settings.get("voxcpm_model", "")),
        "conversion_strategy": strategy,
        "tts_micro_batch_enabled": micro_batch_enabled,
        "tts_micro_batch_size": micro_batch_size,
        **(describe_conversion(conversion_result) if conversion_result is not None else {"conversion_strategy": strategy, "conversion_input_count": 0, "conversion_wall_time_ms": 0, "conversion_process_count": 0, "conversion_fallback_reason": "no_batch_run"}),
    })

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
    max_safe_stretch = float(settings.get("exact_timing_max_safe_stretch", 1.25) or 1.25)

    ffmpeg_path = resolve_tool_path(config, "ffmpeg")
    telemetry = TelemetrySink(config.data_dir, job_id)
    step_started = time.perf_counter()

    with TtsSession(settings, data_dir=config.data_dir, runner=runner, adapter_factory=create_tts_adapter) as session:
        for s in segments:
            segment_started = time.perf_counter()
            idx = s["index"]
            budget = float(s.get("duration_budget") or 0.0)
            tts_dur = float(s.get("tts_duration") or 0.0)
            orig_file = Path(s.get("tts_path") or s.get("tts_raw_path") or (tts_dir / f"tts_{idx}.wav"))
            repaired_file = tts_dir / f"tts_repaired_{idx}.wav"
            repaired_file.unlink(missing_ok=True)

            repair_attempts = 0
            re_synthesis_count = 0
            llm_shorten_ms = 0
            re_synthesis_ms = 0
            atempo_ms = 0
            trim_ms = 0
            fit_methods: list[str] = []
            quality_warning: str | None = None
            duration_repair_risk = "none"
            time_stretch_factor = 1.0
            tail_speech_detected = False

            if not exact_enabled and tts_dur <= budget + 0.1:
                shutil.copy(orig_file, repaired_file)
                s["repaired_method"] = "none"
                s["repaired_duration"] = round(tts_dur, 2)
                s["duration_repair_risk"] = duration_repair_risk
                s["final_timing_error_ms"] = round((tts_dur - budget) * 1000) if budget > 0 else 0
                s["repair_attempts"] = repair_attempts
                s["tail_speech_detected"] = False
                s["time_stretch_factor"] = 1.0
                continue

            input_for_fit = orig_file
            current_duration = get_wav_duration(input_for_fit)
            if budget > 0 and current_duration > budget + tolerance_sec:
                try:
                    gemini_started = time.perf_counter()
                    new_translation, target_words = _shorten_translation_with_gemini(
                        settings,
                        database,
                        text=str(s.get("translation") or ""),
                        budget=budget,
                        current_duration=current_duration,
                    )
                    llm_shorten_ms = round((time.perf_counter() - gemini_started) * 1000)
                    if new_translation and new_translation != s.get("translation"):
                        raw_temp = tts_dir / f"tts_temp_raw_{idx}.wav"
                        temp_wav = tts_dir / f"tts_temp_{idx}.wav"
                        raw_temp.unlink(missing_ok=True)
                        temp_wav.unlink(missing_ok=True)
                        synth_started = time.perf_counter()
                        session.synthesize(new_translation, raw_temp, segment=s)
                        _convert_tts_to_final_wav(ffmpeg_path, raw_temp, temp_wav, job_id, runner)
                        re_synthesis_ms = round((time.perf_counter() - synth_started) * 1000)
                        raw_temp.unlink(missing_ok=True)
                        new_dur = get_wav_duration(temp_wav)
                        repair_attempts += 1
                        re_synthesis_count += 1
                        if new_dur < current_duration:
                            input_for_fit = temp_wav
                            current_duration = new_dur
                            s["translation"] = new_translation
                            fit_methods.append(
                                f"gemini_shorten_to_{target_words}_words"
                                if target_words > 0
                                else "gemini_shorten"
                            )
                        else:
                            temp_wav.unlink(missing_ok=True)
                except Exception:
                    quality_warning = "gemini_shorten_failed"

            if exact_enabled and budget > 0 and current_duration > budget + tolerance_sec:
                raw_factor = current_duration / budget
                speed_factor = min(max_stretch, max(1.0, raw_factor))
                stretch_decision = classify_stretch(speed_factor, max_safe=max_safe_stretch, explicit_allow_danger=True)
                duration_repair_risk = stretch_decision.risk
                if stretch_decision.warning:
                    quality_warning = stretch_decision.warning
                stretched_file = tts_dir / f"tts_stretch_{idx}.wav"
                stretched_file.unlink(missing_ok=True)
                atempo_started = time.perf_counter()
                _run_ffmpeg_audio_filter(
                    ffmpeg_path,
                    input_for_fit,
                    stretched_file,
                    filter_expr=_build_atempo_chain(speed_factor),
                    job_id=job_id,
                    runner=runner,
                )
                atempo_ms = round((time.perf_counter() - atempo_started) * 1000)
                input_for_fit = stretched_file
                current_duration = get_wav_duration(stretched_file)
                time_stretch_factor = round(speed_factor, 3)
                fit_methods.append(f"time_stretch_{round(speed_factor, 2)}x")
                repair_attempts += 1

            if exact_enabled and budget > 0 and abs(current_duration - budget) > tolerance_sec:
                target_dur = max(0.05, float(budget))
                if current_duration > target_dur:
                    tail_speech_detected = _wav_tail_has_speech(input_for_fit)
                    if tail_speech_detected:
                        quality_warning = "tail_speech_detected_skip_hard_trim"
                    else:
                        exact_file = tts_dir / f"tts_exact_{idx}.wav"
                        exact_file.unlink(missing_ok=True)
                        exact_filter = f"apad=pad_dur={target_dur + 0.2:.3f},atrim=0:{target_dur:.3f}"
                        fade_start = max(0.0, target_dur - 0.05)
                        exact_filter = f"afade=t=out:st={fade_start:.3f}:d=0.050,{exact_filter}"
                        trim_started = time.perf_counter()
                        _run_ffmpeg_audio_filter(
                            ffmpeg_path,
                            input_for_fit,
                            exact_file,
                            filter_expr=exact_filter,
                            job_id=job_id,
                            runner=runner,
                        )
                        trim_ms = round((time.perf_counter() - trim_started) * 1000)
                        input_for_fit = exact_file
                        current_duration = get_wav_duration(exact_file)
                        fit_methods.append("exact_trim_pad")
                        repair_attempts += 1
                else:
                    exact_file = tts_dir / f"tts_exact_{idx}.wav"
                    exact_file.unlink(missing_ok=True)
                    trim_started = time.perf_counter()
                    _run_ffmpeg_audio_filter(
                        ffmpeg_path,
                        input_for_fit,
                        exact_file,
                        filter_expr=_centered_silence_pad_filter(current_duration, target_dur),
                        job_id=job_id,
                        runner=runner,
                    )
                    trim_ms = round((time.perf_counter() - trim_started) * 1000)
                    input_for_fit = exact_file
                    current_duration = get_wav_duration(exact_file)
                    fit_methods.append("centered_silence_pad")
                    repair_attempts += 1

            repaired_file.unlink(missing_ok=True)
            shutil.copy(input_for_fit, repaired_file)
            repaired_duration = round(get_wav_duration(repaired_file), 2)
            s["repaired_duration"] = repaired_duration
            s["repaired_method"] = "+".join(fit_methods) if fit_methods else "none"
            s["duration_repair_risk"] = duration_repair_risk
            s["final_timing_error_ms"] = round((repaired_duration - budget) * 1000) if budget > 0 else 0
            s["tail_speech_detected"] = tail_speech_detected
            s["time_stretch_factor"] = time_stretch_factor
            s["repair_attempts"] = repair_attempts
            s["quality_warning"] = quality_warning
            s["re_synthesis_count"] = re_synthesis_count
            telemetry.record("duration_repair_segment", {
                "wall_time_ms": round((time.perf_counter() - segment_started) * 1000),
                "audio_duration_sec": repaired_duration,
                "original_duration": tts_dur,
                "budget": budget,
                "method": s["repaired_method"],
                "llm_shorten_ms": llm_shorten_ms,
                "re_synthesis_ms": re_synthesis_ms,
                "atempo_ms": atempo_ms,
                "trim_ms": trim_ms,
                "re_synthesis_count": re_synthesis_count,
                "segment_index": idx,
            })

    telemetry.record("duration_repair", {
        "wall_time_ms": round((time.perf_counter() - step_started) * 1000),
        "segment_count": len(segments),
        "retry_count": 0,
    })
    checkpoint_data = {
        "schema_version": 2,
        "job_id": job_id,
        "step_name": "duration_repair",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "segments": segments,
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
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    vietnamese_narration = output_dir / "vietnamese_narration.wav"
    ffmpeg_path = resolve_tool_path(config, "ffmpeg")
    settings = _load_settings(database)
    requested_mix_mode = _normalize_mix_mode(settings.get("mix_mode"))

    segment_inputs: list[tuple[Path, float]] = []
    for seg in segments:
        idx = seg["index"]
        seg_path = tts_dir / f"tts_repaired_{idx}.wav"
        if not seg_path.is_file():
            candidate = seg.get("tts_path")
            seg_path = Path(candidate) if candidate else tts_dir / f"tts_{idx}.wav"
        if not seg_path.is_file() and seg.get("tts_raw_path"):
            seg_path = Path(seg["tts_raw_path"])
        if seg_path.is_file():
            segment_inputs.append((seg_path, float(seg.get("start") or 0.0)))

    if segment_inputs:
        cmd_narration = [str(ffmpeg_path), "-y"]
        for path, _start in segment_inputs:
            cmd_narration.extend(["-i", str(path)])
        delayed_labels: list[str] = []
        filters: list[str] = []
        for input_index, (_path, start_time) in enumerate(segment_inputs):
            delay_ms = max(0, round(start_time * 1000))
            label = f"seg{input_index}"
            filters.append(
                f"[{input_index}:a]aresample=48000,aformat=sample_fmts=s16:channel_layouts=stereo,"
                f"adelay={delay_ms}:all=1[{label}]"
            )
            delayed_labels.append(f"[{label}]")
        filters.append(
            f"{''.join(delayed_labels)}amix=inputs={len(delayed_labels)}:"
            f"duration=longest:dropout_transition=0:normalize=0[narration]"
        )
        cmd_narration.extend(["-filter_complex", ";".join(filters), "-map", "[narration]", str(narration_wav)])
    else:
        cmd_narration = [
            str(ffmpeg_path), "-y",
            "-f", "lavfi",
            "-i", "anullsrc=r=48000:cl=stereo",
            "-t", f"{max(get_wav_duration(original_48k), 0.1):.3f}",
            str(narration_wav),
        ]

    try:
        run_subprocess_with_cancel(cmd_narration, job_id, runner)
    except subprocess.CalledProcessError as e:
        raise AppError(
            500,
            ErrorInfo(
                code="NARRATION_MIX_FAILED",
                message="Failed to build Vietnamese narration track.",
                action="Verify FFmpeg audio filters and TTS segment files.",
                detail=e.stderr or e.stdout,
            )
        )
    shutil.copyfile(narration_wav, vietnamese_narration)

    requested_background = Path(audio_cp["bgm_path"]) if audio_cp.get("bgm_path") else artifacts_dir / "bgm.wav"
    if requested_mix_mode == "background_only" and requested_background.is_file():
        background_wav = requested_background
        mix_mode = "background_only"
    else:
        background_wav = original_48k
        mix_mode = "duck"
        if requested_mix_mode == "background_only":
            logger.warning(
                "Background stem missing for job %s; falling back to duck mix.",
                job_id,
            )

    if mix_mode == "background_only":
        filter_graph = (
            "[0:a]loudnorm=I=-24:TP=-4:LRA=7,alimiter=limit=0.72[bg];"
            "[1:a]loudnorm=I=-16:TP=-1.5:LRA=7,alimiter=limit=0.96[fg];"
            "[bg][fg]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[mixed]"
        )
    else:
        filter_graph = (
            "[0:a]loudnorm=I=-24:TP=-4:LRA=7,alimiter=limit=0.72[bg];"
            "[1:a]loudnorm=I=-16:TP=-1.5:LRA=7,alimiter=limit=0.96[fg];"
            "[fg]asplit=2[fg1][fg2];"
            "[bg][fg1]sidechaincompress=threshold=0.015:ratio=12:"
            "attack=12:release=350[ducked];"
            "[ducked][fg2]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[mixed]"
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
        "schema_version": 3,
        "job_id": job_id,
        "step_name": "mix",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "requested_mix_mode": requested_mix_mode,
        "mix_mode": mix_mode,
        "background_source_path": str(background_wav),
        "narration_segment_input_count": len(segment_inputs),
        "narration_wav_path": str(narration_wav),
        "mixed_wav_path": str(mixed_wav),
        "vietnamese_narration_path": str(vietnamese_narration)
    }
    save_checkpoint(config.data_dir, job_id, "mix", checkpoint_data)
    return checkpoint_data


def render_step(job_id: str, config: AppConfig, database: Database, runner) -> dict:
    mix_cp = load_checkpoint(config.data_dir, job_id, "mix")
    repair_cp = load_checkpoint(config.data_dir, job_id, "duration_repair")
    
    if not mix_cp:
        raise AppError(
            400,
            ErrorInfo(
                code="MISSING_UPSTREAM_CHECKPOINTS",
                message="Mix checkpoint is missing.",
                action="Verify upstream steps."
            )
        )
        
    original_mp4 = original_video_path(config, job_id)
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
        "-af", "alimiter=limit=0.98",
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
    duration_repair_distribution: dict[str, int] = {}
    stretch_factor_distribution: dict[str, int] = {}
    risky_trim_count = 0
    suspected_clipped_tails = 0
    synthesis_retry_count = 0

    asr_cp = load_checkpoint(config.data_dir, job_id, "asr") or {}
    vad_cp = load_checkpoint(config.data_dir, job_id, "vad") or {}
    tts_cp = load_checkpoint(config.data_dir, job_id, "tts") or {}
    telemetry_path = Path(config.data_dir) / "jobs" / job_id / "artifacts" / "telemetry.jsonl"
    telemetry_records = []
    if telemetry_path.is_file():
        for line in telemetry_path.read_text(encoding="utf-8").splitlines():
            try:
                telemetry_records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    for s in segments:
        method = s.get("repaired_method", "none")
        duration_repair_distribution[method] = duration_repair_distribution.get(method, 0) + 1
        factor = s.get("time_stretch_factor")
        if factor not in (None, 1, 1.0):
            key = str(round(float(factor), 2))
            stretch_factor_distribution[key] = stretch_factor_distribution.get(key, 0) + 1
        if s.get("tail_speech_detected"):
            suspected_clipped_tails += 1
        if s.get("duration_repair_risk") in {"warning", "danger"} or s.get("tail_speech_detected"):
            risky_trim_count += 1
        synthesis_retry_count += int(s.get("re_synthesis_count") or 0)
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
                "duration_repair_risk": s.get("duration_repair_risk"),
                "quality_warning": s.get("quality_warning"),
            })

    cache_hits = sum(1 for record in telemetry_records if record.get("cache_hit") is True)
    cache_misses = sum(1 for record in telemetry_records if record.get("cache_miss") is True)
    cache_total = cache_hits + cache_misses
    step_rtf: dict[str, float] = {}
    for record in telemetry_records:
        step = str(record.get("step") or "")
        if record.get("real_time_factor") is not None:
            step_rtf[step] = step_rtf.get(step, 0.0) + float(record["real_time_factor"])

    checkpoint_data = {
        "schema_version": 3,
        "job_id": job_id,
        "step_name": "qc",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_segments": total_segments,
        "repaired_count": repaired_segments,
        "shortened_count": shortened_segments,
        "stretched_count": stretched_segments,
        "warnings": warnings,
        "output_video_path": output_path,
        "asr_timing_coverage": None,
        "alignment_availability": asr_cp.get("alignment_status", "unknown"),
        "alignment_mode": asr_cp.get("alignment_mode", "unknown"),
        "vad_speech_ratio": vad_cp.get("vad_speech_ratio"),
        "asr_mode": asr_cp.get("dense_or_sparse_mode", "unknown"),
        "sparse_vs_dense_asr": asr_cp.get("dense_or_sparse_mode", "unknown"),
        "tts_cache_hit_rate": round(cache_hits / cache_total, 4) if cache_total else None,
        "synthesis_retry_count": synthesis_retry_count,
        "duration_repair_distribution": duration_repair_distribution,
        "stretch_factor_distribution": stretch_factor_distribution,
        "risky_trim_count": risky_trim_count,
        "segment_overlap_count": 0,
        "suspected_clipped_tail_count": suspected_clipped_tails,
        "model_cold_start_count": sum(1 for record in telemetry_records if record.get("cold_start") is True),
        "total_rtf_by_step": step_rtf,
        "tts_segment_count": len(tts_cp.get("segments", [])) if tts_cp else total_segments,
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
