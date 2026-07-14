import array
import hashlib
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
from .vendor import VendorManifest, VendorResolver, prefer_macos_ffmpeg_full
from .adapters.translation import GoogleFreeTranslator
from .adapters.tts import TTS_VOICE_INSTRUCT_PREFIX, TtsSession, create_tts_adapter, prepare_spoken_text_for_tts
from .adapters.asr import configure_gpu_manager, reset_model_cache, transcribe_audio
from .adapters.vad_feedback import filter_asr_false_positives
from .adapters.vad_energy import filter_low_vocal_energy_segments
from .adapters.vad_silencedetect import (
    detect_speech_regions_silencedetect,
    model_config_label as silencedetect_model_config_label,
    silencedetect_filter,
)
from .adapters.vad_silero import model_config_label as silero_model_config_label
from .adapters.vad_silero import vad_step_silero
from .gpu_manager import global_gpu_manager
from .hardware import resolve_inference_device
from .audio_probe import get_audio_duration
from .duration_safety import classify_stretch, tail_has_speech
from .duration_fit_policy import (
    acceptable_duration_fit,
    classify_duration_fit,
    classify_stretch_with_policy,
    clamp_automatic_tempo,
    policy_from_settings,
    should_lengthen_for_timing,
    should_shorten_for_timing,
    tempo_factor_for_duration,
    timing_profile_from_segment,
)
from .timing_profile import attach_timing_profiles
from .translation_candidates import translate_segments_with_candidates
from .tts_candidate_retry import synthesize_with_candidate_retry, timing_attempt_limits
from .tts_cache import (
    build_tts_cache_identity,
    cache_key_from_identity,
    segment_wav_cache_valid,
    write_tts_sidecar,
)
from .tts_speech_analysis import attach_speech_metrics, measure_speech_envelope
from .timing_qc_metrics import compute_timing_qc_metrics
from .release_quality_gate import evaluate_release_gate
from .tts_attempt_budget import budget_from_settings
from .voice_duration_profile import update_voice_profile_from_sample
from .voice_profile_policy import effective_voice_profile
from .segmentation import (
    MAX_SEGMENT_SPLIT_SECONDS,
    consolidate_short_segments,
    merge_incomplete_sentence_segments,
    split_long_segments_with_alignment,
    split_segments_by_alignment_pauses,
)
from .sparse_asr import (
    build_sparse_chunks,
    build_stitched_timeline,
    map_stitched_segments_to_source,
    merge_overlapping_segments,
    rebase_sparse_segments,
    should_use_sparse_asr,
    stitched_timeline_duration,
)
from .timing_conflict_repair import repair_conflict_clusters
from .tts_provenance import resolve_voiced_tts_path, spoken_text
from .timing_placement import (
    compute_placement_starts,
    enforce_zero_overlap_placements,
    schedule_soft_placements,
    segments_with_voiced_overlap,
)
from .segment_mix import (
    annotate_segment_mix_caps,
    build_narration_amix_filter,
    build_narration_segment_filter,
)
from .telemetry import TelemetrySink
from .dubbing_languages import default_speaking_rate_wps, dub_language_from_settings, dub_language_label
from .translation_duration import annotate_translation_duration, build_translation_timing_guidance, duration_prompt_suffix
from .adapters.subtitles import (
    build_subtitle_cues,
    ffmpeg_subtitles_filter,
    probe_video_dimensions,
    subtitles_filter_available,
    write_ass_file,
)
from .final_dub_alignment import (
    align_job_segments_final_dub,
    compute_subtitle_qc_metrics,
    refresh_all_segment_dub_word_timestamps,
    segment_has_usable_dub_words,
    summarize_alignment_results,
)
from .adapters.gemini import (
    GeminiKeyPool,
    GeminiTranslator,
)
from .translation_timing_rewrite import (
    lengthen_translation_for_timing,
    shorten_translation_for_timing,
    translation_backend,
)
from .adapters.openai_compat import OpenAiCompatTranslator
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
    yt_dlp_cookie_args_for_file,
)

logger = logging.getLogger(__name__)

ASR_ALIGNMENT_SCHEMA_VERSION = 2
DEFAULT_EXACT_TIMING_TOLERANCE_MS = 40
DEFAULT_EXACT_TIMING_ENABLED = True
DEFAULT_EXACT_TIMING_MAX_STRETCH = 1.2
SHORT_TTS_TAIL_PAD_MAX_GAP_SEC = 1.5


def _speech_slot_duration(segment: dict) -> float:
    """Duration of the detected speech window, excluding pause until the next segment."""
    original = segment.get("original_duration")
    if original is not None:
        return max(float(original), 0.05)
    start = float(segment.get("start", 0.0) or 0.0)
    end = float(segment.get("end", start) or start)
    return max(end - start, 0.05)


def _repair_target_duration(segment: dict, budget: float, tolerance_sec: float) -> float:
    """Cap exact-timing repair to the speech slot; do not pad across inter-segment silence."""
    speech_slot = _speech_slot_duration(segment)
    slack = max(tolerance_sec, SHORT_TTS_TAIL_PAD_MAX_GAP_SEC)
    capped = speech_slot + slack
    if budget <= 0:
        return capped
    return min(float(budget), capped)


def _preferred_timing_budget(segment: dict, settings: dict) -> float:
    budget = float(segment.get("duration_budget") or 0.0)
    exact_enabled, tolerance_sec, _max_stretch = _normalize_exact_timing_settings(settings)
    if not exact_enabled:
        return budget
    return _repair_target_duration(segment, budget, tolerance_sec)


def _wav_has_usable_signal(path: Path, *, min_rms: float = 0.0035) -> bool:
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.readframes(handle.getnframes())
            channels = handle.getnchannels()
        samples = array.array("h")
        samples.frombytes(frames)
        if not samples:
            return False
        if channels > 1:
            mono = [float(samples[index]) / 32768.0 for index in range(0, len(samples), channels)]
        else:
            mono = [float(sample) / 32768.0 for sample in samples]
        if not mono:
            return False
        rms = (sum(sample * sample for sample in mono) / len(mono)) ** 0.5
        return rms >= min_rms
    except Exception:
        return False


def _preferred_recognition_audio(job_dir: Path, fallback_audio_16k: Path) -> tuple[Path, str]:
    audio_cp = load_checkpoint(job_dir.parents[1], job_dir.name, "extract_audio") or {}
    vocals_16k_path = Path(audio_cp["vocals_16k_path"]) if audio_cp.get("vocals_16k_path") else None
    if vocals_16k_path and vocals_16k_path.is_file() and _wav_has_usable_signal(vocals_16k_path):
        return vocals_16k_path, "vocals_16k"
    return fallback_audio_16k, "mixed_audio_16k"


def _speaking_rate_wps(settings: dict) -> float:
    fallback = default_speaking_rate_wps(dub_language_from_settings(settings))
    try:
        rate = float(settings.get("vietnamese_speaking_rate_wps", fallback) or fallback)
    except (TypeError, ValueError):
        rate = fallback
    return max(2.0, min(5.0, rate))


def _update_speaking_rate_calibration(database: Database, segments: list[dict]) -> float | None:
    rates: list[float] = []
    for segment in segments:
        duration = float(segment.get("tts_duration") or 0.0)
        words = _estimate_word_count(str(segment.get("translation") or ""))
        if duration >= 0.25 and words >= 2:
            rates.append(words / duration)
    if not rates:
        return None
    measured = sum(rates) / len(rates)
    settings = _load_settings(database)
    prior = _speaking_rate_wps(settings)
    blended = round((0.7 * prior) + (0.3 * measured), 2)
    blended = max(2.0, min(5.0, blended))
    save_setting(database, "vietnamese_speaking_rate_wps", blended)
    return blended


def _lengthen_min_gap_sec(settings: dict) -> float:
    return max(0.2, float(settings.get("short_tts_lengthen_min_gap_sec", SHORT_TTS_TAIL_PAD_MAX_GAP_SEC) or SHORT_TTS_TAIL_PAD_MAX_GAP_SEC))


def _lengthen_max_ratio(settings: dict) -> float:
    return max(1.05, float(settings.get("short_tts_lengthen_max_ratio", 1.6) or 1.6))


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
    preferred = prefer_macos_ffmpeg_full(tool, resolved)
    return preferred.path if preferred.path is not None else resolved.path


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
    cookies_file: str | Path | None = None,
) -> tuple[subprocess.CompletedProcess, str, list[str]]:
    """Run yt-dlp with cookies.txt first (if set), then browser cookies Firefox → Chrome → Edge → Brave."""
    last_exc: subprocess.CalledProcessError | None = None
    browsers_tried: list[str] = []

    cookies_path = Path(str(cookies_file).strip()) if cookies_file else None
    if cookies_path and cookies_path.is_file():
        source = f"file:{cookies_path}"
        browsers_tried.append(source)
        cmd = [str(yt_dlp_path), *yt_dlp_cookie_args_for_file(cookies_path), *args]
        try:
            result = run_subprocess_with_cancel(cmd, job_id, runner, timeout=timeout)
            return result, source, browsers_tried
        except subprocess.CalledProcessError as exc:
            last_exc = exc

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
    settings = _load_settings(database)
    cookies_file = str(settings.get("cookies_file") or "").strip() or None
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
            cookies_file=cookies_file,
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
    settings = _load_settings(database)
    cookies_file = str(settings.get("cookies_file") or "").strip() or None

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
            cookies_file=cookies_file,
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
    vocals_16k = artifacts_dir / "vocals_16k.wav"
    bgm_16k = artifacts_dir / "bgm_16k.wav"
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
    vocals_16k_path: str | None = None
    bgm_16k_path: str | None = None
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
            cmd_vocals_16k = [
                str(ffmpeg_path), "-y",
                "-i", str(vocals_wav),
                "-acodec", "pcm_s16le",
                "-ac", "1",
                "-ar", "16000",
                str(vocals_16k),
            ]
            run_subprocess_with_cancel(cmd_vocals_16k, job_id, runner)
            vocals_16k_path = str(vocals_16k)
            cmd_bgm_16k = [
                str(ffmpeg_path), "-y",
                "-i", str(bgm_wav),
                "-acodec", "pcm_s16le",
                "-ac", "1",
                "-ar", "16000",
                str(bgm_16k),
            ]
            run_subprocess_with_cancel(cmd_bgm_16k, job_id, runner)
            bgm_16k_path = str(bgm_16k)
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
        "schema_version": 3,
        "job_id": job_id,
        "step_name": "extract_audio",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "original_48k_path": str(original_48k),
        "audio_16k_path": str(audio_16k),
        "bgm_path": bgm_path,
        "vocals_path": vocals_path,
        "vocals_16k_path": vocals_16k_path,
        "bgm_16k_path": bgm_16k_path,
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
        device = resolve_inference_device(str((settings or {}).get("qwen3_device", "cuda:0") or "cuda:0"))
        global_gpu_manager().evict("asr", device, reason="asr_step_complete")
    except Exception:
        logger.debug("ASR GPU lease eviction failed", exc_info=True)


def _release_tts_gpu_resources(settings: dict | None = None) -> None:
    """Shutdown OmniVoice workers before subtitle ASR so Qwen can use VRAM."""
    try:
        from .adapters.omnivoice_client import release_all_clients

        release_all_clients()
    except Exception:
        logger.debug("TTS worker release failed", exc_info=True)
    try:
        device = resolve_inference_device(str((settings or {}).get("omnivoice_device", "cuda:0") or "cuda:0"))
        global_gpu_manager().evict("tts", device, reason="tts_step_complete")
    except Exception:
        logger.debug("TTS GPU lease eviction failed", exc_info=True)
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        logger.debug("CUDA cache cleanup after TTS release failed", exc_info=True)


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

    recognition_audio, recognition_audio_source = _preferred_recognition_audio(job_dir, audio_16k)
    settings = _load_settings(database)
    vad_engine = str(settings.get("vad_engine", "silero") or "silero").strip().lower()
    if vad_engine not in {"silero", "silencedetect"}:
        vad_engine = "silero"

    ffmpeg_path = resolve_tool_path(config, "ffmpeg")
    telemetry = TelemetrySink(config.data_dir, job_id)
    started = time.perf_counter()
    total_duration = get_audio_duration(recognition_audio, ffprobe_path=ffprobe_sibling_for(ffmpeg_path))

    if vad_engine == "silencedetect":
        noise_db = float(settings.get("silencedetect_noise_db", -30) or -30)
        min_silence_sec = float(settings.get("silencedetect_min_silence_sec", 0.5) or 0.5)
        cmd = [
            str(ffmpeg_path),
            "-i", str(recognition_audio),
            "-af", silencedetect_filter(noise_db, min_silence_sec),
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
                "vad_engine": vad_engine,
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
            ) from e

        speech_regions = detect_speech_regions_silencedetect(
            recognition_audio,
            total_duration=total_duration,
            stderr=res.stderr,
        )
        model_config = silencedetect_model_config_label(noise_db, min_silence_sec)
    else:
        try:
            speech_regions = vad_step_silero(
                recognition_audio,
                threshold=float(settings.get("silero_vad_threshold", 0.5) or 0.5),
                min_speech_duration_ms=int(settings.get("silero_vad_min_speech_duration_ms", 250) or 250),
                min_silence_duration_ms=int(settings.get("silero_vad_min_silence_duration_ms", 300) or 300),
                speech_pad_ms=int(settings.get("silero_vad_speech_pad_ms", 150) or 150),
            )
        except Exception as e:
            telemetry.record("vad", {
                "status": "failed",
                "wall_time_ms": round((time.perf_counter() - started) * 1000),
                "audio_duration_sec": total_duration,
                "model_config": "silero_vad",
                "vad_engine": vad_engine,
                "retry_count": 0,
            })
            raise AppError(
                500,
                ErrorInfo(
                    code="VAD_DETECTION_FAILED",
                    message="Failed to run Silero VAD on audio.",
                    action="Verify silero-vad is installed or switch vad_engine to silencedetect.",
                    detail=str(e),
                )
            ) from e
        model_config = silero_model_config_label(
            threshold=float(settings.get("silero_vad_threshold", 0.5) or 0.5),
            min_speech_duration_ms=int(settings.get("silero_vad_min_speech_duration_ms", 250) or 250),
            min_silence_duration_ms=int(settings.get("silero_vad_min_silence_duration_ms", 300) or 300),
            speech_pad_ms=int(settings.get("silero_vad_speech_pad_ms", 150) or 150),
        )

    speech_duration = sum(region["end"] - region["start"] for region in speech_regions)
    speech_ratio = round(speech_duration / total_duration, 4) if total_duration > 0 else 0.0
    telemetry.record("vad", {
        "status": "ok",
        "wall_time_ms": round((time.perf_counter() - started) * 1000),
        "audio_duration_sec": total_duration,
        "model_config": model_config,
        "vad_engine": vad_engine,
        "retry_count": 0,
        "speech_region_count": len(speech_regions),
        "vad_speech_ratio": speech_ratio,
        "recognition_audio_source": recognition_audio_source,
    })

    checkpoint_data = {
        "schema_version": 2,
        "job_id": job_id,
        "step_name": "vad",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_duration": round(total_duration, 2),
        "speech_regions": speech_regions,
        "vad_speech_ratio": speech_ratio,
        "vad_engine": vad_engine,
        "recognition_audio_source": recognition_audio_source,
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

    recognition_audio, recognition_audio_source = _preferred_recognition_audio(job_dir, audio_16k)

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
        from .gpu_lease import clear_gpu_lease_state

        clear_gpu_lease_state(reason=f"asr_step:{job_id}")
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
                if float(region.get("end", 0.0) or 0.0) - float(region.get("start", 0.0) or 0.0) > MAX_SEGMENT_SPLIT_SECONDS
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
            "device": resolve_inference_device(str(settings.get("qwen3_device", "cuda:0") or "cuda:0")),
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
                    merge_gap_sec=float(settings.get("sparse_asr_merge_gap_sec", 0.25) or 0.25),
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
                        "-i", str(recognition_audio),
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
            asr_result = transcribe_audio(recognition_audio, **asr_kwargs)
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
                "recognition_audio_source": recognition_audio_source,
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
            "recognition_audio_source": recognition_audio_source,
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
            "recognition_audio_source": recognition_audio_source,
        }
        save_checkpoint(config.data_dir, job_id, "asr", checkpoint_data)
        return checkpoint_data
    finally:
        _release_asr_gpu_models(settings)


def _split_long_asr_segments_with_vad(
    raw_segments: list[dict],
    speech_regions: list[dict],
    *,
    max_segment_seconds: float = MAX_SEGMENT_SPLIT_SECONDS,
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

        from .segmentation import allocate_text_across_regions

        text_chunks = allocate_text_across_regions(text, overlapping_regions)
        if len(text_chunks) != len(overlapping_regions):
            split_segments.append(segment)
            continue
        for region, chunk_text in zip(overlapping_regions, text_chunks, strict=True):
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
    settings = _load_settings(database)

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

    filter_enabled = bool(settings.get("vad_false_positive_filter_enabled", True))
    raw_segments, rejected_segments = filter_asr_false_positives(
        raw_segments,
        enabled=filter_enabled,
    )

    energy_filter_enabled = bool(settings.get("vad_energy_filter_enabled", True))
    extract_cp = load_checkpoint(config.data_dir, job_id, "extract_audio") or {}
    vocals_16k_path = extract_cp.get("vocals_16k_path")
    bgm_16k_path = extract_cp.get("bgm_16k_path")
    min_vocal_ratio = float(settings.get("vad_energy_min_vocal_ratio", 1.15) or 1.15)
    raw_segments, energy_rejected = filter_low_vocal_energy_segments(
        raw_segments,
        vocals_path=vocals_16k_path,
        bgm_path=bgm_16k_path,
        enabled=energy_filter_enabled,
        min_vocal_ratio=min_vocal_ratio,
    )
    rejected_segments.extend(energy_rejected)
    if energy_rejected:
        for rejected in energy_rejected:
            logger.info(
                "Filtered low vocal-energy segment for job %s: region=%.2f-%.2f text=%r",
                job_id,
                float(rejected.get("start", 0.0) or 0.0),
                float(rejected.get("end", 0.0) or 0.0),
                str(rejected.get("text") or "")[:80],
            )
    if rejected_segments:
        for rejected in rejected_segments:
            logger.info(
                "Filtered likely VAD false positive for job %s: region=%.2f-%.2f reason=%s text=%r",
                job_id,
                float(rejected.get("start", 0.0) or 0.0),
                float(rejected.get("end", 0.0) or 0.0),
                rejected.get("vad_false_positive_reason"),
                str(rejected.get("text") or "")[:80],
            )

    raw_segments = merge_incomplete_sentence_segments(raw_segments)

    raw_segments = split_long_segments_with_alignment(
        raw_segments,
        vad_cp.get("speech_regions", []),
        asr_cp.get("aligned_units", []) or [],
    )
    raw_segments = _split_long_asr_segments_with_vad(
        raw_segments,
        vad_cp.get("speech_regions", []),
    )
    raw_segments = split_segments_by_alignment_pauses(
        raw_segments,
        asr_cp.get("aligned_units", []) or [],
    )
    raw_segments = consolidate_short_segments(raw_segments)

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
        "schema_version": 2,
        "job_id": job_id,
        "step_name": "normalize_segments",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "segments": normalized,
        "vad_false_positive_rejected_count": len(rejected_segments),
        "vad_false_positive_filter_enabled": filter_enabled,
        "vad_energy_filter_enabled": energy_filter_enabled,
        "vad_energy_rejected_count": len(energy_rejected),
    }
    save_checkpoint(config.data_dir, job_id, "normalize_segments", checkpoint_data)
    return checkpoint_data


def _translate_candidates_batch(
    settings: dict,
    database: Database,
    segments: list[dict],
    texts: list[str],
    *,
    source_lang: str,
    target_lang: str,
    timing_profiles: list[dict],
    speaking_rate: float,
) -> list[list[dict]]:
    candidate_count = int(settings.get("timing_translation_candidate_count", 3) or 3)
    backend = settings.get("translation_backend", "google_free")
    if backend == "gemini":
        key_pool = GeminiKeyPool(
            settings.get("gemini_api_keys", []),
            cursor=int(settings.get("gemini_key_cursor", 0)),
        )
        translator = GeminiTranslator(
            key_pool,
            model=settings.get("gemini_translation_model", "gemini-2.5-flash"),
        )
        batches = translator.translate_candidates(
            segments,
            texts,
            source_lang,
            target_lang,
            timing_profiles=timing_profiles,
            speaking_rate_wps=speaking_rate,
            candidate_count=candidate_count,
        )
        save_setting(database, "gemini_key_cursor", translator.key_pool.cursor)
        return batches
    if backend == "openai":
        translator = OpenAiCompatTranslator(
            api_base=str(settings.get("openai_api_base") or ""),
            api_key=str(settings.get("openai_api_key") or ""),
            model=str(settings.get("openai_translation_model") or ""),
        )
        return translator.translate_candidates(
            segments,
            texts,
            source_lang,
            target_lang,
            timing_profiles=timing_profiles,
            speaking_rate_wps=speaking_rate,
            candidate_count=candidate_count,
        )
    return []


def _tts_text_fingerprint(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _tts_cache_identity_for_segment(settings: dict, text: str, *, language: str) -> dict:
    return build_tts_cache_identity(settings, text=str(text or ""), language=language)


def _tts_cache_key(settings: dict, text: str, *, language: str) -> str:
    return cache_key_from_identity(_tts_cache_identity_for_segment(settings, text, language=language))


def _timing_aware_tts_enabled(settings: dict) -> bool:
    return bool(settings.get("timing_candidate_translation_enabled", False))


def _segment_speech_duration(segment: dict, wav_path: Path | None = None) -> float:
    speech = segment.get("tts_speech_duration")
    if speech is not None and float(speech) > 0:
        return float(speech)
    if wav_path is not None and wav_path.is_file():
        envelope = measure_speech_envelope(wav_path)
        attach_speech_metrics(segment, envelope)
        if envelope.speech_duration > 0:
            return float(envelope.speech_duration)
        return float(envelope.raw_wav_duration or 0.0)
    return float(segment.get("tts_duration") or 0.0)


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

    if translation_backend == "openai":
        translator = OpenAiCompatTranslator(
            api_base=str(settings.get("openai_api_base") or ""),
            api_key=str(settings.get("openai_api_key") or ""),
            model=str(settings.get("openai_translation_model") or ""),
        )
        return translator.translate(
            texts,
            source=source_lang,
            target=target_lang,
            duration_budgets=duration_budgets,
            timing_guidance=timing_guidance,
        )

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
    if translation_backend not in {"google_free", "gemini", "openai"}:
        raise AppError(
            400,
            ErrorInfo(
                code="UNSUPPORTED_TRANSLATION_BACKEND",
                message="The selected translation backend is not available.",
                action="Choose Google Translate Free, Gemini, or OpenAPI in Settings."
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

    attach_timing_profiles(segments, total_duration=norm_cp.get("total_duration"), settings=settings)
    texts = [segment["text"] for segment in segments]
    duration_budgets = [_preferred_timing_budget(segment, settings) for segment in segments]
    speaking_rate = _speaking_rate_wps(settings)
    aligned_units = asr_cp.get("aligned_units", []) or []
    timing_guidance = []
    for segment, budget in zip(segments, duration_budgets, strict=True):
        profile = segment.get("timing_profile") or {}
        speech_target = float(profile.get("speech_target_duration") or budget or 0.0)
        segment["repair_target_duration"] = round(speech_target, 2) if speech_target > 0 else 0.0
        guidance = build_translation_timing_guidance(
            {**segment, "repair_target_duration": speech_target},
            aligned_units=_aligned_units_for_segment(segment, aligned_units),
            speaking_rate_wps=speaking_rate,
        )
        segment.update(guidance)
        segment["timing_guidance"] = guidance
        timing_guidance.append(guidance)

    timing_profiles = [segment.get("timing_profile") or {} for segment in segments]

    def _translate_fn(
        settings_arg,
        database_arg,
        texts_arg,
        *,
        source_lang: str,
        target_lang: str,
        duration_budgets: list[float] | None = None,
        timing_guidance: list[dict] | None = None,
    ) -> list[str]:
        return _translate_texts(
            settings_arg,
            database_arg,
            texts_arg,
            source_lang=source_lang,
            target_lang=target_lang,
            duration_budgets=duration_budgets,
            timing_guidance=timing_guidance,
        )

    def _translate_candidates_fn(
        segments_arg,
        texts_arg,
        *,
        source: str,
        target: str,
        timing_profiles: list[dict],
        settings: dict,
        database: Database,
        speaking_rate_wps: float,
    ) -> list[list[dict]]:
        return _translate_candidates_batch(
            settings,
            database,
            segments_arg,
            texts_arg,
            source_lang=source,
            target_lang=target,
            timing_profiles=timing_profiles,
            speaking_rate=speaking_rate_wps,
        )

    translate_segments_with_candidates(
        settings,
        database,
        segments,
        source_lang=source_lang,
        target_lang=target_lang,
        translate_fn=_translate_fn,
        translate_candidates_fn=_translate_candidates_fn,
        data_dir=config.data_dir,
    )

    if any(not str(segment.get("translation") or "").strip() for segment in segments):
        raise AppError(
            502,
            ErrorInfo(
                code="TRANSLATION_COUNT_MISMATCH",
                message="Translation backend returned incomplete segment translations.",
                action="Retry translation or switch translation backend.",
                retryable=True,
            )
        )
    speaking_rate = float(settings.get("vietnamese_speaking_rate_wps") or 3.2)
    voice_profile = effective_voice_profile(settings, language=target_lang, data_dir=config.data_dir)
    for segment in segments:
        segment.update(
            annotate_translation_duration(
                segment,
                speaking_rate_wps=speaking_rate,
                voice_profile=voice_profile,
                language=target_lang,
            )
        )
        segment["voice_profile_key"] = voice_profile.get("profile_key")
        segment["voice_profile_source"] = voice_profile.get("profile_source") or voice_profile.get("source")
        segment["voice_profile_samples"] = voice_profile.get("sample_count_accepted") or voice_profile.get("samples")
        segment["voice_profile_syllables_per_second"] = voice_profile.get("syllables_per_second")
        segment["prediction_method"] = voice_profile.get("prediction_method")

    checkpoint_data = {
        "schema_version": 2,
        "job_id": job_id,
        "step_name": "translate",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "title_vi": title_vi,
        "segments": segments
    }
    save_checkpoint(config.data_dir, job_id, "translate", checkpoint_data)
    return checkpoint_data


def _default_tts_voice(settings: dict) -> str:
    instruct = str(settings.get("omnivoice_instruct") or "").strip()
    if instruct:
        return f"{TTS_VOICE_INSTRUCT_PREFIX}{instruct}"
    ref_audio = str(settings.get("omnivoice_ref_audio") or "").strip()
    if ref_audio:
        return ref_audio
    return "auto"


def _anchor_transcript_for(settings: dict) -> str | None:
    manual = str(settings.get("omnivoice_ref_text") or "").strip()
    if manual:
        return manual
    ref_audio = str(settings.get("omnivoice_ref_audio") or "").strip()
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
    ref_audio = str(settings.get("omnivoice_ref_audio") or "").strip()
    clone = bool(ref_audio)
    clone_mode = "reference"
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


def _apply_global_speed_if_needed(
    *,
    ffmpeg_path: Path,
    input_path: Path,
    output_dir: Path,
    index: int,
    global_speed: float,
    job_id: str,
    runner,
) -> tuple[Path, float]:
    if abs(global_speed - 1.0) <= 0.001:
        return input_path, get_wav_duration(input_path)
    speed_file = output_dir / f"tts_speed_{index}.wav"
    speed_file.unlink(missing_ok=True)
    _run_ffmpeg_audio_filter(
        ffmpeg_path,
        input_path,
        speed_file,
        filter_expr=_build_atempo_chain(global_speed),
        job_id=job_id,
        runner=runner,
    )
    return speed_file, get_wav_duration(speed_file)


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


def _tail_silence_pad_filter(current_duration: float, target_dur: float) -> str:
    duration = max(0.05, float(current_duration))
    fade_out = min(0.03, max(0.01, duration * 0.05))
    fade_start = max(0.0, duration - fade_out)
    return (
        f"afade=t=out:st={fade_start:.3f}:d={fade_out:.3f},"
        f"apad=pad_dur={target_dur + 0.2:.3f},"
        f"atrim=0:{target_dur:.3f}"
    )


def _timing_rewrite_method_prefix(settings: dict) -> str:
    return translation_backend(settings)


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
    micro_batch_size = 4
    timing_enabled = _timing_aware_tts_enabled(settings)
    dub_lang = dub_language_from_settings(settings)
    dub_lang_label = dub_language_label(dub_lang, english=True)
    voice_profile = effective_voice_profile(settings, language=dub_lang, data_dir=config.data_dir)
    global_speed = float(settings.get("tts_global_speed") or 1.0)

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

            measure_path = final_tts if strategy == "per_segment" and final_tts.is_file() else raw_tts
            if measure_path.is_file():
                envelope = measure_speech_envelope(measure_path)
                attach_speech_metrics(s, envelope)

            s["tts_raw_path"] = str(raw_tts)
            s["tts_path"] = str(final_tts) if strategy == "per_segment" else None
            spoken_text = prepare_spoken_text_for_tts(
                str(s.get("translation") or entry.get("text") or ""),
                speech_duration=float(s.get("original_duration") or 0.0),
            )
            s["tts_spoken_text"] = spoken_text
            cache_identity = _tts_cache_identity_for_segment(settings, spoken_text, language=dub_lang)
            s["tts_cache_key"] = cache_key_from_identity(cache_identity)
            s["tts_text_fingerprint"] = cache_identity["translation_text_hash"][:16]
            write_tts_sidecar(raw_tts, cache_identity, extra={
                "segment_index": idx,
                "tts_chunk_count": s.get("tts_chunk_count"),
                "tts_chunking_used": s.get("tts_chunking_used"),
                "tts_fidelity_status": s.get("tts_fidelity_status"),
            })
            s["tts_session_reused"] = True
            telemetry.record("tts_segment", {
                "wall_time_ms": synthesize_ms + conversion_ms,
                "audio_duration_sec": s["tts_duration"],
                "speech_duration_sec": s.get("tts_speech_duration"),
                "tts_session_create_ms": session_create_ms,
                "synthesize_ms": synthesize_ms,
                "conversion_ms": conversion_ms,
                "output_write_ms": conversion_ms,
                "segment_index": idx,
                "retry_count": int(s.get("tts_attempt_count") or 0),
                "cache_hit": None,
                "cache_miss": None,
                "model_config": str(settings.get("omnivoice_model", "")),
                "raw_tts_format": "wav_pcm16le_native",
                "tts_micro_batch_enabled": micro_batch_enabled,
                "tts_micro_batch_size": batch_size,
                "tts_batch_wall_time_ms": batch_wall_time_ms,
                "voice_profile_key": voice_profile.get("profile_key"),
                "voice_profile_source": voice_profile.get("profile_source") or voice_profile.get("source"),
                "prediction_method": voice_profile.get("prediction_method"),
                "chunk_count": s.get("tts_chunk_count"),
                "chunked_segment": bool(s.get("tts_chunking_used")),
                "chunk_retry_count": s.get("tts_chunk_retry_count"),
                "chunk_cache_hits": s.get("tts_chunk_cache_hits"),
                "chunk_cache_misses": s.get("tts_chunk_cache_misses"),
                "fidelity_similarity": s.get("tts_text_similarity"),
                "fidelity_status": s.get("tts_fidelity_status"),
            })
            if (
                bool(settings.get("voice_duration_profile_enabled", True))
                and not timing_enabled
                and raw_tts.is_file()
            ):
                speech_duration = _segment_speech_duration(s, raw_tts)
                update_voice_profile_from_sample(
                    settings,
                    text=str(s.get("translation") or entry.get("text") or ""),
                    speech_duration_sec=speech_duration,
                    data_dir=config.data_dir,
                    language=dub_lang,
                    user_speed_not_unity=abs(global_speed - 1.0) > 0.01,
                    measurement_confidence=float(s.get("tts_speech_measurement_confidence") or 1.0),
                )

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
            text = prepare_spoken_text_for_tts(
                str(s.get("translation") or ""),
                speech_duration=float(s.get("original_duration") or 0.0),
            )
            s["tts_spoken_text"] = text
            cache_identity = _tts_cache_identity_for_segment(settings, str(text or ""), language=dub_lang)
            cache_key = cache_key_from_identity(cache_identity)

            raw_tts = tts_dir / f"tts_raw_{idx}.wav"
            final_tts = tts_dir / f"tts_{idx}.wav"

            if (
                strategy == "per_segment"
                and final_tts.is_file()
                and raw_tts.is_file()
                and segment_wav_cache_valid(
                    raw_tts,
                    cache_identity,
                    text=str(text or ""),
                    settings=settings,
                    tts_dir=tts_dir,
                    segment_index=idx,
                )
            ):
                s["tts_duration"] = round(get_wav_duration(final_tts), 2)
                s["tts_raw_path"] = str(raw_tts) if raw_tts.is_file() else str(final_tts)
                s["tts_path"] = str(final_tts)
                s["tts_cache_key"] = cache_key
                s["tts_cache_hit"] = True
                s["tts_session_reused"] = True
                _segment_speech_duration(s, final_tts)
                continue

            if (
                strategy != "per_segment"
                and raw_tts.is_file()
                and segment_wav_cache_valid(
                    raw_tts,
                    cache_identity,
                    text=str(text or ""),
                    settings=settings,
                    tts_dir=tts_dir,
                    segment_index=idx,
                )
            ):
                s["tts_duration"] = round(get_wav_duration(raw_tts), 2)
                s["tts_raw_path"] = str(raw_tts)
                s["tts_path"] = None
                s["tts_cache_key"] = cache_key
                s["tts_cache_hit"] = True
                s["tts_session_reused"] = True
                _segment_speech_duration(s, raw_tts)
                continue

            if raw_tts.is_file():
                raw_tts.unlink()

            if final_tts.is_file():
                final_tts.unlink()

            entry = {"segment": s, "text": text, "raw_tts": raw_tts, "final_tts": final_tts}
            use_retry = timing_enabled and len(s.get("translation_candidates") or []) > 1
            if use_retry:
                flush_pending()

                def synthesize_one(candidate_text: str, output_path: Path) -> None:
                    session.synthesize(candidate_text, output_path, segment=s)

                segment_budget = budget_from_settings(settings)
                synth_started = time.perf_counter()
                synthesize_with_candidate_retry(
                    s,
                    settings=settings,
                    data_dir=config.data_dir,
                    language=dub_lang,
                    session=session,
                    synthesize_one=synthesize_one,
                    wav_path=raw_tts,
                    database=database,
                    estimate_word_count=_estimate_word_count,
                    dub_lang_label=dub_lang_label,
                    attempt_budget=segment_budget,
                )
                synthesize_ms = round((time.perf_counter() - synth_started) * 1000)
                if not raw_tts.is_file():
                    session.synthesize(str(s.get("translation") or text), raw_tts, segment=s)
                finish_segment(entry, synthesize_ms=synthesize_ms, batch_size=1, batch_wall_time_ms=synthesize_ms)
                continue

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

    from .omnivoice_chunking import segment_text_diagnostics

    chunked_segments = 0
    total_chunks = 0
    chunk_retries = 0
    fidelity_checked = 0
    fidelity_good = 0
    fidelity_poor = 0
    fidelity_failed = 0
    fidelity_scores: list[float] = []
    very_long_count = 0
    for s in segments:
        diag = segment_text_diagnostics(str(s.get("translation") or ""), settings)
        flags = diag.get("segment_diagnostics") or []
        if "very_long_text_segment" in flags:
            very_long_count += 1
        for flag in flags:
            existing = list(s.get("segment_diagnostics") or [])
            if flag not in existing:
                existing.append(flag)
            s["segment_diagnostics"] = existing
        if s.get("tts_chunking_used"):
            chunked_segments += 1
        total_chunks += int(s.get("tts_chunk_count") or 1)
        chunk_retries += int(s.get("tts_chunk_retry_count") or 0)
        status = str(s.get("tts_fidelity_status") or "not_checked")
        if status != "not_checked":
            fidelity_checked += 1
        if status == "good":
            fidelity_good += 1
        elif status in {"poor", "review"}:
            fidelity_poor += 1
        elif status == "failed":
            fidelity_failed += 1
        score = s.get("tts_text_similarity")
        if isinstance(score, (int, float)):
            fidelity_scores.append(float(score))

    calibrated_rate = _update_speaking_rate_calibration(database, segments)

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
        "model_config": str(settings.get("omnivoice_model", "")),
        "conversion_strategy": strategy,
        "tts_micro_batch_enabled": micro_batch_enabled,
        "tts_micro_batch_size": micro_batch_size,
        "calibrated_speaking_rate_wps": calibrated_rate,
        "voice_profile_key": voice_profile.get("profile_key"),
        "voice_profile_source": voice_profile.get("profile_source") or voice_profile.get("source"),
        "voice_profile_samples": voice_profile.get("sample_count_accepted") or voice_profile.get("samples"),
        "voice_profile_syllables_per_second": voice_profile.get("syllables_per_second"),
        "prediction_method": voice_profile.get("prediction_method"),
        "chunked_segment_count": chunked_segments,
        "chunk_count": total_chunks,
        "chunk_retry_count": chunk_retries,
        "tts_fidelity_checked_count": fidelity_checked,
        "tts_fidelity_good_count": fidelity_good,
        "tts_fidelity_poor_count": fidelity_poor,
        "tts_fidelity_failed_count": fidelity_failed,
        "tts_text_similarity_mean": round(sum(fidelity_scores) / len(fidelity_scores), 4) if fidelity_scores else None,
        "very_long_segment_count": very_long_count,
        **(describe_conversion(conversion_result) if conversion_result is not None else {"conversion_strategy": strategy, "conversion_input_count": 0, "conversion_wall_time_ms": 0, "conversion_process_count": 0, "conversion_fallback_reason": "no_batch_run"}),
    })

    checkpoint_data = {
        "schema_version": 1,
        "job_id": job_id,
        "step_name": "tts",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tts_qc": {
            "tts_chunked_segment_count": chunked_segments,
            "tts_total_chunk_count": total_chunks,
            "tts_chunk_retry_count": chunk_retries,
            "tts_fidelity_checked_count": fidelity_checked,
            "tts_fidelity_good_count": fidelity_good,
            "tts_fidelity_poor_count": fidelity_poor,
            "tts_fidelity_failed_count": fidelity_failed,
            "tts_text_similarity_mean": round(sum(fidelity_scores) / len(fidelity_scores), 4) if fidelity_scores else None,
            "very_long_segment_count": very_long_count,
        },
        "segments": segments
    }
    save_checkpoint(config.data_dir, job_id, "tts", checkpoint_data)
    return checkpoint_data


def _scale_attempt_budget_for_overflow(budget, overflow_ratio: float) -> None:
    """Grant extra synthesis attempts when the raw dub badly overflows the free window.

    overflow_ratio = current_speech_duration / fit_max (>1 means overflow). The worse the
    overflow, the more attempts we allow to shorten/re-synthesize into the window (2.4).
    """
    try:
        ratio = float(overflow_ratio)
    except (TypeError, ValueError):
        return
    if ratio <= 1.05:
        return
    extra = 1 if ratio < 1.35 else 2
    budget.max_total_syntheses = min(6, budget.max_total_syntheses + extra)
    budget.max_candidate_attempts = min(4, budget.max_candidate_attempts + extra)
    budget.max_rewrite_attempts = min(3, budget.max_rewrite_attempts + 1)


def _duration_trim_caps(settings: dict) -> tuple[float, int]:
    """Return (max_trim_ratio, max_trim_ms) hard caps for any speech-affecting trim (2.4)."""
    ratio = float(settings.get("duration_trim_max_ratio", 0.15) or 0.15)
    ms = int(settings.get("duration_trim_max_ms", 600) or 600)
    return max(0.0, min(0.5, ratio)), max(0, ms)


def _voiced_tts_segments(segments: list[dict]) -> list[dict]:
    return [
        s
        for s in segments
        if str(s.get("tts_spoken_text") or s.get("translation") or "").strip()
        and not bool(s.get("no_speech"))
    ]


def _ensure_speed_base_wav(segment: dict, tts_dir: Path) -> Path | None:
    """Freeze the current TTS WAV once before the one-shot uniform apply.

    Callers must only freeze *before* any soft/uniform atempo. Later apply always
    reads this frozen base so rates never compound.
    """
    idx = int(segment.get("index", 0) or 0)
    claimed = segment.get("tts_speed_base_path")
    if claimed and Path(str(claimed)).is_file():
        return Path(str(claimed))
    candidates = [
        Path(str(segment.get("tts_path") or "")),
        tts_dir / f"tts_repaired_{idx}.wav",
        Path(str(segment.get("tts_raw_path") or "")),
        tts_dir / f"tts_{idx}.wav",
    ]
    source = next((p for p in candidates if p.is_file()), None)
    if source is None:
        return None
    base = tts_dir / f"tts_speed_base_{idx}.wav"
    if source.resolve() != base.resolve():
        shutil.copyfile(source, base)
    segment["tts_speed_base_path"] = str(base)
    try:
        segment["repaired_duration"] = round(get_wav_duration(base), 2)
    except Exception:
        pass
    return base


def _collect_proposed_speed_factors(
    segments: list[dict],
    *,
    absolute_max_rate: float,
) -> float:
    """Record needed speed per segment without rewriting WAV files.

    Returns the max proposed factor (capped at absolute_max_rate).
    """
    target = 1.0
    for s in _voiced_tts_segments(segments):
        overflow = float(s.get("timing_overflow_sec") or 0.0)
        available = float(s.get("timing_available_duration") or 0.0)
        duration = float(s.get("repaired_duration") or s.get("tts_duration") or 0.0)
        proposed = 1.0
        if overflow > 0.15 and available > 0.05 and duration > 0:
            required = duration / available
            if required > 1.001:
                proposed = min(absolute_max_rate, required)
                if required > absolute_max_rate + 1e-6:
                    s["timing_needs_compact"] = True
                    s["timing_status"] = "SPEED_PARTIAL_NEEDS_COMPACT"
                elif proposed >= absolute_max_rate - 1e-6:
                    s["timing_status"] = "SPEED_PROPOSED_MAX"
                else:
                    s["timing_status"] = "SPEED_PROPOSED"
        prev = float(s.get("proposed_speed_factor") or 1.0)
        s["proposed_speed_factor"] = round(max(prev, proposed), 4)
        target = max(target, float(s["proposed_speed_factor"]))
    return min(absolute_max_rate, target)


def _apply_uniform_reading_speed(
    *,
    segments: list[dict],
    target_rate: float,
    ffmpeg_path: Path,
    tts_dir: Path,
    job_id: str,
    runner,
) -> float:
    """Apply one uniform atempo from frozen base WAVs to every voiced segment."""
    target = max(1.0, float(target_rate))
    if target <= 1.001:
        for s in _voiced_tts_segments(segments):
            s["soft_speed_factor"] = 1.0
        return 1.0

    for s in _voiced_tts_segments(segments):
        base = _ensure_speed_base_wav(s, tts_dir)
        if base is None:
            continue
        idx = int(s.get("index", 0) or 0)
        out = tts_dir / f"tts_repaired_{idx}.wav"
        sync_path = tts_dir / f"tts_speed_sync_{idx}.wav"
        sync_path.unlink(missing_ok=True)
        _run_ffmpeg_audio_filter(
            ffmpeg_path,
            base,
            sync_path,
            filter_expr=_build_atempo_chain(target),
            job_id=job_id,
            runner=runner,
        )
        if not sync_path.is_file():
            continue
        shutil.copyfile(sync_path, out)
        s["tts_path"] = str(out)
        s["repaired_duration"] = round(get_wav_duration(out), 2)
        s["soft_speed_factor"] = round(target, 4)
        method = str(s.get("repaired_method") or "none")
        tag = f"uniform_speed_{target:.3f}x"
        # Drop prior soft_speed/sync tags so method reflects the final one-shot apply.
        cleaned = method
        for junk in ("soft_speed_", "speed_sync_", "uniform_speed_"):
            while f"+{junk}" in cleaned or cleaned.startswith(junk):
                parts = cleaned.split("+")
                parts = [p for p in parts if not p.startswith(junk)]
                cleaned = "+".join(parts) if parts else "none"
        s["repaired_method"] = f"{cleaned}+{tag}" if cleaned and cleaned != "none" else tag
        s["timing_status"] = "SPEED_UNIFORM"

    compute_placement_starts(segments)
    schedule_soft_placements(segments)
    return target


def _propose_then_apply_uniform_speed(
    *,
    segments: list[dict],
    absolute_max_rate: float,
    ffmpeg_path: Path,
    tts_dir: Path,
    job_id: str,
    runner,
) -> float:
    """Measure needed rates (no WAV rewrite) → take max → apply once to all."""
    compute_placement_starts(segments)
    schedule_soft_placements(segments)
    for s in _voiced_tts_segments(segments):
        _ensure_speed_base_wav(s, tts_dir)
        # Reset so propose starts from current base duration metadata.
        s["proposed_speed_factor"] = 1.0
        s["soft_speed_factor"] = 1.0
    target = _collect_proposed_speed_factors(segments, absolute_max_rate=absolute_max_rate)
    return _apply_uniform_reading_speed(
        segments=segments,
        target_rate=target,
        ffmpeg_path=ffmpeg_path,
        tts_dir=tts_dir,
        job_id=job_id,
        runner=runner,
    )


def _apply_soft_placement_speed_and_compact(
    *,
    segments: list[dict],
    settings: dict,
    ffmpeg_path: Path,
    tts_dir: Path,
    job_id: str,
    runner,
    database: Database,
    session: "TtsSession | None",
) -> None:
    """Soft place → propose needed speeds ≤1.2 → apply one uniform max → compact."""
    absolute_max_rate = float(settings.get("edge_tts_overflow_speed_hard_max", 1.2) or 1.2)
    absolute_max_rate = max(1.0, min(1.2, absolute_max_rate))
    dub_lang_label = dub_language_label(dub_language_from_settings(settings), english=True)

    _propose_then_apply_uniform_speed(
        segments=segments,
        absolute_max_rate=absolute_max_rate,
        ffmpeg_path=ffmpeg_path,
        tts_dir=tts_dir,
        job_id=job_id,
        runner=runner,
    )

    # Targeted compact only for remaining overflow after uniform speed.
    if session is None:
        return
    for s in segments:
        overflow = float(s.get("timing_overflow_sec") or 0.0)
        available = float(s.get("timing_available_duration") or 0.0)
        if overflow <= 0.15:
            continue
        # Compact any remaining overflow after speed passes (not only pre-marked).
        text = str(s.get("tts_spoken_text") or s.get("translation") or "").strip()
        if not text or available <= 0.05:
            continue
        try:
            compact_text, _ = shorten_translation_for_timing(
                settings,
                database,
                text=text,
                budget=available,
                current_duration=float(s.get("repaired_duration") or 0.0),
                estimate_word_count=_estimate_word_count,
                language_label=dub_lang_label,
            )
        except Exception:
            logger.exception("Compact rewrite failed for segment %s", s.get("index"))
            continue
        if not compact_text or compact_text.strip() == text:
            continue
        idx = int(s["index"])
        out_path = tts_dir / f"tts_compact_{idx}.wav"
        out_path.unlink(missing_ok=True)
        try:
            session.synthesize(compact_text.strip(), out_path, segment=s)
        except Exception:
            logger.exception("Compact resynth failed for segment %s", idx)
            continue
        if not out_path.is_file():
            continue
        repaired_path = tts_dir / f"tts_repaired_{idx}.wav"
        shutil.copyfile(out_path, repaired_path)
        s["translation"] = compact_text.strip()
        s["tts_spoken_text"] = compact_text.strip()
        s["repaired_duration"] = round(get_wav_duration(repaired_path), 2)
        s["tts_path"] = str(repaired_path)
        s["tts_raw_path"] = str(out_path)
        method = str(s.get("repaired_method") or "none")
        s["repaired_method"] = f"{method}+soft_compact"
        s["timing_status"] = "COMPACTED"
        # Compact replaced speech at natural pace — clear frozen base for re-apply.
        s.pop("tts_speed_base_path", None)
        s["proposed_speed_factor"] = 1.0
        s["soft_speed_factor"] = 1.0

    compute_placement_starts(segments)
    schedule_soft_placements(segments)

    overflow_remaining = sum(1 for s in segments if float(s.get("timing_overflow_sec") or 0) > 0.15)
    overlap_remaining = len(segments_with_voiced_overlap(segments))
    if session is not None and (overflow_remaining > 0 or overlap_remaining > 0):
        video_duration = (
            max(float(s.get("end") or 0.0) for s in segments) + 2.0 if segments else None
        )
        repaired = repair_conflict_clusters(
            segments,
            settings=settings,
            ffmpeg_path=ffmpeg_path,
            tts_dir=tts_dir,
            job_id=job_id,
            runner=runner,
            session=session,
            get_wav_duration=get_wav_duration,
            build_atempo_chain=_build_atempo_chain,
            run_ffmpeg_audio_filter=_run_ffmpeg_audio_filter,
            video_duration=video_duration,
        )
        segments[:] = repaired
        for s in segments:
            s.pop("tts_speed_base_path", None)
            s["proposed_speed_factor"] = 1.0
            s["soft_speed_factor"] = 1.0

    # Always finish with one propose→max→apply so compact/cluster do not leave uneven pace.
    for s in segments:
        if str(s.get("timing_status") or "") == "COMPACTED" or s.get("cluster_source_indices"):
            s.pop("tts_speed_base_path", None)
    _propose_then_apply_uniform_speed(
        segments=segments,
        absolute_max_rate=absolute_max_rate,
        ffmpeg_path=ffmpeg_path,
        tts_dir=tts_dir,
        job_id=job_id,
        runner=runner,
    )
    enforce_zero_overlap_placements(segments)


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
    global_speed = max(1.0, min(2.5, float(settings.get("tts_global_speed", 1.0) or 1.0)))
    rewrite_prefix = _timing_rewrite_method_prefix(settings)
    dub_lang_label = dub_language_label(dub_language_from_settings(settings), english=True)
    fit_policy = policy_from_settings(settings)

    ffmpeg_path = resolve_tool_path(config, "ffmpeg")
    telemetry = TelemetrySink(config.data_dir, job_id)
    step_started = time.perf_counter()

    with TtsSession(settings, data_dir=config.data_dir, runner=runner, adapter_factory=create_tts_adapter) as session:
        for s in segments:
            segment_started = time.perf_counter()
            idx = s["index"]
            profile = timing_profile_from_segment(s)
            s.setdefault("timing_profile", profile)
            segment_budget = budget_from_settings(settings)
            prior_budget = s.get("tts_attempt_budget")
            if isinstance(prior_budget, dict):
                segment_budget.used = int(prior_budget.get("used") or 0)
                segment_budget.candidate_attempts = int(prior_budget.get("candidate_attempts") or 0)
                segment_budget.rewrite_attempts = int(prior_budget.get("rewrite_attempts") or 0)
                segment_budget.cache_hits = int(prior_budget.get("cache_hits") or 0)
            budget = float(s.get("duration_budget") or profile.get("timeline_window") or 0.0)
            tts_dur = float(s.get("tts_duration") or 0.0)
            orig_file = Path(s.get("tts_path") or s.get("tts_raw_path") or (tts_dir / f"tts_{idx}.wav"))
            speech_dur = _segment_speech_duration(s, orig_file if orig_file.is_file() else None)
            s["user_requested_speed"] = round(global_speed, 3)
            repaired_file = tts_dir / f"tts_repaired_{idx}.wav"
            repaired_file.unlink(missing_ok=True)

            repair_attempts = 0
            re_synthesis_count = 0
            llm_shorten_ms = 0
            llm_lengthen_ms = 0
            re_synthesis_ms = 0
            atempo_ms = 0
            trim_ms = 0
            fit_methods: list[str] = []
            quality_warning: str | None = None
            duration_repair_risk = "none"
            time_stretch_factor = 1.0
            tail_speech_detected = False

            speech_fit = classify_duration_fit(speech_dur, profile, policy=fit_policy)
            if not exact_enabled and acceptable_duration_fit(speech_fit):
                input_for_fit = orig_file
                if abs(global_speed - 1.0) > 0.001:
                    speed_started = time.perf_counter()
                    input_for_fit, repaired_duration_value = _apply_global_speed_if_needed(
                        ffmpeg_path=ffmpeg_path,
                        input_path=input_for_fit,
                        output_dir=tts_dir,
                        index=idx,
                        global_speed=global_speed,
                        job_id=job_id,
                        runner=runner,
                    )
                    atempo_ms += round((time.perf_counter() - speed_started) * 1000)
                    fit_methods.append(f"global_speed_{round(global_speed, 2)}x")
                else:
                    repaired_duration_value = tts_dur
                shutil.copy(input_for_fit, repaired_file)
                s["repaired_method"] = "+".join(fit_methods) if fit_methods else "none"
                s["repaired_duration"] = round(repaired_duration_value, 2)
                s["duration_repair_risk"] = duration_repair_risk
                s["repair_severity"] = "none"
                s["needs_review"] = False
                s["final_timing_error_ms"] = round((repaired_duration_value - budget) * 1000) if budget > 0 else 0
                s["repair_attempts"] = repair_attempts
                s["tail_speech_detected"] = False
                s["time_stretch_factor"] = 1.0
                s["final_timing_error_ms"] = round((speech_dur - profile.get("speech_target_duration", budget)) * 1000)
                s["accepted_without_repair"] = True
                continue

            input_for_fit = orig_file
            current_wav_duration = get_wav_duration(input_for_fit) if input_for_fit.is_file() else tts_dur
            current_speech_duration = speech_dur or current_wav_duration
            current_duration = current_wav_duration
            repair_target = float(profile.get("speech_target_duration") or _repair_target_duration(s, budget, tolerance_sec))
            hard_max = float(profile.get("hard_max_duration") or repair_target)
            # Gap-based fit (VẤN ĐỀ 1): the dub only needs to fit inside the free window up to
            # the next segment (fit_max = hard_max). Compress toward fit_max, never toward the
            # shorter original speech duration, so full translated content is preserved whenever
            # there is real free space before the next segment.
            fit_max = max(repair_target, hard_max)
            s["repair_target_duration"] = round(repair_target, 2) if repair_target > 0 else 0.0
            s["fit_max_duration"] = round(fit_max, 2) if fit_max > 0 else 0.0
            needs_review = False
            if fit_max > 0 and current_speech_duration > fit_max + tolerance_sec:
                _scale_attempt_budget_for_overflow(segment_budget, current_speech_duration / fit_max)
            if fit_max > 0 and should_shorten_for_timing(current_speech_duration, s, policy=fit_policy):
                try:
                    llm_started = time.perf_counter()
                    new_translation, target_words = shorten_translation_for_timing(
                        settings,
                        database,
                        text=str(s.get("translation") or ""),
                        budget=fit_max,
                        current_duration=current_speech_duration,
                        estimate_word_count=_estimate_word_count,
                        language_label=dub_lang_label,
                    )
                    llm_shorten_ms = round((time.perf_counter() - llm_started) * 1000)
                    if new_translation and new_translation != s.get("translation") and segment_budget.can_synthesize():
                        raw_temp = tts_dir / f"tts_temp_raw_{idx}.wav"
                        temp_wav = tts_dir / f"tts_temp_{idx}.wav"
                        raw_temp.unlink(missing_ok=True)
                        temp_wav.unlink(missing_ok=True)
                        synth_started = time.perf_counter()
                        session.synthesize(new_translation, raw_temp, segment=s)
                        segment_budget.record_repair_resynth()
                        _convert_tts_to_final_wav(ffmpeg_path, raw_temp, temp_wav, job_id, runner)
                        re_synthesis_ms = round((time.perf_counter() - synth_started) * 1000)
                        raw_temp.unlink(missing_ok=True)
                        new_dur = get_wav_duration(temp_wav)
                        new_speech = _segment_speech_duration(s, temp_wav)
                        repair_attempts += 1
                        re_synthesis_count += 1
                        if new_speech < current_speech_duration:
                            input_for_fit = temp_wav
                            current_duration = new_dur
                            current_speech_duration = new_speech
                            s["translation"] = new_translation
                            fit_methods.append(
                                f"{rewrite_prefix}_shorten_to_{target_words}_words"
                                if target_words > 0
                                else f"{rewrite_prefix}_shorten"
                            )
                        else:
                            temp_wav.unlink(missing_ok=True)
                except Exception:
                    quality_warning = f"{rewrite_prefix}_shorten_failed"

            def _maybe_apply_global_speed() -> None:
                nonlocal input_for_fit, current_duration, current_speech_duration, atempo_ms
                global_speed_method = f"global_speed_{round(global_speed, 2)}x"
                if abs(global_speed - 1.0) <= 0.001 or global_speed_method in fit_methods:
                    return
                speed_started = time.perf_counter()
                input_for_fit, current_duration = _apply_global_speed_if_needed(
                    ffmpeg_path=ffmpeg_path,
                    input_path=input_for_fit,
                    output_dir=tts_dir,
                    index=idx,
                    global_speed=global_speed,
                    job_id=job_id,
                    runner=runner,
                )
                atempo_ms += round((time.perf_counter() - speed_started) * 1000)
                fit_methods.append(global_speed_method)
                current_speech_duration = _segment_speech_duration(s, input_for_fit)

            if exact_enabled and fit_max > 0 and current_speech_duration > fit_max + tolerance_sec:
                _maybe_apply_global_speed()
                raw_factor = tempo_factor_for_duration(current_speech_duration, fit_max)
                automatic_tempo, tempo_risk = clamp_automatic_tempo(
                    raw_factor,
                    policy=fit_policy,
                    user_global_speed=global_speed,
                )
                speed_factor = min(max_stretch, max(1.0, automatic_tempo))
                stretch_decision = classify_stretch_with_policy(
                    speed_factor,
                    settings=settings,
                    explicit_allow_danger=True,
                )
                duration_repair_risk = stretch_decision.risk if stretch_decision.risk != "normal" else tempo_risk
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
                current_speech_duration = _segment_speech_duration(s, stretched_file)
                time_stretch_factor = round(speed_factor, 3)
                s["automatic_tempo_factor"] = time_stretch_factor
                s["effective_speed"] = round(global_speed * time_stretch_factor, 3)
                fit_methods.append(f"time_stretch_{round(speed_factor, 2)}x")
                repair_attempts += 1

            if exact_enabled and fit_max > 0:
                target_dur = max(0.05, float(fit_max))
                lengthen_target = max(0.05, float(repair_target))
                if current_speech_duration > fit_max + tolerance_sec:
                    # Speech itself still overflows the free window after shorten + stretch.
                    # Never trim into speech (2.4 hard cap) — flag the segment for human review
                    # instead of silently cutting content.
                    _maybe_apply_global_speed()
                    tail_speech_detected = _wav_tail_has_speech(input_for_fit)
                    quality_warning = quality_warning or "speech_over_window_needs_review"
                    duration_repair_risk = "danger"
                    needs_review = True
                    s["speech_trimmed"] = False
                elif current_duration > fit_max + tolerance_sec and (
                    current_duration - current_speech_duration
                ) > tolerance_sec:
                    # Only trailing silence overshoots the window: shrink the silence down to the
                    # window boundary (fit_max), keeping the full speech tail intact.
                    _maybe_apply_global_speed()
                    trailing = float(
                        s.get("tts_trailing_silence") or max(0.0, current_duration - current_speech_duration)
                    )
                    trim_ratio, trim_cap_ms = _duration_trim_caps(settings)
                    trim_amount_ms = round(max(0.0, current_duration - target_dur) * 1000)
                    speech_headroom_ms = round(max(0.0, target_dur - current_speech_duration) * 1000)
                    # Guard: only trim silence that sits beyond the speech end. If honoring the
                    # window would require cutting into speech, refuse and flag for review.
                    if speech_headroom_ms < 0 or trim_amount_ms > trim_cap_ms + round(trailing * 1000):
                        quality_warning = quality_warning or "silence_trim_exceeds_cap_needs_review"
                        needs_review = True
                        s["speech_trimmed"] = False
                    else:
                        exact_file = tts_dir / f"tts_exact_{idx}.wav"
                        exact_file.unlink(missing_ok=True)
                        exact_filter = (
                            f"atrim=0:{current_speech_duration + 0.05:.3f},"
                            f"apad=pad_dur={target_dur + 0.2:.3f},atrim=0:{target_dur:.3f}"
                        )
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
                        current_speech_duration = _segment_speech_duration(s, exact_file)
                        fit_methods.append("outer_silence_trim")
                        s["outer_silence_trimmed_ms"] = round(trailing * 1000)
                        repair_attempts += 1
                elif should_lengthen_for_timing(current_speech_duration, s, policy=fit_policy):
                    lengthen_gap_threshold = _lengthen_min_gap_sec(settings)
                    gap = lengthen_target - current_speech_duration
                    if gap > lengthen_gap_threshold:
                        try:
                            llm_started = time.perf_counter()
                            new_translation, target_words = lengthen_translation_for_timing(
                                settings,
                                database,
                                text=str(s.get("translation") or ""),
                                budget=lengthen_target,
                                current_duration=current_speech_duration,
                                min_gap_sec=lengthen_gap_threshold,
                                max_ratio=_lengthen_max_ratio(settings),
                                estimate_word_count=_estimate_word_count,
                                language_label=dub_lang_label,
                            )
                            llm_lengthen_ms = round((time.perf_counter() - llm_started) * 1000)
                            if new_translation and new_translation != s.get("translation") and segment_budget.can_synthesize():
                                raw_temp = tts_dir / f"tts_temp_raw_{idx}.wav"
                                temp_wav = tts_dir / f"tts_temp_{idx}.wav"
                                raw_temp.unlink(missing_ok=True)
                                temp_wav.unlink(missing_ok=True)
                                synth_started = time.perf_counter()
                                session.synthesize(new_translation, raw_temp, segment=s)
                                segment_budget.record_repair_resynth()
                                _convert_tts_to_final_wav(ffmpeg_path, raw_temp, temp_wav, job_id, runner)
                                re_synthesis_ms += round((time.perf_counter() - synth_started) * 1000)
                                raw_temp.unlink(missing_ok=True)
                                new_dur = get_wav_duration(temp_wav)
                                new_speech = _segment_speech_duration(s, temp_wav)
                                repair_attempts += 1
                                re_synthesis_count += 1
                                if new_speech > current_speech_duration:
                                    input_for_fit = temp_wav
                                    current_duration = new_dur
                                    current_speech_duration = new_speech
                                    s["translation"] = new_translation
                                    fit_methods.append(
                                        f"{rewrite_prefix}_lengthen_to_{target_words}_words"
                                        if target_words > 0
                                        else f"{rewrite_prefix}_lengthen"
                                    )
                                else:
                                    temp_wav.unlink(missing_ok=True)
                        except AppError:
                            raise
                        except Exception:
                            quality_warning = f"{rewrite_prefix}_lengthen_failed"

                    _maybe_apply_global_speed()
                    if (
                        should_lengthen_for_timing(current_speech_duration, s, policy=fit_policy)
                        and current_speech_duration < lengthen_target - tolerance_sec
                    ):
                        remaining_gap = lengthen_target - current_speech_duration
                        if remaining_gap > lengthen_gap_threshold and not any(
                            method.endswith("_lengthen") or "_lengthen_to_" in method for method in fit_methods
                        ):
                            quality_warning = quality_warning or "short_tts_gap_exceeds_tail_pad_without_lengthen"
                        exact_file = tts_dir / f"tts_exact_{idx}.wav"
                        exact_file.unlink(missing_ok=True)
                        trim_started = time.perf_counter()
                        _run_ffmpeg_audio_filter(
                            ffmpeg_path,
                            input_for_fit,
                            exact_file,
                            filter_expr=_tail_silence_pad_filter(current_duration, lengthen_target),
                            job_id=job_id,
                            runner=runner,
                        )
                        trim_ms = round((time.perf_counter() - trim_started) * 1000)
                        input_for_fit = exact_file
                        current_duration = get_wav_duration(exact_file)
                        fit_methods.append("tail_silence_pad")
                        repair_attempts += 1
                elif acceptable_duration_fit(speech_fit):
                    _maybe_apply_global_speed()
                    s["accepted_without_repair"] = True
                else:
                    _maybe_apply_global_speed()
            else:
                _maybe_apply_global_speed()

            repaired_file.unlink(missing_ok=True)
            shutil.copy(input_for_fit, repaired_file)
            repaired_duration = round(get_wav_duration(repaired_file), 2)
            # Final residual-overflow guard: if the dub speech still overflows the free window
            # after every safe repair, mark it danger + needs_review instead of hiding the issue.
            final_speech = _segment_speech_duration(s, repaired_file) or current_speech_duration
            if fit_max > 0 and final_speech > fit_max + tolerance_sec:
                duration_repair_risk = "danger"
                needs_review = True
                quality_warning = quality_warning or "residual_speech_over_window"
            if duration_repair_risk == "danger":
                repair_severity = "danger"
            elif duration_repair_risk == "warning" or needs_review:
                repair_severity = "warning"
            else:
                repair_severity = "none"
            s["repaired_duration"] = repaired_duration
            s["repaired_method"] = "+".join(fit_methods) if fit_methods else "none"
            s["duration_repair_risk"] = duration_repair_risk
            s["repair_severity"] = repair_severity
            s["needs_review"] = bool(needs_review)
            s["final_timing_error_ms"] = round((repaired_duration - repair_target) * 1000) if repair_target > 0 else 0
            s["tail_speech_detected"] = tail_speech_detected
            s["time_stretch_factor"] = time_stretch_factor
            s["repair_attempts"] = repair_attempts
            s["quality_warning"] = quality_warning
            s["re_synthesis_count"] = re_synthesis_count
            s["tts_attempt_budget"] = segment_budget.to_dict()
            telemetry.record("duration_repair_segment", {
                "wall_time_ms": round((time.perf_counter() - segment_started) * 1000),
                "audio_duration_sec": repaired_duration,
                "original_duration": tts_dur,
                "budget": budget,
                "repair_target": repair_target,
                "method": s["repaired_method"],
                "llm_shorten_ms": llm_shorten_ms,
                "llm_lengthen_ms": llm_lengthen_ms,
                "re_synthesis_ms": re_synthesis_ms,
                "atempo_ms": atempo_ms,
                "trim_ms": trim_ms,
                "re_synthesis_count": re_synthesis_count,
                "segment_index": idx,
            })

        _apply_soft_placement_speed_and_compact(
            segments=segments,
            settings=settings,
            ffmpeg_path=ffmpeg_path,
            tts_dir=tts_dir,
            job_id=job_id,
            runner=runner,
            database=database,
            session=session,
        )

    telemetry.record("duration_repair", {
        "wall_time_ms": round((time.perf_counter() - step_started) * 1000),
        "segment_count": len(segments),
        "retry_count": 0,
        "overflow_remaining": sum(1 for s in segments if float(s.get("timing_overflow_sec") or 0) > 0.15),
        "voiced_overlaps": len(segments_with_voiced_overlap(segments)),
    })
    checkpoint_data = {
        "schema_version": 2,
        "job_id": job_id,
        "step_name": "duration_repair",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "segments": segments,
        "timing_overflow_count": sum(1 for s in segments if float(s.get("timing_overflow_sec") or 0) > 0.15),
        "voiced_overlap_count": len(segments_with_voiced_overlap(segments)),
    }
    save_checkpoint(config.data_dir, job_id, "duration_repair", checkpoint_data)
    _release_tts_gpu_resources(settings)
    return checkpoint_data


def _load_repaired_segments(config: AppConfig, job_id: str) -> list[dict]:
    align_cp = load_checkpoint(config.data_dir, job_id, "align_final_dub")
    if align_cp and align_cp.get("segments"):
        return list(align_cp["segments"])
    repair_cp = load_checkpoint(config.data_dir, job_id, "duration_repair")
    if repair_cp and repair_cp.get("segments"):
        return list(repair_cp["segments"])
    return []


def align_final_dub_step(job_id: str, config: AppConfig, database: Database, runner) -> dict:
    repair_cp = load_checkpoint(config.data_dir, job_id, "duration_repair")
    if not repair_cp:
        raise AppError(
            400,
            ErrorInfo(
                code="MISSING_DURATION_REPAIR",
                message="Duration repair checkpoint is missing.",
                action="Resume duration_repair step.",
            ),
        )

    segments = [dict(segment) for segment in repair_cp.get("segments", [])]
    settings = _load_settings(database)
    job_dir = config.data_dir / "jobs" / job_id
    cache_dir = job_dir / "artifacts" / "subtitle_asr"
    ffmpeg_path = resolve_tool_path(config, "ffmpeg")
    project_root = Path(__file__).resolve().parents[2]
    vendor_dir = Path(os.environ.get("DV_VENDOR_DIR", project_root / "vendor"))
    configure_gpu_manager(settings)
    _release_tts_gpu_resources(settings)

    language = dub_language_label(dub_language_from_settings(settings), english=True)
    telemetry = TelemetrySink(config.data_dir, job_id)
    step_started = time.perf_counter()
    model_load_started = time.perf_counter()

    aligned_count = 0
    fallback_count = 0
    failed_count = 0
    batch_result = align_job_segments_final_dub(
        segments,
        job_dir=job_dir,
        cache_dir=cache_dir,
        transcribe_fn=transcribe_audio,
        vendor_dir=vendor_dir,
        settings=settings,
        ffmpeg_path=ffmpeg_path,
        language=language,
    )
    cache_hits = int(batch_result.get("cache_hits") or 0)
    cache_misses = int(batch_result.get("cache_misses") or 0)
    model_calls = int(batch_result.get("model_calls") or 0)

    for result in batch_result.get("results") or []:
        status = str(result.get("status") or "skipped")
        if status == "aligned":
            aligned_count += 1
        elif status in {"fallback_interpolated", "no_speech"}:
            fallback_count += 1
        elif status == "failed":
            failed_count += 1

    similarities = [
        float(result["text_similarity"])
        for result in batch_result.get("results") or []
        if result.get("text_similarity") is not None
    ]

    model_load_time = round((time.perf_counter() - model_load_started) * 1000)
    alignment_wall_time = round((time.perf_counter() - step_started) * 1000)
    summary = summarize_alignment_results(segments)

    telemetry.record(
        "align_final_dub",
        {
            "segment_count": len(segments),
            "aligned_count": aligned_count,
            "fallback_count": fallback_count,
            "failed_count": failed_count,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "model_calls": model_calls,
            "alignment_wall_time": alignment_wall_time,
            "model_load_time": model_load_time,
            "average_text_similarity": summary.get("average_text_similarity"),
        },
    )

    checkpoint_data = {
        "schema_version": 1,
        "job_id": job_id,
        "step_name": "align_final_dub",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "segments": segments,
        **summary,
    }
    save_checkpoint(config.data_dir, job_id, "align_final_dub", checkpoint_data)
    if model_calls > 0:
        _release_asr_gpu_models(settings)
    return checkpoint_data


def _evaluate_release_gate_before_render(
    job_id: str,
    config: AppConfig,
    database: Database,
    segments: list[dict],
    settings: dict,
) -> dict:
    """Run the release quality gate after align_final_dub and persist a report.

    This is the first time the gate is wired into the real job flow: previously it only ran
    from CLI scripts after render. If any blocking condition is present (speech trim, semantic
    critical, subtitle overlap/out-of-bounds, danger stretch, or a segment flagged
    needs_review during duration repair), the job is halted before mix/render so the operator
    can review it instead of shipping a broken dubbed.mp4 (2.5).
    """
    metrics = compute_timing_qc_metrics(segments, settings=settings)
    gate = evaluate_release_gate(segments, metrics=metrics, settings=settings)
    # Fold in the per-segment needs_review flag written by duration_repair (2.4).
    review_segments = [
        int(s.get("index")) for s in segments if s.get("needs_review")
    ]
    gate["needs_review_segments"] = review_segments
    if review_segments and "needs_review_segments" not in gate["blocking"]:
        gate = {**gate, "passed": False, "blocking": [*gate["blocking"], "needs_review_segments"]}
    report = {
        "schema_version": 1,
        "job_id": job_id,
        "step_name": "release_gate",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **gate,
    }
    save_checkpoint(config.data_dir, job_id, "release_gate", report)
    return gate


def mix_step(job_id: str, config: AppConfig, database: Database, runner) -> dict:
    segments = _load_repaired_segments(config, job_id)
    audio_cp = load_checkpoint(config.data_dir, job_id, "extract_audio")

    gate_settings = _load_settings(database)
    if bool(gate_settings.get("release_gate_blocking_enabled", True)):
        gate = _evaluate_release_gate_before_render(job_id, config, database, segments, gate_settings)
        if not gate.get("passed", True):
            raise AppError(
                409,
                ErrorInfo(
                    code="RELEASE_GATE_BLOCKED",
                    message="Release quality gate blocked before render: "
                    + ", ".join(gate.get("blocking") or []),
                    action="Review flagged segments (duration overflow / stretch / subtitle), then resume the job.",
                ),
            )

    # ChatGPT TL hard gate: overflow=0, voiced_overlap=0 before mix.
    overflow_count = sum(1 for s in segments if float(s.get("timing_overflow_sec") or 0) > 0.15)
    voiced_overlaps = segments_with_voiced_overlap(segments)
    last_audible_end = 0.0
    for s in segments:
        text = str(s.get("tts_spoken_text") or s.get("translation") or "").strip()
        if not text or bool(s.get("no_speech")):
            continue
        start = float(s.get("placement_start") or s.get("start") or 0.0)
        rep = s.get("repaired_duration")
        if rep is None or rep == "":
            dur = float(s.get("tts_duration") or 0.0)
        else:
            dur = float(rep)
        if dur <= 0:
            continue
        last_audible_end = max(last_audible_end, start + dur)
    timing_blocking: list[str] = []
    if overflow_count:
        timing_blocking.append(f"timing_overflow_count={overflow_count}")
    if voiced_overlaps:
        timing_blocking.append(f"voiced_overlap_count={len(voiced_overlaps)}")
    if timing_blocking and bool(gate_settings.get("timing_placement_gate_enabled", True)):
        raise AppError(
            409,
            ErrorInfo(
                code="TIMING_PLACEMENT_GATE_BLOCKED",
                message="Timing placement gate blocked before mix: " + ", ".join(timing_blocking),
                action="Re-run duration_repair conflict-cluster repair, then resume.",
            ),
        )

    if not audio_cp:
        raise AppError(
            400,
            ErrorInfo(
                code="MISSING_REPAIR_OR_AUDIO",
                message="Original audio checkpoint is missing.",
                action="Verify extract_audio step.",
            ),
        )

    original_48k = Path(audio_cp["original_48k_path"])
    try:
        media_duration = get_wav_duration(original_48k)
    except Exception:
        media_duration = 0.0
    last_cn_end = max((float(s.get("end") or 0.0) for s in segments), default=0.0)
    # ASR windows can slightly overrun extracted WAV length; hold narration to the
    # effective source timeline (wav or last Chinese end), not an unreachable tighter bound.
    source_deadline = max(media_duration, last_cn_end)
    if (
        source_deadline > 0
        and last_audible_end > source_deadline + 0.05
        and bool(gate_settings.get("timing_placement_gate_enabled", True))
    ):
        raise AppError(
            409,
            ErrorInfo(
                code="TIMING_PLACEMENT_GATE_BLOCKED",
                message=(
                    f"Timing placement gate blocked: last_audible_end={last_audible_end:.2f}s "
                    f"> source_deadline={source_deadline:.2f}s "
                    f"(media={media_duration:.2f}s, cn_end={last_cn_end:.2f}s)"
                ),
                action="Re-run duration_repair conflict-cluster repair, then resume.",
            ),
        )

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

    segment_entries: list[dict] = []
    for seg in segments:
        idx = seg["index"]
        seg_path = resolve_voiced_tts_path(seg)
        if seg_path is None and spoken_text(seg):
            raise AppError(
                409,
                ErrorInfo(
                    code="TTS_PROVENANCE_MISSING",
                    message=f"Voiced segment {idx} is missing canonical tts_path.",
                    action="Re-run duration_repair provenance repair, then resume mix.",
                ),
            )
        if seg_path is None:
            continue
        if seg_path.is_file():
            placement_start = float(seg.get("placement_start") or seg.get("start") or 0.0)
            clip_duration = float(seg.get("repaired_duration") or seg.get("tts_duration") or 0.0)
            if clip_duration <= 0:
                clip_duration = get_wav_duration(seg_path)
            segment_entries.append(
                {
                    "path": seg_path,
                    "placement_start": placement_start,
                    "clip_duration": clip_duration,
                }
            )

    annotate_segment_mix_caps(segment_entries)
    allow_hard_clip = bool(settings.get("mix_hard_clip_enabled", False))
    would_clip = [
        entry for entry in segment_entries
        if float(entry.get("mix_would_clip_sec") or 0.0) > 0.15
    ]
    if would_clip and not allow_hard_clip:
        logger.warning(
            "Mix soft-placement residual overflow for job %s: %s segments would have been hard-clipped under legacy policy",
            job_id,
            len(would_clip),
        )

    if segment_entries:
        cmd_narration = [str(ffmpeg_path), "-y"]
        for entry in segment_entries:
            cmd_narration.extend(["-i", str(entry["path"])])
        filters = [
            build_narration_segment_filter(
                input_index,
                placement_start=entry["placement_start"],
                clip_duration=entry["clip_duration"],
                max_duration=entry.get("max_duration"),
                allow_hard_clip=allow_hard_clip,
            )
            for input_index, entry in enumerate(segment_entries)
        ]
        filters.append(build_narration_amix_filter(len(segment_entries)))
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
        "narration_segment_input_count": len(segment_entries),
        "narration_wav_path": str(narration_wav),
        "mixed_wav_path": str(mixed_wav),
        "vietnamese_narration_path": str(vietnamese_narration)
    }
    save_checkpoint(config.data_dir, job_id, "mix", checkpoint_data)
    return checkpoint_data


def render_step(job_id: str, config: AppConfig, database: Database, runner) -> dict:
    mix_cp = load_checkpoint(config.data_dir, job_id, "mix")
    segments = _load_repaired_segments(config, job_id)
    
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
    subtitle_burn_in = False
    if settings.get("subtitles_enabled", True) and segments:
        refresh_all_segment_dub_word_timestamps(segments)
        subtitle_segments = [
            segment
            for segment in segments
            if str(segment.get("translation") or segment.get("target_text") or "").strip()
        ]
        if subtitle_segments:
            width, height = probe_video_dimensions(ffmpeg_path, original_mp4)
            project_root = Path(__file__).resolve().parents[2]
            vendor_dir = Path(os.environ.get("DV_VENDOR_DIR", project_root / "vendor"))
            has_dub_words = any(segment_has_usable_dub_words(segment) for segment in subtitle_segments)
            write_ass_file(
                ass_path,
                subtitle_segments,
                settings,
                play_res_x=width,
                play_res_y=height,
                job_dir=job_dir,
                vendor_dir=vendor_dir,
                ffmpeg_path=ffmpeg_path,
                transcribe_fn=transcribe_audio,
                tts_asr_align=not has_dub_words,
            )
            if not has_dub_words:
                _release_tts_gpu_resources(settings)
                configure_gpu_manager(settings)
                _release_asr_gpu_models(settings)
            if subtitles_filter_available(ffmpeg_path):
                video_filters.append(ffmpeg_subtitles_filter(ass_path))
                subtitle_burn_in = True
            else:
                logger.warning(
                    "ffmpeg at %s lacks libass subtitles filter; rendering without burned-in subtitles. "
                    "Install ffmpeg with libass (e.g. brew install ffmpeg-full on macOS). ASS saved to %s",
                    ffmpeg_path,
                    ass_path,
                )
    
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
                action=(
                    "Ensure original video format is compatible. On macOS, Homebrew ffmpeg may "
                    "lack libass; install ffmpeg-full or disable burned-in subtitles."
                ),
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
        "subtitle_burn_in": subtitle_burn_in,
        "subtitles_path": str(ass_path) if ass_path.is_file() else None,
    }
    save_checkpoint(config.data_dir, job_id, "render", checkpoint_data)
    return checkpoint_data


def qc_step(job_id: str, config: AppConfig, database: Database, runner) -> dict:
    segments = _load_repaired_segments(config, job_id)
    norm_cp = load_checkpoint(config.data_dir, job_id, "normalize_segments")
    render_cp = load_checkpoint(config.data_dir, job_id, "render")
    align_cp = load_checkpoint(config.data_dir, job_id, "align_final_dub")
    
    if not norm_cp or not render_cp:
        raise AppError(
            400,
            ErrorInfo(
                code="MISSING_UPSTREAM_CHECKPOINTS",
                message="Normalized segments or Render checkpoints are missing.",
                action="Verify upstream steps."
            )
        )
        
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
    timing_candidate_count = 0
    candidate_retry_count = 0
    prediction_errors: list[float] = []
    segments_accepted_first_try = 0
    segments_using_extreme_stretch = 0
    segments_using_speech_trim = 0
    segments_rewritten = 0
    segments_accepted_without_repair = 0
    automatic_tempo_factors: list[float] = []
    automatic_tempo_distribution: dict[str, int] = {}

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
        timing_candidate_count += len(s.get("translation_candidates") or [])
        candidate_retry_count += max(0, int(s.get("tts_attempt_count") or 1) - 1)
        predicted = s.get("predicted_duration")
        actual = s.get("tts_speech_duration") or s.get("tts_duration")
        if predicted is not None and actual is not None:
            prediction_errors.append(abs(float(actual) - float(predicted)) * 1000.0)
        if int(s.get("tts_attempt_count") or 1) <= 1 and s.get("accepted_without_repair"):
            segments_accepted_first_try += 1
        tempo = float(s.get("automatic_tempo_factor") or s.get("time_stretch_factor") or 1.0)
        if tempo and (tempo > 1.12 or tempo < 0.9):
            segments_using_extreme_stretch += 1
        if s.get("speech_trimmed"):
            segments_using_speech_trim += 1
        if any(item.get("source") == "rewrite" for item in s.get("tts_attempts") or []):
            segments_rewritten += 1
        if s.get("accepted_without_repair"):
            segments_accepted_without_repair += 1
        if tempo:
            automatic_tempo_factors.append(tempo)
            tempo_key = str(round(tempo, 2))
            automatic_tempo_distribution[tempo_key] = automatic_tempo_distribution.get(tempo_key, 0) + 1
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

    alignment_summary = summarize_alignment_results(segments) if align_cp else {}
    subtitle_segments = [
        segment
        for segment in segments
        if str(segment.get("translation") or segment.get("target_text") or "").strip()
    ]
    subtitle_cues = build_subtitle_cues(subtitle_segments) if subtitle_segments else []
    subtitle_metrics = compute_subtitle_qc_metrics(segments, subtitle_cues) if subtitle_cues else {}

    rows = database.connection.execute("SELECT key, value FROM settings").fetchall()
    qc_settings = {r["key"]: json.loads(r["value"]) for r in rows}
    timing_metrics = compute_timing_qc_metrics(segments, settings=qc_settings)

    checkpoint_data = {
        "schema_version": 4,
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
        "translation_candidate_count": timing_candidate_count,
        "candidate_retry_count": candidate_retry_count,
        "predicted_vs_actual_duration_error_ms": round(sum(prediction_errors) / len(prediction_errors), 1) if prediction_errors else None,
        "speech_duration_error_ms": round(sum(prediction_errors) / len(prediction_errors), 1) if prediction_errors else None,
        "segments_accepted_first_try": segments_accepted_first_try,
        "segments_using_extreme_stretch": segments_using_extreme_stretch,
        "segments_using_speech_trim": segments_using_speech_trim,
        "segments_rewritten": segments_rewritten,
        "segments_accepted_without_repair": segments_accepted_without_repair,
        "automatic_tempo_factor_distribution": automatic_tempo_distribution,
        **timing_metrics,
        **alignment_summary,
        **subtitle_metrics,
        "dub_alignment_segments": alignment_summary.get("per_segment", []),
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
