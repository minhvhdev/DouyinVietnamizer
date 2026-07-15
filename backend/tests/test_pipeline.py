import json
from pathlib import Path
import time
from unittest.mock import patch, MagicMock
import wave

import pytest

from dv_backend.config import AppConfig
from dv_backend.database import Database
from dv_backend.jobs import JobService
from dv_backend.runner import JobRunner
from dv_backend.checkpoints import load_checkpoint, save_checkpoint
from dv_backend.errors import AppError
import dv_backend.pipeline as pipeline


@pytest.fixture
def test_env(tmp_path: Path):
    database = Database(tmp_path / "app.db")
    database.migrate()
    config = AppConfig(tmp_path)
    config.ensure_directories()
    
    # Pre-populate settings
    with database.connection:
        database.connection.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("openai_api_key", json.dumps("fake-key"), "now")
        )
        database.connection.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("tts_api_key", json.dumps("fake-key"), "now")
        )
        database.connection.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("qwen3_asr_model", json.dumps("Qwen/Qwen3-ASR-1.7B"), "now")
        )
        database.connection.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("qwen3_aligner_model", json.dumps("Qwen/Qwen3-ForcedAligner-0.6B"), "now")
        )
        database.connection.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("qwen3_device", json.dumps("cuda:0"), "now")
        )
        database.connection.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("vad_engine", json.dumps("silencedetect"), "now")
        )
        
    job_service = JobService(database, tmp_path)
    runner = JobRunner(config, database)
    return config, database, job_service, runner


def create_local_job(job_service: JobService, data_dir: Path, filename: str = "sample.mp4"):
    video = data_dir / filename
    if not video.is_file():
        video.write_bytes(b"fake-video")
    return job_service.create_imported(video, original_filename=filename)


def test_resolve_tool_path_supports_runtime_tools_subdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_dir = tmp_path / "portable-runtime"
    tools_dir = runtime_dir / "tools" / "ffmpeg"
    tools_dir.mkdir(parents=True)
    tool_path = tools_dir / "ffmpeg.exe"
    tool_path.write_text("binary", encoding="utf-8")
    (runtime_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tools": [
                    {
                        "id": "ffmpeg",
                        "display_name": "FFmpeg",
                        "executable": "ffmpeg/ffmpeg.exe",
                        "dev_command": "ffmpeg",
                        "version_args": ["-version"],
                        "version_contains": "ffmpeg",
                        "required": True,
                        "capability": "media",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DV_VENDOR_DIR", str(runtime_dir))
    monkeypatch.setenv("DV_VENDOR_MANIFEST", str(runtime_dir / "manifest.json"))
    monkeypatch.setenv("DV_ALLOW_PATH_TOOLS", "0")

    resolved = pipeline.resolve_tool_path(AppConfig(tmp_path), "ffmpeg")
    assert resolved == tool_path


def test_resolve_tool_path_uses_vendor_root_when_tool_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vendor_dir = tmp_path / "vendor"
    tool_dir = vendor_dir / "ffmpeg"
    tool_dir.mkdir(parents=True)
    tool_path = tool_dir / "ffmpeg.exe"
    tool_path.write_text("binary", encoding="utf-8")
    (vendor_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tools": [
                    {
                        "id": "ffmpeg",
                        "display_name": "FFmpeg",
                        "executable": "ffmpeg/ffmpeg.exe",
                        "dev_command": "ffmpeg",
                        "version_args": ["-version"],
                        "version_contains": "ffmpeg",
                        "required": True,
                        "capability": "media",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DV_VENDOR_DIR", str(vendor_dir))
    monkeypatch.setenv("DV_VENDOR_MANIFEST", str(vendor_dir / "manifest.json"))
    monkeypatch.setenv("DV_ALLOW_PATH_TOOLS", "0")

    resolved = pipeline.resolve_tool_path(AppConfig(tmp_path), "ffmpeg")
    assert resolved == tool_path


def test_resolve_tool_path_accepts_legacy_tools_prefix(tmp_path: Path) -> None:
    tools_dir = tmp_path / "tools" / "ffmpeg"
    tools_dir.mkdir(parents=True)
    tool_path = tools_dir / "ffmpeg.exe"
    tool_path.write_text("binary", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tools": [
                    {
                        "id": "ffmpeg",
                        "display_name": "FFmpeg",
                        "executable": "tools/ffmpeg/ffmpeg.exe",
                        "dev_command": "ffmpeg",
                        "version_args": ["-version"],
                        "version_contains": "ffmpeg",
                        "required": True,
                        "capability": "media",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    from dv_backend.vendor import VendorManifest, VendorResolver

    manifest = VendorManifest.load(manifest_path)
    resolved = VendorResolver(tmp_path / "tools", allow_path_tools=False).resolve(manifest.tools[0])
    assert resolved.path == tool_path


@patch("dv_backend.pipeline.transcribe_audio")
def test_asr_uses_qwen3_gpu_transcription(mock_transcribe, test_env):
    config, database, job_service, runner = test_env
    job = create_local_job(job_service, config.data_dir)
    job_dir = config.data_dir / "jobs" / job.id
    artifacts_dir = job_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    audio_path = artifacts_dir / "audio_16k.wav"
    audio_path.write_bytes(b"wav")

    mock_transcribe.return_value = [{"start": 0.25, "end": 1.48, "text": "你好"}]

    result = pipeline.asr_step(job.id, config, database, runner)

    assert result["segments"] == [{"start": 0.25, "end": 1.48, "text": "你好"}]
    mock_transcribe.assert_called_once()
    kwargs = mock_transcribe.call_args.kwargs
    assert kwargs["device"] == "cuda:0"
    assert kwargs["language"] == "Chinese"
    assert kwargs["include_alignment"] is True
    assert result["alignment_mode"] == "accurate"
    assert result["dense_or_sparse_mode"] == "dense"


@patch("dv_backend.pipeline.transcribe_audio")
def test_asr_fast_mode_skips_word_alignment(mock_transcribe, test_env):
    config, database, job_service, runner = test_env
    job = create_local_job(job_service, config.data_dir)
    artifacts_dir = config.data_dir / "jobs" / job.id / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "audio_16k.wav").write_bytes(b"wav")
    with database.connection:
        database.connection.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("asr_alignment_mode", json.dumps("fast"), "now"),
        )
    mock_transcribe.return_value = [{"start": 0.0, "end": 1.0, "text": "你好"}]

    result = pipeline.asr_step(job.id, config, database, runner)

    assert mock_transcribe.call_args.kwargs["include_alignment"] is False
    assert result["aligned_units"] == []
    assert result["alignment_mode"] == "fast"
    assert result["alignment_status"] == "skipped"


@patch("dv_backend.pipeline.run_subprocess_with_cancel")
@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.transcribe_audio")
def test_asr_sparse_mode_transcribes_stitched_wav_once(mock_transcribe, mock_resolve, mock_run, test_env):
    config, database, job_service, runner = test_env
    job = create_local_job(job_service, config.data_dir)
    artifacts_dir = config.data_dir / "jobs" / job.id / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "audio_16k.wav").write_bytes(b"wav")
    mock_resolve.return_value = Path("ffmpeg")
    mock_run.return_value = None
    save_checkpoint(config.data_dir, job.id, "vad", {
        "total_duration": 30.0,
        "speech_regions": [{"start": 1.0, "end": 2.0}, {"start": 10.0, "end": 11.0}],
    })
    with database.connection:
        database.connection.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("sparse_asr_enabled", json.dumps(True), "now"),
        )
    mock_transcribe.return_value = {
        "segments": [
            {"start": 0.0, "end": 0.6, "text": "你好"},
            {"start": 0.6, "end": 2.0, "text": "世界"},
        ],
        "aligned_units": [],
    }

    result = pipeline.asr_step(job.id, config, database, runner)

    mock_transcribe.assert_called_once()
    call_path = Path(mock_transcribe.call_args.args[0])
    assert call_path.name == "stitched.wav"
    assert mock_run.call_count == 1
    assert result["dense_or_sparse_mode"] == "sparse"
    assert result["sparse_chunk_count"] == 2
    assert result["stitched_duration_sec"] == 2.8
    assert result["segments"][0]["start"] == 0.8
    assert result["segments"][-1]["end"] == 10.4
    assert result["sparse_asr_fallback_reason"] is None


@patch("dv_backend.pipeline.transcribe_audio")
def test_asr_balanced_mode_aligns_long_vad_segments(mock_transcribe, test_env):
    config, database, job_service, runner = test_env
    job = create_local_job(job_service, config.data_dir)
    artifacts_dir = config.data_dir / "jobs" / job.id / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "audio_16k.wav").write_bytes(b"wav")
    save_checkpoint(config.data_dir, job.id, "vad", {
        "total_duration": 30.0,
        "speech_regions": [{"start": 0.0, "end": 25.0}],
    })
    with database.connection:
        database.connection.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("asr_alignment_mode", json.dumps("balanced"), "now"),
        )
    mock_transcribe.return_value = {"segments": [{"start": 0.0, "end": 1.0, "text": "你好"}], "aligned_units": []}

    result = pipeline.asr_step(job.id, config, database, runner)

    assert mock_transcribe.call_args.kwargs["include_alignment"] is True
    assert result["alignment_mode"] == "balanced"
    assert result["alignment_requested_reason"] == "balanced_long_vad_region"


@patch("dv_backend.pipeline.transcribe_audio")
def test_asr_balanced_mode_skips_alignment_for_short_vad(mock_transcribe, test_env):
    config, database, job_service, runner = test_env
    job = create_local_job(job_service, config.data_dir)
    artifacts_dir = config.data_dir / "jobs" / job.id / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "audio_16k.wav").write_bytes(b"wav")
    save_checkpoint(config.data_dir, job.id, "vad", {
        "total_duration": 8.0,
        "speech_regions": [{"start": 0.0, "end": 2.0}, {"start": 3.0, "end": 5.0}],
    })
    with database.connection:
        database.connection.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("asr_alignment_mode", json.dumps("balanced"), "now"),
        )
    mock_transcribe.return_value = {"segments": [{"start": 0.0, "end": 1.0, "text": "短"}], "aligned_units": []}

    result = pipeline.asr_step(job.id, config, database, runner)

    assert mock_transcribe.call_args.kwargs["include_alignment"] is False
    assert result["alignment_requested_reason"] == "balanced_skip_alignment"


@patch("dv_backend.pipeline.reset_model_cache")
@patch("dv_backend.pipeline.transcribe_audio")
def test_asr_step_releases_gpu_model_after_completion(mock_transcribe, mock_reset, test_env):
    config, database, job_service, runner = test_env
    job = create_local_job(job_service, config.data_dir)
    artifacts_dir = config.data_dir / "jobs" / job.id / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "audio_16k.wav").write_bytes(b"wav")
    mock_transcribe.return_value = [{"start": 0.0, "end": 1.0, "text": "你好"}]

    pipeline.asr_step(job.id, config, database, runner)

    mock_reset.assert_called_once()


@patch("dv_backend.pipeline.reset_model_cache")
@patch("dv_backend.pipeline.transcribe_audio")
def test_asr_step_releases_gpu_model_after_failure(mock_transcribe, mock_reset, test_env):
    config, database, job_service, runner = test_env
    job = create_local_job(job_service, config.data_dir)
    artifacts_dir = config.data_dir / "jobs" / job.id / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "audio_16k.wav").write_bytes(b"wav")
    mock_transcribe.return_value = []

    with pytest.raises(AppError):
        pipeline.asr_step(job.id, config, database, runner)

    mock_reset.assert_called_once()


@patch("dv_backend.pipeline.transcribe_audio")
def test_asr_rejects_empty_transcription(mock_transcribe, test_env):
    config, database, job_service, runner = test_env
    job = create_local_job(job_service, config.data_dir)
    artifacts_dir = config.data_dir / "jobs" / job.id / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    audio_path = artifacts_dir / "audio_16k.wav"
    audio_path.write_bytes(b"wav")

    mock_transcribe.return_value = []

    with pytest.raises(AppError) as error:
        pipeline.asr_step(job.id, config, database, runner)

    assert error.value.info.code == "EMPTY_ASR_TRANSCRIPTION"


@patch("dv_backend.pipeline.transcribe_audio")
def test_asr_prefers_vocals_16k_when_available(mock_transcribe, test_env):
    config, database, job_service, runner = test_env
    job = create_local_job(job_service, config.data_dir)
    artifacts_dir = config.data_dir / "jobs" / job.id / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    mixed_path = artifacts_dir / "audio_16k.wav"
    vocals_path = artifacts_dir / "vocals_16k.wav"
    write_dummy_wav(mixed_path, amplitude=0.01)
    write_dummy_wav(vocals_path, amplitude=0.2)
    save_checkpoint(config.data_dir, job.id, "extract_audio", {"vocals_16k_path": str(vocals_path)})
    mock_transcribe.return_value = [{"start": 0.0, "end": 1.0, "text": "你好"}]

    result = pipeline.asr_step(job.id, config, database, runner)

    assert mock_transcribe.call_args.args[0] == vocals_path
    assert result["recognition_audio_source"] == "vocals_16k"


@patch("dv_backend.pipeline.get_audio_duration")
@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
def test_vad_step_parses_silence_and_writes_telemetry(mock_run, mock_resolve, mock_probe_duration, test_env):
    config, database, job_service, runner = test_env
    mock_resolve.return_value = Path("dummy-ffmpeg")
    mock_probe_duration.return_value = 10.0

    # test_env defaults vad_engine to silencedetect for FFmpeg-mocked integration tests.

    # Create dummy wav file
    job_dir = config.data_dir / "jobs" / "job123"
    (job_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    audio_wav = job_dir / "artifacts" / "audio_16k.wav"
    audio_wav.write_text("")
    
    # FFmpeg silence detection output simulation
    mock_run.return_value = MagicMock(
        stdout="",
        stderr=(
            "  Duration: 00:00:10.00, start: 0.000000, bitrate: 256 kb/s\n"
            "[silencedetect @ 0x1] silence_start: 2.0\n"
            "[silencedetect @ 0x1] silence_end: 4.5 | silence_duration: 2.5\n"
            "[silencedetect @ 0x1] silence_start: 7.0\n"
            "[silencedetect @ 0x1] silence_end: 9.0 | silence_duration: 2.0\n"
        ),
        returncode=0
    )
    
    res = pipeline.vad_step("job123", config, database, runner)
    
    assert res["total_duration"] == 10.0
    # Gaps between silences are speech:
    # 0.0 -> 2.0 (speech)
    # 2.0 -> 4.5 (silence)
    # 4.5 -> 7.0 (speech)
    # 7.0 -> 9.0 (silence)
    # 9.0 -> 10.0 (speech)
    assert len(res["speech_regions"]) == 3
    assert res["speech_regions"][0] == {"start": 0.0, "end": 2.0}
    assert res["speech_regions"][1] == {"start": 4.5, "end": 7.0}
    assert res["speech_regions"][2] == {"start": 9.0, "end": 10.0}
    mock_probe_duration.assert_called_once()
    telemetry = job_dir / "artifacts" / "telemetry.jsonl"
    assert telemetry.is_file()
    assert '"step": "vad"' in telemetry.read_text(encoding="utf-8")


@patch("dv_backend.pipeline.get_audio_duration")
@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
def test_vad_step_prefers_vocals_16k_when_available(mock_run, mock_resolve, mock_probe_duration, test_env):
    config, database, job_service, runner = test_env
    mock_resolve.return_value = Path("dummy-ffmpeg")
    mock_probe_duration.return_value = 6.0
    job_dir = config.data_dir / "jobs" / "job-vad-vocals"
    artifacts_dir = job_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    write_dummy_wav(artifacts_dir / "audio_16k.wav", amplitude=0.01)
    vocals_16k = artifacts_dir / "vocals_16k.wav"
    write_dummy_wav(vocals_16k, amplitude=0.2)
    save_checkpoint(config.data_dir, "job-vad-vocals", "extract_audio", {"vocals_16k_path": str(vocals_16k)})
    mock_run.return_value = MagicMock(
        stdout="",
        stderr="[silencedetect @ 0x1] silence_start: 2.0\n[silencedetect @ 0x1] silence_end: 3.0 | silence_duration: 1.0\n",
        returncode=0,
    )

    res = pipeline.vad_step("job-vad-vocals", config, database, runner)

    cmd = mock_run.call_args.args[0]
    assert str(vocals_16k) in cmd
    assert res["recognition_audio_source"] == "vocals_16k"


def test_normalize_segments_resolves_overlaps(test_env):
    config, database, job_service, runner = test_env
    
    # Create checkpoints
    save_checkpoint(config.data_dir, "job123", "asr", {
        "segments": [
            {"start": 1.0, "end": 3.0, "text": "Hello"},
            {"start": 2.5, "end": 5.0, "text": "World"}
        ]
    })
    save_checkpoint(config.data_dir, "job123", "vad", {
        "total_duration": 10.0,
        "speech_regions": []
    })
    
    res = pipeline.normalize_segments_step("job123", config, database, runner)
    segments = res["segments"]
    
    assert len(segments) == 2
    # First end is shrunk from 3.0 to 2.5
    assert segments[0]["start"] == 1.0
    assert segments[0]["end"] == 2.5
    assert segments[0]["original_duration"] == 1.5
    # First budget is start of next (2.5) - start of current (1.0) = 1.5
    assert segments[0]["duration_budget"] == 1.5
    
    # Second budget is total_duration (10.0) - start of current (2.5) = 7.5


def test_normalize_segments_filters_vad_false_positives(test_env):
    config, database, job_service, runner = test_env

    save_checkpoint(config.data_dir, "job123", "asr", {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "music"},
            {"start": 1.0, "end": 2.0, "text": "music"},
            {"start": 2.0, "end": 3.0, "text": "real line"},
        ],
        "aligned_units": [],
    })
    save_checkpoint(config.data_dir, "job123", "vad", {
        "total_duration": 5.0,
        "speech_regions": [{"start": 0.0, "end": 3.0}],
    })

    res = pipeline.normalize_segments_step("job123", config, database, runner)
    assert res["vad_false_positive_rejected_count"] == 1
    assert len(res["segments"]) == 2
    assert res["segments"][0]["text"] == "music"
    assert res["segments"][1]["text"] == "real line"


def test_normalize_segments_splits_single_long_asr_segment_with_vad(test_env):
    config, database, job_service, runner = test_env

    save_checkpoint(config.data_dir, "job123", "asr", {
        "segments": [
            {"start": 0.0, "end": 60.0, "text": "ABCDEFGHIJKL"}
        ]
    })
    save_checkpoint(config.data_dir, "job123", "vad", {
        "total_duration": 60.0,
        "speech_regions": [
            {"start": 0.0, "end": 10.0},
            {"start": 20.0, "end": 30.0},
            {"start": 40.0, "end": 50.0},
        ]
    })

    res = pipeline.normalize_segments_step("job123", config, database, runner)
    segments = res["segments"]

    assert len(segments) == 3
    assert [segment["start"] for segment in segments] == [0.0, 20.0, 40.0]
    assert [segment["end"] for segment in segments] == [10.0, 30.0, 50.0]
    assert "".join(segment["text"] for segment in segments) == "ABCDEFGHIJKL"


@patch("dv_backend.translation_timing_rewrite.call_openai_chat")
@patch("dv_backend.pipeline.GeminiTranslator")
def test_translate_step(translator_type, mock_chat, test_env):
    config, database, job_service, runner = test_env

    with database.connection:
        database.connection.execute(
            "INSERT INTO jobs (id, source_url, title, status, created_at, updated_at) "
            "VALUES (?, ?, ?, 'running', ?, ?)",
            ("job123", "https://www.bilibili.com/video/BV1", "中文标题", "now", "now"),
        )
        database.connection.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("translation_backend", json.dumps("gemini"), "now"),
        )
        database.connection.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("gemini_api_keys", json.dumps([{"id": "a", "key": "key-a"}]), "now"),
        )
    
    save_checkpoint(config.data_dir, "job123", "normalize_segments", {
        "segments": [
            {"index": 0, "start": 1.0, "end": 3.0, "text": "你好", "duration_budget": 2.0}
        ]
    })
    
    mock_chat.return_value = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "translations": [
                        {"index": 0, "translation": "Xin chào"}
                    ]
                })
            }
        }]
    }
    
    content = mock_chat.return_value["choices"][0]["message"]["content"]
    translator = translator_type.return_value
    translator.key_pool.cursor = 0
    translator.translate.side_effect = [
        ["Tiêu đề tiếng Việt"],
        [json.loads(content)["translations"][0]["translation"]],
    ]
    res = pipeline.translate_step("job123", config, database, runner)
    assert res["segments"][0]["translation"] == "Xin chào"
    assert res["title_vi"] == "Tiêu đề tiếng Việt"
    row = database.connection.execute(
        "SELECT title_vi FROM jobs WHERE id = 'job123'"
    ).fetchone()
    assert row["title_vi"] == "Tiêu đề tiếng Việt"


@patch("dv_backend.pipeline.GeminiTranslator")
def test_translate_step_uses_gemini_pool_and_persists_cursor(translator_type, test_env):
    config, database, _job_service, runner = test_env
    translator = translator_type.return_value
    translator.translate.side_effect = [["Tieu de"], ["Xin chao"]]
    translator.key_pool.cursor = 1
    save_checkpoint(config.data_dir, "job123", "asr", {
        "aligned_units": [
            {"start": 0.0, "end": 0.4, "text": "你"},
            {"start": 0.4, "end": 0.8, "text": "好"},
        ]
    })
    save_checkpoint(config.data_dir, "job123", "normalize_segments", {
        "segments": [
            {"index": 0, "start": 0.0, "end": 1.0, "text": "hello", "duration_budget": 1.0}
        ]
    })
    with database.connection:
        database.connection.execute(
            "INSERT INTO jobs (id, source_url, title, status, created_at, updated_at) "
            "VALUES (?, ?, ?, 'running', ?, ?)",
            ("job123", "https://www.bilibili.com/video/BV1", "Chinese title", "now", "now"),
        )
        database.connection.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("translation_backend", json.dumps("gemini"), "now"),
        )
        database.connection.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("gemini_api_keys", json.dumps([{"id": "a", "key": "key-a"}]), "now"),
        )

    res = pipeline.translate_step("job123", config, database, runner)

    assert res["segments"][0]["translation"] == "Xin chao"
    assert res["segments"][0]["source_speech_units"] == 2
    assert res["segments"][0]["target_vi_syllables"] == 3
    assert res["segments"][0]["target_vi_syllable_range"] == [2, 4]
    segment_call = translator.translate.call_args_list[1]
    assert segment_call.kwargs["timing_guidance"] == [{
        "source_speech_units": 2,
        "target_vi_syllables": 3,
        "target_vi_syllable_range": [2, 4],
    }]
    assert json.loads(database.connection.execute(
        "SELECT value FROM settings WHERE key = 'gemini_key_cursor'"
    ).fetchone()["value"]) == 1


@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
@patch("dv_backend.pipeline.get_wav_duration")
@patch("dv_backend.pipeline.create_tts_adapter")
def test_tts_step_uses_configured_adapter(
    tts_factory, mock_duration, mock_run, mock_resolve, test_env
):
    config, database, _job_service, runner = test_env
    mock_resolve.return_value = Path("ffmpeg")
    mock_duration.return_value = 1.0
    tts = tts_factory.return_value

    def synthesize(_text, output_path, **_kwargs):
        write_dummy_wav(output_path, duration=1.0, sample_rate=24000, channels=1)

    def convert(cmd, *_args, **_kwargs):
        write_dummy_wav(Path(cmd[-1]), duration=1.0, sample_rate=48000, channels=2)
        return MagicMock(stdout="", stderr="", returncode=0)

    tts.synthesize.side_effect = synthesize
    tts.synthesize_batch.side_effect = lambda items: [synthesize(item["text"], item["output_path"]) for item in items]
    mock_run.side_effect = convert
    save_checkpoint(config.data_dir, "job123", "translate", {
        "segments": [
            {"index": 0, "start": 0.0, "end": 1.0, "translation": "Xin chao", "duration_budget": 1.0}
        ]
    })
    with database.connection:
        database.connection.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("vieneu_voice", json.dumps("Xuân Vĩnh"), "now"),
        )

    pipeline.tts_step("job123", config, database, runner)

    tts.synthesize_batch.assert_called_once()
    assert tts.synthesize.call_count == 0
    tts_factory.assert_called_once()


@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
@patch("dv_backend.pipeline.get_wav_duration")
@patch("dv_backend.pipeline.create_tts_adapter")
def test_tts_step_micro_batches_multiple_segments(
    tts_factory, mock_duration, mock_run, mock_resolve, test_env
):
    config, database, _job_service, runner = test_env
    mock_resolve.return_value = Path("ffmpeg")
    mock_duration.return_value = 1.0
    tts = tts_factory.return_value

    def synthesize_batch(items):
        for item in items:
            write_dummy_wav(item["output_path"], duration=1.0, sample_rate=24000, channels=1)

    tts.synthesize_batch.side_effect = synthesize_batch
    save_checkpoint(config.data_dir, "job123", "translate", {
        "segments": [
            {"index": 0, "start": 0.0, "end": 1.0, "translation": "Xin chao", "duration_budget": 1.0},
            {"index": 1, "start": 1.0, "end": 2.0, "translation": "Tam biet", "duration_budget": 1.0},
        ]
    })
    with database.connection:
        database.connection.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("tts_micro_batch_enabled", json.dumps(True), "now"),
        )

    pipeline.tts_step("job123", config, database, runner)

    tts.synthesize_batch.assert_called_once()
    items = tts.synthesize_batch.call_args.args[0]
    assert [item["text"] for item in items] == ["Xin chao", "Tam biet"]
    assert tts.synthesize.call_count == 0


@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
@patch("dv_backend.pipeline.get_wav_duration")
@patch("dv_backend.pipeline.create_tts_adapter")
def test_tts_per_segment_keeps_final_tts_path_after_step(
    tts_factory, mock_duration, mock_run, mock_resolve, test_env
):
    config, database, _job_service, runner = test_env
    mock_resolve.return_value = Path("ffmpeg")
    mock_duration.return_value = 1.0
    tts = tts_factory.return_value

    def synthesize(_text, output_path, **_kwargs):
        write_dummy_wav(output_path, duration=1.0, sample_rate=24000, channels=1)

    def convert(cmd, *_args, **_kwargs):
        out_path = Path(cmd[-1])
        write_dummy_wav(out_path, duration=1.0, sample_rate=48000, channels=2)
        return MagicMock(stdout="", stderr="", returncode=0)

    tts.synthesize.side_effect = synthesize
    tts.synthesize_batch.side_effect = lambda items: [synthesize(item["text"], item["output_path"]) for item in items]
    mock_run.side_effect = convert

    save_checkpoint(config.data_dir, "job123", "translate", {
        "segments": [
            {"index": 0, "start": 0.0, "end": 1.0, "translation": "Xin chao", "duration_budget": 1.0}
        ]
    })
    with database.connection:
        database.connection.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("tts_conversion_strategy", json.dumps("per_segment"), "now"),
        )

    result = pipeline.tts_step("job123", config, database, runner)

    final_path = Path(result["segments"][0]["tts_path"])
    assert final_path.is_file(), f"per_segment tts_path missing: {final_path}"


@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
@patch("dv_backend.pipeline.get_wav_duration")
@patch("dv_backend.pipeline.create_tts_adapter")
def test_tts_step_passes_clone_mode_and_anchor_transcript(
    tts_factory, mock_duration, mock_run, mock_resolve, test_env
):
    config, database, _job_service, runner = test_env
    mock_resolve.return_value = Path("ffmpeg")
    mock_duration.return_value = 1.0
    tts = tts_factory.return_value

    def synthesize(_text, output_path, **_kwargs):
        write_dummy_wav(output_path, duration=1.0, sample_rate=24000, channels=1)

    def convert(cmd, *_args, **_kwargs):
        write_dummy_wav(Path(cmd[-1]), duration=1.0, sample_rate=48000, channels=2)
        return MagicMock(stdout="", stderr="", returncode=0)

    tts.synthesize.side_effect = synthesize
    tts.synthesize_batch.side_effect = lambda items: [synthesize(item["text"], item["output_path"]) for item in items]
    mock_run.side_effect = convert

    cloned_dir = config.data_dir / "cloned_voices_omnivoice"
    cloned_dir.mkdir(parents=True, exist_ok=True)
    voice_path = cloned_dir / "voice-abc.wav"
    write_dummy_wav(voice_path, duration=1.0, sample_rate=16000, channels=1)
    voice_path.with_suffix(".txt").write_text("xin chào", encoding="utf-8")

    save_checkpoint(config.data_dir, "job123", "translate", {
        "segments": [
            {"index": 0, "start": 0.0, "end": 1.0, "translation": "Xin chao", "duration_budget": 1.0}
        ]
    })
    with database.connection:
        database.connection.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("omnivoice_ref_audio", json.dumps(str(voice_path)), "now"),
        )
        database.connection.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("omnivoice_ref_text", json.dumps("xin chào"), "now"),
        )

    pipeline.tts_step("job123", config, database, runner)

    tts.synthesize_batch.assert_called_once()
    batch_items = tts.synthesize_batch.call_args.args[0]
    assert batch_items[0]["text"] == "Xin chao"
    assert batch_items[0]["output_path"].name == "tts_raw_0.wav"

    session_kwargs = tts_factory.call_args.kwargs
    from dv_backend.adapters.tts import TtsSession
    session = TtsSession(settings={"omnivoice_ref_audio": str(voice_path), "omnivoice_ref_text": "xin chào"}, data_dir=config.data_dir, runner=runner, adapter_factory=tts_factory)
    assert session.clone is True
    assert session.clone_mode == "reference"
    assert session.anchor_text == "xin chào"
    assert session.voice == str(voice_path)
    assert session_kwargs["data_dir"] == config.data_dir


@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
@patch("dv_backend.pipeline.get_wav_duration")
def test_duration_repair_time_stretches_fallback(mock_dur, mock_run, mock_resolve, test_env):
    config, database, job_service, runner = test_env
    mock_resolve.return_value = Path("dummy-ffmpeg")
    
    save_checkpoint(config.data_dir, "job123", "tts", {
        "segments": [
            {"index": 0, "start": 0.0, "end": 2.0, "translation": "Vietnamese sentence that is way too long to speak in two seconds.", "duration_budget": 2.0, "tts_duration": 4.0}
        ]
    })
    tts_dir = config.data_dir / "jobs" / "job123" / "artifacts" / "tts"
    write_dummy_wav(tts_dir / "tts_0.wav", duration=4.0)
    
    # Mock duration flow:
    # 1) current duration from original segment
    # 2) duration after 1.2x speed-up
    # 3) duration after exact trim
    # 4) final repaired duration
    mock_dur.side_effect = [4.0, 3.33, 2.0, 2.0]

    def ffmpeg_side_effect(cmd, *_args, **_kwargs):
        out_path = Path(cmd[-1])
        write_dummy_wav(out_path, duration=2.0)
        return MagicMock(stdout="", stderr="", returncode=0)
    mock_run.side_effect = ffmpeg_side_effect
    
    res = pipeline.duration_repair_step("job123", config, database, runner)
    
    assert "time_stretch_1.2x" in res["segments"][0]["repaired_method"]
    assert "exact_trim_pad" not in res["segments"][0]["repaired_method"]
    assert res["segments"][0].get("speech_trimmed") is not True
    assert res["segments"][0]["repaired_duration"] == 2.0
    
    # Assert ffmpeg atempo filter was called
    mock_run.assert_called()
    calls = [" ".join(call.args[0]) for call in mock_run.call_args_list]
    assert any("atempo=" in cmd for cmd in calls)


@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
@patch("dv_backend.pipeline.get_wav_duration")
def test_duration_repair_pads_short_segments_to_budget(mock_dur, mock_run, mock_resolve, test_env):
    config, database, _job_service, runner = test_env
    mock_resolve.return_value = Path("dummy-ffmpeg")
    save_checkpoint(config.data_dir, "job-short", "tts", {
        "segments": [
            {
                "index": 0,
                "start": 0.0,
                "end": 2.0,
                "translation": "Ngắn.",
                "duration_budget": 2.0,
                "tts_duration": 1.0,
            }
        ]
    })
    tts_dir = config.data_dir / "jobs" / "job-short" / "artifacts" / "tts"
    write_dummy_wav(tts_dir / "tts_0.wav", duration=1.0)
    mock_dur.side_effect = [1.0, 2.0, 2.0]
    mock_run.side_effect = lambda cmd, *_args, **_kwargs: (
        write_dummy_wav(Path(cmd[-1]), duration=2.0) or MagicMock(stdout="", stderr="", returncode=0)
    )

    result = pipeline.duration_repair_step("job-short", config, database, runner)

    # Phase 2: short natural speech may remain unpadded when it fits the speech window.
    assert result["segments"][0]["repaired_method"] in {"none", "tail_silence_pad"}
    assert result["segments"][0]["repaired_duration"] == 2.0
    calls = [" ".join(call.args[0]) for call in mock_run.call_args_list]
    assert any("apad=" in cmd for cmd in calls)
    assert not any("adelay=" in cmd for cmd in calls)
    assert not any("atempo=" in cmd for cmd in calls)


@patch("dv_backend.pipeline.TtsSession")
@patch("dv_backend.pipeline.lengthen_translation_for_timing")
@patch("dv_backend.pipeline.get_wav_duration")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
@patch("dv_backend.pipeline.resolve_tool_path")
def test_duration_repair_lengthens_when_gap_exceeds_one_second(
    mock_resolve,
    mock_run,
    mock_dur,
    mock_lengthen,
    mock_tts_session,
    test_env,
):
    config, database, _job_service, runner = test_env
    with database.connection:
        database.connection.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("allow_spoken_text_mutation", json.dumps(True), "now"),
        )
    mock_resolve.return_value = Path("dummy-ffmpeg")
    save_checkpoint(config.data_dir, "job-long-gap", "tts", {
        "segments": [
            {
                "index": 0,
                "start": 0.0,
                "end": 4.0,
                "translation": "Ngắn.",
                "duration_budget": 4.0,
                "tts_duration": 1.0,
            }
        ]
    })
    tts_dir = config.data_dir / "jobs" / "job-long-gap" / "artifacts" / "tts"
    write_dummy_wav(tts_dir / "tts_0.wav", duration=1.0)
    mock_lengthen.return_value = ("Ngắn hơn một chút nhé.", 4)
    mock_dur.side_effect = [1.0, 3.5, 3.5, 3.5, 3.5]
    mock_run.side_effect = lambda cmd, *_args, **_kwargs: (
        write_dummy_wav(Path(cmd[-1]), duration=3.5) or MagicMock(stdout="", stderr="", returncode=0)
    )
    session = MagicMock()
    mock_tts_session.return_value.__enter__.return_value = session

    result = pipeline.duration_repair_step("job-long-gap", config, database, runner)

    mock_lengthen.assert_called_once()
    session.synthesize.assert_called_once()
    assert "_lengthen" in result["segments"][0]["repaired_method"]
    assert result["segments"][0]["translation"] == "Ngắn hơn một chút nhé."


def test_tail_silence_pad_filter_only_appends_silence_at_end() -> None:
    expr = pipeline._tail_silence_pad_filter(1.0, 2.0)
    assert "adelay=" not in expr
    assert "afade=t=out" in expr
    assert "apad=" in expr
    assert "atrim=0:2.000" in expr


def test_repair_target_duration_caps_inter_segment_pause() -> None:
    segment = {"start": 1.0, "end": 3.0, "original_duration": 2.0}
    assert pipeline._repair_target_duration(segment, budget=9.0, tolerance_sec=0.04) == 3.5


@patch("dv_backend.pipeline.get_wav_duration")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
@patch("dv_backend.pipeline.resolve_tool_path")
def test_duration_repair_does_not_pad_to_full_inter_segment_budget(
    mock_resolve,
    mock_run,
    mock_dur,
    test_env,
):
    config, database, _job_service, runner = test_env
    mock_resolve.return_value = Path("dummy-ffmpeg")
    save_checkpoint(config.data_dir, "job-wide-budget", "tts", {
        "segments": [
            {
                "index": 0,
                "start": 1.0,
                "end": 3.0,
                "original_duration": 2.0,
                "translation": "Một câu ngắn.",
                "duration_budget": 9.0,
                "tts_duration": 1.5,
            }
        ]
    })
    tts_dir = config.data_dir / "jobs" / "job-wide-budget" / "artifacts" / "tts"
    write_dummy_wav(tts_dir / "tts_0.wav", duration=1.5)
    mock_dur.side_effect = [1.5, 3.0, 3.0, 3.0, 3.0]
    mock_run.side_effect = lambda cmd, *_args, **_kwargs: (
        write_dummy_wav(Path(cmd[-1]), duration=3.0) or MagicMock(stdout="", stderr="", returncode=0)
    )

    result = pipeline.duration_repair_step("job-wide-budget", config, database, runner)

    # Speech target follows original_duration (2.0s), not the full inter-segment budget (9.0s).
    assert result["segments"][0]["repaired_method"] in {"none", "tail_silence_pad", "outer_silence_trim"}
    assert result["segments"][0]["repaired_duration"] <= 3.0


def write_dummy_wav(
    path: Path,
    duration: float = 5.0,
    sample_rate: int = 48000,
    channels: int = 2,
    amplitude: float = 0.0,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        frame_count = int(duration * sample_rate)
        sample_value = max(-32767, min(32767, int(32767 * amplitude)))
        frame = int(sample_value).to_bytes(2, byteorder="little", signed=True) * channels
        if sample_value == 0:
            w.writeframes(b"\x00" * int(duration * sample_rate * channels * 2))
        else:
            w.writeframes(frame * frame_count)


@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
def test_tts_lazy_mix_defers_per_segment_conversion(mock_run, mock_resolve, test_env):
    config, database, _job_service, runner = test_env
    mock_resolve.return_value = Path("ffmpeg")

    save_checkpoint(config.data_dir, "job123", "translate", {
        "segments": [
            {"index": 0, "start": 0.0, "end": 1.0, "translation": "Xin chao", "duration_budget": 1.0}
        ]
    })
    with database.connection:
        database.connection.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("tts_conversion_strategy", json.dumps("lazy_mix"), "now"),
        )

    def synthesize(_text, output_path, **_kwargs):
        write_dummy_wav(output_path, duration=1.0, sample_rate=24000, channels=1)

    with patch("dv_backend.pipeline.create_tts_adapter") as tts_factory:
        tts_factory.return_value.synthesize.side_effect = synthesize
        tts_factory.return_value.synthesize_batch.side_effect = lambda items: [synthesize(item["text"], item["output_path"]) for item in items]
        result = pipeline.tts_step("job123", config, database, runner)

    assert mock_run.call_count == 0
    segment = result["segments"][0]
    assert Path(segment["tts_raw_path"]).is_file()
    assert segment["tts_path"] is None


@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
def test_extract_audio_creates_background_stems_for_background_only_mix(
    mock_run, mock_resolve, test_env
):
    config, database, _job_service, runner = test_env
    mock_resolve.return_value = Path("ffmpeg")
    job_id = "job-extract"
    artifacts_dir = config.data_dir / "jobs" / job_id / "artifacts"
    original_mp4 = artifacts_dir / "original.mp4"
    original_mp4.parent.mkdir(parents=True, exist_ok=True)
    original_mp4.write_bytes(b"mp4")

    commands: list[list[str]] = []

    def side_effect(cmd, *_args, **_kwargs):
        commands.append(cmd)
        if "demucs.separate" in " ".join(cmd):
            stem_dir = artifacts_dir / "demucs" / "htdemucs" / "original_48k"
            stem_dir.mkdir(parents=True, exist_ok=True)
            write_dummy_wav(stem_dir / "no_vocals.wav")
            write_dummy_wav(stem_dir / "vocals.wav")
        elif str(cmd[-1]).endswith(".wav"):
            write_dummy_wav(Path(cmd[-1]))
        return MagicMock(stdout="", stderr="", returncode=0)

    mock_run.side_effect = side_effect

    result = pipeline.extract_audio_step(job_id, config, database, runner)

    assert result["bgm_path"] == str(artifacts_dir / "bgm.wav")
    assert result["vocals_path"] == str(artifacts_dir / "vocals.wav")
    assert result["vocals_16k_path"] == str(artifacts_dir / "vocals_16k.wav")
    assert result["bgm_16k_path"] == str(artifacts_dir / "bgm_16k.wav")
    assert any("demucs.separate" in " ".join(cmd) for cmd in commands)
    assert any(str(artifacts_dir / "vocals_16k.wav") == str(cmd[-1]) for cmd in commands)
    assert any(str(artifacts_dir / "bgm_16k.wav") == str(cmd[-1]) for cmd in commands)


@patch("dv_backend.pipeline.get_video_stream_duration", return_value=5.0)
@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
def test_mix_lazy_mix_resamples_raw_tts_in_filtergraph(
    mock_run, mock_resolve, _mock_video_dur, test_env
):
    config, database, _job_service, runner = test_env
    mock_resolve.return_value = Path("ffmpeg")
    job_id = "job-lazy-mix"
    artifacts_dir = config.data_dir / "jobs" / job_id / "artifacts"
    original = artifacts_dir / "original_48k.wav"
    original_mp4 = artifacts_dir / "original.mp4"
    raw = artifacts_dir / "tts" / "tts_raw_0.wav"
    write_dummy_wav(original)
    original_mp4.write_bytes(b"mp4")
    write_dummy_wav(raw, duration=1.0, sample_rate=24000, channels=1)
    save_checkpoint(config.data_dir, job_id, "extract_audio", {"original_48k_path": str(original)})
    save_checkpoint(
        config.data_dir,
        job_id,
        "duration_repair",
        {"segments": [{"index": 0, "start": 0.0, "tts_raw_path": str(raw), "tts_path": None}]},
    )

    def write_mix(cmd, *_args, **_kwargs):
        write_dummy_wav(Path(cmd[-1]))
        return MagicMock(stdout="", stderr="", returncode=0)

    mock_run.side_effect = write_mix

    result = pipeline.mix_step(job_id, config, database, runner)

    cmd = mock_run.call_args_list[0].args[0]
    assert str(raw) in cmd
    filter_graph = cmd[cmd.index("-filter_complex") + 1]
    assert "aresample=48000" in filter_graph
    assert "afade=t=in" in filter_graph
    assert "afade=t=out" in filter_graph
    assert result["narration_segment_input_count"] == 1


@patch("dv_backend.pipeline.get_video_stream_duration", return_value=5.0)
@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
def test_mix_uses_background_stem_and_normalizes_levels(
    mock_run, mock_resolve, _mock_video_dur, test_env
):
    config, database, _job_service, runner = test_env
    mock_resolve.return_value = Path("ffmpeg")
    job_id = "job123"
    artifacts_dir = config.data_dir / "jobs" / job_id / "artifacts"
    original = artifacts_dir / "original_48k.wav"
    bgm = artifacts_dir / "bgm.wav"
    narration_segment = artifacts_dir / "tts" / "tts_repaired_0.wav"
    write_dummy_wav(original)
    (artifacts_dir / "original.mp4").write_bytes(b"mp4")
    write_dummy_wav(bgm)
    write_dummy_wav(narration_segment, duration=1.0)
    save_checkpoint(
        config.data_dir,
        job_id,
        "extract_audio",
        {"original_48k_path": str(original), "bgm_path": str(bgm)},
    )
    save_checkpoint(
        config.data_dir,
        job_id,
        "duration_repair",
        {"segments": [{"index": 0, "start": 0.0}]},
    )

    def write_mix(cmd, *_args, **_kwargs):
        write_dummy_wav(Path(cmd[-1]))
        return MagicMock(stdout="", stderr="", returncode=0)

    mock_run.side_effect = write_mix

    result = pipeline.mix_step(job_id, config, database, runner)

    mix_cmd = mock_run.call_args_list[1].args[0]
    filter_graph = mix_cmd[mix_cmd.index("-filter_complex") + 1]
    assert result["mix_mode"] == "background_only"
    assert str(bgm) in mix_cmd
    assert "sidechaincompress" not in filter_graph
    assert "loudnorm=I=-24" in filter_graph
    assert "loudnorm=I=-16" in filter_graph
    assert "apad,atrim=0:5.000000" in filter_graph
    assert result["target_video_duration_sec"] == 5.0


@patch("dv_backend.pipeline.get_video_stream_duration", return_value=5.0)
@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
def test_mix_legacy_separate_setting_uses_background_only(
    mock_run,
    mock_resolve,
    _mock_video_dur,
    test_env,
):
    config, database, _job_service, runner = test_env
    mock_resolve.return_value = Path("ffmpeg")
    job_id = "job123"
    artifacts_dir = config.data_dir / "jobs" / job_id / "artifacts"
    original = artifacts_dir / "original_48k.wav"
    bgm = artifacts_dir / "bgm.wav"
    narration_segment = artifacts_dir / "tts" / "tts_repaired_0.wav"
    write_dummy_wav(original)
    (artifacts_dir / "original.mp4").write_bytes(b"mp4")
    write_dummy_wav(bgm)
    write_dummy_wav(narration_segment, duration=1.0)
    with database.connection:
        database.connection.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("mix_mode", json.dumps("separate"), "now"),
        )

    save_checkpoint(
        config.data_dir,
        job_id,
        "extract_audio",
        {"original_48k_path": str(original), "bgm_path": str(bgm)},
    )
    save_checkpoint(
        config.data_dir,
        job_id,
        "duration_repair",
        {"segments": [{"index": 0, "start": 0.0}]},
    )

    def write_mix(cmd, *_args, **_kwargs):
        write_dummy_wav(Path(cmd[-1]))
        return MagicMock(stdout="", stderr="", returncode=0)

    mock_run.side_effect = write_mix

    result = pipeline.mix_step(job_id, config, database, runner)

    mix_cmd = mock_run.call_args_list[1].args[0]
    filter_graph = mix_cmd[mix_cmd.index("-filter_complex") + 1]
    assert result["requested_mix_mode"] == "background_only"
    assert result["mix_mode"] == "background_only"
    assert "sidechaincompress" not in filter_graph
    assert str(bgm) in mix_cmd


@patch("dv_backend.pipeline.get_video_stream_duration", return_value=5.0)
@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
def test_mix_falls_back_to_ducking_without_background_stem(
    mock_run,
    mock_resolve,
    _mock_video_dur,
    test_env,
):
    config, database, _job_service, runner = test_env
    mock_resolve.return_value = Path("ffmpeg")
    job_id = "job-fallback"
    artifacts_dir = config.data_dir / "jobs" / job_id / "artifacts"
    original = artifacts_dir / "original_48k.wav"
    narration_segment = artifacts_dir / "tts" / "tts_repaired_0.wav"
    write_dummy_wav(original)
    (artifacts_dir / "original.mp4").write_bytes(b"mp4")
    write_dummy_wav(narration_segment, duration=1.0)

    save_checkpoint(
        config.data_dir,
        job_id,
        "extract_audio",
        {"original_48k_path": str(original)},
    )
    save_checkpoint(
        config.data_dir,
        job_id,
        "duration_repair",
        {"segments": [{"index": 0, "start": 0.0}]},
    )

    def write_mix(cmd, *_args, **_kwargs):
        write_dummy_wav(Path(cmd[-1]))
        return MagicMock(stdout="", stderr="", returncode=0)

    mock_run.side_effect = write_mix

    result = pipeline.mix_step(job_id, config, database, runner)

    mix_cmd = mock_run.call_args_list[1].args[0]
    filter_graph = mix_cmd[mix_cmd.index("-filter_complex") + 1]
    assert result["mix_mode"] == "duck"
    assert "sidechaincompress" in filter_graph
    assert str(original) in mix_cmd
    assert "apad,atrim=0:5.000000" in filter_graph


@patch("dv_backend.pipeline.get_video_stream_duration", return_value=5.0)
@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
def test_mix_saves_vietnamese_narration_as_debug_output(
    mock_run, mock_resolve, _mock_video_dur, test_env
):
    config, database, _job_service, runner = test_env
    mock_resolve.return_value = Path("ffmpeg")
    job_id = "job123"
    artifacts_dir = config.data_dir / "jobs" / job_id / "artifacts"
    original = artifacts_dir / "original_48k.wav"
    narration_segment = artifacts_dir / "tts" / "tts_repaired_0.wav"
    write_dummy_wav(original)
    (artifacts_dir / "original.mp4").write_bytes(b"mp4")
    write_dummy_wav(narration_segment, duration=1.0)
    save_checkpoint(
        config.data_dir,
        job_id,
        "extract_audio",
        {"original_48k_path": str(original)},
    )
    save_checkpoint(
        config.data_dir,
        job_id,
        "duration_repair",
        {"segments": [{"index": 0, "start": 0.0}]},
    )

    def write_mix(cmd, *_args, **_kwargs):
        write_dummy_wav(Path(cmd[-1]))
        return MagicMock(stdout="", stderr="", returncode=0)

    mock_run.side_effect = write_mix

    result = pipeline.mix_step(job_id, config, database, runner)

    debug_path = config.data_dir / "jobs" / job_id / "output" / "vietnamese_narration.wav"
    assert debug_path.is_file()
    assert result["vietnamese_narration_path"] == str(debug_path)


@patch("dv_backend.pipeline.probe_video_dimensions")
@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
def test_render_step_burns_subtitles_when_enabled(
    mock_run,
    mock_resolve,
    mock_probe,
    test_env,
):
    config, database, _job_service, runner = test_env
    mock_resolve.return_value = Path("ffmpeg")
    mock_probe.return_value = (1080, 1920)
    job_id = "job-subtitles"
    artifacts_dir = config.data_dir / "jobs" / job_id / "artifacts"
    output_dir = config.data_dir / "jobs" / job_id / "output"
    original = artifacts_dir / "original.mp4"
    mixed = artifacts_dir / "mixed.wav"
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(b"mp4")
    write_dummy_wav(mixed)

    with database.connection:
        database.connection.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("subtitles_enabled", json.dumps(True), "now"),
        )
        database.connection.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("subtitle_font_size", json.dumps(52), "now"),
        )
        database.connection.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("subtitle_font_color", json.dumps("#FFFF00"), "now"),
        )
        database.connection.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("subtitle_background_color", json.dumps("#000000"), "now"),
        )
        database.connection.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("subtitle_background_opacity", json.dumps(80), "now"),
        )
        database.connection.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("subtitle_position", json.dumps("center"), "now"),
        )

    save_checkpoint(
        config.data_dir,
        job_id,
        "download",
        {"output_path": str(original)},
    )
    save_checkpoint(
        config.data_dir,
        job_id,
        "mix",
        {"mixed_wav_path": str(mixed)},
    )
    save_checkpoint(
        config.data_dir,
        job_id,
        "duration_repair",
        {
            "segments": [
                {
                    "index": 0,
                    "start": 0.0,
                    "end": 2.0,
                    "translation": "Xin chào mọi người",
                    "repaired_duration": 2.0,
                }
            ]
        },
    )

    render_calls: list[list[str]] = []

    def side_effect(cmd, *_args, **_kwargs):
        if "-c:v libx264" in " ".join(cmd):
            render_calls.append(cmd)
            Path(cmd[-1]).write_bytes(b"rendered")
        elif str(cmd[-1]).endswith(".wav"):
            write_dummy_wav(Path(cmd[-1]))
        return MagicMock(stdout="", stderr="", returncode=0)

    mock_run.side_effect = side_effect

    result = pipeline.render_step(job_id, config, database, runner)

    assert render_calls
    render_cmd = render_calls[0]
    vf_index = render_cmd.index("-vf")
    assert "subtitles=" in render_cmd[vf_index + 1]
    assert (output_dir / "subtitles.ass").is_file()
    assert "Xin chào mọi người" in (output_dir / "subtitles.ass").read_text(encoding="utf-8-sig")
    style_line = next(
        line for line in (output_dir / "subtitles.ass").read_text(encoding="utf-8-sig").splitlines()
        if line.startswith("Style:")
    )
    assert ",4," in style_line
    assert "\\bord" in (output_dir / "subtitles.ass").read_text(encoding="utf-8-sig")
    assert result["subtitles_enabled"] is True
    assert result["subtitles_path"] == str(output_dir / "subtitles.ass")


@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
@patch("dv_backend.pipeline.transcribe_audio")
@patch("dv_backend.pipeline.GeminiTranslator")
@patch("dv_backend.pipeline.create_tts_adapter")
@patch("dv_backend.pipeline.get_video_stream_duration", return_value=5.0)
def test_full_runner_execution_and_resume(
    _mock_video_dur,
    tts_factory,
    translator_type,
    mock_transcribe,
    mock_run,
    mock_resolve,
    test_env,
    tmp_path: Path,
):
    config, database, job_service, runner = test_env
    with database.connection:
        database.connection.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("translation_backend", json.dumps("gemini"), "now"),
        )
        database.connection.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("gemini_api_keys", json.dumps([{"id": "a", "key": "key-a"}]), "now"),
        )
    mock_resolve.return_value = Path("dummy-tool")
    mock_transcribe.return_value = [
        {"start": 0.0, "end": 1.0, "text": "Hello"},
        {"start": 2.0, "end": 3.0, "text": "World"},
    ]

    def side_effect(cmd, job_id, runner_instance, timeout=None):
        cmd_str = " ".join(cmd)

        if "demucs.separate" in cmd_str:
            source_wav = Path(cmd[-1])
            stem_dir = source_wav.parent / "demucs" / "htdemucs" / source_wav.stem
            stem_dir.mkdir(parents=True, exist_ok=True)
            write_dummy_wav(stem_dir / "no_vocals.wav", duration=5.0, sample_rate=48000, channels=2)
            write_dummy_wav(stem_dir / "vocals.wav", duration=5.0, sample_rate=48000, channels=2)
            return MagicMock(stdout="", stderr="", returncode=0)

        if "-o" in cmd:
            idx = cmd.index("-o")
            out_path = Path(cmd[idx + 1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("dummy mp4")
            return MagicMock(stdout="", stderr="", returncode=0)

        if "silencedetect" in cmd_str:
            return MagicMock(
                stdout="",
                stderr="Duration: 00:00:05.00\nsilence_start: 1.0\nsilence_end: 2.0\n",
                returncode=0
            )

        if "-oj" in cmd:
            wav_path = Path(cmd[4])
            json_path = wav_path.with_suffix(".wav.json")
            json_path.write_text(json.dumps({
                "result": {
                    "transcription": [
                        {"offsets": {"from": 0, "to": 1000}, "text": "Hello"},
                        {"offsets": {"from": 2000, "to": 3000}, "text": "World"}
                    ]
                }
            }))
            return MagicMock(stdout="", stderr="", returncode=0)

        if "-c:v libx264" in cmd_str:
            out_path = Path(cmd[-1])
            out_path.write_text("dummy dubbed mp4")
            return MagicMock(stdout="", stderr="", returncode=0)

        if cmd[-1].endswith(".wav"):
            out_path = Path(cmd[-1])
            if "audio_16k.wav" in str(out_path):
                write_dummy_wav(out_path, duration=5.0, sample_rate=16000, channels=1)
            elif "mixed.wav" in str(out_path) or "normalized.wav" in str(out_path) or "narration.wav" in str(out_path):
                write_dummy_wav(out_path, duration=5.0, sample_rate=48000, channels=2)
            else:
                write_dummy_wav(out_path, duration=1.0, sample_rate=48000, channels=2)
            return MagicMock(stdout="", stderr="", returncode=0)

        return MagicMock(stdout="", stderr="", returncode=0)

    mock_run.side_effect = side_effect
    translator = translator_type.return_value
    translator.key_pool.cursor = 0
    translator.translate.return_value = ["tr0", "tr1"]

    def tts_synthesize_side_effect(text, output_path, **kwargs):
        write_dummy_wav(output_path, duration=1.0, sample_rate=24000, channels=1)

    tts_factory.return_value.synthesize.side_effect = tts_synthesize_side_effect
    tts_factory.return_value.synthesize_batch.side_effect = lambda items: [tts_synthesize_side_effect(item["text"], item["output_path"]) for item in items]

    with patch("dv_backend.translation_timing_rewrite.call_openai_chat") as mock_chat:
        mock_chat.return_value = {"choices": [{"message": {"content": json.dumps({"translations": [{"index": 0, "translation": "tr0"}, {"index": 1, "translation": "tr1"}]})}}]}

        source_video = tmp_path / "sample.mp4"
        source_video.write_text("dummy mp4")
        job = job_service.create_imported(source_video, original_filename="sample.mp4")
        runner.start_job(job.id)

        for _ in range(30):
            time.sleep(0.05)
            job_db = job_service.get(job.id)
            if job_db.status in {"completed", "failed"}:
                break

        job_db = job_service.get(job.id)
        if job_db.status != "completed":
            events = database.connection.execute("SELECT * FROM events").fetchall()
            print("EVENTS:", [dict(e) for e in events])
            print("JOB ERROR:", job_db.last_error_code, job_db.last_error_message)
            for step in job_db.steps:
                print("STEP:", step.name, step.status)
        assert job_db.status == "completed"
        assert all(step.status == "completed" for step in job_db.steps)


@patch("dv_backend.subtitle_timing.transcribe_tts_clip_details_for_subtitles")
def test_align_final_dub_integration(mock_transcribe, test_env, tmp_path: Path) -> None:
    config, database, _job_service, runner = test_env
    job_id = "job-align-final-dub"
    artifacts_dir = config.data_dir / "jobs" / job_id / "artifacts" / "tts"
    wav0 = artifacts_dir / "tts_repaired_0.wav"
    wav1 = artifacts_dir / "tts_repaired_1.wav"
    write_dummy_wav(wav0, duration=1.0)
    write_dummy_wav(wav1, duration=1.0)

    segments = [
        {
            "index": 0,
            "start": 0.0,
            "placement_start": 1.0,
            "repaired_duration": 1.0,
            "translation": "Hôm nay thử món này.",
            "tts_path": str(wav0),
        },
        {
            "index": 1,
            "start": 2.0,
            "placement_start": 5.0,
            "repaired_duration": 1.0,
            "translation": "Rất ngon.",
            "tts_path": str(wav1),
        },
    ]
    save_checkpoint(config.data_dir, job_id, "duration_repair", {"segments": segments})

    def fake_transcribe(wav_path, **_kwargs):
        if wav_path.stem.endswith("_0") or "repaired_0" in wav_path.stem:
            return {
                "aligned_units": [
                    {"text": "hôm", "start": 0.0, "end": 0.15},
                    {"text": "nay", "start": 0.15, "end": 0.3},
                    {"text": "thử", "start": 0.3, "end": 0.5},
                    {"text": "món", "start": 0.5, "end": 0.7},
                    {"text": "nầy", "start": 0.7, "end": 0.95},
                ],
                "segments": [],
                "from_forced_aligner": True,
            }
        return {"aligned_units": [], "segments": [], "from_forced_aligner": False}

    mock_transcribe.side_effect = fake_transcribe

    result = pipeline.align_final_dub_step(job_id, config, database, runner)
    assert result["aligned_count"] >= 1
    assert result["dub_alignment_fallback_count"] >= 1

    aligned = load_checkpoint(config.data_dir, job_id, "align_final_dub")
    assert aligned is not None
    seg0 = aligned["segments"][0]
    seg1 = aligned["segments"][1]
    assert seg0["dub_words"]
    assert seg0["dub_words"][-1]["text"] == "này."
    assert seg0["dub_words"][0]["absolute_start"] == pytest.approx(1.0, abs=0.05)
    assert seg1["dub_alignment_status"] in {"fallback_interpolated", "no_speech"}

    from dv_backend.adapters.subtitles import build_subtitle_cues

    cues = build_subtitle_cues(aligned["segments"])
    assert cues
    assert "này." in cues[0]["text"]
    assert all(cues[index]["start"] >= cues[index - 1]["end"] - 0.001 for index in range(1, len(cues)))

    pipeline.align_final_dub_step(job_id, config, database, runner)
    assert mock_transcribe.call_count == 2

