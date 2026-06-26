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
        
    job_service = JobService(database, tmp_path)
    runner = JobRunner(config, database)
    return config, database, job_service, runner


@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
def test_resolve_step_single_video(mock_run, mock_resolve, test_env):
    config, database, job_service, runner = test_env
    mock_resolve.return_value = Path("dummy-yt-dlp")
    
    # Mock single video dump from yt-dlp
    mock_run.return_value = MagicMock(
        stdout=json.dumps({
            "_type": "video",
            "id": "douyin123",
            "title": "Funny cat video",
            "webpage_url": "https://www.douyin.com/video/douyin123",
            "duration": 15.5,
            "thumbnail": "https://img.douyin.com/thumb.jpg"
        }),
        stderr="",
        returncode=0
    )
    
    job = job_service.create("https://www.douyin.com/video/douyin123")
    res = pipeline.resolve_step(job.id, config, database, runner)
    
    assert res["is_playlist"] is False
    assert len(res["videos"]) == 1
    assert res["selected_video"]["id"] == "douyin123"
    
    # Verify title is saved to jobs table
    job_db = job_service.get(job.id)
    assert job_db.title == "Funny cat video"


def test_normalize_douyin_jingxuan_modal_url() -> None:
    assert pipeline.normalize_douyin_url(
        "https://www.douyin.com/jingxuan?modal_id=7639476837437699301"
    ) == "https://www.douyin.com/video/7639476837437699301"


def test_is_douyin_user_profile_url() -> None:
    url = (
        "https://www.douyin.com/user/"
        "MS4wLjABAAAAOnRpvxiasUeDLCX4WG94yZ3LA6ogPP7MJ6rNzi7bFy8m6QrRR9orTshL80q-1cUc"
    )
    assert pipeline.is_douyin_user_profile_url(url) is True
    assert pipeline.is_douyin_user_profile_url("https://www.douyin.com/video/123") is False


def test_resolve_step_rejects_douyin_user_profile_url(test_env) -> None:
    config, database, job_service, runner = test_env
    job = job_service.create(
        "https://www.douyin.com/user/MS4wLjABAAAAOnRpvxiasUeDLCX4WG94yZ3LA6ogPP7MJ6rNzi7bFy8m6QrRR9orTshL80q-1cUc"
    )
    with pytest.raises(AppError) as exc:
        pipeline.resolve_step(job.id, config, database, runner)
    assert exc.value.info.code == "DOUYIN_USER_URL_NOT_SUPPORTED"


@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
def test_resolve_step_bilibili_single_video(mock_run, mock_resolve, test_env):
    config, database, job_service, runner = test_env
    mock_resolve.return_value = Path("dummy-yt-dlp")
    mock_run.return_value = MagicMock(
        stdout=json.dumps({
            "_type": "video",
            "id": "BV1MEJw6qE8b",
            "title": "Bilibili sample",
            "webpage_url": "https://www.bilibili.com/video/BV1MEJw6qE8b/",
            "duration": 120.0,
            "thumbnail": "https://i0.hdslb.com/bfs/archive/sample.jpg",
        }),
        stderr="",
        returncode=0,
    )

    job = job_service.create("https://www.bilibili.com/video/BV1MEJw6qE8b/")
    res = pipeline.resolve_step(job.id, config, database, runner)

    assert res["is_playlist"] is False
    assert len(res["videos"]) == 1
    assert res["selected_video"]["id"] == "BV1MEJw6qE8b"
    assert res["selected_video"]["url"] == "https://www.bilibili.com/video/BV1MEJw6qE8b/"


@patch("dv_backend.pipeline.transcribe_audio")
def test_asr_uses_qwen3_gpu_transcription(mock_transcribe, test_env):
    config, database, job_service, runner = test_env
    job = job_service.create("https://www.douyin.com/video/douyin123")
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


@patch("dv_backend.pipeline.transcribe_audio")
def test_asr_rejects_empty_transcription(mock_transcribe, test_env):
    config, database, job_service, runner = test_env
    job = job_service.create("https://www.douyin.com/video/douyin123")
    artifacts_dir = config.data_dir / "jobs" / job.id / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    audio_path = artifacts_dir / "audio_16k.wav"
    audio_path.write_bytes(b"wav")

    mock_transcribe.return_value = []

    with pytest.raises(AppError) as error:
        pipeline.asr_step(job.id, config, database, runner)

    assert error.value.info.code == "EMPTY_ASR_TRANSCRIPTION"


@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
def test_resolve_step_uses_selected_browser_cookies(mock_run, mock_resolve, test_env):
    config, database, job_service, runner = test_env
    mock_resolve.return_value = Path("dummy-yt-dlp")
    mock_run.return_value = MagicMock(
        stdout=json.dumps({
            "_type": "video",
            "id": "douyin123",
            "title": "Cookie video",
            "webpage_url": "https://www.douyin.com/video/douyin123",
            "duration": 15.5,
        }),
        stderr="",
        returncode=0,
    )
    with database.connection:
        database.connection.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("cookies_browser", json.dumps("edge"), "now"),
        )

    job = job_service.create("https://www.douyin.com/video/douyin123")
    pipeline.resolve_step(job.id, config, database, runner)

    command = mock_run.call_args.args[0]
    assert command[:3] == ["dummy-yt-dlp", "--cookies-from-browser", "edge"]


@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
def test_resolve_step_playlist(mock_run, mock_resolve, test_env):
    config, database, job_service, runner = test_env
    mock_resolve.return_value = Path("dummy-yt-dlp")
    
    # Mock playlist dump from yt-dlp
    mock_run.return_value = MagicMock(
        stdout=json.dumps({
            "_type": "playlist",
            "entries": [
                {"id": "v1", "title": "First", "url": "url1", "duration": 10},
                {"id": "v2", "title": "Second", "url": "url2", "duration": 20}
            ]
        }),
        stderr="",
        returncode=0
    )
    
    job = job_service.create("https://www.douyin.com/video/playlist")
    res = pipeline.resolve_step(job.id, config, database, runner)
    
    assert res["is_playlist"] is True
    assert len(res["videos"]) == 2
    assert res["selected_video"] is None
    
    # Running download step should fail because no video selected
    with pytest.raises(AppError) as ex:
        pipeline.download_step(job.id, config, database, runner)
    assert ex.value.info.code == "NO_VIDEO_SELECTED"


@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
def test_vad_step_parses_silence(mock_run, mock_resolve, test_env):
    config, database, job_service, runner = test_env
    mock_resolve.return_value = Path("dummy-ffmpeg")
    
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
    assert segments[1]["start"] == 2.5
    assert segments[1]["end"] == 5.0
    assert segments[1]["original_duration"] == 2.5
    assert segments[1]["duration_budget"] == 7.5


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


@patch("dv_backend.pipeline.call_openai_chat")
@patch("dv_backend.pipeline.GoogleFreeTranslator")
def test_translate_step(translator_type, mock_chat, test_env):
    config, database, job_service, runner = test_env

    with database.connection:
        database.connection.execute(
            "INSERT INTO jobs (id, source_url, title, status, created_at, updated_at) "
            "VALUES (?, ?, ?, 'running', ?, ?)",
            ("job123", "https://www.douyin.com/video/1", "中文标题", "now", "now"),
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
    translator_type.return_value.translate.side_effect = [
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
    save_checkpoint(config.data_dir, "job123", "normalize_segments", {
        "segments": [
            {"index": 0, "start": 0.0, "end": 1.0, "text": "hello", "duration_budget": 1.0}
        ]
    })
    with database.connection:
        database.connection.execute(
            "INSERT INTO jobs (id, source_url, title, status, created_at, updated_at) "
            "VALUES (?, ?, ?, 'running', ?, ?)",
            ("job123", "https://www.douyin.com/video/1", "Chinese title", "now", "now"),
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

    tts.synthesize.assert_called_once()
    tts_factory.assert_called_once()


@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
@patch("dv_backend.pipeline.get_wav_duration")
@patch("dv_backend.pipeline.call_openai_chat")
def test_duration_repair_time_stretches_fallback(mock_chat, mock_dur, mock_run, mock_resolve, test_env):
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
    # 2) duration after stretch
    # 3) duration after exact trim/pad
    # 4) final repaired duration
    mock_dur.side_effect = [4.0, 2.86, 2.0, 2.0]
    
    # Mock LLM shortening to fail or return still too long
    mock_chat.return_value = {
        "choices": [{
            "message": {
                "content": "Still too long..."
            }
        }]
    }
    def ffmpeg_side_effect(cmd, *_args, **_kwargs):
        out_path = Path(cmd[-1])
        write_dummy_wav(out_path, duration=2.0)
        return MagicMock(stdout="", stderr="", returncode=0)
    mock_run.side_effect = ffmpeg_side_effect
    
    res = pipeline.duration_repair_step("job123", config, database, runner)
    
    assert "time_stretch_" in res["segments"][0]["repaired_method"]
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

    assert "time_stretch_" in result["segments"][0]["repaired_method"]
    assert result["segments"][0]["repaired_duration"] == 2.0


def write_dummy_wav(path: Path, duration: float = 5.0, sample_rate: int = 48000, channels: int = 2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00" * int(duration * sample_rate * channels * 2))


@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
def test_mix_ducks_original_and_includes_vietnamese_narration(
    mock_run, mock_resolve, test_env
):
    config, database, _job_service, runner = test_env
    mock_resolve.return_value = Path("ffmpeg")
    job_id = "job123"
    artifacts_dir = config.data_dir / "jobs" / job_id / "artifacts"
    original = artifacts_dir / "original_48k.wav"
    narration_segment = artifacts_dir / "tts" / "tts_repaired_0.wav"
    write_dummy_wav(original)
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

    pipeline.mix_step(job_id, config, database, runner)

    filter_graph = mock_run.call_args.args[0][
        mock_run.call_args.args[0].index("-filter_complex") + 1
    ]
    assert "sidechaincompress" in filter_graph
    assert "[ducked][fg2]amix=inputs=2" in filter_graph


@patch("dv_backend.pipeline.separate_vocals")
@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
def test_mix_separate_mode_uses_bgm_without_ducking(
    mock_run,
    mock_resolve,
    mock_separate,
    test_env,
):
    config, database, _job_service, runner = test_env
    mock_resolve.return_value = Path("ffmpeg")
    job_id = "job123"
    artifacts_dir = config.data_dir / "jobs" / job_id / "artifacts"
    original = artifacts_dir / "original_48k.wav"
    bgm = artifacts_dir / "bgm.wav"
    vocals = artifacts_dir / "vocals.wav"
    narration_segment = artifacts_dir / "tts" / "tts_repaired_0.wav"
    write_dummy_wav(original)
    write_dummy_wav(bgm)
    write_dummy_wav(vocals)
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

    pipeline.mix_step(job_id, config, database, runner)

    mock_separate.assert_not_called()
    filter_graph = mock_run.call_args.args[0][
        mock_run.call_args.args[0].index("-filter_complex") + 1
    ]
    assert "sidechaincompress" not in filter_graph
    assert str(bgm) in mock_run.call_args.args[0]


@patch("dv_backend.pipeline.separate_vocals")
@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
def test_mix_separate_mode_runs_demucs_when_bgm_missing(
    mock_run,
    mock_resolve,
    mock_separate,
    test_env,
):
    config, database, _job_service, runner = test_env
    mock_resolve.return_value = Path("ffmpeg")
    job_id = "job456"
    artifacts_dir = config.data_dir / "jobs" / job_id / "artifacts"
    original = artifacts_dir / "original_48k.wav"
    narration_segment = artifacts_dir / "tts" / "tts_repaired_0.wav"
    write_dummy_wav(original)
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
        {"original_48k_path": str(original)},
    )
    save_checkpoint(
        config.data_dir,
        job_id,
        "duration_repair",
        {"segments": [{"index": 0, "start": 0.0}]},
    )

    def fake_separate(input_wav, **kwargs):
        write_dummy_wav(kwargs["bgm_out"])
        write_dummy_wav(kwargs["vocals_out"])

    mock_separate.side_effect = fake_separate

    def write_mix(cmd, *_args, **_kwargs):
        write_dummy_wav(Path(cmd[-1]))
        return MagicMock(stdout="", stderr="", returncode=0)

    mock_run.side_effect = write_mix

    pipeline.mix_step(job_id, config, database, runner)

    mock_separate.assert_called_once()


@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
def test_mix_saves_vietnamese_narration_as_debug_output(
    mock_run, mock_resolve, test_env
):
    config, database, _job_service, runner = test_env
    mock_resolve.return_value = Path("ffmpeg")
    job_id = "job123"
    artifacts_dir = config.data_dir / "jobs" / job_id / "artifacts"
    original = artifacts_dir / "original_48k.wav"
    narration_segment = artifacts_dir / "tts" / "tts_repaired_0.wav"
    write_dummy_wav(original)
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
@patch("dv_backend.pipeline.GoogleFreeTranslator")
@patch("dv_backend.pipeline.create_tts_adapter")
def test_full_runner_execution_and_resume(tts_factory, translator_type, mock_transcribe, mock_run, mock_resolve, test_env):
    config, database, job_service, runner = test_env
    mock_resolve.return_value = Path("dummy-tool")
    mock_transcribe.return_value = [
        {"start": 0.0, "end": 1.0, "text": "Hello"},
        {"start": 2.0, "end": 3.0, "text": "World"},
    ]
    
    def side_effect(cmd, job_id, runner_instance, timeout=None):
        cmd_str = " ".join(cmd)
        
        if "--dump-single-json" in cmd_str:
            return MagicMock(
                stdout=json.dumps({
                    "_type": "video",
                    "id": "v",
                    "title": "t",
                    "webpage_url": "url",
                    "duration": 5.0,
                    "thumbnail": "thumb"
                }),
                stderr="",
                returncode=0
            )
            
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
            
        # General WAV file writer for any audio tool/FFmpeg commands
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
    translator_type.return_value.translate.return_value = ["tr0", "tr1"]

    def tts_synthesize_side_effect(text, output_path, **kwargs):
        write_dummy_wav(output_path, duration=1.0, sample_rate=24000, channels=1)

    tts_factory.return_value.synthesize.side_effect = tts_synthesize_side_effect
    
    with patch("dv_backend.pipeline.call_openai_chat") as mock_chat, \
         patch("dv_backend.pipeline.call_openai_tts") as mock_tts:
         
        mock_chat.return_value = {"choices": [{"message": {"content": json.dumps({"translations": [{"index": 0, "translation": "tr0"}, {"index": 1, "translation": "tr1"}]})}}]}
        
        # Mock call_openai_tts to write a dummy WAV file
        def tts_side_effect(api_base, api_key, model, voice, text, output_path):
            write_dummy_wav(output_path, duration=1.0, sample_rate=48000, channels=2)
            
        mock_tts.side_effect = tts_side_effect
        
        job = job_service.create("https://www.douyin.com/video/123")
        runner.start_job(job.id)
        
        # Wait for thread to finish (up to 1.5 seconds)
        for _ in range(30):
            time.sleep(0.05)
            job_db = job_service.get(job.id)
            if job_db.status in {"completed", "failed"}:
                break
                
        # Verify job is completed
        job_db = job_service.get(job.id)
        if job_db.status != "completed":
            events = database.connection.execute("SELECT * FROM events").fetchall()
            print("EVENTS:", [dict(e) for e in events])
            print("JOB ERROR:", job_db.last_error_code, job_db.last_error_message)
            for step in job_db.steps:
                print("STEP:", step.name, step.status)
        assert job_db.status == "completed"
        assert all(step.status == "completed" for step in job_db.steps)
