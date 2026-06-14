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
            ("whisper_model_path", json.dumps(str(tmp_path / "ggml-tiny.bin")), "now")
        )
        database.connection.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("piper_model_path", json.dumps(str(tmp_path / "voice.onnx")), "now")
        )
        
    # Create files to satisfy path checks
    (tmp_path / "ggml-tiny.bin").write_text("dummy model")
    (tmp_path / "voice.onnx").write_text("dummy voice")
    
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


@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
def test_asr_parses_current_whisper_json_and_uses_chinese(
    mock_run, mock_resolve, test_env
):
    config, database, job_service, runner = test_env
    mock_resolve.return_value = Path("whisper-cli")
    job = job_service.create("https://www.douyin.com/video/douyin123")
    job_dir = config.data_dir / "jobs" / job.id
    artifacts_dir = job_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    audio_path = artifacts_dir / "audio_16k.wav"
    audio_path.write_bytes(b"wav")

    def write_whisper_json(*_args, **_kwargs):
        audio_path.with_suffix(".wav.json").write_text(
            json.dumps(
                {
                    "transcription": [
                        {
                            "offsets": {"from": 250, "to": 1480},
                            "text": " 你好 ",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return MagicMock(returncode=0, stdout="", stderr="")

    mock_run.side_effect = write_whisper_json

    result = pipeline.asr_step(job.id, config, database, runner)

    assert result["segments"] == [{"start": 0.25, "end": 1.48, "text": "你好"}]
    command = mock_run.call_args.args[0]
    assert command[-3:] == ["-l", "zh", "-oj"]


@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
def test_asr_rejects_empty_transcription(mock_run, mock_resolve, test_env):
    config, database, job_service, runner = test_env
    mock_resolve.return_value = Path("whisper-cli")
    job = job_service.create("https://www.douyin.com/video/douyin123")
    artifacts_dir = config.data_dir / "jobs" / job.id / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    audio_path = artifacts_dir / "audio_16k.wav"
    audio_path.write_bytes(b"wav")

    def write_empty_whisper_json(*_args, **_kwargs):
        audio_path.with_suffix(".wav.json").write_text(
            json.dumps({"transcription": []}),
            encoding="utf-8",
        )
        return MagicMock(returncode=0, stdout="", stderr="")

    mock_run.side_effect = write_empty_whisper_json

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


@patch("dv_backend.pipeline.call_openai_chat")
@patch("dv_backend.pipeline.GoogleFreeTranslator")
def test_translate_step(translator_type, mock_chat, test_env):
    config, database, job_service, runner = test_env
    
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
    translator_type.return_value.translate.return_value = [
        json.loads(content)["translations"][0]["translation"]
    ]
    res = pipeline.translate_step("job123", config, database, runner)
    assert res["segments"][0]["translation"] == "Xin chào"


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
    
    # Mock get_wav_duration for the repaired output
    mock_dur.return_value = 2.0
    
    # Mock LLM shortening to fail or return still too long
    mock_chat.return_value = {
        "choices": [{
            "message": {
                "content": "Still too long..."
            }
        }]
    }
    
    res = pipeline.duration_repair_step("job123", config, database, runner)
    
    assert res["segments"][0]["repaired_method"] == "time_stretch_1.4x"
    assert res["segments"][0]["repaired_duration"] == 2.0
    
    # Assert ffmpeg atempo filter was called
    mock_run.assert_called()
    called_cmd = mock_run.call_args[0][0]
    assert "atempo=1.4" in " ".join(called_cmd)


def write_dummy_wav(path: Path, duration: float = 5.0, sample_rate: int = 48000, channels: int = 2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00" * int(duration * sample_rate * channels * 2))


@patch("dv_backend.pipeline.resolve_tool_path")
@patch("dv_backend.pipeline.run_subprocess_with_cancel")
@patch("dv_backend.pipeline.GoogleFreeTranslator")
@patch("dv_backend.pipeline.EdgeTtsAdapter")
def test_full_runner_execution_and_resume(edge_tts_type, translator_type, mock_run, mock_resolve, test_env):
    config, database, job_service, runner = test_env
    mock_resolve.return_value = Path("dummy-tool")
    
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
            
        if "-c:v copy" in cmd_str:
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

    def edge_tts_side_effect(text, output_path, **kwargs):
        output_path.write_bytes(b"fake mp3")

    edge_tts_type.return_value.synthesize.side_effect = edge_tts_side_effect
    
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
