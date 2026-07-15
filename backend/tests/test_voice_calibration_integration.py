"""Integration tests for voice calibration runner with mocked TTS."""

from __future__ import annotations

import array
import threading
import time
import wave
from pathlib import Path
from unittest.mock import patch

import pytest

from dv_backend.database import Database
from dv_backend.settings import SettingsService
from dv_backend.voice_calibration_dataset import CalibrationSample, select_calibration_samples, load_calibration_dataset
from dv_backend.duration_predictor import count_vietnamese_syllables
from dv_backend.voice_calibration_runner import VoiceCalibrationRunner
from dv_backend.voice_duration_profile import load_profiles
from dv_backend.voice_identity import settings_for_cloned_voice
from dv_backend.voice_profile_policy import effective_voice_profile


def _write_tone(path: Path, *, duration_sec: float = 1.0, amplitude: float = 0.25) -> None:
    rate = 16000
    frames = int(rate * duration_sec)
    samples = array.array("h", [int(amplitude * 32767) for index in range(frames)])
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())


@pytest.fixture()
def calibration_env(tmp_path: Path):
    db_path = tmp_path / "app.db"
    database = Database(db_path)
    database.migrate()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cloned_dir = data_dir / "cloned_voices_omnivoice"
    cloned_dir.mkdir(parents=True)
    voice_id = "voice-test-1"
    wav_path = cloned_dir / f"{voice_id}.wav"
    _write_tone(wav_path, duration_sec=2.0)
    transcript = "Xin chào đây là mẫu giọng tham chiếu."
    wav_path.with_suffix(".txt").write_text(transcript, encoding="utf-8")
    now = "2026-07-12T00:00:00Z"
    with database.connection:
        database.connection.execute(
            """
            INSERT INTO cloned_voices (
                id, backend, name, wav_filename, transcript, created_at,
                voice_status, duration_profile_status
            ) VALUES (?, 'omnivoice', 'Test Voice', ?, ?, ?, 'ready', 'not_started')
            """,
            (voice_id, f"{voice_id}.wav", transcript, now),
        )
    settings = SettingsService(database)
    settings.update({"voice_duration_profile_enabled": True, "tts_backend": "omnivoice", "omnivoice_model": "test-model"})
    runner = VoiceCalibrationRunner(data_dir=data_dir, database=database, settings_getter=settings.get_raw_all)
    return {
        "database": database,
        "data_dir": data_dir,
        "runner": runner,
        "voice_id": voice_id,
        "wav_path": wav_path,
        "transcript": transcript,
        "settings": settings,
    }


def test_calibration_full_job_completes_with_mock_tts(calibration_env) -> None:
    env = calibration_env
    runner: VoiceCalibrationRunner = env["runner"]
    dataset = load_calibration_dataset()
    full_samples = select_calibration_samples(dataset, "full")

    with patch("dv_backend.voice_calibration_runner.TtsSession") as mock_tts_session:
        session = mock_tts_session.return_value
        session.__enter__.return_value = session
        session.__exit__.return_value = None

        def synthesize(text: str, output_path: Path, segment=None) -> None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if "003" in path.stem:
                with wave.open(str(path), "w") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(16000)
                    handle.writeframes(b"\x00\x00" * 8000)
            else:
                syllables = max(2, count_vietnamese_syllables(text))
                duration = max(0.8, syllables / 4.2)
                _write_tone(path, duration_sec=duration)

        session.synthesize.side_effect = synthesize

        result = runner.start_calibration(env["voice_id"], "full")
        job_id = result["job_id"]
        thread = runner.threads.get(job_id)
        if thread:
            thread.join(timeout=20)

    status = runner.get_status(env["voice_id"]) or {}
    assert status["status"] in {"ready", "partial"}
    assert status["completed"] == len(full_samples)

    row = env["database"].connection.execute(
        "SELECT duration_profile_status, duration_profile_key, voice_status FROM cloned_voices WHERE id = ?",
        (env["voice_id"],),
    ).fetchone()
    assert row["voice_status"] == "ready"
    assert row["duration_profile_status"] in {"ready", "partial"}
    assert row["duration_profile_key"]

    settings = settings_for_cloned_voice(
        env["settings"].get_raw_all(),
        voice_id=env["voice_id"],
        wav_path=env["wav_path"],
        transcript=env["transcript"],
    )
    profile = effective_voice_profile(settings, data_dir=env["data_dir"])
    assert profile.get("prediction_method") != "default_insufficient_samples"
    store = load_profiles(env["data_dir"])
    assert store["profiles"]


def test_reset_profile_keeps_voice_ready(calibration_env) -> None:
    env = calibration_env
    runner: VoiceCalibrationRunner = env["runner"]
    with env["database"].connection:
        env["database"].connection.execute(
            """
            UPDATE cloned_voices
            SET duration_profile_status = 'ready', duration_profile_key = 'id:test', duration_profile_sample_count = 20
            WHERE id = ?
            """,
            (env["voice_id"],),
        )
    runner.delete_profile(env["voice_id"])
    row = env["database"].connection.execute(
        "SELECT voice_status, duration_profile_status FROM cloned_voices WHERE id = ?",
        (env["voice_id"],),
    ).fetchone()
    assert row["voice_status"] == "ready"
    assert row["duration_profile_status"] == "not_started"
