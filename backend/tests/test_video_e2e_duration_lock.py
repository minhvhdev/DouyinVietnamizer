"""Video E2E: short source with A/V duration mismatch + mix duration-lock + render."""
from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

from dv_backend.audio_probe import get_video_stream_duration
from dv_backend.checkpoints import save_checkpoint
from dv_backend.config import AppConfig
from dv_backend.database import Database
from dv_backend.pipeline import mix_step
from dv_backend.runner import JobRunner
from dv_backend.segment_mix import build_background_narration_mix_filter


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise AssertionError(f"cmd failed: {' '.join(cmd)}\n{result.stderr}")


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def _rms_tail(path: Path, *, seconds: float = 0.4) -> float:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        assert width == 2
        frames = handle.readframes(handle.getnframes())
    import array

    samples = array.array("h")
    samples.frombytes(frames)
    if channels > 1:
        samples = array.array("h", (samples[i] for i in range(0, len(samples), channels)))
    tail = max(1, int(seconds * rate))
    window = samples[-tail:]
    if not window:
        return 0.0
    mean_sq = sum(int(s) * int(s) for s in window) / float(len(window))
    return mean_sq ** 0.5


def _make_mismatched_mp4(path: Path, *, video_sec: float = 8.0, audio_sec: float = 5.0) -> None:
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=320x240:d={video_sec}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=48000:duration={audio_sec}",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ]
    )


def test_video_e2e_mix_lock_and_render_preserves_duration(tmp_path: Path) -> None:
    """~8s video / ~5s audio → mix locks to video; render with -shortest keeps video length."""
    src = tmp_path / "src_v8_a5.mp4"
    _make_mismatched_mp4(src, video_sec=8.0, audio_sec=5.0)
    video_dur = get_video_stream_duration(src)
    assert video_dur >= 7.8

    # Extract short audio as if from source audio stream, then lock like mix_step.
    short_bg = tmp_path / "bg_short.wav"
    short_fg = tmp_path / "fg_short.wav"
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-vn",
            "-ac",
            "2",
            "-ar",
            "48000",
            str(short_bg),
        ]
    )
    assert _wav_duration(short_bg) <= 5.2
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:sample_rate=48000:duration=1.0",
            "-ac",
            "2",
            str(short_fg),
        ]
    )

    mixed = tmp_path / "mixed.wav"
    graph = build_background_narration_mix_filter(duck=False, target_duration_sec=video_dur)
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(short_bg),
            "-i",
            str(short_fg),
            "-filter_complex",
            graph,
            "-map",
            "[mixed]",
            str(mixed),
        ]
    )
    assert abs(_wav_duration(mixed) - video_dur) <= 0.15
    assert _rms_tail(mixed, seconds=0.5) < 50

    out = tmp_path / "dubbed.mp4"
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-i",
            str(mixed),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(out),
        ]
    )
    out_dur = get_video_stream_duration(out)
    # Output must not be shorter than video beyond tolerance after pad + -shortest.
    assert out_dur + 0.15 >= video_dur
    assert abs(out_dur - video_dur) < 0.35


def test_mix_step_duration_lock_with_fixtures(tmp_path: Path) -> None:
    """mix_step uses video stream duration lock when original audio is shorter."""
    database = Database(tmp_path / "app.db")
    database.migrate()
    config = AppConfig(tmp_path)
    config.ensure_directories()
    with database.connection:
        database.connection.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("release_gate_blocking_enabled", json.dumps(False), "now"),
        )
        database.connection.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("timing_placement_gate_enabled", json.dumps(False), "now"),
        )
        database.connection.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("mix_mode", json.dumps("duck"), "now"),
        )

    job_id = "job-e2e-mix"
    job_dir = config.data_dir / "jobs" / job_id
    artifacts = job_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    src = artifacts / "original.mp4"
    _make_mismatched_mp4(src, video_sec=8.0, audio_sec=5.0)
    video_dur = get_video_stream_duration(src)

    original_48k = artifacts / "original_48k.wav"
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-vn",
            "-ac",
            "2",
            "-ar",
            "48000",
            str(original_48k),
        ]
    )
    # One short narration clip
    tts = artifacts / "tts" / "tts_repaired_0.wav"
    tts.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=660:sample_rate=48000:duration=1.0",
            "-ac",
            "2",
            str(tts),
        ]
    )

    save_checkpoint(
        config.data_dir,
        job_id,
        "extract_audio",
        {
            "original_48k_path": str(original_48k),
            "bgm_path": None,
        },
    )
    save_checkpoint(
        config.data_dir,
        job_id,
        "duration_repair",
        {
            "release_eligible": True,
            "segments": [
                {
                    "index": 0,
                    "start": 0.5,
                    "end": 1.5,
                    "placement_start": 0.5,
                    "translation": "Xin chào",
                    "tts_spoken_text": "Xin chào",
                    "repaired_duration": 1.0,
                    "tts_path": str(tts),
                    "timing_overflow_sec": 0.0,
                }
            ],
        },
    )

    runner = JobRunner(config, database)

    with patch("dv_backend.pipeline.resolve_tool_path", return_value=Path("ffmpeg")), patch(
        "dv_backend.pipeline.run_subprocess_with_cancel"
    ) as mock_run:

        def real_ffmpeg(cmd, *_a, **_k):
            _run([str(c) for c in cmd])
            return MagicMock(stdout="", stderr="", returncode=0)

        mock_run.side_effect = real_ffmpeg
        result = mix_step(job_id, config, database, runner)

    mixed = Path(result["mixed_wav_path"])
    assert mixed.is_file()
    target = float(result["target_video_duration_sec"])
    assert abs(target - video_dur) <= 0.15
    # loudnorm + AAC-extracted bg can land a few frames under exact atrim target
    assert abs(_wav_duration(mixed) - target) <= 0.12
    assert _wav_duration(mixed) + 0.05 >= float(_wav_duration(original_48k))
    assert _rms_tail(mixed, seconds=0.4) < 80
