from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path

import pytest

from dv_backend.audio_probe import get_video_stream_duration
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
        # Use first channel only for silence check.
        samples = array.array("h", (samples[i] for i in range(0, len(samples), channels)))
    tail = max(1, int(seconds * rate))
    window = samples[-tail:]
    if not window:
        return 0.0
    mean_sq = sum(int(s) * int(s) for s in window) / float(len(window))
    return mean_sq ** 0.5


def test_get_video_stream_duration_uses_video_not_audio(tmp_path: Path) -> None:
    mp4_v = tmp_path / "v5_a3.mp4"
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:d=5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=3",
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
            str(mp4_v),
        ]
    )
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,duration:format=duration",
            "-of",
            "json",
            str(mp4_v),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(probe.stdout)
    streams = {s["codec_type"]: s for s in payload["streams"]}
    video_dur = float(streams["video"].get("duration") or payload["format"]["duration"])
    measured = get_video_stream_duration(mp4_v)
    assert abs(measured - video_dur) < 0.05
    assert measured >= 4.8
    # Audio stream is shorter — SoT must still be video.
    audio_dur = float(streams["audio"].get("duration") or 0)
    if audio_dur > 0:
        assert measured > audio_dur + 1.0



def test_mix_filter_pads_short_background_to_video_target(tmp_path: Path) -> None:
    bg = tmp_path / "bg.wav"
    fg = tmp_path / "fg.wav"
    out = tmp_path / "mixed.wav"
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:sample_rate=48000:duration=2.5",
            "-ac",
            "2",
            str(bg),
        ]
    )
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
            str(fg),
        ]
    )
    target = 5.0
    graph = build_background_narration_mix_filter(duck=False, target_duration_sec=target)
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(bg),
            "-i",
            str(fg),
            "-filter_complex",
            graph,
            "-map",
            "[mixed]",
            str(out),
        ]
    )
    duration = _wav_duration(out)
    assert abs(duration - target) <= 0.001
    assert _rms_tail(out, seconds=0.5) < 50  # silence pad at the end


def test_mix_filter_trims_long_narration_to_video_target(tmp_path: Path) -> None:
    bg = tmp_path / "bg.wav"
    fg = tmp_path / "fg.wav"
    out = tmp_path / "mixed.wav"
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:sample_rate=48000:duration=5.0",
            "-ac",
            "2",
            str(bg),
        ]
    )
    # Narration longer than target after delay would previously stretch output.
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:sample_rate=48000:duration=3.0",
            "-af",
            "adelay=4000:all=1",
            "-ac",
            "2",
            str(fg),
        ]
    )
    target = 5.0
    graph = build_background_narration_mix_filter(duck=False, target_duration_sec=target)
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(bg),
            "-i",
            str(fg),
            "-filter_complex",
            graph,
            "-map",
            "[mixed]",
            str(out),
        ]
    )
    duration = _wav_duration(out)
    assert abs(duration - target) <= 0.001


def test_render_with_padded_mix_preserves_video_duration(tmp_path: Path) -> None:
    mp4 = tmp_path / "src.mp4"
    mixed = tmp_path / "mixed.wav"
    out = tmp_path / "dubbed.mp4"
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:d=5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=3",
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
            str(mp4),
        ]
    )
    video_dur = get_video_stream_duration(mp4)
    # Short mix as if old path used extracted audio only
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=330:sample_rate=48000:duration=3",
            "-ac",
            "2",
            str(mixed),
        ]
    )
    # Apply duration lock as mix_step now does before render -shortest
    locked = tmp_path / "locked.wav"
    lock_graph = f"[0:a]apad,atrim=0:{video_dur:.6f},asetpts=PTS-STARTPTS[a]"
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(mixed),
            "-filter_complex",
            lock_graph,
            "-map",
            "[a]",
            str(locked),
        ]
    )
    assert abs(_wav_duration(locked) - video_dur) <= 0.001
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(mp4),
            "-i",
            str(locked),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(out),
        ]
    )
    out_dur = get_video_stream_duration(out)
    assert out_dur + 0.05 >= video_dur
    assert abs(out_dur - video_dur) < 0.15
