#!/usr/bin/env python3
"""Run TTS -> render smoke test with VoxCPM2 on a synthetic local job."""

from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path


def write_silent_wav(path: Path, *, duration: float = 2.0, sample_rate: int = 48000, channels: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(duration * sample_rate)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count * channels)


def make_test_video(video_path: Path, ffmpeg: Path) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ffmpeg),
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=4:size=640x360:rate=24",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=4",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        str(video_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def main() -> int:
    backend_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(backend_dir))

    from dv_backend.config import AppConfig
    from dv_backend.database import Database
    from dv_backend.jobs import JobService
    from dv_backend.voxcpm_env import is_voxcpm_available
    from dv_backend.pipeline import duration_repair_step, mix_step, render_step, tts_step
    from dv_backend.runner import JobRunner
    from dv_backend.settings import SettingsService
    from dv_backend.vendor import VendorManifest, VendorResolver
    from dv_backend.checkpoints import load_checkpoint, save_checkpoint

    if not is_voxcpm_available():
        print("VoxCPM2 env missing. Run: python scripts/setup_voxcpm.py", file=sys.stderr)
        return 1

    data_dir = backend_dir / ".data-voxcpm-smoke"
    if data_dir.exists():
        import shutil

        shutil.rmtree(data_dir)

    config = AppConfig(data_dir)
    config.ensure_directories()

    project_root = backend_dir.parent
    vendor_dir = project_root / "vendor"
    manifest = VendorManifest.load(vendor_dir / "manifest.json")
    resolver = VendorResolver(vendor_dir, allow_path_tools=True)
    ffmpeg = resolver.resolve(next(tool for tool in manifest.tools if tool.id == "ffmpeg")).path
    if ffmpeg is None:
        print("ffmpeg not found in vendor manifest", file=sys.stderr)
        return 1

    database = Database(config.database_path)
    database.migrate()
    settings = SettingsService(database)
    settings.update(
        {
            "tts_backend": "voxcpm",
            "voxcpm_auto_voice": True,
            "speaker_diarization": False,
            "subtitles_enabled": False,
            "exact_timing_enabled": False,
        }
    )

    jobs = JobService(database, config.data_dir)
    job = jobs.create("https://www.douyin.com/video/0000000000000000000")
    job_id = job.id

    job_dir = config.data_dir / "jobs" / job_id
    artifacts = job_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    source_video = artifacts / "source.mp4"
    make_test_video(source_video, ffmpeg)
    write_silent_wav(artifacts / "original_48k.wav", duration=4.0)
    original_48k = artifacts / "original_48k.wav"

    save_checkpoint(
        config.data_dir,
        job_id,
        "extract_audio",
        {"original_48k_path": str(original_48k)},
    )
    save_checkpoint(
        config.data_dir,
        job_id,
        "download",
        {
            "output_path": str(source_video),
            "selected_video": {
                "path": str(source_video),
                "title": "VoxCPM2 Smoke",
            },
        },
    )
    save_checkpoint(
        config.data_dir,
        job_id,
        "translate",
        {
            "title_vi": "Thử VoxCPM2",
            "segments": [
                {
                    "index": 0,
                    "start": 0.0,
                    "end": 3.5,
                    "text": "你好",
                    "translation": "Xin chào, đây là thử nghiệm lồng tiếng bằng VoxCPM2.",
                    "duration_budget": 3.5,
                }
            ],
        },
    )

    runner = JobRunner(config, database)
    print("Running TTS with VoxCPM2...")
    tts_step(job_id, config, database, runner)
    print("Running duration repair...")
    duration_repair_step(job_id, config, database, runner)
    print("Running mix...")
    mix_step(job_id, config, database, runner)
    print("Running render...")
    render_step(job_id, config, database, runner)

    render_cp = load_checkpoint(config.data_dir, job_id, "render")
    output_path = Path(render_cp["output_path"]) if render_cp else job_dir / "output" / "dubbed.mp4"
    if not output_path.is_file():
        print(f"No output video produced at {output_path}.", file=sys.stderr)
        return 1

    print(f"SUCCESS: {output_path} ({output_path.stat().st_size} bytes)")
    print(json.dumps({"job_id": job_id, "output": str(output_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
