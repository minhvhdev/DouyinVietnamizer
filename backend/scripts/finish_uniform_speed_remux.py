#!/usr/bin/env python3
"""Finish a stuck duration_repair by applying uniform max speed once, then remux."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def main() -> int:
    job_id = (sys.argv[1] if len(sys.argv) > 1 else "f7620299-9f3c-42bd-a1e6-6f3ec0b542a2").strip()
    backend_dir = Path(__file__).resolve().parents[1]
    repo_root = backend_dir.parent
    sys.path.insert(0, str(backend_dir))
    os.environ.setdefault("DV_VENDOR_DIR", str(repo_root / "vendor"))

    from dv_backend.adapters.omnivoice_client import release_all_clients
    from dv_backend.checkpoints import load_checkpoint, save_checkpoint
    from dv_backend.config import AppConfig
    from dv_backend.database import Database
    from dv_backend.jobs import JobService
    from dv_backend.pipeline import (
        _propose_then_apply_uniform_speed,
        get_wav_duration,
        resolve_tool_path,
    )
    from dv_backend.runner import JobRunner
    from dv_backend.settings import SettingsService
    from dv_backend.timing_placement import compute_placement_starts, schedule_soft_placements

    config = AppConfig.from_env()
    database = Database(config.database_path)
    database.migrate()
    jobs = JobService(database, config.data_dir)
    jobs.reconcile_interrupted()
    settings = SettingsService(database).get_all()
    absolute_max_rate = float(settings.get("edge_tts_overflow_speed_hard_max", 1.25) or 1.25)
    absolute_max_rate = max(1.0, min(1.25, absolute_max_rate))

    tts_cp = load_checkpoint(config.data_dir, job_id, "tts")
    if not tts_cp:
        print("Missing TTS checkpoint", file=sys.stderr)
        return 1
    segments = list(tts_cp.get("segments") or [])
    tts_dir = config.data_dir / "jobs" / job_id / "artifacts" / "tts"

    for s in segments:
        idx = int(s.get("index", 0) or 0)
        repaired = tts_dir / f"tts_repaired_{idx}.wav"
        compact = tts_dir / f"tts_compact_{idx}.wav"
        raw = Path(str(s.get("tts_raw_path") or (tts_dir / f"tts_raw_{idx}.wav")))
        if compact.is_file() and compact.stat().st_mtime >= (
            repaired.stat().st_mtime if repaired.is_file() else 0
        ):
            source = compact
            s["tts_raw_path"] = str(compact)
        elif repaired.is_file():
            base = tts_dir / f"tts_speed_base_{idx}.wav"
            source = base if base.is_file() else repaired
        elif raw.is_file():
            source = raw
        else:
            continue
        s.pop("tts_speed_base_path", None)
        s["tts_path"] = str(source)
        s["repaired_duration"] = round(get_wav_duration(source), 2)
        s["proposed_speed_factor"] = 1.0
        s["soft_speed_factor"] = 1.0

    compute_placement_starts(segments)
    schedule_soft_placements(segments)

    runner = JobRunner(config, database)
    ffmpeg_path = resolve_tool_path(config, "ffmpeg")
    target = _propose_then_apply_uniform_speed(
        segments=segments,
        absolute_max_rate=absolute_max_rate,
        ffmpeg_path=ffmpeg_path,
        tts_dir=tts_dir,
        job_id=job_id,
        runner=runner,
    )
    print(f"Applied uniform speed target={target}", flush=True)

    factors = sorted(
        {
            round(float(s.get("soft_speed_factor") or 1.0), 3)
            for s in segments
            if str(s.get("tts_spoken_text") or s.get("translation") or "").strip()
            and not bool(s.get("no_speech"))
        }
    )
    print(f"soft_speed_factor values: {factors}", flush=True)

    checkpoint_data = {
        "schema_version": 1,
        "job_id": job_id,
        "step_name": "duration_repair",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "uniform_speed_target": target,
        "segments": segments,
    }
    save_checkpoint(config.data_dir, job_id, "duration_repair", checkpoint_data)

    now = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    with database.connection:
        database.connection.execute(
            """
            UPDATE job_steps
            SET status = 'completed', completed_at = ?, error_code = NULL, error_message = NULL
            WHERE job_id = ? AND name = 'duration_repair'
            """,
            (now, job_id),
        )
        database.connection.execute(
            """
            UPDATE jobs
            SET status = 'interrupted', current_step = 'duration_repair',
                last_error_code = NULL, last_error_message = NULL, updated_at = ?
            WHERE id = ?
            """,
            (now, job_id),
        )

    keep = [
        "resolve",
        "download",
        "extract_audio",
        "vad",
        "asr",
        "normalize_segments",
        "translate",
        "tts",
        "duration_repair",
    ]
    jobs.rerun(job_id, keep)
    print("Rerun from align_final_dub (kept duration_repair)", flush=True)
    runner.start_job(job_id)

    deadline = time.time() + 2 * 3600
    while time.time() < deadline:
        hydrated = jobs.get(job_id)
        if hydrated.status in {"completed", "failed", "cancelled"}:
            print(f"Final status: {hydrated.status}", flush=True)
            if hydrated.last_error_code:
                print(
                    f"Error: {hydrated.last_error_code} — {hydrated.last_error_message}",
                    flush=True,
                )
            release_all_clients()
            return 0 if hydrated.status == "completed" else 1
        time.sleep(3)

    print("Timed out", file=sys.stderr)
    release_all_clients()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
