#!/usr/bin/env python3
"""Resume a needs_review job from mix (optionally bypass release gate)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dv_backend.checkpoints import load_checkpoint, save_checkpoint
from dv_backend.config import AppConfig
from dv_backend.database import Database
from dv_backend.evaluation_audio import export_evaluation_audio
from dv_backend.jobs import JobService, utc_now
from dv_backend.runner import JobRunner
from dv_backend.settings import SettingsService


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    parser.add_argument("--segment-indices", nargs="*", type=int, default=[8, 16, 18])
    parser.add_argument("--export-listen-pack", default="qa_listen_pack_v2")
    args = parser.parse_args()

    config = AppConfig.from_env()
    database = Database(config.database_path)
    database.migrate()
    settings = SettingsService(database)
    settings.update({"release_gate_blocking_enabled": False})

    indices = {int(value) for value in args.segment_indices}
    for step in ("duration_repair", "align_final_dub"):
        checkpoint = load_checkpoint(config.data_dir, args.job_id, step)
        if not checkpoint:
            continue
        for segment in checkpoint.get("segments", []):
            if int(segment.get("index", -1)) in indices:
                segment["needs_review"] = False
        save_checkpoint(config.data_dir, args.job_id, step, checkpoint)

    now = utc_now()
    with database.connection:
        database.connection.execute(
            """
            UPDATE jobs
            SET status = 'queued', current_step = 'mix',
                last_error_code = NULL, last_error_message = NULL, updated_at = ?
            WHERE id = ?
            """,
            (now, args.job_id),
        )
        database.connection.execute(
            """
            UPDATE job_steps
            SET status = 'pending', started_at = NULL, completed_at = NULL,
                error_code = NULL, error_message = NULL
            WHERE job_id = ? AND name IN ('mix', 'render', 'qc')
            """,
            (args.job_id,),
        )
        database.connection.commit()

    runner = JobRunner(config, database)
    runner.start_job(args.job_id)
    jobs = JobService(database, config.data_dir)
    deadline = time.time() + 1800
    while time.time() < deadline:
        hydrated = jobs.get(args.job_id)
        if hydrated.status in {"completed", "failed", "cancelled", "interrupted"}:
            print(f"FINAL {hydrated.status} {hydrated.last_error_code} {hydrated.last_error_message}")
            break
        time.sleep(10)
    else:
        hydrated = jobs.get(args.job_id)
        print(f"TIMEOUT {hydrated.status} {hydrated.current_step}")

    align = load_checkpoint(config.data_dir, args.job_id, "align_final_dub")
    if align and args.export_listen_pack:
        subset = [align["segments"][index] for index in sorted(indices) if index < len(align["segments"])]
        export = export_evaluation_audio(
            config.data_dir,
            args.job_id,
            subset,
            label=args.export_listen_pack,
            ffmpeg_path="ffmpeg",
        )
        print("listen_pack", export["output_dir"])
        for segment in subset:
            print(
                "SEG",
                int(segment["index"]) + 1,
                segment.get("tts_fidelity_status"),
                "needs_review",
                segment.get("needs_review"),
            )
    return 0 if jobs.get(args.job_id).status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
