#!/usr/bin/env python3
"""Rerun a job from duration_repair with updated global TTS speed."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    parser.add_argument("--speed", type=float, default=2.5)
    parser.add_argument(
        "--from-step",
        default="duration_repair",
        choices=["duration_repair", "mix", "render"],
        help="First pipeline step to rerun.",
    )
    args = parser.parse_args()

    backend_dir = Path(__file__).resolve().parents[1]
    repo_root = backend_dir.parent
    sys.path.insert(0, str(backend_dir))
    os.environ.setdefault("DV_VENDOR_DIR", str(repo_root / "vendor"))

    from dv_backend.adapters.omnivoice_client import release_all_clients
    from dv_backend.checkpoints import PIPELINE_STEPS
    from dv_backend.config import AppConfig
    from dv_backend.database import Database
    from dv_backend.jobs import JobService
    from dv_backend.runner import JobRunner
    from dv_backend.settings import SettingsService

    config = AppConfig.from_env()
    database = Database(config.database_path)
    database.migrate()
    settings = SettingsService(database)
    settings.update({"tts_global_speed": args.speed})
    print(f"Set tts_global_speed={args.speed}")

    jobs = JobService(database, config.data_dir)
    job_id = args.job_id.strip()
    keep = list(PIPELINE_STEPS[: PIPELINE_STEPS.index(args.from_step)])
    jobs.rerun(job_id, keep)
    print(f"Rerun job {job_id} from {args.from_step} (kept: {keep[-1]})")

    runner = JobRunner(config, database)
    runner.start_job(job_id)

    deadline = time.time() + 6 * 3600
    while time.time() < deadline:
        hydrated = jobs.get(job_id)
        if hydrated.status in {"completed", "failed", "cancelled"}:
            print(f"Final status: {hydrated.status}")
            if hydrated.last_error_code:
                print(f"Error: {hydrated.last_error_code} — {hydrated.last_error_message}")
            release_all_clients()
            return 0 if hydrated.status == "completed" else 1
        time.sleep(5)

    print("Timed out waiting for job completion.", file=sys.stderr)
    release_all_clients()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
