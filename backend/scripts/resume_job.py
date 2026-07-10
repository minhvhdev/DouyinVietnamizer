#!/usr/bin/env python3
"""Resume a failed job from the first incomplete pipeline step."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", help="Job UUID to resume")

    args = parser.parse_args()

    backend_dir = Path(__file__).resolve().parents[1]
    repo_root = backend_dir.parent
    sys.path.insert(0, str(backend_dir))

    os.environ.setdefault("DV_VENDOR_DIR", str(repo_root / "vendor"))

    from dv_backend.adapters.omnivoice_client import release_all_clients
    from dv_backend.config import AppConfig
    from dv_backend.database import Database
    from dv_backend.jobs import JobService
    from dv_backend.runner import JobRunner

    config = AppConfig.from_env()
    database = Database(config.database_path)
    database.migrate()
    jobs = JobService(database, config.data_dir)
    job_id = args.job_id.strip()

    job = jobs.get(job_id)
    print(f"Resuming job {job_id} (status={job.status})")
    jobs.prepare_job_for_resume(job_id)
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
        time.sleep(2)

    print("Timed out waiting for job completion.", file=sys.stderr)
    release_all_clients()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
