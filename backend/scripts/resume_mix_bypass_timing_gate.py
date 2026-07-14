#!/usr/bin/env python3
"""Resume mix/render for listening while temporarily bypassing placement gate."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def main() -> int:
    job_id = (sys.argv[1] if len(sys.argv) > 1 else "").strip()
    if not job_id:
        print("job_id required", file=sys.stderr)
        return 1

    backend_dir = Path(__file__).resolve().parents[1]
    repo_root = backend_dir.parent
    sys.path.insert(0, str(backend_dir))
    os.environ.setdefault("DV_VENDOR_DIR", str(repo_root / "vendor"))

    from dv_backend.adapters.omnivoice_client import release_all_clients
    from dv_backend.config import AppConfig
    from dv_backend.database import Database
    from dv_backend.jobs import JobService, utc_now
    from dv_backend.runner import JobRunner
    from dv_backend.settings import SettingsService

    config = AppConfig.from_env()
    database = Database(config.database_path)
    database.migrate()
    settings = SettingsService(database)
    jobs = JobService(database, config.data_dir)
    jobs.reconcile_interrupted()

    settings.update(
        {
            "timing_placement_gate_enabled": False,
            "release_gate_blocking_enabled": False,
        }
    )
    now = utc_now()
    with database.connection:
        database.connection.execute(
            """
            UPDATE jobs
            SET status = 'queued', current_step = 'mix',
                last_error_code = NULL, last_error_message = NULL, updated_at = ?
            WHERE id = ?
            """,
            (now, job_id),
        )
        database.connection.execute(
            """
            UPDATE job_steps
            SET status = 'completed',
                completed_at = COALESCE(completed_at, ?)
            WHERE job_id = ? AND name IN ('align_final_dub', 'duration_repair')
            """,
            (now, job_id),
        )
        database.connection.execute(
            """
            UPDATE job_steps
            SET status = 'pending', started_at = NULL, completed_at = NULL,
                error_code = NULL, error_message = NULL
            WHERE job_id = ? AND name IN ('mix', 'render', 'qc')
            """,
            (job_id,),
        )

    runner = JobRunner(config, database)
    runner.start_job(job_id)

    deadline = time.time() + 1800
    status = "timeout"
    while time.time() < deadline:
        hydrated = jobs.get(job_id)
        if hydrated.status in {"completed", "failed", "cancelled", "interrupted"}:
            print(
                f"FINAL {hydrated.status} {hydrated.last_error_code} {hydrated.last_error_message}",
                flush=True,
            )
            status = hydrated.status
            break
        time.sleep(4)

    settings.update({"timing_placement_gate_enabled": True})
    release_all_clients()
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
