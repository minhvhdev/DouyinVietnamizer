#!/usr/bin/env python3
"""CLI entrypoint for voice duration calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from dv_backend.config import AppConfig
from dv_backend.database import Database
from dv_backend.settings import SettingsService
from dv_backend.voice_calibration_runner import VoiceCalibrationRunner
from dv_backend.voice_calibration_store import job_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate voice duration profile for a cloned voice.")
    parser.add_argument("voice_id", help="Cloned voice UUID")
    parser.add_argument("--mode", choices=["quick", "standard", "full"], default="standard")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dataset-version", default=None)
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--export-html", action="store_true")
    args = parser.parse_args()

    config = AppConfig.from_env()
    config.ensure_directories()
    database = Database(config.database_path)
    database.migrate()
    settings = SettingsService(database)
    runner = VoiceCalibrationRunner(data_dir=config.data_dir, database=database, settings_getter=settings.get_raw_all)

    if args.resume:
        status = runner.get_status(args.voice_id) or {}
        job_id = status.get("job_id")
        if not job_id:
            print("No calibration job to resume.", file=sys.stderr)
            return 1
        result = runner.resume_calibration(args.voice_id, job_id)
    else:
        if not args.force:
            runner.preflight(args.voice_id, args.mode)
        result = runner.start_calibration(args.voice_id, args.mode)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    job_id = result.get("job_id")
    if job_id:
        report_path = job_dir(config.data_dir, job_id) / "report.json"
        if report_path.is_file():
            print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
