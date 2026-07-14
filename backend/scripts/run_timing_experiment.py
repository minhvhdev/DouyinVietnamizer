#!/usr/bin/env python3
"""Run timing A/B experiment from a source job."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("DV_VENDOR_DIR", str(ROOT.parent / "vendor"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run timing A/B experiment")
    parser.add_argument("source_job_id")
    parser.add_argument("--name", default="phase2-ab")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--experiment", action="store_true")
    parser.add_argument("--export-html", action="store_true")
    parser.add_argument("--resume", default=None, help="Experiment ID to resume")
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()

    from dv_backend.config import AppConfig
    from dv_backend.database import Database
    from dv_backend.jobs import JobService
    from dv_backend.runner import JobRunner
    from dv_backend.timing_eval_dashboard import build_dashboard_payload, export_dashboard_html
    from dv_backend.timing_experiment import (
        apply_settings,
        build_manifest,
        capture_settings,
        clone_job_prefix,
        default_phase2_config,
        experiment_dir,
        load_experiment_config,
        load_manifest,
        run_experiment_arm,
        save_manifest,
        validate_fixed_settings_match,
    )

    config = AppConfig.from_env() if args.data_dir is None else AppConfig(args.data_dir)
    database = Database(config.database_path)
    database.migrate()
    jobs = JobService(database, config.data_dir)
    runner = JobRunner(config, database)

    if args.config:
        exp_config = load_experiment_config(args.config)
    else:
        exp_config = default_phase2_config()

    run_baseline = args.baseline or (not args.experiment)
    run_experiment_arm_flag = args.experiment or (not args.baseline)

    experiment_id = args.resume or f"{args.name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    manifest = load_manifest(config.data_dir, experiment_id) if args.resume else None

    original_settings = capture_settings(database)
    fixed = dict(exp_config.get("fixed_settings") or {})
    baseline_settings = {**fixed, **(exp_config.get("baseline_settings") or {})}
    experiment_settings = {**fixed, **(exp_config.get("experiment_settings") or {})}

    mismatches = validate_fixed_settings_match(
        {**original_settings, **baseline_settings},
        {**original_settings, **experiment_settings},
        fixed,
    )
    if mismatches:
        print(f"Warning: fixed settings mismatch keys: {mismatches}", file=sys.stderr)

    baseline_job_id = manifest.get("baseline_job_id") if manifest else None
    experiment_job_id = manifest.get("experiment_job_id") if manifest else None

    try:
        if run_baseline and not baseline_job_id:
            baseline_job_id = clone_job_prefix(jobs, args.source_job_id, label="baseline")
            print(f"Cloned baseline job: {baseline_job_id}")
            merged = {**original_settings, **baseline_settings}
            run_experiment_arm(job_service=jobs, runner=runner, job_id=baseline_job_id, settings_snapshot=merged, database=database)
            print(f"Baseline completed: {baseline_job_id}")

        if run_experiment_arm_flag and not experiment_job_id:
            experiment_job_id = clone_job_prefix(jobs, args.source_job_id, label="experiment")
            print(f"Cloned experiment job: {experiment_job_id}")
            merged = {**original_settings, **experiment_settings}
            run_experiment_arm(job_service=jobs, runner=runner, job_id=experiment_job_id, settings_snapshot=merged, database=database)
            print(f"Experiment completed: {experiment_job_id}")

        manifest = build_manifest(
            experiment_id=experiment_id,
            source_job_id=args.source_job_id,
            baseline_job_id=baseline_job_id or "",
            experiment_job_id=experiment_job_id or "",
            data_dir=config.data_dir,
            config=exp_config,
            fixed_settings=fixed,
            baseline_settings=baseline_settings,
            experiment_settings=experiment_settings,
            status="completed",
        )
        save_manifest(config.data_dir, experiment_id, manifest)
        print(f"Manifest: {experiment_dir(config.data_dir, experiment_id) / 'manifest.json'}")

        if args.export_html and baseline_job_id and experiment_job_id:
            payload = build_dashboard_payload(
                config.data_dir,
                experiment_job_id,
                baseline_job_id=baseline_job_id,
                baseline_settings={**original_settings, **baseline_settings},
                experiment_settings={**original_settings, **experiment_settings},
                experiment_id=experiment_id,
                include_audio=True,
            )
            out = experiment_dir(config.data_dir, experiment_id) / "dashboard.html"
            export_dashboard_html(out, payload)
            print(f"Dashboard: {out}")
    finally:
        apply_settings(database, original_settings)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
