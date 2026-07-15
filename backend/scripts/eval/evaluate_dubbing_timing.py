#!/usr/bin/env python3
"""Evaluate dubbing timing quality for a completed or partial job."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dv_backend.checkpoints import load_checkpoint  # noqa: E402
from dv_backend.database import Database  # noqa: E402
from dv_backend.config import AppConfig  # noqa: E402
from dv_backend.eval.timing_eval_dashboard import build_dashboard_payload, export_dashboard_html  # noqa: E402
from dv_backend.timing_qc_metrics import compare_timing_metrics, compute_timing_qc_metrics  # noqa: E402


def _load_segments(data_dir: Path, job_id: str) -> list[dict]:
    for step in ("duration_repair", "tts", "translate"):
        cp = load_checkpoint(data_dir, job_id, step)
        if cp and cp.get("segments"):
            return list(cp["segments"])
    return []


def _load_settings(data_dir: Path) -> dict:
    database = Database(AppConfig(data_dir).database_path)
    database.migrate()
    rows = database.connection.execute("SELECT key, value FROM settings").fetchall()
    return {r["key"]: json.loads(r["value"]) for r in rows}


def _build_job_payload(data_dir: Path, job_id: str) -> dict:
    segments = _load_segments(data_dir, job_id)
    if not segments:
        raise ValueError(f"No segments found for job {job_id}")
    return {
        "job_id": job_id,
        "summary": compute_timing_qc_metrics(segments),
        "segments": segments,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate dubbing timing for a job")
    parser.add_argument("job_id")
    parser.add_argument("--compare", dest="compare_job_id", default=None)
    parser.add_argument("--data-dir", type=Path, default=Path.home() / ".douyin-vietnamizer")
    parser.add_argument("--export-html", action="store_true")
    parser.add_argument("--include-audio", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        if args.export_html and args.compare_job_id:
            settings = _load_settings(args.data_dir)
            payload = build_dashboard_payload(
                args.data_dir,
                args.job_id,
                baseline_job_id=args.compare_job_id,
                baseline_settings=settings,
                experiment_settings=settings,
                include_audio=args.include_audio,
            )
            out = args.data_dir / "jobs" / args.job_id / "artifacts" / "timing_eval.html"
            export_dashboard_html(out, payload)
            print(f"Wrote {out}")
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        payload = _build_job_payload(args.data_dir, args.job_id)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    if args.compare_job_id:
        try:
            compare_payload = _build_job_payload(args.data_dir, args.compare_job_id)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 1
        payload["comparison"] = compare_timing_metrics(
            compare_payload["summary"],
            payload["summary"],
        )
        payload["baseline_job_id"] = args.compare_job_id

    if args.export_html:
        out = args.data_dir / "jobs" / args.job_id / "artifacts" / "timing_eval.html"
        payload_dash = build_dashboard_payload(
            args.data_dir,
            args.job_id,
            baseline_job_id=args.compare_job_id,
            experiment_segments=payload.get("segments"),
            experiment_summary=payload.get("summary"),
            include_audio=args.include_audio,
        )
        export_dashboard_html(out, payload_dash)
        print(f"Wrote {out}")
    if args.json or (not args.export_html and not args.compare_job_id):
        slim = {k: v for k, v in payload.items() if k != "segments"} if not args.json else payload
        if "segments" in payload and args.json:
            slim = payload
        print(json.dumps(slim if not args.json and "comparison" in payload else payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
