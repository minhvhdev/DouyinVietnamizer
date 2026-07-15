#!/usr/bin/env python3
"""Batch production evaluator for multiple jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dv_backend.checkpoints import load_checkpoint  # noqa: E402
from dv_backend.eval.dubbing_quality_score import score_segment_quality  # noqa: E402
from dv_backend.release_quality_gate import evaluate_release_gate  # noqa: E402
from dv_backend.timing_qc_metrics import compute_timing_qc_metrics  # noqa: E402


def _load_segments(data_dir: Path, job_id: str) -> list[dict]:
    for step in ("duration_repair", "tts", "translate"):
        cp = load_checkpoint(data_dir, job_id, step)
        if cp and cp.get("segments"):
            return list(cp["segments"])
    return []


def evaluate_job(data_dir: Path, job_id: str, *, category: str | None = None) -> dict:
    segments = _load_segments(data_dir, job_id)
    metrics = compute_timing_qc_metrics(segments)
    gate = evaluate_release_gate(segments, metrics=metrics)
    scored = []
    for segment in segments:
        quality = score_segment_quality(segment)
        scored.append({"index": segment.get("index"), **quality})
    scored.sort(key=lambda row: row.get("quality_score", 1.0))
    return {
        "job_id": job_id,
        "category": category,
        "metrics": metrics,
        "release_gate": gate,
        "worst_segments": scored[:5],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch production evaluation")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--job", action="append", default=[], help="job_id[:category]")
    parser.add_argument("--data-dir", type=Path, default=Path.home() / ".douyin-vietnamizer")
    parser.add_argument("--export-html", action="store_true")
    args = parser.parse_args()

    jobs: list[tuple[str, str | None]] = []
    if args.manifest and args.manifest.is_file():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        for video in manifest.get("videos") or []:
            if video.get("job_id"):
                jobs.append((video["job_id"], video.get("category")))
    for item in args.job:
        if ":" in item:
            job_id, category = item.split(":", 1)
        else:
            job_id, category = item, None
        jobs.append((job_id, category))

    if not jobs:
        print("No jobs specified (--job or manifest with job_id)", file=sys.stderr)
        return 1

    results = [evaluate_job(args.data_dir, job_id, category=category) for job_id, category in jobs]
    by_category: dict[str, list] = {}
    for row in results:
        cat = row.get("category") or "uncategorized"
        by_category.setdefault(cat, []).append(row)

    payload = {"jobs": results, "by_category": by_category}
    out = args.data_dir / "artifacts" / "batch_eval.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.export_html:
        html_path = args.data_dir / "artifacts" / "batch_eval.html"
        html_path.write_text(
            f"<html><body><h1>Batch Eval</h1><pre>{json.dumps(payload, ensure_ascii=False, indent=2)}</pre></body></html>",
            encoding="utf-8",
        )
        print(f"Wrote {html_path}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
