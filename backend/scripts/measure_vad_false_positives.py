"""Measure ASR segments likely rejected as VAD false positives.

Reads completed job checkpoints (vad + asr) and reports the fraction of ASR
segments that `is_likely_vad_false_positive` would drop. Use this as a quick
proxy when comparing `vad_engine` settings before/after a change.

Examples:
    python -m scripts.measure_vad_false_positives --data-dir data --job-id abc123
    python -m scripts.measure_vad_false_positives --data-dir data --all-jobs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from dv_backend.adapters.vad_feedback import filter_asr_false_positives  # noqa: E402
from dv_backend.checkpoints import load_checkpoint  # noqa: E402


def _measure_job(data_dir: Path, job_id: str) -> dict | None:
    asr_cp = load_checkpoint(data_dir, job_id, "asr")
    vad_cp = load_checkpoint(data_dir, job_id, "vad")
    if not asr_cp or not vad_cp:
        return None

    segments = list(asr_cp.get("segments") or [])
    kept, rejected = filter_asr_false_positives(segments, enabled=True)
    total = len(segments)
    return {
        "job_id": job_id,
        "vad_engine": vad_cp.get("vad_engine", "unknown"),
        "speech_region_count": len(vad_cp.get("speech_regions") or []),
        "asr_segment_count": total,
        "false_positive_rejected_count": len(rejected),
        "false_positive_ratio": round(len(rejected) / total, 4) if total else 0.0,
        "kept_segment_count": len(kept),
        "rejected_reasons": {
            reason: sum(1 for item in rejected if item.get("vad_false_positive_reason") == reason)
            for reason in ("empty_asr", "duplicate_asr")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--job-id", action="append", default=[])
    parser.add_argument("--all-jobs", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    jobs_root = args.data_dir / "jobs"
    if not jobs_root.is_dir():
        print(f"No jobs directory at {jobs_root}", file=sys.stderr)
        return 1

    job_ids = list(args.job_id)
    if args.all_jobs:
        job_ids = sorted(path.name for path in jobs_root.iterdir() if path.is_dir())
    if not job_ids:
        print("Provide --job-id or --all-jobs", file=sys.stderr)
        return 1

    reports = [item for job_id in job_ids if (item := _measure_job(args.data_dir, job_id))]
    if not reports:
        print("No jobs with both vad and asr checkpoints found.", file=sys.stderr)
        return 1

    summary = {
        "jobs_measured": len(reports),
        "total_asr_segments": sum(item["asr_segment_count"] for item in reports),
        "total_false_positive_rejected": sum(item["false_positive_rejected_count"] for item in reports),
        "reports": reports,
    }
    if summary["total_asr_segments"]:
        summary["aggregate_false_positive_ratio"] = round(
            summary["total_false_positive_rejected"] / summary["total_asr_segments"],
            4,
        )
    else:
        summary["aggregate_false_positive_ratio"] = 0.0

    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.out is not None:
        args.out.write_text(payload, encoding="utf-8")
        print(f"Wrote report to {args.out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
