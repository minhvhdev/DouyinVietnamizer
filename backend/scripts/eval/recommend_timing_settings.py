#!/usr/bin/env python3
"""Recommend timing settings from experiment metrics (does not auto-apply)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dv_backend.production_thresholds import DEFAULT_THRESHOLDS  # noqa: E402
from dv_backend.eval.timing_experiment import experiment_dir, load_manifest  # noqa: E402
from dv_backend.timing_qc_metrics import compute_timing_qc_metrics  # noqa: E402
from dv_backend.checkpoints import load_checkpoint  # noqa: E402


def _load_segments(data_dir: Path, job_id: str) -> list[dict]:
    for step in ("duration_repair", "tts", "translate"):
        cp = load_checkpoint(data_dir, job_id, step)
        if cp and cp.get("segments"):
            return list(cp["segments"])
    return []


def recommend(metrics: dict, *, sample_count: int, video_count: int = 1) -> dict:
    recommendations: list[dict] = []
    min_segments = DEFAULT_THRESHOLDS["min_reviewed_segments_for_recommendation"]
    min_videos = DEFAULT_THRESHOLDS["min_videos_for_recommendation"]
    if sample_count < min_segments or video_count < min_videos:
        return {
            "recommendations": [],
            "sample_count": sample_count,
            "video_count": video_count,
            "confidence": "insufficient_data",
            "evidence": f"Need >={min_segments} reviewed segments and >={min_videos} videos",
        }

    p90_tempo = metrics.get("p90_effective_tempo")
    if p90_tempo is not None and p90_tempo > 1.08:
        recommendations.append(
            {
                "setting": "timing_preferred_tempo_max",
                "current": 1.08,
                "recommended": 1.06,
                "confidence": "medium",
                "reason": "P90 effective tempo above preferred range",
                "sample_count": sample_count,
                "evidence": {"p90_effective_tempo": p90_tempo},
            }
        )
    if metrics.get("rewrite_rate") is not None and metrics["rewrite_rate"] > 0.15:
        recommendations.append(
            {
                "setting": "timing_max_llm_rewrite_attempts",
                "current": 1,
                "recommended": 0,
                "confidence": "low",
                "reason": "Rewrite rate above 15%",
                "sample_count": sample_count,
                "evidence": {"rewrite_rate": metrics["rewrite_rate"]},
            }
        )
    if metrics.get("p90_prediction_error_ms") is not None and metrics["p90_prediction_error_ms"] > 700:
        recommendations.append(
            {
                "setting": "voice_duration_profile_enabled",
                "current": True,
                "recommended": True,
                "confidence": "medium",
                "reason": "High prediction P90; ensure profile convergence with clean samples",
                "sample_count": sample_count,
                "evidence": {"p90_prediction_error_ms": metrics["p90_prediction_error_ms"]},
            }
        )
    return {
        "recommendations": recommendations,
        "sample_count": sample_count,
        "video_count": video_count,
        "confidence": "medium" if recommendations else "none",
        "evidence": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Recommend timing settings from experiment")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path.home() / ".douyin-vietnamizer")
    parser.add_argument("--reviewed-segments", type=int, default=0)
    args = parser.parse_args()

    manifest = load_manifest(args.data_dir, args.experiment)
    if not manifest:
        print(f"Experiment manifest not found: {args.experiment}", file=sys.stderr)
        return 1
    job_id = manifest.get("experiment_job_id")
    segments = _load_segments(args.data_dir, job_id)
    metrics = compute_timing_qc_metrics(segments)
    sample_count = args.reviewed_segments or metrics.get("tts_segment_count") or 0
    result = recommend(metrics, sample_count=sample_count)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    out = experiment_dir(args.data_dir, args.experiment) / "recommendations.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
