#!/usr/bin/env python3
"""Import human timing review JSON exported from dashboard HTML."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def validate_review(payload: dict, *, expected_segments: set[int] | None = None) -> list[str]:
    errors: list[str] = []
    if not payload.get("experiment_id"):
        errors.append("missing experiment_id")
    segments = payload.get("segments")
    if not isinstance(segments, dict):
        errors.append("segments must be an object keyed by index")
        return errors
    for key, row in segments.items():
        try:
            index = int(key)
        except ValueError:
            errors.append(f"invalid segment key: {key}")
            continue
        if expected_segments is not None and index not in expected_segments:
            errors.append(f"unknown segment index: {index}")
        for field in ("naturalness", "timing", "subtitle_sync"):
            if field in row and row[field] is not None:
                val = float(row[field])
                if val < 1 or val > 5:
                    errors.append(f"segment {index} {field} out of range")
    return errors


def aggregate_human_metrics(reviews: list[dict]) -> dict:
    naturalness: list[float] = []
    timing: list[float] = []
    subtitle: list[float] = []
    meaning_fail = 0
    pref = {"baseline": 0, "experiment": 0, "tie": 0}
    for review in reviews:
        segments = review.get("segments") or {}
        if isinstance(segments, dict):
            items = segments.values()
        else:
            items = segments
        for row in items:
            if row.get("naturalness") is not None:
                naturalness.append(float(row["naturalness"]))
            if row.get("timing") is not None:
                timing.append(float(row["timing"]))
            if row.get("subtitle_sync") is not None:
                subtitle.append(float(row["subtitle_sync"]))
            if row.get("meaning_preserved") is False:
                meaning_fail += 1
            p = str(row.get("preferred") or "").lower()
            if p in pref:
                pref[p] += 1
    total_pref = sum(pref.values()) or 1
    return {
        "mean_naturalness": round(sum(naturalness) / len(naturalness), 3) if naturalness else None,
        "mean_timing_score": round(sum(timing) / len(timing), 3) if timing else None,
        "mean_subtitle_sync": round(sum(subtitle) / len(subtitle), 3) if subtitle else None,
        "meaning_preservation_failure_count": meaning_fail,
        "baseline_preference_rate": round(pref["baseline"] / total_pref, 4),
        "experiment_preference_rate": round(pref["experiment"] / total_pref, 4),
        "tie_rate": round(pref["tie"] / total_pref, 4),
        "reviewed_segment_count": len(naturalness),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Import human timing review JSON")
    parser.add_argument("experiment_id")
    parser.add_argument("review_path", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path.home() / ".douyin-vietnamizer")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.review_path.is_file():
        print(f"Review file not found: {args.review_path}", file=sys.stderr)
        return 1

    payload = json.loads(args.review_path.read_text(encoding="utf-8"))
    errors = validate_review(payload)
    if errors:
        print("Validation errors:", "; ".join(errors), file=sys.stderr)
        return 1

    out_dir = args.data_dir / "experiments" / args.experiment_id / "reviews"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"review_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    if list(out_dir.glob("review_*.json")) and not args.force:
        print("Existing review found. Use --force to add another version.", file=sys.stderr)
        return 1

    payload["imported_at"] = datetime.now(timezone.utc).isoformat()
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    reviews = [payload]
    for path in out_dir.glob("review_*.json"):
        if path == target:
            continue
        try:
            reviews.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    aggregate = aggregate_human_metrics(reviews)
    agg_path = out_dir / "aggregate.json"
    agg_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"Saved {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
