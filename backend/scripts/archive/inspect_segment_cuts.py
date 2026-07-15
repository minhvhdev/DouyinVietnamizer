#!/usr/bin/env python3
"""Inspect mid-sentence cuts and pause candidates between aligned units."""
from __future__ import annotations

import sys

from dv_backend.checkpoints import load_checkpoint
from dv_backend.config import AppConfig


def main(job_id: str) -> None:
    cfg = AppConfig.from_env()
    asr = load_checkpoint(cfg.data_dir, job_id, "asr") or {}
    norm = load_checkpoint(cfg.data_dir, job_id, "normalize_segments") or {}
    tr = load_checkpoint(cfg.data_dir, job_id, "translate") or {}
    units = asr.get("aligned_units") or []
    segs = tr.get("segments") or norm.get("segments") or []

    targets = {7, 8, 9, 10, 13, 14, 23, 24, 35, 36}  # 0-based from UI 1-based
    print(f"aligned_units={len(units)} segments={len(segs)}")
    for s in segs:
        idx = int(s.get("index", 0) or 0)
        if idx not in targets:
            continue
        start = float(s.get("start", 0))
        end = float(s.get("end", 0))
        span = [u for u in units if start - 0.05 <= float(u.get("start", 0)) and float(u.get("end", 0)) <= end + 0.05]
        gaps = []
        for a, b in zip(span, span[1:]):
            gap = float(b.get("start", 0)) - float(a.get("end", 0))
            if gap >= 0.12:
                gaps.append((gap, float(a.get("end", 0)), "".join(str(x.get("text") or "") for x in span[: span.index(b)])[-8:]))
        gaps.sort(reverse=True)
        print(f"\n#{idx+1} {start:.2f}-{end:.2f}s CN={s.get('text','')[:50]}")
        print(f"  VI={s.get('translation','')[:80]}")
        print(f"  units={len(span)} top_gaps={[(round(g,3), round(t,2), txt) for g,t,txt in gaps[:5]]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "f7620299-9f3c-42bd-a1e6-6f3ec0b542a2")
