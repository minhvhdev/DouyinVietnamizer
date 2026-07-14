#!/usr/bin/env python3
from dv_backend.checkpoints import load_checkpoint
from dv_backend.config import AppConfig


def main() -> None:
    job = "f7620299-9f3c-42bd-a1e6-6f3ec0b542a2"
    cfg = AppConfig.from_env()
    tr = load_checkpoint(cfg.data_dir, job, "translate") or {}
    segs = tr.get("segments") or []
    print(f"segments={len(segs)}")
    for s in segs:
        start = float(s.get("start", 0))
        if 50 <= start <= 80 or 90 <= start <= 110 or 150 <= start <= 165 or 225 <= start <= 245:
            idx = int(s.get("index", 0)) + 1
            od = float(s.get("end", 0)) - start
            print(f"#{idx} {start:.2f}-{s.get('end')} ({od:.2f}s)")
            print(f"  CN: {s.get('text','')[:60]}")
            print(f"  VI: {s.get('translation','')[:90]}")


if __name__ == "__main__":
    main()
