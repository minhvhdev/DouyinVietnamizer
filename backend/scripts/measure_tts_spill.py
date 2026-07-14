#!/usr/bin/env python3
from dv_backend.checkpoints import load_checkpoint
from dv_backend.config import AppConfig

job = "f7620299-9f3c-42bd-a1e6-6f3ec0b542a2"
cfg = AppConfig.from_env()
segs = (load_checkpoint(cfg.data_dir, job, "duration_repair") or {}).get("segments") or []
over = spill = 0
print("idx | od | rd | avail | overflow | method | spoken")
for i, s in enumerate(segs):
    od = float(s.get("original_duration") or 0)
    rd = float(s.get("repaired_duration") or 0)
    db = float(s.get("duration_budget") or 0)
    ps = float(s.get("placement_start") or s.get("start") or 0)
    nxt = segs[i + 1] if i + 1 < len(segs) else None
    next_ps = float(nxt.get("placement_start") or nxt.get("start") or 0) if nxt else None
    avail = (next_ps - ps) if next_ps is not None else db
    overflow = rd - avail
    if rd > od + 0.3:
        over += 1
    if overflow > 0.15:
        spill += 1
        if spill <= 15:
            spoken = (s.get("tts_spoken_text") or s.get("translation") or "")[:55]
            print(
                f"{s.get('index'):3} | {od:4.2f} | {rd:4.2f} | {avail:4.2f} | "
                f"+{overflow:4.2f} | {s.get('repaired_method')} | {spoken}"
            )
print(f"tts_longer_than_orig {over}/{len(segs)}")
print(f"spill_into_next_window {spill}/{len(segs)}")
