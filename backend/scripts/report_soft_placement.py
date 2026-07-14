#!/usr/bin/env python3
from dv_backend.checkpoints import load_checkpoint
from dv_backend.config import AppConfig
from dv_backend.timing_placement import segments_with_voiced_overlap

job = "f7620299-9f3c-42bd-a1e6-6f3ec0b542a2"
cfg = AppConfig.from_env()
cp = load_checkpoint(cfg.data_dir, job, "duration_repair") or {}
segs = cp.get("segments") or []
shifted = sum(1 for s in segs if abs(float(s.get("placement_drift_sec") or 0)) > 0.02)
overflow = sum(1 for s in segs if float(s.get("timing_overflow_sec") or 0) > 0.15)
statuses: dict[str, int] = {}
for s in segs:
    st = str(s.get("timing_status") or "missing")
    statuses[st] = statuses.get(st, 0) + 1
print("segments", len(segs))
print("shifted", shifted)
print("overflow", overflow)
print("overlaps", segments_with_voiced_overlap(segs))
print("statuses", statuses)
for s in segs:
    if 50 <= float(s.get("start") or 0) <= 75:
        print(
            f"#{int(s['index'])+1} start={s.get('start')} place={s.get('placement_start')} "
            f"rd={s.get('repaired_duration')} drift={s.get('placement_drift_sec')} "
            f"ov={s.get('timing_overflow_sec')} st={s.get('timing_status')}"
        )
ends = [
    float(s.get("placement_start") or 0) + float(s.get("repaired_duration") or 0)
    for s in segs
]
print("last_audible_end", round(max(ends) if ends else 0, 2))
print("last_cn_end", max(float(s.get("end") or 0) for s in segs) if segs else 0)
