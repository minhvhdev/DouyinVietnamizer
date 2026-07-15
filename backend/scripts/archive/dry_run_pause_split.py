#!/usr/bin/env python3
from dv_backend.checkpoints import load_checkpoint
from dv_backend.config import AppConfig
from dv_backend.pipeline import _split_long_asr_segments_with_vad
from dv_backend.segmentation import (
    consolidate_short_segments,
    merge_incomplete_sentence_segments,
    split_long_segments_with_alignment,
    split_segments_by_alignment_pauses,
)

job = "f7620299-9f3c-42bd-a1e6-6f3ec0b542a2"
cfg = AppConfig.from_env()
asr = load_checkpoint(cfg.data_dir, job, "asr")
vad = load_checkpoint(cfg.data_dir, job, "vad")
raw = split_long_segments_with_alignment(
    asr["segments"], vad["speech_regions"], asr.get("aligned_units") or []
)
raw = _split_long_asr_segments_with_vad(raw, vad["speech_regions"])
raw = merge_incomplete_sentence_segments(raw)
raw = split_long_segments_with_alignment(
    raw, vad["speech_regions"], asr.get("aligned_units") or []
)
raw = _split_long_asr_segments_with_vad(raw, vad["speech_regions"])
raw = split_segments_by_alignment_pauses(raw, asr.get("aligned_units") or [])
raw = consolidate_short_segments(raw)
durs = [s["end"] - s["start"] for s in raw]
print("segments", len(raw), "max", round(max(durs), 2), "avg", round(sum(durs) / len(durs), 2))
print(">4.5s", sum(1 for d in durs if d > 4.5), ">6s", sum(1 for d in durs if d > 6))
for s in raw:
    if 50 <= float(s["start"]) <= 80 or 90 <= float(s["start"]) <= 110 or 150 <= float(s["start"]) <= 165:
        print(
            f"{s['start']:.2f}-{s['end']:.2f} ({s['end']-s['start']:.2f}s) "
            f"{s.get('split_method')} | {s['text'][:42]}"
        )
