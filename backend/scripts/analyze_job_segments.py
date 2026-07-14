#!/usr/bin/env python3
"""Analyze ASR segmentation and subtitle cues for a job."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from dv_backend.checkpoints import load_checkpoint
from dv_backend.config import AppConfig
from dv_backend.subtitle_timing import build_segment_subtitle_cues, subtitle_layout_from_settings
from dv_backend.database import Database


def parse_ass(path: Path) -> list[dict]:
    dialogues = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        if len(parts) < 10:
            continue

        def parse_t(t: str) -> float:
            h, m, s = t.split(":")
            sec, cs = s.split(".")
            return int(h) * 3600 + int(m) * 60 + int(sec) + int(cs) / 100

        start = parse_t(parts[1].strip())
        end = parse_t(parts[2].strip())
        text = re.sub(r"\{[^}]*\}", "", parts[9].strip())
        dialogues.append({"start": start, "end": end, "dur": end - start, "text": text})
    return dialogues


def word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", text.strip()) if w])


def main(job_id: str) -> None:
    cfg = AppConfig.from_env()
    job_dir = Path(cfg.data_dir) / "jobs" / job_id

    db = Database(cfg.database_path)
    rows = db.connection.execute("SELECT key, value FROM settings").fetchall()
    settings = {r["key"]: json.loads(r["value"]) for r in rows}
    layout = subtitle_layout_from_settings(settings)

    print(f"job={job_id}")
    print(f"subtitle layout: {layout}")

    for step in ("asr", "normalize_segments"):
        cp = load_checkpoint(cfg.data_dir, job_id, step)
        segs = cp.get("segments") or []
        print(f"\n=== {step}: {len(segs)} segments ===")
        for i, s in enumerate(segs):
            od = float(s.get("end", 0)) - float(s.get("start", 0))
            txt = str(s.get("text") or "")
            units = len(s.get("aligned_units") or [])
            dbudget = s.get("duration_budget")
            print(
                f"  {i:02d} od={od:5.2f}s db={dbudget} chars={len(txt):3d} units={units:3d} | {txt[:70]}"
            )

    align = load_checkpoint(cfg.data_dir, job_id, "align_final_dub")
    segs = (align or {}).get("segments") or []
    dub_ok = sum(
        1
        for s in segs
        if (s.get("dub_words") or [])
        and s.get("dub_alignment_status") not in ("failed", "skipped")
    )
    print(f"\n=== align_final_dub: dub_words usable {dub_ok}/{len(segs)} ===")

    ass_path = job_dir / "output" / "subtitles.ass"
    if ass_path.is_file():
        dialogues = parse_ass(ass_path)
        print(f"\n=== ASS: {len(dialogues)} cues ===")
        single_word = [d for d in dialogues if word_count(d["text"]) <= 1]
        rapid = [d for d in dialogues if d["dur"] <= 0.5]
        print(f"  single-word cues: {len(single_word)}")
        print(f"  rapid cues (<=0.5s): {len(rapid)}")
        print("  sample single-word:")
        for d in single_word[:15]:
            print(f"    {d['start']:.2f}-{d['end']:.2f} ({d['dur']:.2f}s) | {d['text']}")
        print("  sample rapid:")
        for d in rapid[:10]:
            print(f"    {d['start']:.2f}-{d['end']:.2f} ({d['dur']:.2f}s) | {d['text'][:60]}")

    repair = load_checkpoint(cfg.data_dir, job_id, "duration_repair")
    repair_segs = (repair or {}).get("segments") or []
    print(f"\n=== subtitle cue rebuild (dub_words path) ===")
    total_cues = 0
    single_word_cues = 0
    for seg in repair_segs:
        cues = build_segment_subtitle_cues(
            seg,
            job_dir=job_dir,
            settings=settings,
            vendor_dir=None,
            ffmpeg_path=None,
            transcribe_fn=None,
            tts_asr_align=False,
        )
        total_cues += len(cues)
        for c in cues:
            if word_count(str(c.get("text") or "")) <= 1:
                single_word_cues += 1
                idx = seg.get("index")
                print(
                    f"  seg{idx} {c['start']:.2f}-{c['end']:.2f} "
                    f"({c['end']-c['start']:.2f}s) | {c.get('text')}"
                )
    print(f"  total cues={total_cues} single-word={single_word_cues}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "f7620299-9f3c-42bd-a1e6-6f3ec0b542a2")
