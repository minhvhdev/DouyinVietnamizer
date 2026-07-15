"""Debug subtitle timing for a job."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from dv_backend.subtitle_timing import (
    build_subtitle_cues,
    segment_subtitle_end,
    segment_subtitle_start,
    split_for_subtitle_display,
)


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


def load_segments(job: Path) -> list[dict]:
    repair = json.loads((job / "checkpoints/duration_repair.json").read_text(encoding="utf-8"))
    for key in ("segments",):
        if repair.get(key):
            return repair[key]
    if repair.get("data", {}).get("segments"):
        return repair["data"]["segments"]
    if repair.get("payload", {}).get("segments"):
        return repair["payload"]["segments"]
    return []


def main(job_id: str) -> None:
    job = Path.home() / "AppData/Local/DouyinVietnamizer/jobs" / job_id
    segments = load_segments(job)
    dialogues = parse_ass(job / "output/subtitles.ass")

    print(f"job={job_id}")
    print(f"segments={len(segments)} dialogues={len(dialogues)}")

    rapid = [d for d in dialogues if d["dur"] <= 0.26]
    print(f"rapid cues (<=0.26s): {len(rapid)}")
    if rapid:
        print("  first rapid cluster:")
        for d in rapid[:5]:
            print(f"    {d['start']:.2f}-{d['end']:.2f} ({d['dur']:.2f}s) {d['text'][:70]}")

    print("\n--- segments with many chunks or tight windows ---")
    for seg in segments:
        tr = str(seg.get("translation") or "").strip()
        if not tr:
            continue
        chunks = split_for_subtitle_display(tr)
        if len(chunks) < 2:
            continue
        ws = segment_subtitle_start(seg)
        we = segment_subtitle_end(seg)
        wd = we - ws
        if wd < 0.5:
            continue
        per_chunk = wd / len(chunks)
        if len(chunks) >= 3 or per_chunk < 0.8:
            idx = seg.get("index")
            print(
                f"\nseg {idx}: window {ws:.2f}-{we:.2f} ({wd:.2f}s) "
                f"chunks={len(chunks)} avg={per_chunk:.2f}s "
                f"placement={seg.get('placement_start')} repaired={seg.get('repaired_duration')}"
            )
            for i, c in enumerate(chunks[:8]):
                print(f"  chunk {i}: {c[:80]}")
            if len(chunks) > 8:
                print(f"  ... +{len(chunks)-8} more")

    # Rebuild cues without overlap resolver to compare
    rebuilt = build_subtitle_cues(segments, tts_asr_align=False)
    rebuilt_rapid = [c for c in rebuilt if c["end"] - c["start"] <= 0.26]
    print(f"\nrebuilt proportional-only rapid: {len(rebuilt_rapid)}/{len(rebuilt)}")

    # Find overlap clusters in rebuilt before resolve - we need raw per-segment
    from dv_backend.subtitle_timing import build_segment_subtitle_cues, resolve_overlapping_cues

    raw: list[dict] = []
    for seg in segments:
        raw.extend(
            build_segment_subtitle_cues(
                seg,
                job_dir=None,
                settings=None,
                vendor_dir=None,
                ffmpeg_path=None,
                transcribe_fn=None,
                tts_asr_align=False,
            )
        )
    overlaps = 0
    ordered = sorted(raw, key=lambda x: (x["start"], x["end"]))
    for prev, cur in zip(ordered, ordered[1:], strict=False):
        if cur["start"] < prev["end"] - 0.02:
            overlaps += 1
    print(f"raw cue overlaps before resolve: {overlaps}/{max(0,len(ordered)-1)}")

    resolved = resolve_overlapping_cues(raw)
    resolved_rapid = [c for c in resolved if c["end"] - c["start"] <= 0.26]
    print(f"after resolve rapid: {len(resolved_rapid)}/{len(resolved)}")

    # Window vs mix cap analysis
    from dv_backend.segment_mix import annotate_segment_mix_caps, effective_clip_duration

    entries = []
    for seg in sorted(segments, key=lambda s: int(s.get("index", 0) or 0)):
        ps = float(seg.get("placement_start") or seg.get("start") or 0.0)
        rd = float(seg.get("repaired_duration") or 0.0)
        entries.append(
            {
                "index": seg.get("index"),
                "placement_start": ps,
                "clip_duration": rd,
                "subtitle_end": segment_subtitle_end(seg),
            }
        )
    annotate_segment_mix_caps(entries)
    print("\n--- subtitle window vs audio mix cap ---")
    for i, entry in enumerate(entries):
        next_ps = entries[i + 1]["placement_start"] if i + 1 < len(entries) else None
        effective = effective_clip_duration(entry["clip_duration"], entry.get("max_duration"))
        sub_overflow = entry["subtitle_end"] - (entry["placement_start"] + effective)
        overlap = bool(next_ps and entry["subtitle_end"] > next_ps + 0.02)
        if overlap or sub_overflow > 0.5:
            print(
                f"seg {entry['index']}: sub_end={entry['subtitle_end']:.2f} "
                f"effective_audio={entry['placement_start'] + effective:.2f} "
                f"max_dur={entry.get('max_duration')} overflow={sub_overflow:.2f}s overlap_next={overlap}"
            )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ebef2dac-3e2e-4dc2-9f8b-575d7e50342d")
