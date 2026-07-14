"""Speed repaired segments to kill residual overlaps, then resume mix."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

JOB = "f7620299-9f3c-42bd-a1e6-6f3ec0b542a2"


def main() -> int:
    backend_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(backend_dir))
    import os

    os.environ.setdefault("DV_VENDOR_DIR", str(backend_dir.parent / "vendor"))

    from dv_backend.checkpoints import load_checkpoint, save_checkpoint
    from dv_backend.config import AppConfig
    from dv_backend.database import Database
    from dv_backend.jobs import JobService
    from dv_backend.pipeline import _build_atempo_chain, _run_ffmpeg_audio_filter, get_wav_duration, resolve_tool_path
    from dv_backend.runner import JobRunner
    from dv_backend.timing_placement import (
        BOUNDARY_MARGIN_SEC,
        compute_placement_starts,
        enforce_zero_overlap_placements,
        schedule_soft_placements,
        segments_with_voiced_overlap,
    )
    from dv_backend.tts_provenance import spoken_text

    config = AppConfig.from_env()
    database = Database(config.database_path)
    repair = load_checkpoint(config.data_dir, JOB, "duration_repair") or {}
    segments = [dict(s) for s in repair.get("segments") or []]
    tts_dir = config.data_dir / "jobs" / JOB / "artifacts" / "tts"
    ffmpeg_path = resolve_tool_path(config, "ffmpeg")
    runner = JobRunner(config, database)

    # Drop silent units that still point at old audio.
    for seg in segments:
        if not spoken_text(seg):
            seg["tts_path"] = None
            seg["tts_raw_path"] = None
            seg["repaired_duration"] = 0.0
            seg["no_speech"] = True

    ordered = sorted([s for s in segments if spoken_text(s) and s.get("tts_path")], key=lambda s: int(s["index"]))
    for i, seg in enumerate(ordered):
        path = Path(str(seg["tts_path"]))
        if not path.is_file():
            continue
        duration = get_wav_duration(path)
        next_start = None
        if i + 1 < len(ordered):
            next_start = float(ordered[i + 1].get("placement_start") or ordered[i + 1].get("start") or 0.0)
        start = float(seg.get("placement_start") or seg.get("start") or 0.0)
        if next_start is None:
            seg["repaired_duration"] = round(duration, 2)
            continue
        alloc = max(0.4, next_start - start - BOUNDARY_MARGIN_SEC)
        if duration <= alloc + 0.05:
            seg["repaired_duration"] = round(duration, 2)
            continue
        rate = min(1.15, duration / alloc)
        out = tts_dir / f"tts_fit_{seg['index']}.wav"
        out.unlink(missing_ok=True)
        _run_ffmpeg_audio_filter(
            ffmpeg_path,
            path,
            out,
            filter_expr=_build_atempo_chain(rate),
            job_id=JOB,
            runner=runner,
        )
        if out.is_file():
            shutil.copyfile(out, path)
            duration = get_wav_duration(path)
            seg["repaired_duration"] = round(duration, 2)
            seg["soft_speed_factor"] = round(float(seg.get("soft_speed_factor") or 1.0) * rate, 4)
            print(f"fit index={seg['index']} rate={rate:.3f} dur={duration:.2f} alloc={alloc:.2f}")

    compute_placement_starts(segments)
    schedule_soft_placements([s for s in segments if spoken_text(s)])
    enforce_zero_overlap_placements([s for s in segments if spoken_text(s)])
    voiced = [s for s in segments if spoken_text(s) and s.get("tts_path")]
    overlaps = segments_with_voiced_overlap(voiced)
    print("overlaps_after_fit", overlaps)
    if overlaps:
        # Second pass: force place zero-overlap then speed again against forced windows.
        cursor = 0.0
        for seg in sorted(voiced, key=lambda s: float(s.get("placement_start") or 0)):
            pref = float(seg.get("preferred_placement_start") or seg.get("start") or 0)
            start = max(pref, cursor)
            seg["placement_start"] = round(start, 3)
            dur = float(seg.get("repaired_duration") or 0)
            cursor = start + dur + 0.05
        # Recreate order and fit using neighboring placement starts after enforce.
        enforce_zero_overlap_placements(voiced)
        ordered = sorted(voiced, key=lambda s: float(s.get("placement_start") or 0))
        for i, seg in enumerate(ordered):
            if i + 1 >= len(ordered):
                continue
            start = float(seg["placement_start"])
            nxt = float(ordered[i + 1]["placement_start"])
            alloc = max(0.35, nxt - start - 0.05)
            path = Path(str(seg["tts_path"]))
            duration = get_wav_duration(path)
            if duration <= alloc + 0.05:
                continue
            rate = min(1.15, duration / alloc)
            out = tts_dir / f"tts_fit2_{seg['index']}.wav"
            out.unlink(missing_ok=True)
            _run_ffmpeg_audio_filter(
                ffmpeg_path,
                path,
                out,
                filter_expr=_build_atempo_chain(rate),
                job_id=JOB,
                runner=runner,
            )
            if out.is_file():
                shutil.copyfile(out, path)
                seg["repaired_duration"] = round(get_wav_duration(path), 2)
                print(f"fit2 index={seg['index']} rate={rate:.3f}")
        enforce_zero_overlap_placements(voiced)
        overlaps = segments_with_voiced_overlap(voiced)
        print("overlaps_final", overlaps)

    repair["segments"] = segments
    save_checkpoint(config.data_dir, JOB, "duration_repair", repair)

    if overlaps:
        print("still overlapping; abort", file=sys.stderr)
        return 1

    jobs = JobService(database, config.data_dir)
    jobs.prepare_job_for_resume(JOB)
    runner.start_job(JOB)
    deadline = time.time() + 3600
    while time.time() < deadline:
        hydrated = jobs.get(JOB)
        if hydrated.status in {"completed", "failed", "cancelled"}:
            print("Final status:", hydrated.status)
            if hydrated.last_error_code:
                print("Error:", hydrated.last_error_code, hydrated.last_error_message)
            return 0 if hydrated.status == "completed" else 1
        time.sleep(2)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
