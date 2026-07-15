"""Targeted P0 repair: provenance + seam heal + #27 re-TTS, then remux."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

JOB = "f7620299-9f3c-42bd-a1e6-6f3ec0b542a2"


def main() -> int:
    backend_dir = Path(__file__).resolve().parents[2]
    repo_root = backend_dir.parent
    sys.path.insert(0, str(backend_dir))
    import os

    os.environ.setdefault("DV_VENDOR_DIR", str(repo_root / "vendor"))

    from dv_backend.adapters.tts import TtsSession, create_tts_adapter
    from dv_backend.checkpoints import PIPELINE_STEPS, load_checkpoint, save_checkpoint
    from dv_backend.config import AppConfig
    from dv_backend.database import Database
    from dv_backend.jobs import JobService
    from dv_backend.pipeline import get_wav_duration
    from dv_backend.runner import JobRunner
    from dv_backend.tts_provenance import (
        detect_clause_seams,
        sha256_file,
        spoken_text,
        validate_segments_tts_provenance,
    )

    config = AppConfig.from_env()
    database = Database(config.database_path)
    database.migrate()
    rows = database.connection.execute("SELECT key, value FROM settings").fetchall()
    settings = {r["key"]: json.loads(r["value"]) for r in rows}

    job_dir = config.data_dir / "jobs" / JOB
    tts_dir = job_dir / "artifacts" / "tts"
    repair = load_checkpoint(config.data_dir, JOB, "duration_repair")
    if not repair:
        print("missing duration_repair", file=sys.stderr)
        return 1
    segments = [dict(s) for s in repair.get("segments") or []]
    by_index = {int(s["index"]): s for s in segments}

    runner = JobRunner(config, database)

    def resynth(seg: dict, text: str, tag: str) -> None:
        idx = int(seg["index"])
        out = tts_dir / f"tts_repaired_{idx}.wav"
        raw = tts_dir / f"tts_targeted_{tag}_{idx}.wav"
        out.unlink(missing_ok=True)
        raw.unlink(missing_ok=True)
        session.synthesize(text.strip(), raw, segment=seg)
        shutil.copyfile(raw, out)
        dur = get_wav_duration(out)
        seg["tts_path"] = str(out)
        seg["tts_raw_path"] = str(raw)
        seg["tts_spoken_text"] = text.strip()
        seg["translation"] = text.strip()
        seg["repaired_duration"] = round(dur, 2)
        seg["tts_duration"] = round(dur, 2)
        seg["tts_sha256"] = sha256_file(out)
        seg["placement_start"] = float(seg.get("placement_start") or seg.get("start") or 0.0)
        seg["preferred_placement_start"] = float(
            seg.get("preferred_placement_start") or seg.get("placement_start") or seg.get("start") or 0.0
        )
        seg["repaired_method"] = f"{seg.get('repaired_method') or 'none'}+targeted_{tag}"
        seg["timing_status"] = "TARGETED_REPAIRED"

    with TtsSession(settings, data_dir=config.data_dir, runner=runner, adapter_factory=create_tts_adapter) as session:
        # 1) Missing provenance → re-synth from spoken text
        for seg in segments:
            text = spoken_text(seg)
            if not text:
                continue
            path = Path(str(seg.get("tts_path") or ""))
            if path.is_file():
                seg["tts_sha256"] = sha256_file(path)
                continue
            print(f"resynth missing provenance index={seg['index']}: {text[:60]}")
            resynth(seg, text.lstrip(".… ").replace(". . .", "").strip() if text.startswith(("...", "…", ". . .")) else text, "provenance")

        # 2) Segment #27 (index 26) forced re-TTS from claimed spoken text
        if 26 in by_index:
            seg = by_index[26]
            text = spoken_text(seg)
            print(f"force re-TTS #27: {text}")
            resynth(seg, text, "seg27")

        # 3) Seam heal targeted pairs
        seams = detect_clause_seams(segments)
        print("detected seams", seams)
        target_pairs = {(35, 36), (71, 72)}
        for seam in seams:
            pair = (int(seam["left_index"]), int(seam["right_index"]))
            if pair not in target_pairs:
                continue
            left = by_index[pair[0]]
            right = by_index[pair[1]]
            left_text = spoken_text(left).rstrip(".… ").replace(". . .", "").strip()
            right_text = spoken_text(right).lstrip(".… ").replace(". . .", "").strip()
            # Join mid-clause: "…gãy" + "chân." → "…gãy chân."
            if left_text.endswith("gãy") and right_text.lower().startswith("chân"):
                merged = f"{left_text} {right_text}"
            elif "còn lại" in left_text and "trăm" in right_text:
                merged = f"{left_text.rstrip('.')} {right_text}".replace("  ", " ")
                # Prefer complete readable clause
                merged = "Thọ nguyên yêu ma còn lại một trăm chín mươi chín năm."
            else:
                merged = f"{left_text} {right_text}".replace("  ", " ").strip()
            # Keep left window expanded through right end; drop right speech by silking to empty? better: put full clause on left, move remainder to right if needed.
            print(f"heal seam {pair}: {merged[:100]}")
            # Place full healed clause on LEFT; right carries the remainder after first sentence if any.
            # For #72-73: entire "chạy gãy chân. Nhớ đãi..." should stay as readable units.
            if pair == (71, 72):
                # Keep first sentence complete on left; rest on right.
                # "Để gom đủ số ngài cần, mấy hôm nay anh em suýt chạy gãy chân."
                first = "Để gom đủ số ngài cần, mấy hôm nay anh em suýt chạy gãy chân."
                # Right: continue from original without orphan "chân."
                rest = spoken_text(right)
                if rest.lower().startswith("chân"):
                    rest = rest.split(".", 1)[-1].strip() if "." in rest else rest
                    rest = rest.lstrip(". ").strip()
                if not rest:
                    rest = "Nhớ đãi một bữa. Đưa nhóc của Trần Quý tới thôn Lục Lý Yêu chăn thả."
                # Also pull trailing fragment from #74 into right if short continuation.
                if 73 in by_index:
                    cont = spoken_text(by_index[73])
                    if cont and cont[:1].islower():
                        rest = f"{rest.rstrip('.')} {cont}".replace("  ", " ").strip()
                        # Clear #74 text to avoid duplicate — leave audio handled by re-synth of right only for now; mark 73 for provenance resynth of rewritten? Keep #74 as-is if already voiced continuation absorbed:
                        # Absorb into right and silence duplicate by resynth short pause? Safer: rewrite #74 spoken to empty? No — remove duplicate later by setting #74 to empty and skip mix if empty.
                        by_index[73]["tts_spoken_text"] = ""
                        by_index[73]["translation"] = ""
                        by_index[73]["no_speech"] = True
                resynth(left, first, "seam72")
                resynth(right, rest if rest.endswith((".", "!", "?")) else rest + ".", "seam73")
            else:
                # #36-37 quantity seam: one complete clause spanning left window end through right.
                # Put full sentence on left, silence/remove orphan right number fragment by merging duration windows.
                left_end = float(right.get("end") or left.get("end") or 0)
                left["end"] = left_end
                left["duration_budget"] = round(left_end - float(left.get("start") or 0), 2)
                resynth(left, merged if merged.endswith((".", "!", "?")) else merged + ".", "seam36")
                right["tts_spoken_text"] = ""
                right["translation"] = ""
                right["no_speech"] = True
                right["tts_path"] = None
                right["tts_raw_path"] = None

    # Drop no_speech units from mix list? Keep in checkpoint but skip mix — set empty spoken already.
    # Recompute hashes for remaining
    for seg in segments:
        path = Path(str(seg.get("tts_path") or ""))
        if path.is_file():
            seg["tts_sha256"] = sha256_file(path)

    report = validate_segments_tts_provenance(segments)
    seams_after = detect_clause_seams(
        [s for s in segments if spoken_text(s)]
    )
    print("provenance_report", report)
    print("seams_after", seams_after)

    repair["segments"] = segments
    repair["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    repair["targeted_p0"] = {
        "provenance": report,
        "seams_after": seams_after,
    }
    save_checkpoint(config.data_dir, JOB, "duration_repair", repair)

    # Clear align+mix+render so remux rebuilds from duration_repair
    jobs = JobService(database, config.data_dir)
    keep = list(PIPELINE_STEPS[: PIPELINE_STEPS.index("align_final_dub")])
    jobs.rerun(JOB, keep)
    # Restore the repaired duration_repair we just wrote (rerun may not delete it if in keep)
    # keep includes duration_repair — good.
    runner.start_job(JOB)
    deadline = time.time() + 2 * 3600
    while time.time() < deadline:
        hydrated = jobs.get(JOB)
        if hydrated.status in {"completed", "failed", "cancelled"}:
            print("Final status:", hydrated.status)
            if hydrated.last_error_code:
                print("Error:", hydrated.last_error_code, hydrated.last_error_message)
            return 0 if hydrated.status == "completed" else 1
        time.sleep(3)
    print("timeout", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
