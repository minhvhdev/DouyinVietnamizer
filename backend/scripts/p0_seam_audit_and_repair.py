"""P0: reconstruct #73, classify/heal hard seams, sanitize text, remux."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

JOB = "f7620299-9f3c-42bd-a1e6-6f3ec0b542a2"


def main() -> int:
    backend_dir = Path(__file__).resolve().parents[1]
    repo_root = backend_dir.parent
    sys.path.insert(0, str(backend_dir))
    import os

    os.environ.setdefault("DV_VENDOR_DIR", str(repo_root / "vendor"))

    from dv_backend.adapters.tts import TtsSession, create_tts_adapter
    from dv_backend.checkpoints import load_checkpoint, save_checkpoint
    from dv_backend.config import AppConfig
    from dv_backend.database import Database
    from dv_backend.jobs import JobService
    from dv_backend.pipeline import (
        _build_atempo_chain,
        _run_ffmpeg_audio_filter,
        get_wav_duration,
        resolve_tool_path,
    )
    from dv_backend.runner import JobRunner
    from dv_backend.text_sanitation import sanitize_spoken_text, validate_segments_text_sanitation
    from dv_backend.timing_placement import (
        BOUNDARY_MARGIN_SEC,
        compute_placement_starts,
        enforce_zero_overlap_placements,
        schedule_soft_placements,
        segments_with_voiced_overlap,
    )
    from dv_backend.tts_provenance import (
        classify_clause_seams,
        hard_seam_clusters,
        join_clause_texts,
        sha256_file,
        spoken_text,
        validate_segments_tts_provenance,
    )

    config = AppConfig.from_env()
    database = Database(config.database_path)
    rows = database.connection.execute("SELECT key, value FROM settings").fetchall()
    settings = {r["key"]: json.loads(r["value"]) for r in rows}
    job_dir = config.data_dir / "jobs" / JOB
    tts_dir = job_dir / "artifacts" / "tts"
    ffmpeg_path = resolve_tool_path(config, "ffmpeg")
    runner = JobRunner(config, database)

    repair = load_checkpoint(config.data_dir, JOB, "duration_repair") or {}
    segments = [dict(s) for s in repair.get("segments") or []]
    by_index = {int(s["index"]): s for s in segments}
    translate = (load_checkpoint(config.data_dir, JOB, "translate") or {}).get("segments") or []
    translate_by = {int(s["index"]): s for s in translate}

    def silence(seg: dict) -> None:
        seg["tts_spoken_text"] = ""
        seg["translation"] = ""
        seg["target_text"] = ""
        seg["no_speech"] = True
        seg["tts_path"] = None
        seg["tts_raw_path"] = None
        seg["repaired_duration"] = 0.0
        seg["tts_duration"] = 0.0
        seg["tts_sha256"] = None

    def resynth(seg: dict, text: str, tag: str) -> None:
        idx = int(seg["index"])
        clean = sanitize_spoken_text(text, strip_leading_ellipsis=True)
        if clean and not clean.endswith((".", "!", "?", "...")):
            clean += "."
        out = tts_dir / f"tts_repaired_{idx}.wav"
        raw = tts_dir / f"tts_targeted_{tag}_{idx}.wav"
        out.unlink(missing_ok=True)
        raw.unlink(missing_ok=True)
        session.synthesize(clean, raw, segment=seg)
        shutil.copyfile(raw, out)
        dur = get_wav_duration(out)
        seg["tts_path"] = str(out)
        seg["tts_raw_path"] = str(raw)
        seg["tts_spoken_text"] = clean
        seg["translation"] = clean
        seg["no_speech"] = False
        seg["repaired_duration"] = round(dur, 2)
        seg["tts_duration"] = round(dur, 2)
        seg["tts_sha256"] = sha256_file(out)
        seg["repaired_method"] = f"{seg.get('repaired_method') or 'none'}+{tag}"
        seg["timing_status"] = "TARGETED_REPAIRED"
        print(f"resynth #{idx + 1}: {clean[:100]}")

    # --- P0.1 reconstruct #73 from translate lineage orig 82+83 (minus chân) ---
    t82 = str((translate_by.get(82) or {}).get("translation") or "")
    t83 = str((translate_by.get(83) or {}).get("translation") or "")
    # Drop orphan chân./ellipsis head from 82, keep meal/send clauses.
    body82 = sanitize_spoken_text(t82, strip_leading_ellipsis=True)
    if body82.lower().startswith("chân"):
        body82 = body82.split(".", 1)[-1].strip() if "." in body82 else body82
        body82 = sanitize_spoken_text(body82, strip_leading_ellipsis=True)
    body83 = sanitize_spoken_text(t83, strip_leading_ellipsis=True)
    intended_73 = join_clause_texts([body82, body83])
    print("intended #73 lineage:", intended_73)

    with TtsSession(settings, data_dir=config.data_dir, runner=runner, adapter_factory=create_tts_adapter) as session:
        if 72 in by_index:
            resynth(by_index[72], intended_73, "recon73")

        # Classify seams before heal.
        classified = classify_clause_seams(segments)
        report = {
            "job_id": JOB,
            "classified_seams": classified,
            "hard_clusters": hard_seam_clusters(classified),
            "intended_73": intended_73,
        }
        audit_path = job_dir / "artifacts" / "clause_seam_audit.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)

        for cluster in report["hard_clusters"]:
            texts = [spoken_text(by_index[i]) for i in cluster if i in by_index]
            # Skip empty right tails.
            texts = [t for t in texts if t]
            if len(cluster) < 2 or len(texts) < 2:
                continue
            keep = cluster[0]
            # Prefer leave #72 untouched if already reconstructed and only soft neighbors
            merged = join_clause_texts(texts)
            print(f"heal hard cluster {cluster}: {merged[:120]}")
            resynth(by_index[keep], merged, f"hardseam{keep}")
            for idx in cluster[1:]:
                silence(by_index[idx])
                by_index[idx]["end"] = float(by_index[idx].get("end") or by_index[keep].get("end") or 0)
            # Expand keeper window through last silenced end for subtitle cover.
            last = by_index[cluster[-1]]
            by_index[keep]["end"] = max(float(by_index[keep].get("end") or 0), float(last.get("end") or 0))

        # Typography sanitize remaining voiced texts; re-TTS if changed.
        for seg in segments:
            text = spoken_text(seg)
            if not text:
                if seg.get("no_speech"):
                    silence(seg)
                continue
            clean = sanitize_spoken_text(text, strip_leading_ellipsis=False)
            if clean != text:
                print(f"sanitize #{int(seg['index']) + 1}: {text!r} -> {clean!r}")
                resynth(seg, clean, "sanitize")

    # Fit overlaps after longer merges.
    voiced = [s for s in segments if spoken_text(s) and s.get("tts_path")]
    compute_placement_starts(segments)
    schedule_soft_placements(voiced)
    enforce_zero_overlap_placements(voiced)
    ordered = sorted(voiced, key=lambda s: float(s.get("placement_start") or 0))
    for i, seg in enumerate(ordered):
        path = Path(str(seg["tts_path"]))
        if not path.is_file():
            continue
        duration = get_wav_duration(path)
        next_start = None
        if i + 1 < len(ordered):
            next_start = float(ordered[i + 1].get("placement_start") or ordered[i + 1].get("start") or 0)
        start = float(seg.get("placement_start") or seg.get("start") or 0)
        if next_start is None:
            seg["repaired_duration"] = round(duration, 2)
            seg["tts_sha256"] = sha256_file(path)
            continue
        alloc = max(0.4, next_start - start - BOUNDARY_MARGIN_SEC)
        if duration <= alloc + 0.05:
            seg["repaired_duration"] = round(duration, 2)
            seg["tts_sha256"] = sha256_file(path)
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
            seg["tts_sha256"] = sha256_file(path)
            print(f"fit index={seg['index']} rate={rate:.3f} dur={duration:.2f}")

    compute_placement_starts(segments)
    voiced = [s for s in segments if spoken_text(s) and s.get("tts_path")]
    schedule_soft_placements(voiced)
    enforce_zero_overlap_placements(voiced)

    classified_after = classify_clause_seams(segments)
    hard_after = [s for s in classified_after if s["severity"] == "hard"]
    report["classified_after"] = classified_after
    report["hard_after"] = hard_after
    report["provenance"] = validate_segments_tts_provenance(segments)
    report["sanitation"] = validate_segments_text_sanitation(segments)
    report["voiced_overlap"] = segments_with_voiced_overlap(segments)
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("audit written", audit_path)
    print("hard_after", hard_after)
    print("prov", report["provenance"]["passed"], "sanitation", report["sanitation"]["passed"])
    print("overlap", report["voiced_overlap"])

    repair["segments"] = segments
    save_checkpoint(config.data_dir, JOB, "duration_repair", repair)
    # Invalidate align so mix picks repaired durations/texts; force full re-align for ASS cover?
    # ChatGPT: remux from mix with force subtitle. Keep align but sync segment payload.
    align = load_checkpoint(config.data_dir, JOB, "align_final_dub") or {}
    if align:
        align["segments"] = segments
        save_checkpoint(config.data_dir, JOB, "align_final_dub", align)

    if hard_after:
        print("HARD seams remain; abort remux", file=sys.stderr)
        return 1
    if not report["provenance"]["passed"] or not report["sanitation"]["passed"]:
        print("gates failed; abort remux", file=sys.stderr)
        return 1

    jobs = JobService(database, config.data_dir)
    # Rerun from mix: keep through align_final_dub
    keep = []
    from dv_backend.checkpoints import PIPELINE_STEPS

    for step in PIPELINE_STEPS:
        if step == "mix":
            break
        keep.append(step)
    jobs.rerun(JOB, keep_steps=keep)
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
