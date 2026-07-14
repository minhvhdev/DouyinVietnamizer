"""Recover from audit + sanitize log, heal hard seams, fit overflow, remux."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

JOB = "f7620299-9f3c-42bd-a1e6-6f3ec0b542a2"

# Exact pre-sanitize texts captured from the failed sanitation pass (UI index = key+1).
SANITIZE_ORIGINS: dict[int, str] = {
    6: "Rõ ràng giây trước. . . còn đang hàn huyên, giây sau đã như hành quyết.",
    20: "Hắn cúi mắt nhìn bàn tay, dùng sức. . . siết thành quyền, đến khi đầu ngón tay ngừng run.",
    21: "Bình tĩnh, nhất định phải cố đến khi Trấn Ma Ty. . . tới đây.",
    23: "Còn vấn đề Trần Tịch đưa ra. . . Thẩm Thất Dạ thật ra cũng từng suy nghĩ.",
    24: "Nhưng cuối cùng hắn đã hiểu một chuyện: có hay không. . . cách giải quyết khác?",
    25: "Câu trả lời là không. Nếu mình khoanh tay mặc kệ, dân làng này cũng. . . phải chết.",
    27: "Vậy cứ làm tròn bổn phận, cần gì tự chuốc. . . phiền não?",
    33: "Khuyển yêu đã khai trí, chưa nhập Sơ cảnh. Tổng. . . thọ một trăm năm mươi hai năm, còn sáu mươi mốt năm. Hấp thu hoàn tất. Khuyển yêu chưa khôn.",
    38: "Tổng cộng vừa đúng hai trăm năm. Sau một đêm bận rộn. . .",
    42: "Chỉ khi nhìn qua chiếc xe. . . đôi mắt thất thần ấy mới ánh lên chút cảm xúc.",
    56: "Nếu đại nhân Thẩm hứng thú, thuộc hạ có thể giúp ngài. . .",
    63: "Lão đại Thẩm. . . dậy sớm vậy sao?",
    66: "Lão đại Thẩm, việc ngài giao. . . đã làm xong rồi. Việc gì cơ?",
    68: "Nhờ phần của con gái nhà Đinh Lưu. . .",
}


def main() -> int:
    backend_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(backend_dir))
    import os

    os.environ.setdefault("DV_VENDOR_DIR", str(backend_dir.parent / "vendor"))

    from dv_backend.adapters.tts import TtsSession, create_tts_adapter
    from dv_backend.checkpoints import PIPELINE_STEPS, load_checkpoint, save_checkpoint
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
    tts_by = {
        int(s["index"]): s
        for s in ((load_checkpoint(config.data_dir, JOB, "tts") or {}).get("segments") or [])
    }

    audit_path = job_dir / "artifacts" / "clause_seam_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    classified = audit.get("classified_seams") or classify_clause_seams(segments)
    # Prefer original pre-heal classifications if present.
    if audit.get("classified_seams"):
        # Rebuild hard clusters from that snapshot.
        clusters = hard_seam_clusters(classified)
    else:
        clusters = hard_seam_clusters(classify_clause_seams(segments))

    # Rebuild per-index text map from audit seam edges (pre-mangle).
    edge_text: dict[int, str] = {}
    for seam in classified:
        edge_text[int(seam["left_index"])] = str(seam["left_text"])
        edge_text[int(seam["right_index"])] = str(seam["right_text"])

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
        seg["timing_overflow_sec"] = 0.0

    planned: dict[int, str] = {}
    silenced: set[int] = set()

    for cluster in clusters:
        parts = []
        for idx in cluster:
            text = edge_text.get(idx) or spoken_text(by_index[idx])
            if text:
                parts.append(text)
        if len(parts) < 2:
            continue
        keep = cluster[0]
        planned[keep] = join_clause_texts(parts)
        for idx in cluster[1:]:
            silenced.add(idx)
        last = by_index[cluster[-1]]
        by_index[keep]["end"] = max(float(by_index[keep].get("end") or 0), float(last.get("end") or 0))
        print(f"cluster {cluster} -> #{keep + 1}: {planned[keep][:100]}")

    # Restore sanitize victims (normalize . . . → ... via sanitize).
    for idx, raw in SANITIZE_ORIGINS.items():
        if idx in silenced or idx in planned:
            continue
        planned[idx] = sanitize_spoken_text(raw)

    # #72 / #73 targets
    body82 = sanitize_spoken_text(
        str((tts_by.get(82) or {}).get("translation") or ""), strip_leading_ellipsis=True
    )
    if body82.lower().startswith("chân"):
        body82 = body82.split(".", 1)[-1].strip() if "." in body82 else body82
        body82 = sanitize_spoken_text(body82, strip_leading_ellipsis=True)
    body83 = sanitize_spoken_text(
        str((tts_by.get(83) or {}).get("translation") or ""), strip_leading_ellipsis=True
    )
    planned[72] = join_clause_texts([body82, body83])
    planned[71] = "Để gom đủ số ngài cần, mấy hôm nay anh em suýt chạy gãy chân."
    # Ensure #27 kept
    if 26 not in planned:
        planned[26] = "Đằng nào cũng chết, đây là thế cờ vô giải."
    if 35 not in planned:
        planned[35] = "Thọ nguyên yêu ma còn lại một trăm chín mươi chín năm."
    silenced.update({36, 73})  # absorbed orphans

    with TtsSession(settings, data_dir=config.data_dir, runner=runner, adapter_factory=create_tts_adapter) as session:

        def resynth(seg: dict, text: str, tag: str) -> None:
            clean = sanitize_spoken_text(text, strip_leading_ellipsis=True)
            if clean and not clean.endswith((".", "!", "?", "...")):
                clean += "."
            idx = int(seg["index"])
            out = tts_dir / f"tts_repaired_{idx}.wav"
            raw = tts_dir / f"tts_targeted_{tag}_{idx}.wav"
            out.unlink(missing_ok=True)
            raw.unlink(missing_ok=True)
            session.synthesize(clean, raw, segment=seg)
            shutil.copyfile(raw, out)
            dur = get_wav_duration(out)
            seg.update(
                {
                    "tts_path": str(out),
                    "tts_raw_path": str(raw),
                    "tts_spoken_text": clean,
                    "translation": clean,
                    "no_speech": False,
                    "repaired_duration": round(dur, 2),
                    "tts_duration": round(dur, 2),
                    "tts_sha256": sha256_file(out),
                }
            )
            print(f"resynth #{idx + 1}: {clean[:110]}")

        for idx in sorted(silenced):
            if idx in by_index:
                silence(by_index[idx])

        for idx, text in sorted(planned.items()):
            if idx not in by_index:
                continue
            target = sanitize_spoken_text(text, strip_leading_ellipsis=True)
            current = spoken_text(by_index[idx])
            path = Path(str(by_index[idx].get("tts_path") or ""))
            if current == target and path.is_file() and abs(float(by_index[idx].get("repaired_duration") or 0)) > 0.2:
                by_index[idx]["tts_sha256"] = sha256_file(path)
                continue
            resynth(by_index[idx], target, "recover")

    # Fit to zero-overlap placements and stamp overflow from allocation.
    for _round in range(2):
        compute_placement_starts(segments)
        voiced = [s for s in segments if spoken_text(s) and s.get("tts_path")]
        schedule_soft_placements(voiced)
        enforce_zero_overlap_placements(voiced)
        ordered = sorted(voiced, key=lambda s: float(s.get("placement_start") or 0))
        for i, seg in enumerate(ordered):
            path = Path(str(seg["tts_path"]))
            duration = get_wav_duration(path)
            start = float(seg.get("placement_start") or 0)
            next_start = (
                float(ordered[i + 1].get("placement_start") or 0) if i + 1 < len(ordered) else None
            )
            if next_start is None:
                seg["repaired_duration"] = round(duration, 2)
                seg["timing_overflow_sec"] = 0.0
                seg["tts_sha256"] = sha256_file(path)
                continue
            alloc = max(0.35, next_start - start - BOUNDARY_MARGIN_SEC)
            if duration > alloc + 0.05:
                rate = min(1.15, duration / max(alloc, 0.35))
                out = tts_dir / f"tts_fit_{seg['index']}_{_round}.wav"
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
                    print(f"fit r{_round} idx={seg['index']} rate={rate:.3f} dur={duration:.2f} alloc={alloc:.2f}")
            seg["repaired_duration"] = round(duration, 2)
            seg["tts_sha256"] = sha256_file(path)
            seg["timing_overflow_sec"] = round(max(0.0, duration - alloc), 3)

    compute_placement_starts(segments)
    voiced = [s for s in segments if spoken_text(s) and s.get("tts_path")]
    schedule_soft_placements(voiced)
    enforce_zero_overlap_placements(voiced)
    ordered = sorted(voiced, key=lambda s: float(s.get("placement_start") or 0))
    for i, seg in enumerate(ordered):
        start = float(seg.get("placement_start") or 0)
        dur = float(seg.get("repaired_duration") or 0)
        if i + 1 < len(ordered):
            alloc = max(0.35, float(ordered[i + 1].get("placement_start") or 0) - start - BOUNDARY_MARGIN_SEC)
            seg["timing_overflow_sec"] = round(max(0.0, dur - alloc), 3)
        else:
            seg["timing_overflow_sec"] = 0.0

    classified_after = classify_clause_seams(segments)
    hard_after = [s for s in classified_after if s["severity"] == "hard"]
    soft_after = [s for s in classified_after if s["severity"] == "soft"]
    overflow_n = sum(1 for s in segments if float(s.get("timing_overflow_sec") or 0) > 0.15)
    report = {
        "hard_clusters": clusters,
        "hard_after": hard_after,
        "soft_after": soft_after,
        "unclassified": 0,
        "provenance": validate_segments_tts_provenance(segments),
        "sanitation": validate_segments_text_sanitation(segments),
        "voiced_overlap": segments_with_voiced_overlap(segments),
        "overflow_count": overflow_n,
        "spoken_27": spoken_text(by_index.get(26, {})),
        "spoken_36": spoken_text(by_index.get(35, {})),
        "spoken_37": spoken_text(by_index.get(36, {})),
        "spoken_72": spoken_text(by_index.get(71, {})),
        "spoken_73": spoken_text(by_index.get(72, {})),
        "seam_report": classified_after,
    }
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "hard_after": hard_after,
                "soft_after": soft_after,
                "overflow": overflow_n,
                "overlap": report["voiced_overlap"],
                "prov": report["provenance"]["passed"],
                "sanitation": report["sanitation"],
                "spoken_73": report["spoken_73"],
                "spoken_72": report["spoken_72"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    repair["segments"] = segments
    save_checkpoint(config.data_dir, JOB, "duration_repair", repair)
    align = load_checkpoint(config.data_dir, JOB, "align_final_dub") or {}
    if align:
        align["segments"] = segments
        save_checkpoint(config.data_dir, JOB, "align_final_dub", align)

    if hard_after or not report["provenance"]["passed"]:
        return 1
    if report["sanitation"]["count"]:
        for item in report["sanitation"]["blocking"]:
            seg = by_index[int(item["index"])]
            cleaned = sanitize_spoken_text(spoken_text(seg))
            seg["tts_spoken_text"] = cleaned
            seg["translation"] = cleaned
        if validate_segments_text_sanitation(segments)["count"]:
            return 1

    if overflow_n:
        print(f"overflow_remaining={overflow_n}", file=sys.stderr)
        return 2

    jobs = JobService(database, config.data_dir)
    keep = [step for step in PIPELINE_STEPS if step != "mix" and PIPELINE_STEPS.index(step) < PIPELINE_STEPS.index("mix")]
    jobs.rerun(JOB, keep_steps=keep)
    runner.start_job(JOB)
    deadline = time.time() + 3600
    while time.time() < deadline:
        hydrated = jobs.get(JOB)
        if hydrated.status in {"completed", "failed", "cancelled"}:
            print("Final status:", hydrated.status, hydrated.last_error_code, hydrated.last_error_message)
            return 0 if hydrated.status == "completed" else 1
        time.sleep(2)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
