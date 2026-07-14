"""Rollback to narrow P0 only: #27 + #36/#37 + #72/#73. Undo hard-seam/sanitize mass re-TTS."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

JOB = "f7620299-9f3c-42bd-a1e6-6f3ec0b542a2"

# Pre-hard-seam spoken texts (from targeted_p0 seams_after + sanitize origins).
RESTORE_TEXT: dict[int, str] = {
    # Sanitize victims (intentional mid-clause ellipsis kept as ". . .")
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
    # Hard-seam undo (split again)
    7: "Ngồi đó mà giết con khuyển yêu. Vì sao lúc ra tay. . .",
    8: "trên mặt ngài vẫn treo nụ cười đáng ghét ấy?",
    10: "Ngài thấy yêu ma ra sao? Trần Tịch. . .",
    11: "lạnh lùng nhìn đồng loại bị gặm, rồi. . .",
    16: "Đành chọn cách dung hòa để mọi người đều hài lòng. Cái. . .",
    17: "cái đó cũng tính là lý do sao? Mới vài ngày không gặp. . .",
    18: "hắn đã nhận ra mình không còn hiểu nổi vị đại nhân này. Khi xung quanh không còn",
    28: "Thay vì lấy đại cục làm trọng. . .",
    29: "hay nhịn một lúc sóng yên biển lặng để tự ru ngủ mình, chi bằng. . .",
    30: "thu thêm vài năm thọ nguyên yêu ma để tăng thực lực.",
    40: "Bất kể đại nhân Thẩm tính toán gì, những dân làng lác đác trên bờ ruộng. . .",
    41: "vẫn mang vẻ mặt tê dại như cũ. Họ buông thõng tay như những xác sống.",
    46: "Nếu là trước đây, dù đại nhân Thẩm chỉ đi ngang cửa. . .",
    47: "cũng hận không thể lột một lớp da của dân chúng.",
    49: "Đây là cái cớ tốt biết bao, vậy mà lại nhẹ nhàng. . .",
    50: "… bỏ đi vậy. Lạ thật.",
    51: "Trần Quý tự hỏi, nếu mình ngồi cạnh khuyển yêu khi ấy, e rằng cũng không thể. . .",
    52: "ra tay dứt khoát như vậy.",
    53: "Nhớ cảnh Thẩm Nhất tùy tiện lật xem bản chép võ học, Trần Quý hỏi: Đại. . .",
    54: "nhân cũng hứng thú với võ học của Trấn Ma Ty sao?",
    59: "Phải biết một nguyên nhân quan trọng khiến hắn khổ luyện võ công năm xưa. . .",
    60: "là muốn dùng trường đao chém tên súc sinh trước mặt.",
    75: "Đến lúc đó, ta cứ lịch sự với em gái hắn trước, rồi. . .",
    76: "sau đó đưa lên núi cho vị kia.",
    # Narrow P0 keeps
    26: "Đằng nào cũng chết, đây là thế cờ vô giải.",
    35: "Thọ nguyên yêu ma còn lại một trăm chín mươi chín năm.",
    71: "Để gom đủ số ngài cần, mấy hôm nay anh em suýt chạy gãy chân.",
    72: "Nhớ đãi một bữa. Đưa nhóc của Trần Quý tới thôn Lục Lý Yêu chăn thả.",
}

SILENCE = {36, 73}  # absorbed orphans after narrow seam heal

# Prefer existing good WAV copies (no re-TTS) when available.
WAV_COPIES: dict[int, str] = {
    0: "tts_targeted_provenance_0.wav",
    1: "tts_targeted_provenance_1.wav",
    2: "tts_targeted_provenance_2.wav",
    3: "tts_targeted_provenance_3.wav",
    4: "tts_targeted_provenance_4.wav",
    26: "tts_fit_26.wav",  # fitted #27 after seg27
    35: "tts_targeted_seam36_35.wav",
    71: "tts_fit_71.wav",
    72: "tts_targeted_seam73_72.wav",
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
    from dv_backend.pipeline import get_wav_duration
    from dv_backend.runner import JobRunner
    from dv_backend.timing_placement import (
        compute_placement_starts,
        enforce_zero_overlap_placements,
        schedule_soft_placements,
        segments_with_voiced_overlap,
    )
    from dv_backend.tts_provenance import sha256_file, spoken_text, validate_segments_tts_provenance

    config = AppConfig.from_env()
    database = Database(config.database_path)
    rows = database.connection.execute("SELECT key, value FROM settings").fetchall()
    settings = {r["key"]: json.loads(r["value"]) for r in rows}
    job_dir = config.data_dir / "jobs" / JOB
    tts_dir = job_dir / "artifacts" / "tts"
    runner = JobRunner(config, database)

    repair = load_checkpoint(config.data_dir, JOB, "duration_repair") or {}
    segments = [dict(s) for s in repair.get("segments") or []]
    by_index = {int(s["index"]): s for s in segments}

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
        seg["soft_speed_factor"] = 1.0

    def apply_wav(seg: dict, src: Path, text: str, tag: str) -> None:
        idx = int(seg["index"])
        out = tts_dir / f"tts_repaired_{idx}.wav"
        if src.resolve() != out.resolve():
            shutil.copyfile(src, out)
        dur = get_wav_duration(out)
        seg["tts_path"] = str(out)
        seg["tts_spoken_text"] = text
        seg["translation"] = text
        seg["no_speech"] = False
        seg["repaired_duration"] = round(dur, 2)
        seg["tts_duration"] = round(dur, 2)
        seg["tts_sha256"] = sha256_file(out)
        seg["timing_overflow_sec"] = 0.0
        seg["repaired_method"] = f"rollback_narrow_{tag}"
        print(f"restore wav #{idx + 1} from {src.name} dur={dur:.2f}")

    need_synth: list[tuple[dict, str]] = []

    for idx, text in RESTORE_TEXT.items():
        if idx not in by_index:
            continue
        seg = by_index[idx]
        copy_name = WAV_COPIES.get(idx)
        if copy_name:
            src = tts_dir / copy_name
            if src.is_file():
                apply_wav(seg, src, text, "copy")
                continue
        # Prefer intact repaired wav if timestamp suggests pre-mass-heal AND text length similar?
        # Always re-synth restored hard/sanitize texts so audio matches text exactly.
        need_synth.append((seg, text))

    for idx in SILENCE:
        if idx in by_index:
            silence(by_index[idx])

    if need_synth:
        with TtsSession(
            settings, data_dir=config.data_dir, runner=runner, adapter_factory=create_tts_adapter
        ) as session:
            for seg, text in need_synth:
                idx = int(seg["index"])
                out = tts_dir / f"tts_repaired_{idx}.wav"
                raw = tts_dir / f"tts_rollback_{idx}.wav"
                out.unlink(missing_ok=True)
                raw.unlink(missing_ok=True)
                session.synthesize(text, raw, segment=seg)
                shutil.copyfile(raw, out)
                apply_wav(seg, out, text, "synth")

    # Keep other units: if they still have speech text, refresh hash/duration from file.
    for seg in segments:
        idx = int(seg["index"])
        if idx in RESTORE_TEXT or idx in SILENCE:
            continue
        text = spoken_text(seg)
        if not text:
            silence(seg)
            continue
        path = Path(str(seg.get("tts_path") or ""))
        if not path.is_file():
            path = tts_dir / f"tts_repaired_{idx}.wav"
        if path.is_file():
            seg["tts_path"] = str(path)
            seg["repaired_duration"] = round(get_wav_duration(path), 2)
            seg["tts_sha256"] = sha256_file(path)
            seg["no_speech"] = False
            seg["timing_overflow_sec"] = 0.0
        else:
            print(f"WARNING missing wav for kept index {idx}", file=sys.stderr)

    compute_placement_starts(segments)
    voiced = [s for s in segments if spoken_text(s) and s.get("tts_path")]
    schedule_soft_placements(voiced)
    enforce_zero_overlap_placements(voiced)
    for seg in voiced:
        start = float(seg.get("placement_start") or 0)
        dur = float(seg.get("repaired_duration") or 0)
        seg["timing_overflow_sec"] = 0.0
        # recompute overflow vs next later in remux gate after soft place
    ordered = sorted(voiced, key=lambda s: float(s.get("placement_start") or 0))
    for i, seg in enumerate(ordered):
        if i + 1 >= len(ordered):
            seg["timing_overflow_sec"] = 0.0
            continue
        start = float(seg.get("placement_start") or 0)
        nxt = float(ordered[i + 1].get("placement_start") or 0)
        dur = float(seg.get("repaired_duration") or 0)
        seg["timing_overflow_sec"] = round(max(0.0, dur - max(0.35, nxt - start - 0.025)), 3)

    overlaps = segments_with_voiced_overlap(voiced)
    print("overlap", overlaps)
    print("prov", validate_segments_tts_provenance(segments))
    for i in (26, 35, 36, 71, 72, 73):
        print(f"#{i + 1}", repr(spoken_text(by_index[i])[:90]), "path", bool(by_index[i].get("tts_path")))

    repair["segments"] = segments
    save_checkpoint(config.data_dir, JOB, "duration_repair", repair)
    align = load_checkpoint(config.data_dir, JOB, "align_final_dub") or {}
    if align:
        align["segments"] = segments
        save_checkpoint(config.data_dir, JOB, "align_final_dub", align)

    if overlaps:
        print("overlap remains; still remuxing after soft place", file=sys.stderr)

    jobs = JobService(database, config.data_dir)
    keep = [s for s in PIPELINE_STEPS if PIPELINE_STEPS.index(s) < PIPELINE_STEPS.index("mix")]
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
