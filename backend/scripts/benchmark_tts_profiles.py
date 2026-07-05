from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import time
import wave
from pathlib import Path
from typing import Any


TEXTS = [
    (
        "Xin chào mọi người. Hôm nay mình chia sẻ một mẹo học tiếng Trung. "
        "Bạn không cần học quá nhiều từ mới mỗi ngày. Chỉ cần nghe lại các câu ngắn nhiều lần. "
        "Khi quen nhịp nói, bạn sẽ nhớ tự nhiên hơn."
    ),
    (
        "Buổi sáng nay trời khá mát. Tôi pha một ly cà phê rồi mở cửa sổ. "
        "Ngoài đường bắt đầu đông hơn bình thường. Vài người ghé quán ăn sáng ở đầu hẻm. "
        "Khung cảnh quen thuộc nhưng vẫn rất dễ chịu."
    ),
    (
        "Nếu bạn làm video ngắn, nhịp kể chuyện là điều quan trọng nhất. "
        "Câu đầu phải đủ rõ để người xem hiểu vấn đề. Sau đó nên thêm một chi tiết gây tò mò. "
        "Phần cuối hãy chốt lại bằng một ý thật gọn. Đừng nhồi quá nhiều thông tin."
    ),
    (
        "Hôm qua tôi thử nấu một món mới theo công thức trên mạng. Lúc đầu mọi thứ có vẻ khá đơn giản. "
        "Đến bước nêm gia vị thì tôi bắt đầu lúng túng. Sau vài lần nếm thử, hương vị cũng ổn hơn. "
        "Dù chưa hoàn hảo, cả nhà vẫn ăn rất vui vẻ."
    ),
    (
        "Nhiều người nghĩ rằng làm việc hiệu quả là phải luôn bận rộn. Thật ra không hẳn như vậy. "
        "Khi lịch làm việc quá dày, đầu óc dễ mệt hơn. Nghỉ ngắn đúng lúc lại giúp mình quay lại nhanh hơn. "
        "Giữ nhịp ổn định cả ngày thường tốt hơn cố quá sức."
    ),
]


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def normalize_text(text: str) -> str:
    lowered = text.casefold()
    lowered = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    return re.sub(r"\s+", " ", lowered).strip()


def levenshtein(seq_a: list[str], seq_b: list[str]) -> int:
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)
    prev = list(range(len(seq_b) + 1))
    for i, item_a in enumerate(seq_a, start=1):
        curr = [i]
        for j, item_b in enumerate(seq_b, start=1):
            if item_a == item_b:
                curr.append(prev[j - 1])
            else:
                curr.append(1 + min(prev[j - 1], prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def char_error_rate(reference: str, hypothesis: str) -> float:
    ref = list(normalize_text(reference).replace(" ", ""))
    hyp = list(normalize_text(hypothesis).replace(" ", ""))
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def load_settings() -> tuple[Path, dict[str, Any]]:
    from dv_backend.config import AppConfig
    from dv_backend.database import Database
    from dv_backend.settings import SettingsService

    config = AppConfig.from_env()
    config.ensure_directories()
    database = Database(config.database_path)
    database.migrate()
    settings = SettingsService(database).get_all()
    return config.data_dir, settings


def asr_roundtrip(
    audio_files: list[Path],
    references: list[str],
    *,
    settings: dict[str, Any],
) -> dict[str, Any]:
    from dv_backend.adapters.asr import transcribe_audio

    project_root = Path(__file__).resolve().parents[2]
    vendor_dir = project_root / "vendor"
    transcripts: list[str] = []
    cer_values: list[float] = []
    wer_values: list[float] = []
    per_item: list[dict[str, Any]] = []

    for audio_path, reference in zip(audio_files, references, strict=True):
        result = transcribe_audio(
            audio_path,
            vendor_dir=vendor_dir,
            asr_model=str(settings.get("qwen3_asr_model", "") or ""),
            aligner_model=str(settings.get("qwen3_aligner_model", "") or ""),
            device=str(settings.get("qwen3_device", "cuda:0") or "cuda:0"),
            language="Vietnamese",
            include_alignment=False,
        )
        if isinstance(result, dict):
            segments = result.get("segments", [])
        else:
            segments = result
        transcript = " ".join(str(item.get("text") or "").strip() for item in segments if str(item.get("text") or "").strip())
        transcripts.append(transcript)
        cer = char_error_rate(reference, transcript)
        wer = word_error_rate(reference, transcript)
        cer_values.append(cer)
        wer_values.append(wer)
        per_item.append(
            {
                "audio_file": str(audio_path),
                "reference": reference,
                "transcript": transcript,
                "char_error_rate": round(cer, 4),
                "word_error_rate": round(wer, 4),
            }
        )

    return {
        "avg_char_error_rate": round(statistics.mean(cer_values), 4) if cer_values else None,
        "avg_word_error_rate": round(statistics.mean(wer_values), 4) if wer_values else None,
        "items": per_item,
    }


def run_profile(
    name: str,
    *,
    settings: dict[str, Any],
    texts: list[str],
    out_dir: Path,
) -> dict[str, Any]:
    from dv_backend.adapters.tts import TtsSession
    from dv_backend.adapters.voxcpm_client import release_all_clients

    profile_dir = out_dir / name
    if profile_dir.exists():
        shutil.rmtree(profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)

    release_all_clients()
    segment_reports: list[dict[str, Any]] = []
    warmup_path = profile_dir / "_warmup.wav"
    total_started = time.perf_counter()

    with TtsSession(settings, data_dir=profile_dir, runner=None) as session:
        session.synthesize("Xin chào, đây là câu khởi động benchmark.", warmup_path, segment={"text": "你好，预热。"})
        warmup_path.unlink(missing_ok=True)

        for index, text in enumerate(texts):
            output_path = profile_dir / f"seg_{index:02d}.wav"
            started = time.perf_counter()
            session.synthesize(text, output_path, segment={"text": f"segment-{index}"})
            wall_ms = round((time.perf_counter() - started) * 1000)
            audio_sec = round(wav_duration(output_path), 3)
            rtf = round((wall_ms / 1000.0) / audio_sec, 4) if audio_sec > 0 else None
            segment_reports.append(
                {
                    "index": index,
                    "output_path": str(output_path),
                    "wall_time_ms": wall_ms,
                    "audio_duration_sec": audio_sec,
                    "real_time_factor": rtf,
                    "file_size_bytes": output_path.stat().st_size,
                }
            )

    release_all_clients()
    total_wall_ms = round((time.perf_counter() - total_started) * 1000)
    synth_wall_ms = sum(item["wall_time_ms"] for item in segment_reports)
    total_audio_sec = round(sum(item["audio_duration_sec"] for item in segment_reports), 3)
    return {
        "name": name,
        "num_steps": int(settings["voxcpm_num_steps"]),
        "batch_size": int(settings["voxcpm_batch_size"]),
        "flush_ms": int(settings["voxcpm_batch_flush_ms"]),
        "clone_mode": str(settings.get("voxcpm_clone_mode") or "reference"),
        "reference_audio": str(settings.get("voxcpm_ref_audio") or ""),
        "total_wall_time_ms": total_wall_ms,
        "synthesis_wall_time_ms": synth_wall_ms,
        "total_audio_duration_sec": total_audio_sec,
        "avg_segment_wall_ms": round(statistics.mean(item["wall_time_ms"] for item in segment_reports), 1),
        "avg_segment_rtf": round(statistics.mean(item["real_time_factor"] for item in segment_reports if item["real_time_factor"] is not None), 4),
        "segments": segment_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark VoxCPM TTS profiles on five medium-length Vietnamese passages.")
    parser.add_argument("--steps", default="10,8,6", help="Comma-separated num_steps values to benchmark.")
    parser.add_argument("--batch-sizes", default="4,8,12", help="Comma-separated batch_size values to benchmark.")
    parser.add_argument("--flush-ms", type=int, default=150, help="Flush window forwarded to the worker client.")
    parser.add_argument("--finalists", type=int, default=3, help="How many fastest profiles to round-trip through ASR.")
    parser.add_argument("--out", type=Path, default=Path("benchmark_artifacts") / "tts_profiles", help="Output directory.")
    args = parser.parse_args()

    app_data_dir, base_settings = load_settings()
    ref_audio = str(base_settings.get("voxcpm_ref_audio") or "").strip()
    if ref_audio and not Path(ref_audio).is_file():
        base_settings["voxcpm_ref_audio"] = ""
        base_settings["voxcpm_clone_mode"] = "reference"

    base_settings["tts_backend"] = "voxcpm"
    base_settings["tts_session_reuse_enabled"] = True
    base_settings["tts_conversion_strategy"] = "lazy_mix"
    base_settings["voxcpm_cache_enabled"] = False

    steps_values = [int(item.strip()) for item in args.steps.split(",") if item.strip()]
    batch_sizes = [int(item.strip()) for item in args.batch_sizes.split(",") if item.strip()]
    args.out.mkdir(parents=True, exist_ok=True)

    print("Benchmark settings")
    print(json.dumps(
        {
            "app_data_dir": str(app_data_dir),
            "reference_audio": str(base_settings.get("voxcpm_ref_audio") or ""),
            "clone_mode": str(base_settings.get("voxcpm_clone_mode") or "reference"),
            "texts": len(TEXTS),
            "steps": steps_values,
            "batch_sizes": batch_sizes,
            "flush_ms": args.flush_ms,
            "cache_enabled": False,
            "conversion_strategy": "lazy_mix",
        },
        ensure_ascii=False,
        indent=2,
    ))

    results: list[dict[str, Any]] = []
    for steps in steps_values:
        for batch_size in batch_sizes:
            profile_settings = dict(base_settings)
            profile_settings["voxcpm_model"] = str(base_settings.get("voxcpm_model") or "openbmb/VoxCPM2")
            profile_settings["voxcpm_device"] = str(base_settings.get("voxcpm_device") or "cuda:0")
            profile_settings["voxcpm_num_steps"] = steps
            profile_settings["voxcpm_batch_size"] = batch_size
            profile_settings["voxcpm_batch_flush_ms"] = args.flush_ms
            profile_name = f"steps{steps}_batch{batch_size}"
            print(f"\nRunning {profile_name} ...")
            report = run_profile(profile_name, settings=profile_settings, texts=TEXTS, out_dir=args.out)
            print(
                f"  synth_ms={report['synthesis_wall_time_ms']} total_ms={report['total_wall_time_ms']} "
                f"audio_sec={report['total_audio_duration_sec']} avg_rtf={report['avg_segment_rtf']}"
            )
            results.append(report)

    ranked = sorted(results, key=lambda item: (item["synthesis_wall_time_ms"], item["avg_segment_rtf"]))
    finalists = ranked[: max(1, min(args.finalists, len(ranked)))]
    baseline = next((item for item in ranked if item["num_steps"] == steps_values[0] and item["batch_size"] == batch_sizes[0]), None)
    if baseline is not None and baseline not in finalists:
        finalists = [baseline, *finalists[:-1]]

    print("\nRunning ASR quality proxy on finalists ...")
    for item in finalists:
        profile_dir = args.out / item["name"]
        audio_files = [profile_dir / f"seg_{index:02d}.wav" for index in range(len(TEXTS))]
        item["quality_proxy"] = asr_roundtrip(audio_files, TEXTS, settings=base_settings)
        quality = item["quality_proxy"]
        print(
            f"  {item['name']}: avg_CER={quality['avg_char_error_rate']} "
            f"avg_WER={quality['avg_word_error_rate']}"
        )

    summary = {
        "texts": TEXTS,
        "results": ranked,
        "finalists": finalists,
    }
    out_json = args.out / "summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nTop speed ranking")
    for index, item in enumerate(ranked[:5], start=1):
        quality = item.get("quality_proxy") or {}
        print(
            f"  {index}. {item['name']} "
            f"synth_ms={item['synthesis_wall_time_ms']} avg_rtf={item['avg_segment_rtf']} "
            f"avg_CER={quality.get('avg_char_error_rate')}"
        )
    print(f"\nSaved summary to {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
