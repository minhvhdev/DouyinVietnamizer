"""Benchmark OmniVoice num_steps with speed and fidelity quality metrics."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import statistics
import time
import uuid
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure ASR vendor tools resolve for fidelity checks when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("DV_VENDOR_DIR", str(_REPO_ROOT / "vendor"))

from dv_backend.adapters.omnivoice_tts import OmniVoiceTtsAdapter
from dv_backend.omnivoice_steps_eval import (
    evaluate_quality_gate,
    extract_segment_row,
    percentile,
    recommend_steps,
    summarize_by_group,
)

STEP_LEVELS = (32, 24, 20, 16)

QUALITY_CORPUS: list[dict[str, str]] = [
    {"group": "short", "text": "Xin chào các bạn."},
    {"group": "short", "text": "Hôm nay trời đẹp quá."},
    {"group": "short", "text": "Tôi đang thử nghiệm giọng nói."},
    {"group": "short", "text": "OmniVoice chạy trên GPU."},
    {"group": "short", "text": "Queued batch giúp giảm round-trip."},
    {"group": "short", "text": "Segment ngắn thường nhanh hơn."},
    {"group": "short", "text": "Chúng ta đo throughput."},
    {"group": "short", "text": "Benchmark cần warm-up."},
    {"group": "short", "text": "Clone prompt cache được giữ nóng."},
    {"group": "short", "text": "Không bật external chunking."},
    {"group": "short", "text": "Fidelity check theo production default."},
    {"group": "short", "text": "Voice design instruct nữ trầm."},
    {"group": "short", "text": "Mỗi run lưu JSON raw."},
    {"group": "short", "text": "Flush reason được ghi lại."},
    {"group": "short", "text": "Worker batch size histogram."},
    {"group": "medium", "text": " ".join(["Trong video Douyin gốc, nhân vật nói rất nhanh và câu dài hơn bình thường,", "nên pipeline cần giữ nhịp dubbing ổn định mà vẫn đảm bảo chất lượng tiếng Việt."])},
    {"group": "medium", "text": " ".join(["Khi chạy job thật, micro-batch size bốn giúp adapter gom các segment direct", "và gửi worker một lần thay vì submit-wait tuần tự từng câu."])},
    {"group": "medium", "text": " ".join(["Explicit batch boundary cho phép worker flush ngay khi biết block đã đủ,", "tránh chờ flush timeout một trăm năm mươi millisecond không cần thiết."])},
    {"group": "medium", "text": " ".join(["Benchmark medium corpus kiểm tra câu trung bình khoảng mười lăm đến hai mươi từ,", "gần với lời thoại thực tế trong clip review hoặc giải thích sản phẩm."])},
    {"group": "medium", "text": " ".join(["Chúng ta giữ fidelity policy production, không bật external chunking", "trong phase đo num_steps này."])},
    {"group": "medium", "text": " ".join(["Warm-up ba batch đầu giúp model và clone prompt cache ổn định", "trước khi ghi nhận các run measured."])},
    {"group": "medium", "text": " ".join(["Mỗi measured run ghi wall time, throughput, RTF, batch p50/p95,", "và histogram worker_batch_size từ perf metadata."])},
    {"group": "medium", "text": " ".join(["Nếu queued không nhanh hơn sequential, báo cáo phải chứng minh", "model inference chiếm phần lớn wall time hoặc flush đã được loại bỏ."])},
    {"group": "medium", "text": " ".join(["Mixed corpus mô phỏng job thật: xen kẽ câu ngắn, câu dài và block direct nhỏ", "do segment chunked tách khỏi batch."])},
    {"group": "medium", "text": " ".join(["Telemetry pipeline ghi tts_batch_mode_final và flush_reasons", "để QA biết đường chạy thực tế trong production job."])},
    {"group": "medium", "text": " ".join(["Edge TTS và Google TTS không bị ảnh hưởng bởi thay đổi OmniVoice queued batch,", "regression suite phải pass toàn bộ."])},
    {"group": "medium", "text": " ".join(["Kết quả JSON lưu vào data/benchmarks để Technical Leader review", "và quyết định mức num_steps tối ưu."])},
    {"group": "medium", "text": " ".join(["Audio RTF nhỏ hơn một nghĩa là synthesize nhanh hơn realtime playback,", "lớn hơn một nghĩa là chậm hơn thời lượng audio output."])},
    {"group": "medium", "text": " ".join(["Per-segment completion p95 giúp phát hiện outlier do retry fidelity", "hoặc worker restart giữa batch."])},
    {"group": "medium", "text": " ".join(["Retry và error count phải bằng zero trong benchmark sạch;", "nếu không, run bị loại khỏi aggregate."])},
    {"group": "long", "text": " ".join(["Theo báo cáo của Bộ Y tế, tỷ lệ tiêm chủng đạt 95,7% tại thành phố Hồ Chí Minh", "vào tháng 3 năm 2026, cao hơn mức trung bình cả nước khoảng 2,3 điểm phần trăm."])},
    {"group": "long", "text": " ".join(["Khi review sản phẩm trên Douyin, creator thường nói: 'Các bạn nhìn này, chất liệu cotton 100%,", "giặt máy thoải mái, size M cân nặng từ 50 đến 58 kilogram đều mặc vừa.'"])},
    {"group": "long", "text": " ".join(["Nguyễn Văn An, Trần Thị Bích và Lê Hoàng Nam sẽ tham dự hội thảo AI tại Hà Nội,", "Tokyo và Singapore vào tuần tới — bạn có muốn đăng ký sớm không?"])},
    {"group": "long", "text": " ".join(["Wow, deal này giảm tới 40% luôn á!", "Nhưng mà ship về Việt Nam mất bao lâu, và có được đổi trả trong 7 ngày không bạn?"])},
    {"group": "long", "text": " ".join(["Nếu bạn muốn dubbing tự nhiên hơn, hãy chia câu dài thành hai segment ngắn,", "giữ dấu câu đầy đủ, và tránh gom quá nhiều số liệu vào một câu duy nhất."])},
    {"group": "long", "text": " ".join(["Trong thử nghiệm A/B, pipeline mới giảm 10,29% thời gian TTS so với sequential,", "trong khi fidelity similarity trung bình chỉ giảm 0,003 so với baseline 32 steps."])},
    {"group": "long", "text": " ".join(["Câu hỏi quan trọng là: liệu 16 steps có đủ cho tiếng Việt có dấu, tên riêng,", "và các từ vay mượn tiếng Anh như benchmark, throughput, fidelity hay không?"])},
    {"group": "long", "text": " ".join(["Blind listening evaluation yêu cầu ít nhất mười câu ngắn, mười câu vừa,", "và năm câu dài, bao gồm mọi sample có fidelity retry hoặc score thấp."])},
    {"group": "long", "text": " ".join(["Đừng quên kiểm tra missing-number: nếu target có '95,7%' hoặc '2026',", "ASR output phải giữ lại các con số quan trọng, không được cắt cụt cuối câu."])},
    {"group": "long", "text": " ".join(["Prosody và timing rất quan trọng cho dubbing Douyin:", "câu cuối phải kết thúc trọn vẹn, không bị nuốt âm cuối hoặc rè loa."])},
]


def _wav_duration(path: Path) -> float:
    if not path.is_file():
        return 0.0
    with wave.open(str(path), "rb") as handle:
        rate = int(handle.getframerate() or 0)
        frames = int(handle.getnframes())
        if rate <= 0:
            return 0.0
        return frames / float(rate)


def build_blind_manifest(
    *,
    out_dir: Path,
    steps_levels: list[int],
    corpus: list[dict[str, str]],
    seed: int = 42,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    entries: list[dict[str, Any]] = []
    for steps in steps_levels:
        for index, item in enumerate(corpus):
            blind_id = f"sample_{uuid.uuid4().hex[:8]}"
            entries.append(
                {
                    "blind_id": blind_id,
                    "num_steps": steps,
                    "segment_index": index,
                    "group": item["group"],
                    "target_text": item["text"],
                    "audio_path": str(audio_dir / f"{blind_id}.wav"),
                }
            )
    rng.shuffle(entries)
    manifest_path = out_dir / "listening_manifest_blind.csv"
    key_path = out_dir / "listening_key.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["blind_id", "audio_path", "group", "target_text"])
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "blind_id": entry["blind_id"],
                    "audio_path": entry["audio_path"],
                    "group": entry["group"],
                    "target_text": entry["target_text"],
                }
            )
    with key_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["blind_id", "num_steps", "segment_index", "group"])
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "blind_id": entry["blind_id"],
                    "num_steps": entry["num_steps"],
                    "segment_index": entry["segment_index"],
                    "group": entry["group"],
                }
            )
    listening_template = out_dir / "listening_scores_template.csv"
    with listening_template.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "blind_id",
                "text_correctness",
                "pronunciation",
                "voice_similarity",
                "naturalness",
                "prosody_timing",
                "audio_artifacts",
                "notes",
            ],
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow({"blind_id": entry["blind_id"]})
    return manifest_path, key_path, entries


def populate_listening_audio(
    *,
    out_dir: Path,
    manifest_entries: list[dict[str, Any]],
    num_steps: int,
    source_run_dir: Path,
) -> int:
    copied = 0
    for entry in manifest_entries:
        if int(entry["num_steps"]) != num_steps:
            continue
        source = source_run_dir / f"seg_{int(entry['segment_index']):03d}.wav"
        target = Path(entry["audio_path"])
        if not source.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
    return copied


def run_steps_benchmark(
    *,
    num_steps: int,
    corpus: list[dict[str, str]],
    out_dir: Path,
    runs: int,
    warmup: int,
    voice: str,
    mock: bool,
) -> dict[str, Any]:
    from scripts.benchmark_omnivoice_queued_batch import _BenchClient

    settings = {
        "omnivoice_fidelity_check_enabled": True,
        "omnivoice_fidelity_check_all_segments": True,
        "omnivoice_external_chunking_enabled": False,
        "omnivoice_tts_include_perf": True,
        "omnivoice_num_steps": num_steps,
        "tts_micro_batch_size": 4,
        "tts_micro_batch_enabled": True,
    }
    measured_runs: list[dict[str, Any]] = []
    segment_results: list[dict[str, Any]] = []
    last_measured_run_dir: Path | None = None

    for run_index in range(warmup + runs):
        if mock:
            adapter = OmniVoiceTtsAdapter(model="k2-fsa/OmniVoice", _client=_BenchClient(), settings=settings)
        else:
            adapter = OmniVoiceTtsAdapter(
                model="k2-fsa/OmniVoice",
                device="cuda:0",
                num_step=num_steps,
                data_dir=out_dir,
                settings=settings,
            )
        run_dir = out_dir / f"steps_{num_steps}" / f"run_{run_index:02d}"
        items = []
        for index, entry in enumerate(corpus):
            segment: dict[str, Any] = {"index": index, "group": entry["group"]}
            items.append(
                {
                    "text": entry["text"],
                    "output_path": run_dir / f"seg_{index:03d}.wav",
                    "voice": voice,
                    "segment": segment,
                }
            )
        started = time.perf_counter()
        adapter.synthesize_batch(items)
        wall_sec = time.perf_counter() - started
        if run_index < warmup:
            continue
        last_measured_run_dir = run_dir
        audio_durations = [_wav_duration(Path(item["output_path"])) for item in items]
        audio_sec = sum(audio_durations)
        perf = dict(getattr(adapter, "last_batch_perf", {}) or {})
        diagnostics = dict(getattr(adapter, "last_batch_diagnostics", {}) or {})
        similarities = [
            float(segment.get("tts_text_similarity"))
            for item in items
            if isinstance((segment := item.get("segment", {})).get("tts_text_similarity"), (int, float))
        ]
        retries = sum(int(item.get("segment", {}).get("tts_chunk_retry_count") or 0) for item in items)
        failures = sum(1 for item in items if item.get("segment", {}).get("tts_fidelity_status") == "failed")
        measured_runs.append(
            {
                "run_index": run_index,
                "wall_sec": round(wall_sec, 4),
                "segments_per_sec": round(len(corpus) / wall_sec, 4) if wall_sec > 0 else 0.0,
                "audio_rtf": round(wall_sec / audio_sec, 4) if audio_sec > 0 else 0.0,
                "audio_duration_sec": round(audio_sec, 4),
                "retry_count": retries,
                "failure_count": failures,
                "mean_fidelity_similarity": round(statistics.mean(similarities), 4) if similarities else None,
                "p5_fidelity_similarity": round(percentile(similarities, 5), 4) if similarities else None,
                "batch_diagnostics": diagnostics,
                "perf": perf,
            }
        )
        for index, item in enumerate(items):
            segment = item.get("segment") or {}
            segment_results.append(
                extract_segment_row(
                    num_steps=num_steps,
                    run_index=run_index,
                    segment_index=index,
                    group=str(entry["group"]) if (entry := corpus[index]) else "unknown",
                    target_text=str(item["text"]),
                    output_path=Path(item["output_path"]),
                    segment=segment,
                    audio_duration_sec=_wav_duration(Path(item["output_path"])),
                )
            )

    walls = [float(run["wall_sec"]) for run in measured_runs]
    rtfs = [float(run["audio_rtf"]) for run in measured_runs if run.get("audio_rtf")]
    model_ms = [
        float((run.get("perf") or {}).get("model_synthesis_ms_mean") or 0.0)
        for run in measured_runs
        if (run.get("perf") or {}).get("model_synthesis_ms_mean")
    ]
    return {
        "num_steps": num_steps,
        "warmup_runs": warmup,
        "measured_runs": runs,
        "wall_mean_sec": round(statistics.mean(walls), 4) if walls else 0.0,
        "wall_p50_sec": round(percentile(walls, 50), 4),
        "wall_p95_sec": round(percentile(walls, 95), 4),
        "segments_per_sec_mean": round(statistics.mean([run["segments_per_sec"] for run in measured_runs]), 4)
        if measured_runs
        else 0.0,
        "audio_rtf_mean": round(statistics.mean(rtfs), 4) if rtfs else None,
        "model_synthesis_ms_mean": round(statistics.mean(model_ms), 2) if model_ms else None,
        "retry_count_mean": round(statistics.mean([run["retry_count"] for run in measured_runs]), 2)
        if measured_runs
        else 0.0,
        "failure_count_total": sum(run["failure_count"] for run in measured_runs),
        "missing_number_total": sum(1 for row in segment_results if row.get("missing_number")),
        "truncated_ending_total": sum(1 for row in segment_results if row.get("truncated_ending")),
        "mean_fidelity_similarity": round(
            statistics.mean([run["mean_fidelity_similarity"] for run in measured_runs if run["mean_fidelity_similarity"] is not None]),
            4,
        )
        if any(run["mean_fidelity_similarity"] is not None for run in measured_runs)
        else None,
        "p5_fidelity_similarity": round(
            statistics.mean([run["p5_fidelity_similarity"] for run in measured_runs if run["p5_fidelity_similarity"] is not None]),
            4,
        )
        if any(run["p5_fidelity_similarity"] is not None for run in measured_runs)
        else None,
        "runs": measured_runs,
        "segment_results": segment_results,
        "last_measured_run_dir": str(last_measured_run_dir) if last_measured_run_dir else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--steps", type=str, default="32,24,20,16")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--voice", type=str, default="instruct:female, low pitch")
    parser.add_argument("--out-dir", type=Path, default=Path("data/benchmarks/omnivoice_steps"))
    args = parser.parse_args()

    mock = not args.real
    step_levels = [int(part.strip()) for part in args.steps.split(",") if part.strip()]
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    _manifest_path, _key_path, manifest_entries = build_blind_manifest(
        out_dir=out_dir,
        steps_levels=step_levels,
        corpus=QUALITY_CORPUS,
    )
    all_segment_results: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for steps in step_levels:
        case = run_steps_benchmark(
            num_steps=steps,
            corpus=QUALITY_CORPUS,
            out_dir=out_dir,
            runs=max(1, args.runs),
            warmup=max(0, args.warmup),
            voice=args.voice,
            mock=mock,
        )
        segment_results = list(case.pop("segment_results", []))
        all_segment_results.extend(segment_results)
        case["group_summary"] = summarize_by_group(segment_results)
        last_run_dir = case.pop("last_measured_run_dir", None)
        if last_run_dir:
            populate_listening_audio(
                out_dir=out_dir,
                manifest_entries=manifest_entries,
                num_steps=steps,
                source_run_dir=Path(last_run_dir),
            )
        results.append(case)

    baseline = next((item for item in results if item["num_steps"] == 32), None)
    baseline_issues = (
        {
            "missing_number": int(baseline.get("missing_number_total") or 0),
            "truncated_ending": int(baseline.get("truncated_ending_total") or 0),
        }
        if baseline
        else {}
    )
    summary_rows = []
    for item in results:
        speedup = None
        if baseline and item["wall_mean_sec"] > 0:
            speedup = round((baseline["wall_mean_sec"] - item["wall_mean_sec"]) / baseline["wall_mean_sec"] * 100.0, 2)
        item["speedup_vs_32_pct"] = speedup
        if baseline and item["num_steps"] != 32:
            item["quality_gate"] = evaluate_quality_gate(
                candidate=item,
                baseline=baseline,
                baseline_issues=baseline_issues,
            )
        else:
            item["quality_gate"] = {"passed": True, "violations": [], "baseline": True}
        summary_rows.append(
            {
                "num_steps": item["num_steps"],
                "wall_mean_sec": item["wall_mean_sec"],
                "segments_per_sec": item["segments_per_sec_mean"],
                "audio_rtf_mean": item.get("audio_rtf_mean"),
                "speedup_vs_32_pct": speedup,
                "mean_fidelity": item["mean_fidelity_similarity"],
                "p5_fidelity": item["p5_fidelity_similarity"],
                "retry_mean": item["retry_count_mean"],
                "failure_total": item["failure_count_total"],
                "missing_number_total": item.get("missing_number_total"),
                "quality_gate_passed": item["quality_gate"]["passed"],
            }
        )

    segment_jsonl = out_dir / "segment_results.jsonl"
    with segment_jsonl.open("w", encoding="utf-8") as handle:
        for row in all_segment_results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    recommendation = recommend_steps(results)
    payload = {
        "generated_at": timestamp,
        "mode": "mock" if mock else "real_gpu",
        "corpus_size": len(QUALITY_CORPUS),
        "configured_batch_size": 4,
        "step_levels": step_levels,
        "cases": results,
        "summary": summary_rows,
        "recommendation": recommendation,
    }
    json_path = out_dir / "benchmark_results.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["num_steps"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(
        json.dumps(
            {
                "json_path": str(json_path),
                "summary_path": str(summary_path),
                "segment_jsonl": str(segment_jsonl),
                "recommendation": recommendation,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
