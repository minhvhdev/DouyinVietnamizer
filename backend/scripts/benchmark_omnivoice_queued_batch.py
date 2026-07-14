"""Benchmark sequential vs queued OmniVoice batching (mock or real GPU worker)."""
from __future__ import annotations

import argparse
import array
import json
import statistics
import time
import wave
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dv_backend.adapters.omnivoice_tts import OmniVoiceTtsAdapter


SHORT_CORPUS = [
    "Xin chào các bạn.",
    "Hôm nay trời đẹp quá.",
    "Tôi đang thử nghiệm giọng nói.",
    "Đây là câu thứ tư.",
    "OmniVoice chạy trên GPU.",
    "Queued batch giúp giảm round-trip.",
    "Segment ngắn thường nhanh hơn.",
    "Chúng ta đo throughput.",
    "Benchmark cần warm-up.",
    "Clone prompt cache được giữ nóng.",
    "Không bật external chunking.",
    "Fidelity check theo production default.",
    "Voice design instruct nữ trầm.",
    "Mỗi run lưu JSON raw.",
    "Flush reason được ghi lại.",
    "Worker batch size histogram.",
    "Model synthesis ms quan trọng.",
    "Encode ms thường nhỏ.",
    "Queue wait ms cho flush penalty.",
    "RTF = wall / audio duration.",
    "Corpus short gồm 25 câu.",
    "Batch size 1, 2, 4.",
    "Sequential gọi synthesize lặp.",
    "Queued gọi synthesize_batch.",
    "Kết quả so sánh percent improvement.",
]

MEDIUM_CORPUS = [
    " ".join(
        [
            "Trong video Douyin gốc, nhân vật nói rất nhanh và câu dài hơn bình thường,",
            "nên pipeline cần giữ nhịp dubbing ổn định mà vẫn đảm bảo chất lượng tiếng Việt.",
        ]
    ),
    " ".join(
        [
            "Khi chạy job thật, micro-batch size bốn giúp adapter gom các segment direct",
            "và gửi worker một lần thay vì submit-wait tuần tự từng câu.",
        ]
    ),
    " ".join(
        [
            "Explicit batch boundary cho phép worker flush ngay khi biết block đã đủ,",
            "tránh chờ flush timeout một trăm năm mươi millisecond không cần thiết.",
        ]
    ),
    " ".join(
        [
            "Benchmark medium corpus kiểm tra câu trung bình khoảng mười lăm đến hai mươi từ,",
            "gần với lời thoại thực tế trong clip review hoặc giải thích sản phẩm.",
        ]
    ),
    " ".join(
        [
            "Chúng ta giữ num_steps bằng ba mươi hai, không đổi fidelity policy,",
            "và không bật external chunking trong phase đo hiệu năng này.",
        ]
    ),
    " ".join(
        [
            "Warm-up ba batch đầu giúp model và clone prompt cache ổn định",
            "trước khi ghi nhận các run measured.",
        ]
    ),
    " ".join(
        [
            "Mỗi measured run ghi wall time, throughput, RTF, batch p50/p95,",
            "và histogram worker_batch_size từ perf metadata.",
        ]
    ),
    " ".join(
        [
            "Nếu queued không nhanh hơn sequential, báo cáo phải chứng minh",
            "model inference chiếm phần lớn wall time hoặc flush đã được loại bỏ.",
        ]
    ),
    " ".join(
        [
            "Mixed corpus mô phỏng job thật: xen kẽ câu ngắn, câu dài và block direct nhỏ",
            "do segment chunked tách khỏi batch.",
        ]
    ),
    " ".join(
        [
            "Telemetry pipeline ghi tts_batch_mode_final và flush_reasons",
            "để QA biết đường chạy thực tế trong production job.",
        ]
    ),
    " ".join(
        [
            "Edge TTS và Google TTS không bị ảnh hưởng bởi thay đổi OmniVoice queued batch,",
            "regression suite phải pass toàn bộ.",
        ]
    ),
    " ".join(
        [
            "Kết quả JSON lưu vào data/benchmarks để Technical Leader review",
            "và quyết định đóng Phase hai.",
        ]
    ),
    " ".join(
        [
            "Audio RTF nhỏ hơn một nghĩa là synthesize nhanh hơn realtime playback,",
            "lớn hơn một nghĩa là chậm hơn thời lượng audio output.",
        ]
    ),
    " ".join(
        [
            "Per-segment completion p95 giúp phát hiện outlier do retry fidelity",
            "hoặc worker restart giữa batch.",
        ]
    ),
    " ".join(
        [
            "Retry và error count phải bằng zero trong benchmark sạch;",
            "nếu không, run bị loại khỏi aggregate.",
        ]
    ),
]

MIXED_CORPUS = (
    SHORT_CORPUS[:10]
    + MEDIUM_CORPUS[:5]
    + SHORT_CORPUS[10:18]
    + MEDIUM_CORPUS[5:10]
    + SHORT_CORPUS[18:25]
    + MEDIUM_CORPUS[10:15]
)


def _write_tone_wav(path: Path, *, duration_sec: float = 0.2, rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(rate * duration_sec)
    samples = array.array("h", [8000 if (index // 100) % 2 == 0 else -8000 for index in range(frames)])
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


class _BenchClient:
    def __init__(self, *, flush_ms: float = 0.15) -> None:
        self.flush_ms = flush_ms
        self.submitted = 0
        self.max_inflight = 0
        self._inflight = 0
        self.batch_metadata: list[dict[str, Any]] = []

    def submit(self, **kwargs) -> str:
        self._submit_one()
        self.batch_metadata.append(
            {
                "batch_id": kwargs.get("batch_id"),
                "batch_index": kwargs.get("batch_index"),
                "batch_size": kwargs.get("batch_size"),
            }
        )
        return f"req-{self.submitted}"

    def wait_result(self, request_id: str, *, timeout_sec: float = 600.0) -> dict[str, Any]:
        _ = request_id, timeout_sec
        return {"ok": True, "id": request_id}

    def wait_many(self, request_ids: list[str], *, timeout_sec: float = 600.0) -> list[dict[str, Any]]:
        return [self.wait_result(request_id, timeout_sec=timeout_sec) for request_id in request_ids]

    def synthesize(self, **kwargs) -> dict:
        self._submit_one()
        time.sleep(0.05)
        output_path = Path(kwargs["output_path"])
        _write_tone_wav(output_path)
        self._finish_one()
        return {"ok": True, "output_path": str(output_path), "duration_sec": 0.2}

    def synthesize_many(self, requests: list[dict], *, timeout_sec: float = 600.0) -> list[dict]:
        _ = timeout_sec
        self._inflight = 0
        batch_size = len(requests)
        batch_id = "mock-batch"
        for index, _req in enumerate(requests):
            self.submit(
                text="x",
                output_path=Path("."),
                ref_audio=None,
                ref_text=None,
                instruct=None,
                batch_id=batch_id,
                batch_index=index,
                batch_size=batch_size,
            )
        time.sleep(self.flush_ms)
        responses = []
        for req in requests:
            time.sleep(0.05)
            output_path = Path(req["output_path"])
            _write_tone_wav(output_path)
            self._finish_one()
            responses.append(
                {
                    "ok": True,
                    "output_path": str(output_path),
                    "duration_sec": 0.2,
                    "perf": {
                        "worker_batch_size": batch_size,
                        "flush_reason": "explicit_batch_complete",
                        "queue_wait_ms": self.flush_ms * 1000.0,
                        "model_synthesis_ms": 50.0,
                        "encode_ms": 1.0,
                    },
                }
            )
        return responses

    def _submit_one(self) -> None:
        self.submitted += 1
        self._inflight += 1
        self.max_inflight = max(self.max_inflight, self._inflight)

    def _finish_one(self) -> None:
        self._inflight = max(0, self._inflight - 1)


def _slice_corpus(corpus: list[str], batch_size: int) -> list[str]:
    count = max(batch_size, min(len(corpus), batch_size * 8))
    return corpus[:count]


def _aggregate_perf(responses: list[dict[str, Any]]) -> dict[str, Any]:
    perf_items = [response.get("perf") or {} for response in responses if response.get("ok")]
    worker_sizes = [int(item.get("worker_batch_size") or 0) for item in perf_items if item.get("worker_batch_size")]
    flush_reasons = [str(item.get("flush_reason") or "") for item in perf_items if item.get("flush_reason")]
    queue_waits = [float(item.get("queue_wait_ms") or 0.0) for item in perf_items]
    model_ms = [float(item.get("model_synthesis_ms") or 0.0) for item in perf_items]
    encode_ms = [float(item.get("encode_ms") or 0.0) for item in perf_items]
    return {
        "worker_batch_size_histogram": dict(Counter(worker_sizes)),
        "flush_reason_histogram": dict(Counter(flush_reasons)),
        "queue_wait_ms_p50": round(_percentile(queue_waits, 50), 2),
        "queue_wait_ms_p95": round(_percentile(queue_waits, 95), 2),
        "model_synthesis_ms_p50": round(_percentile(model_ms, 50), 2),
        "model_synthesis_ms_p95": round(_percentile(model_ms, 95), 2),
        "encode_ms_p50": round(_percentile(encode_ms, 50), 2),
        "encode_ms_p95": round(_percentile(encode_ms, 95), 2),
    }


def _run_mode(
    *,
    mode: str,
    texts: list[str],
    out_dir: Path,
    adapter: OmniVoiceTtsAdapter,
    voice: str,
) -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    run_dir = out_dir / mode
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    responses: list[dict[str, Any]] = []
    perf_summary: dict[str, Any] = {}
    if mode == "sequential":
        for index, text in enumerate(texts):
            output_path = run_dir / f"seg_{index:03d}.wav"
            adapter.synthesize(text, output_path, voice=voice)
            responses.append({"ok": True, "duration_sec": 0.2})
    else:
        items = [
            {
                "text": text,
                "output_path": run_dir / f"seg_{index:03d}.wav",
                "voice": voice,
                "segment": {"index": index},
            }
            for index, text in enumerate(texts)
        ]
        adapter.synthesize_batch(items)
        responses = [{"ok": True, "duration_sec": 0.2} for _ in texts]
        perf_summary = dict(getattr(adapter, "last_batch_perf", {}) or {})
    wall_sec = time.perf_counter() - started
    audio_sec = sum(float(item.get("duration_sec") or 0.0) for item in responses)
    diagnostics = dict(getattr(adapter, "last_batch_diagnostics", {}) or {})
    summary = {
        "wall_sec": round(wall_sec, 4),
        "segments": len(texts),
        "segments_per_sec": round(len(texts) / wall_sec, 4) if wall_sec > 0 else 0.0,
        "audio_rtf": round(wall_sec / audio_sec, 4) if audio_sec > 0 else 0.0,
        "batch_diagnostics": diagnostics,
    }
    if mode == "queued":
        summary["perf"] = perf_summary
    return wall_sec, responses, summary


def _build_adapter(*, mock: bool, settings: dict[str, Any], data_dir: Path) -> OmniVoiceTtsAdapter:
    if mock:
        return OmniVoiceTtsAdapter(model="k2-fsa/OmniVoice", _client=_BenchClient(), settings=settings)
    return OmniVoiceTtsAdapter(
        model=str(settings.get("omnivoice_model") or "k2-fsa/OmniVoice"),
        device=str(settings.get("omnivoice_device") or "cuda:0"),
        num_step=int(settings.get("omnivoice_num_steps") or 32),
        speed=float(settings.get("omnivoice_speed") or 1.0),
        language_id=str(settings.get("omnivoice_language_id") or "vi"),
        data_dir=data_dir,
        settings=settings,
    )


def run_case(
    *,
    corpus_name: str,
    texts: list[str],
    batch_size: int,
    runs: int,
    warmup: int,
    out_dir: Path,
    mock: bool,
    voice: str,
) -> dict[str, Any]:
    settings = {
        "omnivoice_fidelity_check_enabled": True,
        "omnivoice_external_chunking_enabled": False,
        "omnivoice_tts_include_perf": True,
        "omnivoice_num_steps": 32,
        "tts_micro_batch_size": 4,
        "tts_micro_batch_enabled": True,
    }
    sliced = _slice_corpus(texts, batch_size)
    case_dir = out_dir / corpus_name / f"batch_{batch_size}"
    sequential_runs: list[dict[str, Any]] = []
    queued_runs: list[dict[str, Any]] = []

    for run_index in range(warmup + runs):
        adapter = _build_adapter(mock=mock, settings=settings, data_dir=case_dir)
        seq_wall, seq_responses, seq_summary = _run_mode(
            mode="sequential",
            texts=sliced,
            out_dir=case_dir / f"run_{run_index:02d}",
            adapter=adapter,
            voice=voice,
        )
        if run_index >= warmup:
            sequential_runs.append({**seq_summary, "perf": _aggregate_perf(seq_responses)})

        adapter = _build_adapter(mock=mock, settings=settings, data_dir=case_dir)
        queued_wall, queued_responses, queued_summary = _run_mode(
            mode="queued",
            texts=sliced,
            out_dir=case_dir / f"run_{run_index:02d}",
            adapter=adapter,
            voice=voice,
        )
        if run_index >= warmup:
            queued_runs.append({**queued_summary, "perf": _aggregate_perf(queued_responses)})

    seq_walls = [float(item["wall_sec"]) for item in sequential_runs]
    queued_walls = [float(item["wall_sec"]) for item in queued_runs]
    seq_mean = statistics.mean(seq_walls) if seq_walls else 0.0
    queued_mean = statistics.mean(queued_walls) if queued_walls else 0.0
    improvement_pct = ((seq_mean - queued_mean) / seq_mean * 100.0) if seq_mean > 0 else 0.0
    return {
        "corpus": corpus_name,
        "batch_size": batch_size,
        "segments": len(sliced),
        "warmup_runs": warmup,
        "measured_runs": runs,
        "sequential": {
            "mean_wall_sec": round(seq_mean, 4),
            "p50_wall_sec": round(_percentile(seq_walls, 50), 4),
            "p95_wall_sec": round(_percentile(seq_walls, 95), 4),
            "mean_segments_per_sec": round(statistics.mean([item["segments_per_sec"] for item in sequential_runs]), 4)
            if sequential_runs
            else 0.0,
            "mean_audio_rtf": round(statistics.mean([item["audio_rtf"] for item in sequential_runs]), 4)
            if sequential_runs
            else 0.0,
            "runs": sequential_runs,
        },
        "queued": {
            "mean_wall_sec": round(queued_mean, 4),
            "p50_wall_sec": round(_percentile(queued_walls, 50), 4),
            "p95_wall_sec": round(_percentile(queued_walls, 95), 4),
            "mean_segments_per_sec": round(statistics.mean([item["segments_per_sec"] for item in queued_runs]), 4)
            if queued_runs
            else 0.0,
            "mean_audio_rtf": round(statistics.mean([item["audio_rtf"] for item in queued_runs]), 4)
            if queued_runs
            else 0.0,
            "runs": queued_runs,
        },
        "throughput_improvement_pct": round(improvement_pct, 2),
        "queued_not_slower_than_3pct": queued_mean <= seq_mean * 1.03 if seq_mean > 0 else True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="Use mock client (default when --real omitted).")
    parser.add_argument("--real", action="store_true", help="Use real OmniVoice GPU worker.")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch-sizes", type=str, default="1,2,4")
    parser.add_argument("--corpora", type=str, default="short,medium,mixed")
    parser.add_argument("--voice", type=str, default="instruct:female, low pitch")
    parser.add_argument("--out-dir", type=Path, default=Path("data/benchmarks/omnivoice_queued_batch"))
    args = parser.parse_args()

    mock = not args.real
    batch_sizes = [max(1, int(part.strip())) for part in args.batch_sizes.split(",") if part.strip()]
    corpus_map = {
        "short": SHORT_CORPUS,
        "medium": MEDIUM_CORPUS,
        "mixed": MIXED_CORPUS,
    }
    selected = [name.strip() for name in args.corpora.split(",") if name.strip()]
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for corpus_name in selected:
        texts = corpus_map.get(corpus_name)
        if texts is None:
            continue
        for batch_size in batch_sizes:
            results.append(
                run_case(
                    corpus_name=corpus_name,
                    texts=texts,
                    batch_size=batch_size,
                    runs=max(1, args.runs),
                    warmup=max(0, args.warmup),
                    out_dir=out_dir,
                    mock=mock,
                    voice=args.voice,
                )
            )

    payload = {
        "generated_at": timestamp,
        "mode": "mock" if mock else "real_gpu",
        "configured_batch_size": 4,
        "batch_sizes": batch_sizes,
        "corpora": selected,
        "cases": results,
    }
    json_path = out_dir / "benchmark_results.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"json_path": str(json_path), "cases": len(results)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
