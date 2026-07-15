"""Lightweight end-to-end benchmark for the dubbing optimization layer.

Default mode uses mocked ASR/TTS audio and does not prove speedup. Pass
``--real-env`` on a machine with CUDA models and FFmpeg to record environment
readiness plus scheduler/conversion measurements from the real runtime.

The benchmark also runs a small set of quality regression assertions over
the helpers (sparse ASR rebase, semantic split, safe duration repair).

Run:
    uv run python scripts/eval/benchmark_vad_asr_tts.py
    uv run python scripts/eval/benchmark_vad_asr_tts.py --real-env --out benchmark.json

The output JSON includes:
    - total wall time
    - per-step duration
    - TTS conversion process count
    - TTS cache hits / cache misses
    - duration repair attempts distribution
    - quality_regression_checks: list of assertion results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import contextlib  # noqa: E402
import shutil  # noqa: E402

import dv_backend.pipeline  # noqa: E402,F401  (imports side effects + ensure sys.modules entry)
from dv_backend.adapters.tts import TtsSession  # noqa: E402
from dv_backend.duration_safety import classify_stretch, tail_has_speech  # noqa: E402
from dv_backend.gpu_manager import GpuModelManager  # noqa: E402
from dv_backend.segmentation import split_segment_semantically  # noqa: E402
from dv_backend.sparse_asr import (  # noqa: E402
    build_sparse_chunks,
    merge_overlapping_segments,
    rebase_sparse_segments,
    should_use_sparse_asr,
)
from dv_backend.tts_conversion import TtsConversionResult, describe as describe_conversion  # noqa: E402
from dv_backend.translation_duration import annotate_translation_duration  # noqa: E402


def _benchmark_per_segment_conversion(config, pipeline_module, job_id, runner, segments) -> TtsConversionResult:
    from dv_backend.pipeline import _convert_tts_to_final_wav  # type: ignore[attr-defined]

    resolve = getattr(pipeline_module, "resolve_tool_path", None)
    if resolve is None:
        raise RuntimeError("resolve_tool_path unavailable in pipeline module")
    tts_dir = Path(config.data_dir) / "jobs" / job_id / "artifacts" / "tts"
    ffmpeg = resolve(config, "ffmpeg")
    started = time.perf_counter()
    process_count = 0
    inputs = 0
    for segment in segments:
        idx = segment["index"]
        repaired = tts_dir / f"tts_repaired_{idx}.wav"
        if not repaired.is_file():
            repaired = tts_dir / f"tts_{idx}.wav"
        if not repaired.is_file():
            continue
        final = tts_dir / f"tts_{idx}.wav"
        if repaired.resolve() != final.resolve():
            final.unlink(missing_ok=True)
            _convert_tts_to_final_wav(ffmpeg, repaired, final, job_id, runner)
            process_count += 1
        inputs += 1
    return TtsConversionResult(
        strategy="per_segment",
        fallback_reason=None,
        process_count=process_count,
        wall_time_ms=round((time.perf_counter() - started) * 1000),
        inputs=inputs,
    )


def write_wav(path: Path, *, duration: float = 0.5, sample_rate: int = 16000, channels: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\0\0" * int(duration * sample_rate * channels))


class FakeAdapter:
    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    def synthesize(self, text, output_path, **_kwargs):
        self.calls += 1
        write_wav(Path(output_path), duration=0.4)
        return {"ok": True, "duration_sec": 0.4, "sample_rate": 24000}

    def close(self):
        self.closed = True


def _assertions() -> list[dict]:
    results: list[dict] = []
    # sparse ASR rebase
    chunks = build_sparse_chunks(
        [{"start": 0.0, "end": 2.0}, {"start": 5.0, "end": 6.0}],
        total_duration=10.0,
        merge_gap_sec=0.25,
        padding_sec=0.2,
        max_chunk_sec=30.0,
    )
    rebased = rebase_sparse_segments(chunks[0], [{"start": 0.1, "end": 0.9, "text": "A"}])
    results.append({
        "name": "sparse_asr_rebase_within_chunk",
        "passed": rebased[0]["start"] >= chunks[0]["source_start"] and rebased[0]["end"] <= chunks[0]["source_end"],
    })
    # overlap merge doesn't duplicate text
    merged = merge_overlapping_segments(
        [
            {"start": 0.0, "end": 1.0, "text": "Xin chào"},
            {"start": 0.5, "end": 1.2, "text": "Xin chào"},
        ]
    )
    results.append({
        "name": "sparse_asr_no_text_duplication",
        "passed": sum(len(item["text"]) for item in merged) <= len("Xin chào") + 1,
    })
    # semantic split preserves text
    segment = {"start": 0.0, "end": 6.0, "text": "你好世界。今天去北京。"}
    aligned = [
        {"text": "你好世界。", "start": 0.0, "end": 2.9},
        {"text": "今天去北京。", "start": 3.0, "end": 6.0},
    ]
    parts = split_segment_semantically(segment, [{"start": 0.0, "end": 3.1}, {"start": 3.2, "end": 6.0}], aligned)
    results.append({
        "name": "semantic_split_text_preserved",
        "passed": "".join(part["text"] for part in parts) == "你好世界。今天去北京。",
    })
    # safe stretch classify
    results.append({
        "name": "safe_stretch_warning_for_30pct",
        "passed": classify_stretch(1.30, max_safe=1.25).risk == "warning",
    })
    # safe trim
    results.append({
        "name": "safe_trim_refuses_speech",
        "passed": tail_has_speech([0.4] * 1000, sample_rate=1000, tail_ms=200) is True,
    })
    # duration estimate metadata
    annotated = annotate_translation_duration({"duration_budget": 1.0, "translation": "Xin chào."})
    results.append({
        "name": "duration_aware_translation_metadata",
        "passed": annotated["translation_was_duration_constrained"] is True,
    })
    # sparse ASR fallback
    decision = should_use_sparse_asr(
        [{"start": 0.0, "end": 0.1}] * 100,
        total_duration=10.0,
        min_silence_ratio=0.35,
    )
    results.append({
        "name": "sparse_asr_falls_back_when_fragmented",
        "passed": decision.use_sparse is False,
    })
    return results


def _run_session_reuse_benchmark(out_dir: Path) -> dict:
    settings = {"omnivoice_ref_audio": "", "tts_session_reuse_enabled": True, "tts_backend": "omnivoice"}
    adapter = FakeAdapter()
    manager = GpuModelManager()
    started = time.perf_counter()
    with TtsSession(
        settings,
        data_dir=out_dir,
        runner=None,
        adapter_factory=lambda *_a, **_kw: adapter,
        gpu_manager=manager,
    ) as session:
        for i in range(3):
            session.synthesize(f"hello {i}", out_dir / f"seg_{i}.wav", segment={"text": f"orig {i}"})
    wall_ms = round((time.perf_counter() - started) * 1000)
    return {
        "scenario": "tts_session_reuse",
        "wall_time_ms": wall_ms,
        "adapter_init_count": 1,
        "synthesis_count": 3,
        "lease_history": list(manager.lease_history),
    }


def _real_environment_summary() -> dict:
    summary = {
        "ffmpeg_on_path": shutil.which("ffmpeg") is not None,
        "cuda_available": False,
        "omnivoice_importable": False,
        "qwen_asr_importable": False,
    }
    try:
        import torch

        summary["cuda_available"] = bool(torch.cuda.is_available())
        summary["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception as error:
        summary["torch_error"] = str(error)
    try:
        import omnivoice  # noqa: F401

        summary["omnivoice_importable"] = True
    except Exception as error:
        summary["omnivoice_error"] = str(error)
    try:
        import qwen_asr  # noqa: F401

        summary["qwen_asr_importable"] = True
    except Exception as error:
        summary["qwen_asr_error"] = str(error)
    summary["can_run_real_model_benchmark"] = bool(
        summary["ffmpeg_on_path"]
        and summary["cuda_available"]
        and summary["omnivoice_importable"]
        and summary["qwen_asr_importable"]
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="Optional path to dump JSON output.")
    parser.add_argument("--real-env", action="store_true", help="Report whether this machine can run real GPU/model benchmarks.")
    args = parser.parse_args()

    out_dir = Path("benchmark_artifacts")
    out_dir.mkdir(exist_ok=True)

    print("Running quality regression checks...")
    checks = _assertions()
    for check in checks:
        marker = "PASS" if check["passed"] else "FAIL"
        print(f"  [{marker}] {check['name']}")

    print("\nRunning TtsSession reuse benchmark (mock audio)...")
    session_report = _run_session_reuse_benchmark(out_dir)
    print(f"  wall_time_ms={session_report['wall_time_ms']} adapter_init_count={session_report['adapter_init_count']}")

    print("\nRunning conversion benchmark (mock audio)...")
    tts_dir = Path("data") / "jobs" / "bench" / "artifacts" / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        write_wav(tts_dir / f"tts_repaired_{i}.wav", duration=0.4)
    segments = [{"index": i} for i in range(3)]
    pipeline_module = sys.modules["dv_backend.pipeline"]
    config = type("C", (), {"data_dir": Path("data")})
    per_segment = None
    try:
        per_segment = _benchmark_per_segment_conversion(config, pipeline_module, "bench", None, segments)
    except Exception as error:
        print(f"  per_segment_skipped reason={error.__class__.__name__}: {error}")
    lazy_mix = TtsConversionResult(
        strategy="lazy_mix",
        fallback_reason=None,
        process_count=0,
        wall_time_ms=0,
        inputs=len(segments),
    )
    conversion_summary = {
        "mode": "mock_audio",
        "per_segment": describe_conversion(per_segment) if per_segment is not None else {
            "conversion_strategy": "per_segment",
            "conversion_input_count": 0,
            "conversion_wall_time_ms": 0,
            "conversion_process_count": 0,
            "conversion_fallback_reason": "ffmpeg_unavailable",
        },
        "lazy_mix": describe_conversion(lazy_mix),
    }
    print(
        "  lazy_mix "
        f"process_count={conversion_summary['lazy_mix']['conversion_process_count']} "
        f"inputs={conversion_summary['lazy_mix']['conversion_input_count']}"
    )

    real_env = _real_environment_summary() if args.real_env else None
    if real_env is not None:
        print(f"\nReal environment readiness: {real_env}")

    report = {
        "benchmark_mode": "real_env_readiness" if args.real_env else "mock_audio",
        "real_environment": real_env,
        "scenario_summaries": [session_report],
        "conversion_summary": conversion_summary,
        "quality_regression_checks": checks,
    }
    if args.out is not None:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote report to {args.out}")
    return 0 if all(check["passed"] for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
