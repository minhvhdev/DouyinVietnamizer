#!/usr/bin/env python3
"""Run a real OmniVoice synthesis smoke test on Apple Silicon MPS."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import traceback

from dv_backend.adapters.omnivoice_worker import OmniVoiceEngine
from dv_backend.omnivoice_env import OMNIVOICE_DEFAULT_MODEL, OMNIVOICE_DEFAULT_SAMPLE_RATE
from dv_backend.omnivoice_mps import omnivoice_runtime_versions, plan_omnivoice_device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="omnivoice_mps_smoke")
    parser.add_argument("--report", default="")
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--ref-text", required=True)
    parser.add_argument("--text", default="Xin chào, đây là kiểm tra OmniVoice trên Apple Silicon.")
    parser.add_argument("--model", default=OMNIVOICE_DEFAULT_MODEL)
    parser.add_argument("--num-step", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else output_dir / "report.json"
    )
    report: dict = {
        "status": "failed",
        "platform": sys.platform,
        "machine": platform.machine().lower(),
        "versions": omnivoice_runtime_versions(),
        "model_id": args.model,
        "outputs": [],
    }
    engine = OmniVoiceEngine()
    try:
        if sys.platform != "darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
            raise RuntimeError("This smoke test requires native macOS on Apple Silicon.")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "").strip() == "1":
            raise RuntimeError(
                "PYTORCH_ENABLE_MPS_FALLBACK must be unset for MPS acceptance."
            )
        ref_audio = Path(args.ref_audio).expanduser().resolve()
        if not ref_audio.is_file():
            raise FileNotFoundError(f"Reference audio not found: {ref_audio}")
        if not args.ref_text.strip():
            raise ValueError("--ref-text must match the complete reference audio transcript.")
        import soundfile as sf

        ref_info = sf.info(str(ref_audio))
        report["input"] = {
            "ref_audio": str(ref_audio),
            "ref_duration_sec": float(ref_info.duration),
            "ref_text_chars": len(args.ref_text.strip()),
            "target_text_chars": len(args.text.strip()),
        }

        plan = plan_omnivoice_device("auto")
        if plan.resolved_device != "mps" or plan.model_dtype != "float16":
            raise RuntimeError(f"Unexpected MPS plan: {plan}")
        engine.get(model=args.model, device=plan.resolved_device)
        placement = engine._placement_diagnostics or {}
        tokenizer = placement.get("audio_tokenizer") or {}
        if placement.get("main_model_device", "").split(":", 1)[0] != "mps":
            raise RuntimeError(f"Main model is not on MPS: {placement}")
        if placement.get("main_model_dtype") not in {"torch.float16", "float16"}:
            raise RuntimeError(f"Main model is not float16: {placement}")
        if tokenizer.get("violations"):
            raise RuntimeError(f"Audio tokenizer placement is invalid: {tokenizer}")
        if tokenizer.get("floating_dtypes") != ["torch.float32"]:
            raise RuntimeError(f"Audio tokenizer is not entirely float32: {tokenizer}")

        report["device_plan"] = {
            "device": plan.resolved_device,
            "model_dtype": plan.model_dtype,
            "audio_tokenizer_device": plan.audio_tokenizer_device,
        }
        report["placement"] = placement
        model_source = str(placement.get("model_source") or args.model)
        report["model_source"] = model_source
        report["model_snapshot_revision"] = resolved_snapshot_revision(model_source)

        for label in ("cold", "warm"):
            output = output_dir / f"{label}.wav"
            duration, sample_rate, perf = engine.synthesize(
                args.text,
                str(output),
                ref_audio=str(ref_audio),
                ref_text=args.ref_text,
                anchor_text=args.ref_text,
                instruct=None,
                num_step=max(4, args.num_step),
                speed=1.0,
                language_id="vi",
                include_perf=True,
            )
            analysis = analyze_wav(output)
            analysis["worker_duration_sec"] = duration
            analysis["worker_sample_rate"] = sample_rate
            analysis["performance"] = perf
            synthesis_ms = float((perf or {}).get("model_synthesis_ms", 0.0))
            analysis["rtf"] = (
                synthesis_ms / 1000.0 / duration if duration > 0 and synthesis_ms > 0 else None
            )
            if analysis["errors"]:
                raise RuntimeError(f"{label} WAV quality checks failed: {analysis['errors']}")
            report["outputs"].append({"label": label, **analysis})

        if not report.get("model_snapshot_revision"):
            raise RuntimeError(
                "Could not resolve the Hugging Face snapshot revision from the loaded model."
            )
        report["status"] = "passed"
        report["manual_listening_required"] = True
    except Exception as exc:  # noqa: BLE001
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        engine.release()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        return 1
    print("Manual gate: listen to cold.wav and warm.wav for muffling, noise, or clipping.")
    return 0


def analyze_wav(path: Path) -> dict:
    import numpy as np
    import soundfile as sf

    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    finite = bool(values.size and np.isfinite(values).all())
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0
    clip_ratio = float(np.mean(np.abs(values) >= 0.999)) if values.size else 1.0
    duration_sec = float(values.size / sample_rate) if sample_rate else 0.0
    errors: list[str] = []
    if sample_rate != OMNIVOICE_DEFAULT_SAMPLE_RATE:
        errors.append(f"sample_rate={sample_rate}")
    if not finite:
        errors.append("samples_not_finite")
    if duration_sec < 0.2:
        errors.append(f"duration_too_short={duration_sec:.3f}")
    if rms < 1e-4:
        errors.append(f"audio_silent_rms={rms:.6f}")
    if peak > 1.0:
        errors.append(f"peak_out_of_range={peak:.6f}")
    if clip_ratio >= 0.01:
        errors.append(f"clip_ratio={clip_ratio:.6f}")
    return {
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "sample_rate": int(sample_rate),
        "duration_sec": duration_sec,
        "finite": finite,
        "peak": peak,
        "rms": rms,
        "clip_ratio": clip_ratio,
        "errors": errors,
    }


def resolved_snapshot_revision(model_source: str) -> str | None:
    parts = Path(model_source).parts
    if "snapshots" not in parts:
        return None
    index = parts.index("snapshots")
    return parts[index + 1] if index + 1 < len(parts) else None


if __name__ == "__main__":
    raise SystemExit(main())
