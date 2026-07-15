"""Opt-in OmniVoice diagnostics (env-gated). Never changes synthesis behavior."""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from .audio_probe import (
    DIAG_ENV,
    capture_inputs_enabled,
    diagnostics_dir,
    diagnostics_enabled,
    file_content_hash,
    probe_text_metrics,
    probe_wav_path,
    probe_waveform,
    short_hash,
)

logger = logging.getLogger(__name__)


def voice_mode(*, ref_audio: str | None, instruct: str | None) -> str:
    if ref_audio:
        return "clone"
    if instruct and str(instruct).strip():
        return "instruct"
    return "auto"


def log_event(stage: str, payload: dict[str, Any]) -> None:
    if not diagnostics_enabled():
        return
    safe = dict(payload)
    # Never log raw transcripts / prompt content.
    for banned in ("ref_text", "anchor_text", "transcript", "text", "instruct", "prompt"):
        safe.pop(banned, None)
    logger.info("omnivoice_diag stage=%s %s", stage, json.dumps(safe, ensure_ascii=True, default=str))


def copy_artifact(source: Path, *, request_id: str, label: str) -> Path | None:
    if not diagnostics_enabled():
        return None
    try:
        target = diagnostics_dir() / f"{request_id}_{label}.wav"
        shutil.copy2(source, target)
        return target
    except OSError as exc:
        log_event("artifact_copy_failed", {"request_id": request_id, "label": label, "error": str(exc)})
        return None


def write_waveform_artifact(
    samples: Any,
    *,
    sample_rate: int,
    request_id: str,
    label: str,
) -> Path | None:
    if not diagnostics_enabled():
        return None
    try:
        import soundfile as sf

        target = diagnostics_dir() / f"{request_id}_{label}.wav"
        sf.write(str(target), samples, int(sample_rate))
        return target
    except Exception as exc:  # noqa: BLE001
        log_event(
            "artifact_write_failed",
            {"request_id": request_id, "label": label, "error": str(exc)},
        )
        return None


def describe_target_conditioning(*, text: str | None, mode: str) -> dict[str, Any]:
    """Hash/summary of target text immediately before model.generate()."""
    from .omnivoice_content_fidelity import describe_target_text_for_generate

    return describe_target_text_for_generate(str(text or ""), mode=mode)


def describe_ref_conditioning(
    *,
    ref_audio: str | None,
    ref_text: str | None,
    cache_hit: bool | None = None,
) -> dict[str, Any]:
    text = str(ref_text or "")
    path = Path(ref_audio) if ref_audio else None
    return {
        "ref_audio_path_hash": short_hash(str(path) if path else ""),
        "ref_audio_content_hash": file_content_hash(path) if path else None,
        "ref_audio_file_size": path.stat().st_size if path and path.is_file() else None,
        "ref_probe": probe_wav_path(path) if path else None,
        "ref_text_length": len(text),
        "ref_text_hash": short_hash(text),
        "cache_hit": cache_hit,
        "cache_key_hash": short_hash(f"{path}|{text}") if ref_audio else None,
        "diag_env": DIAG_ENV,
    }


def probe_adapter_output(
    output_path: Path,
    *,
    mode: str,
    request_id: str | None = None,
    worker_duration: float | None = None,
) -> dict[str, Any]:
    probe = probe_wav_path(output_path)
    payload = {
        "mode": mode,
        "request_id": request_id,
        "worker_duration_sec": worker_duration,
        "file_probe": probe,
        "duration_mismatch": (
            abs(float(worker_duration) - float(probe.get("duration_sec") or 0.0)) > 0.05
            if worker_duration is not None and probe.get("ok")
            else None
        ),
    }
    log_event("adapter_output", payload)
    return payload


def maybe_capture_clone_failure_bundle(
    *,
    request_id: str,
    mode: str,
    ref_audio: str | None,
    ref_text: str | None,
    target_text: str | None,
    generate_output_path: Path | None,
    written_output_path: Path | None,
    generate_probe: dict[str, Any] | None,
    written_probe: dict[str, Any] | None,
    failure_stage: str,
    anchor_source: str | None = None,
    clone_prompt_cache_hit: bool | None = None,
    clone_prompt_cache_key_hash: str | None = None,
    language: str | None = None,
    generation_config: dict[str, Any] | None = None,
    model_identity: dict[str, Any] | None = None,
) -> Path | None:
    """Write local opt-in failure bundle; never raises into synthesis path."""
    try:
        if mode != "clone" or not capture_inputs_enabled():
            return None
        silent = False
        for probe in (generate_probe, written_probe):
            if probe and (probe.get("speech_detected") is False or probe.get("suspect")):
                silent = True
                break
        if not silent:
            return None
        if not ref_audio or not Path(ref_audio).is_file():
            return None

        bundle_dir = diagnostics_dir() / f"{request_id}_clone_failure"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        ref_src = Path(ref_audio)
        ref_dst = bundle_dir / f"ref_audio{ref_src.suffix or '.wav'}"
        shutil.copy2(ref_src, ref_dst)
        (bundle_dir / "ref_text.txt").write_text(str(ref_text or ""), encoding="utf-8")
        (bundle_dir / "target_text.txt").write_text(str(target_text or ""), encoding="utf-8")
        if generate_output_path and generate_output_path.is_file():
            shutil.copy2(generate_output_path, bundle_dir / "generate_output.wav")
        if written_output_path and written_output_path.is_file():
            shutil.copy2(written_output_path, bundle_dir / "written_output.wav")

        import platform
        import sys

        env_info = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "diag_env": DIAG_ENV,
        }
        try:
            import torch

            env_info["torch"] = getattr(torch, "__version__", None)
            env_info["cuda_available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                env_info["cuda_device"] = torch.cuda.get_device_name(0)
        except Exception:
            pass

        text_metrics = probe_text_metrics(ref_text)
        ref_probe = probe_wav_path(ref_dst)
        words = max(1, int(text_metrics.get("ref_text_words") or 1))
        speech_sec = float(ref_probe.get("detectable_speech_duration_sec") or 0.0)
        text_metrics["speech_seconds_per_word"] = round(speech_sec / words, 6)

        manifest = {
            "schema_version": 1,
            "request_id": request_id,
            "mode": mode,
            "failure_stage": failure_stage,
            "ref_audio_sha256": file_content_hash(ref_dst, length=64),
            "ref_audio_original_path_hash": short_hash(str(ref_src)),
            "ref_text_sha256": short_hash(str(ref_text or ""), length=64),
            "target_text_sha256": short_hash(str(target_text or ""), length=64),
            "anchor_source": anchor_source,
            "clone_prompt_cache_hit": clone_prompt_cache_hit,
            "clone_prompt_cache_key_hash": clone_prompt_cache_key_hash,
            "language": language,
            "generation_config": generation_config or {},
            "model_identity": model_identity or {},
            "environment": env_info,
            "ref_text_metrics": text_metrics,
            "probes": {
                "ref_audio": ref_probe,
                "generate_output": generate_probe,
                "written_output": written_probe,
            },
        }
        (bundle_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log_event("clone_failure_bundle", {"request_id": request_id, "bundle_dir": str(bundle_dir)})
        return bundle_dir
    except Exception as exc:  # noqa: BLE001 — never break synthesis
        log_event("clone_failure_bundle_error", {"request_id": request_id, "error": str(exc)})
        return None


__all__ = [
    "copy_artifact",
    "describe_ref_conditioning",
    "describe_target_conditioning",
    "diagnostics_dir",
    "diagnostics_enabled",
    "file_content_hash",
    "log_event",
    "probe_adapter_output",
    "probe_wav_path",
    "probe_waveform",
    "short_hash",
    "voice_mode",
    "write_waveform_artifact",
]
