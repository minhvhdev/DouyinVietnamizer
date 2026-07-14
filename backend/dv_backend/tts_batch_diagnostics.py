"""Diagnostics for TTS batching paths (baseline / profiling)."""
from __future__ import annotations

from typing import Any


def adapter_supports_synthesize_batch(adapter: object | None) -> bool:
    fn = getattr(adapter, "synthesize_batch", None)
    return callable(fn)


def resolve_effective_tts_batch_mode(
    *,
    backend: str,
    micro_batch_enabled: bool,
    micro_batch_size: int,
    adapter: object | None,
) -> str:
    """Describe how pipeline micro-batching maps to adapter/worker execution."""
    if not micro_batch_enabled or micro_batch_size <= 1:
        return "single_segment"
    if adapter_supports_synthesize_batch(adapter):
        return "adapter_synthesize_batch"
    if backend == "omnivoice":
        return "omnivoice_sequential_fallback"
    return "sequential_fallback"


def omnivoice_job_baseline(settings: dict[str, Any]) -> dict[str, Any]:
    """Snapshot OmniVoice knobs relevant to speed vs quality trade-offs."""
    return {
        "tts_backend": str(settings.get("tts_backend") or "omnivoice"),
        "omnivoice_num_steps": int(settings.get("omnivoice_num_steps") or 32),
        "omnivoice_device": str(settings.get("omnivoice_device") or "cuda:0"),
        "omnivoice_external_chunking_enabled": bool(
            settings.get("omnivoice_external_chunking_enabled", True)
        ),
        "omnivoice_audio_chunk_threshold": float(
            settings.get("omnivoice_audio_chunk_threshold") or 30.0
        ),
        "omnivoice_audio_chunk_duration": float(
            settings.get("omnivoice_audio_chunk_duration") or 15.0
        ),
        "omnivoice_fidelity_check_enabled": bool(
            settings.get("omnivoice_fidelity_check_enabled", True)
        ),
        "omnivoice_fidelity_check_min_chars": int(
            settings.get("omnivoice_fidelity_check_min_chars") or 240
        ),
        "tts_micro_batch_enabled": bool(settings.get("tts_micro_batch_enabled", True)),
        "tts_micro_batch_size": int(settings.get("tts_micro_batch_size") or 4),
        "tts_session_reuse_enabled": bool(settings.get("tts_session_reuse_enabled", True)),
    }
