"""Tests for TTS batching baseline diagnostics."""
from __future__ import annotations

from pathlib import Path

from dv_backend.adapters.omnivoice_tts import OmniVoiceTtsAdapter
from dv_backend.tts_batch_diagnostics import (
    adapter_supports_synthesize_batch,
    omnivoice_job_baseline,
    resolve_effective_tts_batch_mode,
)


class _BatchAdapter:
    def synthesize_batch(self, items: list[dict]) -> None:
        _ = items


def test_omnivoice_adapter_has_synthesize_batch() -> None:
    adapter = OmniVoiceTtsAdapter(model="k2-fsa/OmniVoice", settings={})
    assert adapter_supports_synthesize_batch(adapter)


def test_batch_adapter_detected() -> None:
    assert adapter_supports_synthesize_batch(_BatchAdapter())


def test_effective_mode_omnivoice_queued_batch() -> None:
    adapter = OmniVoiceTtsAdapter(model="k2-fsa/OmniVoice", settings={})
    mode = resolve_effective_tts_batch_mode(
        backend="omnivoice",
        micro_batch_enabled=True,
        micro_batch_size=4,
        adapter=adapter,
    )
    assert mode == "adapter_synthesize_batch"


def test_effective_mode_adapter_batch() -> None:
    mode = resolve_effective_tts_batch_mode(
        backend="omnivoice",
        micro_batch_enabled=True,
        micro_batch_size=4,
        adapter=_BatchAdapter(),
    )
    assert mode == "adapter_synthesize_batch"


def test_omnivoice_job_baseline_snapshot() -> None:
    baseline = omnivoice_job_baseline(
        {
            "omnivoice_num_steps": 16,
            "omnivoice_external_chunking_enabled": False,
            "tts_micro_batch_enabled": True,
        }
    )
    assert baseline["omnivoice_num_steps"] == 16
    assert baseline["tts_micro_batch_enabled"] is True
    assert baseline["omnivoice_external_chunking_enabled"] is False
