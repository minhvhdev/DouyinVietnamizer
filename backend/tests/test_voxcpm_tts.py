"""Tests for the VoxCPM2 TTS adapter, cache, and worker.

This file accumulates the tests for the VoxCPM2 migration (Tasks 1-5).
"""

import wave
from pathlib import Path

import pytest

from dv_backend.adapters.tts import (
    VOXCPM_INSTRUCT_PREFIX,
    VoxCPMTtsAdapter,
    create_tts_adapter,
    parse_voxcpm_voice,
    split_tts_text,
)
from dv_backend.errors import AppError


# ---------------------------------------------------------------------------
# Voice parsing
# ---------------------------------------------------------------------------


def test_parse_voxcpm_voice_modes() -> None:
    assert parse_voxcpm_voice("auto") == (None, None, None)
    assert parse_voxcpm_voice(f"{VOXCPM_INSTRUCT_PREFIX}female, low pitch") == (
        None,
        None,
        "female, low pitch",
    )


def test_parse_voxcpm_voice_with_ref_audio(tmp_path: Path) -> None:
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    assert parse_voxcpm_voice(str(ref)) == (str(ref), None, None)


# ---------------------------------------------------------------------------
# create_tts_adapter factory
# ---------------------------------------------------------------------------


def test_create_tts_adapter_always_selects_voxcpm() -> None:
    adapter = create_tts_adapter({"tts_backend": "other", "voxcpm_device": "cuda:0"})
    assert type(adapter).__name__ == "VoxCPMTtsAdapter"
