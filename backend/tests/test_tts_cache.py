"""Tests for TTS cache identity."""

from __future__ import annotations

import json
from pathlib import Path

from dv_backend.tts_cache import (
    CACHE_SCHEMA_VERSION,
    build_tts_cache_identity,
    cache_key_from_identity,
    fidelity_status_cacheable,
    sidecar_path,
    wav_cache_valid,
    write_tts_sidecar,
)


def test_cache_key_changes_when_text_changes() -> None:
    settings = {"tts_backend": "omnivoice", "omnivoice_model": "test"}
    a = cache_key_from_identity(build_tts_cache_identity(settings, text="hello"))
    b = cache_key_from_identity(build_tts_cache_identity(settings, text="world"))
    assert a != b


def test_cache_key_changes_when_voice_changes() -> None:
    settings_a = {"tts_backend": "edge", "edge_tts_voice": "vi-VN-HoaiMyNeural"}
    settings_b = {"tts_backend": "edge", "edge_tts_voice": "vi-VN-NamMinhNeural"}
    a = cache_key_from_identity(build_tts_cache_identity(settings_a, text="hello"))
    b = cache_key_from_identity(build_tts_cache_identity(settings_b, text="hello"))
    assert a != b


def test_cache_key_changes_when_external_chunking_toggle_changes() -> None:
    base = {"tts_backend": "omnivoice", "omnivoice_model": "test"}
    legacy = {**base, "omnivoice_external_chunking_enabled": False}
    adaptive = {**base, "omnivoice_external_chunking_enabled": True}
    key_legacy = cache_key_from_identity(build_tts_cache_identity(legacy, text="x" * 300))
    key_adaptive = cache_key_from_identity(build_tts_cache_identity(adaptive, text="x" * 300))
    assert key_legacy != key_adaptive


def test_cache_key_changes_when_retry_ladder_changes() -> None:
    base = {
        "tts_backend": "omnivoice",
        "omnivoice_model": "test",
        "omnivoice_external_chunking_enabled": True,
    }
    a = {
        **base,
        "omnivoice_chunk_max_chars": 220,
        "omnivoice_chunk_retry_max_chars_2": 140,
        "omnivoice_chunk_retry_max_chars_3": 90,
    }
    b = {
        **base,
        "omnivoice_chunk_max_chars": 220,
        "omnivoice_chunk_retry_max_chars_2": 120,
        "omnivoice_chunk_retry_max_chars_3": 80,
    }
    assert cache_key_from_identity(build_tts_cache_identity(a, text="hello")) != cache_key_from_identity(
        build_tts_cache_identity(b, text="hello")
    )


def test_cache_key_changes_when_fidelity_retry_flag_changes() -> None:
    base = {"tts_backend": "omnivoice", "omnivoice_model": "test"}
    on = {**base, "omnivoice_chunk_retry_on_fidelity_failure": True}
    off = {**base, "omnivoice_chunk_retry_on_fidelity_failure": False}
    assert cache_key_from_identity(build_tts_cache_identity(on, text="hello")) != cache_key_from_identity(
        build_tts_cache_identity(off, text="hello")
    )


def test_identity_includes_synthesis_policy_schema_v4() -> None:
    identity = build_tts_cache_identity(
        {"tts_backend": "omnivoice", "omnivoice_model": "test"},
        text="xin chào",
    )
    assert identity["schema_version"] == CACHE_SCHEMA_VERSION
    assert identity["schema_version"] >= 4
    assert "synthesis_policy" in identity
    assert identity["omnivoice_external_chunking_enabled"] is True


def test_sidecar_validation(tmp_path: Path) -> None:
    wav = tmp_path / "tts_raw_0.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 64)
    settings = {"tts_backend": "omnivoice", "omnivoice_model": "test"}
    identity = build_tts_cache_identity(settings, text="xin chào")
    write_tts_sidecar(wav, identity, extra={"tts_fidelity_status": "good"})
    assert sidecar_path(wav).is_file()
    assert wav_cache_valid(wav, identity) is True
    other = build_tts_cache_identity(settings, text="khác")
    assert wav_cache_valid(wav, other) is False


def test_failed_fidelity_sidecar_is_not_cache_hit(tmp_path: Path) -> None:
    wav = tmp_path / "tts_raw_fail.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 64)
    settings = {"tts_backend": "omnivoice", "omnivoice_model": "test"}
    identity = build_tts_cache_identity(settings, text="xin chào")
    write_tts_sidecar(wav, identity, extra={"tts_fidelity_status": "failed"})
    assert fidelity_status_cacheable("failed") is False
    assert wav_cache_valid(wav, identity) is False
    write_tts_sidecar(wav, identity, extra={"tts_fidelity_status": "poor"})
    assert wav_cache_valid(wav, identity) is False


def test_legacy_schema_sidecar_does_not_collide_with_adaptive(tmp_path: Path) -> None:
    """Schema bump invalidates pre-policy single-shot caches."""
    wav = tmp_path / "tts_raw_legacy.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 64)
    settings = {
        "tts_backend": "omnivoice",
        "omnivoice_model": "test",
        "omnivoice_external_chunking_enabled": True,
    }
    identity = build_tts_cache_identity(settings, text="long text " * 40)
    # Simulate an older sidecar missing synthesis policy / older schema.
    sidecar_path(wav).write_text(
        json.dumps(
            {
                "schema_version": 3,
                "cache_key": "deadbeef",
                "translation_text_hash": identity["translation_text_hash"],
                "tts_fidelity_status": "good",
            }
        ),
        encoding="utf-8",
    )
    assert wav_cache_valid(wav, identity) is False


def test_missing_sidecar_invalidates(tmp_path: Path) -> None:
    wav = tmp_path / "tts_raw_1.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 64)
    settings = {"tts_backend": "omnivoice", "omnivoice_model": "test"}
    identity = build_tts_cache_identity(settings, text="a")
    assert wav_cache_valid(wav, identity) is False


def test_ref_audio_fingerprint_changes_key(tmp_path: Path) -> None:
    ref_a = tmp_path / "ref_a.wav"
    ref_b = tmp_path / "ref_b.wav"
    ref_a.write_bytes(b"a")
    ref_b.write_bytes(b"b")
    settings_a = {"tts_backend": "omnivoice", "omnivoice_ref_audio": str(ref_a), "omnivoice_model": "m"}
    settings_b = {"tts_backend": "omnivoice", "omnivoice_ref_audio": str(ref_b), "omnivoice_model": "m"}
    key_a = cache_key_from_identity(build_tts_cache_identity(settings_a, text="x"))
    key_b = cache_key_from_identity(build_tts_cache_identity(settings_b, text="x"))
    assert key_a != key_b
