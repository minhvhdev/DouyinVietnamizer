"""Tests for canonical voice identity helpers."""

from __future__ import annotations

from pathlib import Path

from dv_backend.voice_identity import (
    generation_config_hash,
    identity_from_settings,
    identity_profile_key,
    resolve_tts_voice_identity,
)


def test_same_voice_same_key(tmp_path: Path) -> None:
    ref = tmp_path / "voice.wav"
    ref.write_bytes(b"same")
    identity_a = resolve_tts_voice_identity(
        voice_id="voice-1",
        reference_audio_path=ref,
        reference_text="Xin chào",
        model="m1",
    )
    identity_b = resolve_tts_voice_identity(
        voice_id="voice-1",
        reference_audio_path=ref,
        reference_text="Xin chào",
        model="m1",
    )
    assert identity_profile_key(identity_a) == identity_profile_key(identity_b)


def test_reference_audio_change_changes_key(tmp_path: Path) -> None:
    ref_a = tmp_path / "a.wav"
    ref_b = tmp_path / "b.wav"
    ref_a.write_bytes(b"a")
    ref_b.write_bytes(b"b")
    key_a = identity_profile_key(resolve_tts_voice_identity(voice_id="v", reference_audio_path=ref_a, reference_text="t"))
    key_b = identity_profile_key(resolve_tts_voice_identity(voice_id="v", reference_audio_path=ref_b, reference_text="t"))
    assert key_a != key_b


def test_rename_voice_does_not_change_key(tmp_path: Path) -> None:
    ref = tmp_path / "voice.wav"
    ref.write_bytes(b"x")
    key = identity_profile_key(resolve_tts_voice_identity(voice_id="uuid-123", reference_audio_path=ref, reference_text="t"))
    key_renamed = identity_profile_key(
        resolve_tts_voice_identity(voice_id="uuid-123", reference_audio_path=ref, reference_text="t")
    )
    assert key == key_renamed


def test_ref_text_change_changes_key(tmp_path: Path) -> None:
    ref = tmp_path / "voice.wav"
    ref.write_bytes(b"x")
    key_a = identity_profile_key(resolve_tts_voice_identity(voice_id="v", reference_audio_path=ref, reference_text="A"))
    key_b = identity_profile_key(resolve_tts_voice_identity(voice_id="v", reference_audio_path=ref, reference_text="B"))
    assert key_a != key_b


def test_model_change_changes_key(tmp_path: Path) -> None:
    ref = tmp_path / "voice.wav"
    ref.write_bytes(b"x")
    key_a = identity_profile_key(resolve_tts_voice_identity(voice_id="v", reference_audio_path=ref, reference_text="t", model="m1"))
    key_b = identity_profile_key(resolve_tts_voice_identity(voice_id="v", reference_audio_path=ref, reference_text="t", model="m2"))
    assert key_a != key_b


def test_backend_change_changes_key() -> None:
    key_a = identity_profile_key(resolve_tts_voice_identity(tts_backend="omnivoice", voice_id="v"))
    key_b = identity_profile_key(resolve_tts_voice_identity(tts_backend="edge_tts", voice_id="v"))
    assert key_a != key_b


def test_generation_config_change_changes_key() -> None:
    key_a = identity_profile_key(
        resolve_tts_voice_identity(voice_id="v", generation_config={"speed": 1.0, "clone_mode": "reference"})
    )
    key_b = identity_profile_key(
        resolve_tts_voice_identity(voice_id="v", generation_config={"speed": 1.1, "clone_mode": "reference"})
    )
    assert key_a != key_b
    assert generation_config_hash({"speed": 1.0}) != generation_config_hash({"speed": 1.1})


def test_identity_from_settings_uses_ref_hash_when_no_clone_id(tmp_path: Path) -> None:
    ref = tmp_path / "voice.wav"
    ref.write_bytes(b"abc")
    settings = {"tts_backend": "omnivoice", "omnivoice_ref_audio": str(ref), "omnivoice_ref_text": "hello"}
    identity = identity_from_settings(settings)
    assert identity["reference_audio_sha256"]
    assert identity["voice_id"].startswith("ref:")
