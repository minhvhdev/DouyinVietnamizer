"""Tests for voice duration profile learning rules."""

from __future__ import annotations

import json
from pathlib import Path

from dv_backend.voice_duration_profile import (
    load_profiles,
    profile_key,
    resolve_voice_id,
    resolve_voice_profile,
    update_voice_profile_from_sample,
)


def test_repaired_audio_not_learned(tmp_path: Path) -> None:
    settings = {"voice_duration_profile_enabled": True, "tts_backend": "omnivoice", "omnivoice_model": "m"}
    result = update_voice_profile_from_sample(
        settings,
        text="Xin chào các bạn",
        speech_duration_sec=1.2,
        data_dir=tmp_path,
        from_repaired_audio=True,
    )
    assert result is None
    profiles = load_profiles(tmp_path)["profiles"]
    key = next(iter(profiles))
    assert profiles[key].get("rejected_sample_count", 0) >= 1


def test_raw_accepted_sample_updates_profile(tmp_path: Path) -> None:
    settings = {"voice_duration_profile_enabled": True, "tts_backend": "omnivoice", "omnivoice_model": "m"}
    updated = update_voice_profile_from_sample(
        settings,
        text="Xin chào các bạn",
        speech_duration_sec=1.2,
        data_dir=tmp_path,
    )
    assert updated is not None
    assert updated["samples"] == 1


def test_ref_audio_fingerprint_changes_voice_id(tmp_path: Path) -> None:
    ref_a = tmp_path / "a.wav"
    ref_b = tmp_path / "b.wav"
    ref_a.write_bytes(b"a")
    ref_b.write_bytes(b"b")
    id_a = resolve_voice_id({"omnivoice_ref_audio": str(ref_a)})
    id_b = resolve_voice_id({"omnivoice_ref_audio": str(ref_b)})
    assert id_a != id_b


def test_corrupt_profile_recovers(tmp_path: Path) -> None:
    path = tmp_path / "artifacts" / "voice_duration_profiles.json"
    path.parent.mkdir(parents=True)
    path.write_text("{broken", encoding="utf-8")
    store = load_profiles(tmp_path)
    assert store.get("recovered_from_corruption") is True


def test_language_specific_profile_key() -> None:
    key_vi = profile_key(tts_backend="omnivoice", voice_id="v1", language="vi", model="m")
    key_th = profile_key(tts_backend="omnivoice", voice_id="v1", language="th", model="m")
    assert key_vi != key_th


def test_disabled_profile_does_not_read(tmp_path: Path) -> None:
    settings = {"voice_duration_profile_enabled": False, "tts_backend": "omnivoice"}
    profile = resolve_voice_profile(settings, data_dir=tmp_path)
    assert profile.get("samples") == 0
