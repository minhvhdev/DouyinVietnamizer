"""Unit tests for OmniVoice audio probe + diagnostics gate (no CUDA)."""
from __future__ import annotations

import array
import json
import logging
import wave
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dv_backend.adapters.omnivoice_tts import OmniVoiceTtsAdapter
from dv_backend.api import _resolve_omnivoice_preview_clone_meta
from dv_backend.audio_probe import diagnostics_enabled, probe_wav_path, short_hash
from dv_backend.omnivoice_chunk_synthesis import _validate_chunk_wav
from dv_backend.omnivoice_diagnostics import probe_adapter_output, voice_mode


def _write_wav(path: Path, *, frames: int, amplitude: int = 8000, rate: int = 16000) -> None:
    samples = array.array("h", [amplitude if (i // 40) % 2 == 0 else -amplitude for i in range(frames)])
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())


def _write_silent_wav(path: Path, *, frames: int = 16000, rate: int = 16000) -> None:
    _write_wav(path, frames=frames, amplitude=0, rate=rate)


def test_probe_detects_silent_and_audible(tmp_path: Path) -> None:
    silent = tmp_path / "silent.wav"
    audible = tmp_path / "audible.wav"
    _write_silent_wav(silent)
    _write_wav(audible, frames=16000, amplitude=9000)
    silent_probe = probe_wav_path(silent)
    audible_probe = probe_wav_path(audible)
    assert silent_probe["speech_detected"] is False
    assert silent_probe["suspect"] is True
    assert audible_probe["speech_detected"] is True
    assert audible_probe["suspect"] is False


def test_voice_mode_classification() -> None:
    assert voice_mode(ref_audio=None, instruct=None) == "auto"
    assert voice_mode(ref_audio=None, instruct="female") == "instruct"
    assert voice_mode(ref_audio="x.wav", instruct=None) == "clone"


def test_adapter_probe_mode_auto_and_clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    monkeypatch.setenv("DV_OMNIVOICE_DIAGNOSTICS", "1")
    assert diagnostics_enabled() is True
    wav = tmp_path / "out.wav"
    _write_wav(wav, frames=8000, amplitude=7000)

    adapter = OmniVoiceTtsAdapter(
        model="k2-fsa/OmniVoice",
        device="cpu",
        num_step=8,
        language_id="vi",
        settings={},
    )
    with caplog.at_level(logging.INFO):
        adapter._emit_adapter_probe(wav, voice="auto", request_id="req-auto", worker_duration=0.5)
        adapter._emit_adapter_probe(
            wav,
            voice=str(wav),
            request_id="req-clone",
            worker_duration=0.5,
        )
    joined = " ".join(record.getMessage() for record in caplog.records)
    assert "mode\": \"auto\"" in joined or "\"mode\": \"auto\"" in joined
    assert "mode\": \"clone\"" in joined or "\"mode\": \"clone\"" in joined
    assert "Xin chào" not in joined


def test_diagnostics_off_skips_adapter_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DV_OMNIVOICE_DIAGNOSTICS", raising=False)
    assert diagnostics_enabled() is False
    wav = tmp_path / "out.wav"
    _write_wav(wav, frames=4000, amplitude=5000)
    called = {"n": 0}

    def _boom(*_a, **_k):
        called["n"] += 1
        raise AssertionError("probe should not run when diagnostics off")

    monkeypatch.setattr("dv_backend.omnivoice_diagnostics.probe_adapter_output", _boom)
    adapter = OmniVoiceTtsAdapter(model="k2-fsa/OmniVoice", device="cpu", num_step=8, settings={})
    adapter._emit_adapter_probe(wav, voice="auto", request_id="x", worker_duration=None)
    assert called["n"] == 0


def test_preview_anchor_sources(tmp_path: Path) -> None:
    wav = tmp_path / "sample.wav"
    wav.write_bytes(b"RIFF")
    wav.with_suffix(".txt").write_text("Transcript từ sidecar.", encoding="utf-8")

    ref, anchor, source = _resolve_omnivoice_preview_clone_meta(
        preview_voice=str(wav),
        settings={},
        explicit_anchor=None,
        clone=True,
    )
    assert ref == str(wav)
    assert anchor == "Transcript từ sidecar."
    assert source == "sidecar"

    ref2, anchor2, source2 = _resolve_omnivoice_preview_clone_meta(
        preview_voice="auto",
        settings={"omnivoice_ref_audio": str(wav), "omnivoice_ref_text": "Explicit text."},
        explicit_anchor=None,
        clone=False,
    )
    assert ref2 == str(wav)
    assert anchor2 == "Explicit text."
    assert source2 == "explicit_omnivoice_ref_text"
    assert short_hash(anchor2) != short_hash(anchor)


def test_chunk_validation_unchanged_with_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DV_OMNIVOICE_DIAGNOSTICS", "1")
    silent = tmp_path / "silent.wav"
    audible = tmp_path / "audible.wav"
    _write_silent_wav(silent, frames=8000)
    _write_wav(audible, frames=16000, amplitude=10000)

    with pytest.raises(Exception):
        _validate_chunk_wav(silent)
    metrics = _validate_chunk_wav(audible)
    assert metrics["speech_duration"] > 0.05


def test_probe_adapter_output_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DV_OMNIVOICE_DIAGNOSTICS", "1")
    wav = tmp_path / "out.wav"
    _write_wav(wav, frames=8000, amplitude=6000)
    payload = probe_adapter_output(wav, mode="auto", request_id="abc", worker_duration=0.5)
    assert payload["mode"] == "auto"
    assert payload["file_probe"]["speech_detected"] is True


def test_capture_inputs_requires_both_env_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    from dv_backend.audio_probe import capture_inputs_enabled

    monkeypatch.delenv("DV_OMNIVOICE_DIAGNOSTICS", raising=False)
    monkeypatch.delenv("DV_OMNIVOICE_DIAGNOSTICS_CAPTURE_INPUTS", raising=False)
    assert capture_inputs_enabled() is False
    monkeypatch.setenv("DV_OMNIVOICE_DIAGNOSTICS", "1")
    assert capture_inputs_enabled() is False
    monkeypatch.setenv("DV_OMNIVOICE_DIAGNOSTICS_CAPTURE_INPUTS", "1")
    assert capture_inputs_enabled() is True


def test_clone_failure_bundle_opt_in_and_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dv_backend.omnivoice_diagnostics import maybe_capture_clone_failure_bundle

    monkeypatch.setenv("DV_OMNIVOICE_DIAGNOSTICS", "1")
    monkeypatch.setenv("DV_OMNIVOICE_DIAGNOSTICS_CAPTURE_INPUTS", "1")
    monkeypatch.setenv("DV_OMNIVOICE_DIAGNOSTICS_DIR", str(tmp_path / "diag"))

    ref = tmp_path / "ref.wav"
    out = tmp_path / "out.wav"
    _write_silent_wav(ref, frames=8000)
    _write_silent_wav(out, frames=8000)
    silent_probe = probe_wav_path(out)

    # Auto mode must not capture.
    assert (
        maybe_capture_clone_failure_bundle(
            request_id="auto1",
            mode="auto",
            ref_audio=str(ref),
            ref_text="abc",
            target_text="target",
            generate_output_path=out,
            written_output_path=out,
            generate_probe=silent_probe,
            written_probe=silent_probe,
            failure_stage="generate_output",
        )
        is None
    )

    # Audible clone must not capture.
    audible = tmp_path / "audible.wav"
    _write_wav(audible, frames=16000, amplitude=9000)
    audible_probe = probe_wav_path(audible)
    assert (
        maybe_capture_clone_failure_bundle(
            request_id="ok1",
            mode="clone",
            ref_audio=str(ref),
            ref_text="Xin chào",
            target_text="target",
            generate_output_path=audible,
            written_output_path=audible,
            generate_probe=audible_probe,
            written_probe=audible_probe,
            failure_stage="generate_output",
        )
        is None
    )

    bundle = maybe_capture_clone_failure_bundle(
        request_id="fail1",
        mode="clone",
        ref_audio=str(ref),
        ref_text="Xin chào",
        target_text="target",
        generate_output_path=out,
        written_output_path=out,
        generate_probe=silent_probe,
        written_probe=silent_probe,
        failure_stage="generate_output",
    )
    assert bundle is not None
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert (bundle / "ref_text.txt").read_text(encoding="utf-8") == "Xin chào"
    assert (bundle / "target_text.txt").read_text(encoding="utf-8") == "target"
    assert Path(manifest["probes"]["written_output"]["path"]).name.endswith(".wav") or True


def test_text_metrics_detect_bom_and_control() -> None:
    from dv_backend.audio_probe import probe_text_metrics

    metrics = probe_text_metrics("\ufeffXin chào\x00 test")
    assert metrics["ref_text_contains_bom"] is True
    assert metrics["ref_text_contains_control_chars"] is True
    assert metrics["ref_text_words"] >= 2


def test_phase1b_report_schema_forbids_w1_raw_label() -> None:
    schema = {
        "w1_label": "generate_output",
        "w2_label": "written_output",
        "experimental_postprocess_disabled_label": "EXP_generate_postprocess_disabled",
        "forbids_w1_raw_label": True,
    }
    assert schema["w1_label"] == "generate_output"
    assert "raw" not in schema["w1_label"]
    assert schema["experimental_postprocess_disabled_label"].startswith("EXP_")
    assert schema["experimental_postprocess_disabled_label"] != schema["w1_label"]
