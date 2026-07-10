from pathlib import Path

from dv_backend.api import _resolve_omnivoice_preview_clone


def test_resolve_preview_clone_from_settings_ref_audio(tmp_path: Path) -> None:
    wav = tmp_path / "sample.wav"
    wav.write_bytes(b"RIFF")
    settings = {
        "omnivoice_ref_audio": str(wav),
        "omnivoice_ref_text": "Xin chào đây là mẫu giọng.",
    }
    ref_audio, anchor = _resolve_omnivoice_preview_clone(
        preview_voice="auto",
        settings=settings,
        explicit_anchor=None,
        clone=False,
    )
    assert ref_audio == str(wav)
    assert anchor == "Xin chào đây là mẫu giọng."


def test_resolve_preview_clone_from_sidecar(tmp_path: Path) -> None:
    wav = tmp_path / "sample.wav"
    wav.write_bytes(b"RIFF")
    wav.with_suffix(".txt").write_text("Transcript từ file txt.", encoding="utf-8")
    ref_audio, anchor = _resolve_omnivoice_preview_clone(
        preview_voice=str(wav),
        settings={},
        explicit_anchor=None,
        clone=True,
    )
    assert ref_audio == str(wav)
    assert anchor == "Transcript từ file txt."


def test_resolve_preview_clone_returns_none_for_auto_voice() -> None:
    ref_audio, anchor = _resolve_omnivoice_preview_clone(
        preview_voice="auto",
        settings={},
        explicit_anchor=None,
        clone=False,
    )
    assert ref_audio is None
    assert anchor is None
