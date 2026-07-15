import array
import wave
from pathlib import Path
from unittest.mock import patch

from dv_backend.api import _resolve_omnivoice_preview_clone, _synthesize_voice_preview


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


def test_preview_schedules_vram_release_after_generated_audio(tmp_path: Path) -> None:
    output_holder: dict[str, Path] = {}

    class _FakeAdapter:
        def synthesize(self, *, output_path: Path, **_kwargs) -> None:
            path = Path(output_path)
            output_holder["path"] = path
            samples = array.array("h", [8000] * 32000)
            with wave.open(str(path), "w") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(samples.tobytes())

    with (
        patch("dv_backend.adapters.tts.create_tts_adapter", return_value=_FakeAdapter()),
        patch("dv_backend.api.begin_omnivoice_preview", return_value=42) as begin,
        patch("dv_backend.api.complete_omnivoice_preview") as complete,
    ):
        output = _synthesize_voice_preview(
            voice="auto",
            text="Xin chào.",
            settings={"tts_backend": "omnivoice"},
            output_suffix="test",
            backend="omnivoice",
        )

    assert output == output_holder["path"]
    begin.assert_called_once_with()
    complete.assert_called_once_with(42, output)
