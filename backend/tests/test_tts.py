from pathlib import Path

import pytest

from dv_backend.adapters.tts import VieNeuTtsAdapter
from dv_backend.errors import AppError


class FakeVieneu:
    def __init__(self) -> None:
        self.infers = []
        self.saves = []

    def infer(self, text: str, voice: str = None, ref_audio: str = None):
        self.infers.append((text, voice, ref_audio))
        return b"pcm_audio"

    def save(self, audio: bytes, output_path: str):
        self.saves.append((audio, output_path))
        Path(output_path).write_bytes(audio)


def test_vieneu_tts_synthesize_preset(tmp_path: Path) -> None:
    fake = FakeVieneu()
    adapter = VieNeuTtsAdapter(vieneu_class=lambda: fake)
    output = tmp_path / "vieneu_preset.wav"
    adapter.synthesize("Xin chao", output, voice="Xuân Vĩnh")

    assert output.read_bytes() == b"pcm_audio"
    assert fake.infers == [("Xin chao", "Xuân Vĩnh", None)]


def test_vieneu_tts_unknown_preset_voice(tmp_path: Path) -> None:
    class RaisingVieneu:
        def infer(self, text: str, voice: str = None, ref_audio: str = None):
            raise ValueError(
                "Voice 'Phương Trang' not found. Available: ['Xuân Vĩnh']"
            )

        def save(self, audio, output_path: str):
            pass

    adapter = VieNeuTtsAdapter(vieneu_class=lambda: RaisingVieneu())
    with pytest.raises(AppError) as exc:
        adapter.synthesize("Xin chao", tmp_path / "out.wav", voice="Phương Trang")

    assert exc.value.info.code == "VIENEU_VOICE_NOT_FOUND"


def test_vieneu_tts_synthesize_clone(tmp_path: Path) -> None:
    fake = FakeVieneu()
    adapter = VieNeuTtsAdapter(vieneu_class=lambda: fake)
    output = tmp_path / "vieneu_clone.wav"
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"dummy")

    adapter.synthesize("Xin chao", output, voice=str(ref_audio))

    assert output.read_bytes() == b"pcm_audio"
    assert fake.infers == [("Xin chao", None, str(ref_audio))]
