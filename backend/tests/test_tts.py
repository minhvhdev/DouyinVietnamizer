from pathlib import Path
import wave

import pytest

from dv_backend.adapters.tts import VieNeuTtsAdapter, split_tts_text
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


def test_split_tts_text_breaks_long_translation() -> None:
    text = "Cau mot. " + ("Cau hai dai. " * 80)
    chunks = split_tts_text(text, max_chars=120)
    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)


def test_vieneu_tts_synthesize_long_text_in_chunks(tmp_path: Path) -> None:
    class TrackingVieneu:
        def __init__(self) -> None:
            self.infers: list[str] = []
            self.call_index = 0

        def infer(self, text: str, voice: str = None, ref_audio: str = None):
            self.infers.append(text)
            return b"pcm"

        def save(self, audio: bytes, output_path: str) -> None:
            self.call_index += 1
            with wave.open(output_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(24000)
                wav_file.writeframes(b"\x00\x00" * (2400 * self.call_index))

    engine = TrackingVieneu()
    adapter = VieNeuTtsAdapter(vieneu_class=lambda: engine)
    output = tmp_path / "long.wav"
    long_text = "Xin chao. " * 120

    adapter.synthesize(long_text, output, voice="Xuân Vĩnh")

    assert len(engine.infers) > 1
    assert output.is_file()
    with wave.open(str(output), "rb") as merged:
        assert merged.getnchannels() == 1
        assert merged.getframerate() == 24000
        expected_frames = sum(2400 * index for index in range(1, len(engine.infers) + 1))
        assert merged.getnframes() == expected_frames
