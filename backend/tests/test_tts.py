from pathlib import Path

import pytest

from dv_backend.adapters.tts import EdgeTtsAdapter
from dv_backend.errors import AppError


class FakeCommunicate:
    def __init__(self, text: str, voice: str, rate: str, pitch: str, volume: str) -> None:
        self.text = text

    async def save(self, output_path: str) -> None:
        Path(output_path).write_bytes(b"audio")


class EmptyCommunicate(FakeCommunicate):
    async def save(self, output_path: str) -> None:
        Path(output_path).write_bytes(b"")


def test_edge_tts_writes_audio_with_selected_voice(tmp_path: Path) -> None:
    adapter = EdgeTtsAdapter(communicate_factory=FakeCommunicate)
    output = tmp_path / "speech.mp3"

    adapter.synthesize(
        "Xin chao",
        output,
        voice="vi-VN-HoaiMyNeural",
        rate="+10%",
        pitch="+0Hz",
        volume="+0%",
    )

    assert output.read_bytes() == b"audio"


def test_edge_tts_rejects_empty_audio(tmp_path: Path) -> None:
    adapter = EdgeTtsAdapter(communicate_factory=EmptyCommunicate)

    with pytest.raises(AppError) as error:
        adapter.synthesize(
            "Xin chao",
            tmp_path / "speech.mp3",
            voice="vi-VN-HoaiMyNeural",
            rate="+0%",
            pitch="+0Hz",
            volume="+0%",
        )

    assert error.value.info.code == "EDGE_TTS_EMPTY_OUTPUT"
