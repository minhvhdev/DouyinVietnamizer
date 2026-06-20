import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from dv_backend.adapters.asr import (
    DEFAULT_ASR_MODEL,
    _group_time_stamps,
    _parse_funasr_segments,
    _result_to_segments,
    resolve_model_reference,
    transcribe_audio,
)


def test_resolve_model_reference_prefers_bundled_vendor_model(tmp_path: Path) -> None:
    vendor_dir = tmp_path / "vendor"
    bundled = vendor_dir / "qwen3-asr" / "Qwen3-ASR-1.7B"
    bundled.mkdir(parents=True)
    (bundled / "config.json").write_text("{}", encoding="utf-8")

    resolved = resolve_model_reference(vendor_dir, "", DEFAULT_ASR_MODEL)

    assert resolved == str(bundled)


def test_group_time_stamps_splits_on_chinese_punctuation() -> None:
    stamps = [
        SimpleNamespace(text="你好", start_time=0.0, end_time=0.4),
        SimpleNamespace(text="世界。", start_time=0.4, end_time=0.9),
        SimpleNamespace(text="再见", start_time=1.0, end_time=1.4),
    ]

    segments = _group_time_stamps(stamps)

    assert segments == [
        {"start": 0.0, "end": 0.9, "text": "你好世界。"},
        {"start": 1.0, "end": 1.4, "text": "再见"},
    ]


def test_result_to_segments_falls_back_to_full_text() -> None:
    result = SimpleNamespace(text="完整句子", time_stamps=[])

    assert _result_to_segments(result) == [{"start": 0.0, "end": 0.0, "text": "完整句子"}]


@patch("dv_backend.adapters.asr._load_model")
def test_transcribe_audio_returns_normalized_segments(mock_load_model: MagicMock, tmp_path: Path) -> None:
    audio_path = tmp_path / "audio_16k.wav"
    audio_path.write_bytes(b"wav")
    stamp = SimpleNamespace(text="你好。", start_time=0.1, end_time=0.8)
    mock_load_model.return_value.transcribe.return_value = [
        SimpleNamespace(text="你好。", time_stamps=[stamp])
    ]

    segments = transcribe_audio(
        audio_path,
        vendor_dir=tmp_path / "vendor",
        asr_model="Qwen/Qwen3-ASR-1.7B",
        aligner_model="Qwen/Qwen3-ForcedAligner-0.6B",
        device="cuda:0",
    )

    assert segments == [{"start": 0.1, "end": 0.8, "text": "你好。"}]
    mock_load_model.return_value.transcribe.assert_called_once_with(
        audio=str(audio_path),
        language="Chinese",
        return_time_stamps=True,
    )


def test_parse_funasr_segments_includes_speaker_id() -> None:
    result = {
        "sentence_info": [
            {"text": "你好。", "start": 100, "end": 900, "spk": 0},
            {"text": "再见。", "start": 1200, "end": 1800, "spk": 1},
        ]
    }

    segments = _parse_funasr_segments(result)

    assert segments == [
        {"start": 0.1, "end": 0.9, "text": "你好。", "speaker_id": "0"},
        {"start": 1.2, "end": 1.8, "text": "再见。", "speaker_id": "1"},
    ]


@patch("dv_backend.adapters.asr._load_funasr_model")
def test_transcribe_audio_uses_funasr_when_diarization_enabled(
    mock_load_funasr: MagicMock,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "audio_16k.wav"
    audio_path.write_bytes(b"wav")
    mock_load_funasr.return_value.generate.return_value = [
        {
            "sentence_info": [
                {"text": "你好。", "start": 0.1, "end": 0.8, "spk": 0},
            ]
        }
    ]

    segments = transcribe_audio(
        audio_path,
        vendor_dir=tmp_path / "vendor",
        speaker_diarization=True,
    )

    assert segments == [{"start": 0.1, "end": 0.8, "text": "你好。", "speaker_id": "0"}]
    mock_load_funasr.return_value.generate.assert_called_once()
