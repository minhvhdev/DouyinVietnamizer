from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from dv_backend.adapters.asr import (
    DEFAULT_ASR_MODEL,
    _group_time_stamps,
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


def test_group_time_stamps_splits_long_unpunctuated_alignment() -> None:
    stamps = [
        SimpleNamespace(text=f"句{i}", start_time=float(i), end_time=float(i + 1))
        for i in range(30)
    ]

    segments = _group_time_stamps(stamps)

    assert len(segments) >= 2
    assert all(segment["end"] - segment["start"] <= 13.0 for segment in segments)
    assert "".join(str(segment["text"]) for segment in segments) == "".join(
        stamp.text for stamp in stamps
    )


def test_group_time_stamps_splits_at_internal_punctuation_before_hard_limit() -> None:
    stamps = [
        SimpleNamespace(text="前面很长", start_time=0.0, end_time=1.0),
        SimpleNamespace(text="的内容。", start_time=1.0, end_time=2.0),
        SimpleNamespace(text="后面", start_time=2.0, end_time=3.0),
        SimpleNamespace(text="继续。", start_time=3.0, end_time=4.0),
    ]

    segments = _group_time_stamps(stamps)

    assert [segment["text"] for segment in segments] == ["前面很长的内容。", "后面继续。"]


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


@patch("dv_backend.adapters.asr._transcribe_details_with_qwen")
@patch("dv_backend.adapters.asr._load_model")
def test_transcribe_audio_with_diarization_returns_alignment(
    mock_load_model: MagicMock,
    mock_qwen_details: MagicMock,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "audio_16k.wav"
    audio_path.write_bytes(b"wav")
    mock_qwen_details.return_value = {
        "segments": [{"start": 0.1, "end": 0.8, "text": "你好。"}],
        "aligned_units": [{"text": "你", "start": 0.1, "end": 0.3}, {"text": "好。", "start": 0.3, "end": 0.8}],
    }

    result = transcribe_audio(
        audio_path,
        vendor_dir=tmp_path / "vendor",
        speaker_diarization=True,
    )

    assert isinstance(result, dict)
    assert result["segments"][0]["text"] == "你好。"
    assert len(result["aligned_units"]) == 2
    mock_qwen_details.assert_called_once()
    mock_load_model.assert_called_once()
