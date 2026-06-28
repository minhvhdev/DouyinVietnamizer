import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from dv_backend.adapters.asr import (
    DEFAULT_ASR_MODEL,
    SPEAKER_CONFIDENCE_LOW,
    _assign_speaker_ids_by_overlap,
    _cluster_speaker_embeddings,
    _group_time_stamps,
    _parse_funasr_segments,
    _result_to_segments,
    resolve_model_reference,
    transcribe_audio,
)
from dv_backend.errors import AppError
from dv_backend.models import ErrorInfo


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
    assert all(segment["end"] - segment["start"] <= 7.0 for segment in segments)
    assert "".join(str(segment["text"]) for segment in segments) == "".join(
        stamp.text for stamp in stamps
    )


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
            {"sentence": "你好。", "start": 100, "end": 900, "spk": 0},
            {"text": "再见。", "start": 1200, "end": 1800, "spk": 1},
        ]
    }

    segments = _parse_funasr_segments(result)

    assert segments == [
        {"start": 0.1, "end": 0.9, "text": "你好。", "speaker_id": "0"},
        {"start": 1.2, "end": 1.8, "text": "再见。", "speaker_id": "1"},
    ]


def test_assign_speaker_ids_by_overlap_uses_best_matching_window() -> None:
    qwen_segments = [
        {"start": 0.0, "end": 2.0, "text": "A"},
        {"start": 2.0, "end": 4.0, "text": "B"},
    ]
    diar_segments = [
        {"start": 0.0, "end": 2.5, "text": "...", "speaker_id": "0"},
        {"start": 2.5, "end": 5.0, "text": "...", "speaker_id": "1"},
    ]

    merged = _assign_speaker_ids_by_overlap(qwen_segments, diar_segments)

    assert merged[0]["speaker_id"] == "0"
    assert merged[1]["speaker_id"] == "1"


def test_assign_speaker_ids_by_overlap_prefers_louder_label(tmp_path: Path) -> None:
    import numpy as np

    import dv_backend.adapters.asr as asr_module

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"wav")
    samples = np.zeros(16000, dtype=np.float32)
    samples[8000:12000] = 0.8
    qwen_segments = [{"start": 0.0, "end": 1.0, "text": "A"}]
    diar_segments = [
        {"start": 0.0, "end": 0.5, "text": "...", "speaker_id": "0"},
        {"start": 0.5, "end": 1.0, "text": "...", "speaker_id": "1"},
    ]

    with patch.object(asr_module, "_load_mono_audio_16k", return_value=samples):
        merged = _assign_speaker_ids_by_overlap(
            qwen_segments,
            diar_segments,
            audio_path=audio_path,
        )

    assert merged[0]["speaker_id"] == "1"


def test_cluster_speaker_embeddings_assigns_distinct_clusters() -> None:
    import numpy as np

    embeddings = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0, 0.0], dtype=np.float32),
    ]

    cluster_ids, confidences = _cluster_speaker_embeddings(embeddings)

    assert cluster_ids == [0, 1, 0]
    assert confidences[0] == 1.0
    assert confidences[1] == 1.0
    assert confidences[2] < SPEAKER_CONFIDENCE_LOW


def test_remap_speaker_ids_to_ten_slots_with_shared_minor_voice() -> None:
    from dv_backend.adapters.asr import (
        MAX_SPEAKER_VOICE_SLOTS,
        _remap_speaker_ids_to_slots,
    )

    segments = [{"start": 0.0, "end": 100.0, "text": "lead", "speaker_id": "0"}]
    for speaker_id in range(1, 9):
        segments.append(
            {
                "start": float(speaker_id * 10),
                "end": float(speaker_id * 10 + 40),
                "text": f"major-{speaker_id}",
                "speaker_id": str(speaker_id),
            }
        )
    for speaker_id in range(9, 12):
        segments.append(
            {
                "start": float(200 + speaker_id),
                "end": float(201 + speaker_id),
                "text": f"minor-{speaker_id}",
                "speaker_id": str(speaker_id),
            }
        )

    remapped = _remap_speaker_ids_to_slots(segments)
    ids = {segment["speaker_id"] for segment in remapped}
    assert ids.issubset({str(index) for index in range(MAX_SPEAKER_VOICE_SLOTS)})
    assert remapped[0]["speaker_id"] == "0"
    minor_segments = [segment for segment in remapped if str(segment["text"]).startswith("minor-")]
    assert minor_segments
    assert all(segment["speaker_id"] == "9" for segment in minor_segments)


def test_merge_qwen_and_diarization_splits_single_qwen_segment() -> None:
    from dv_backend.adapters.asr import _merge_qwen_and_diarization_segments

    qwen_segments = [{"start": 0.0, "end": 10.0, "text": "ABCDEFGHIJ"}]
    diar_segments = [
        {"start": 0.0, "end": 4.0, "text": "...", "speaker_id": "0"},
        {"start": 4.0, "end": 10.0, "text": "...", "speaker_id": "1"},
    ]

    merged = _merge_qwen_and_diarization_segments(qwen_segments, diar_segments)

    assert len(merged) == 2
    assert merged[0]["speaker_id"] == "0"
    assert merged[1]["speaker_id"] == "1"
    assert "".join(segment["text"] for segment in merged) == "ABCDEFGHIJ"


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
