"""Tests for voice WPS catalog helpers."""

from __future__ import annotations

import array
import wave
from pathlib import Path
from unittest.mock import patch

import pytest

from dv_backend.voice_duration_profile import load_profiles
from dv_backend.duration_predictor import count_vietnamese_syllables, count_speech_units
from dv_backend.voice_wps_catalog import (
    make_catalog_key,
    measure_voice_wps,
    parse_catalog_key,
    set_voice_wps,
    wps_from_profile,
)


def test_parse_catalog_key_roundtrip():
    key = make_catalog_key("edge_tts", "preset", "vi-VN-HoaiMyNeural")
    assert parse_catalog_key(key) == ("edge_tts", "preset", "vi-VN-HoaiMyNeural")


def test_wps_from_profile_prefers_words_per_second():
    wps, source = wps_from_profile({"words_per_second": 3.5, "source": "manual"}, language="vi")
    assert wps == 3.5
    assert source == "manual"


def test_wps_from_profile_derives_from_syllables():
    wps, source = wps_from_profile({"syllables_per_second": 3.68, "source": "calibration"}, language="vi")
    assert wps == 3.2
    assert source == "calibration"


def test_set_voice_wps_persists_manual_profile(tmp_path: Path):
    data_dir = tmp_path
    catalog_key = make_catalog_key("edge_tts", "preset", "vi-VN-HoaiMyNeural")
    result = set_voice_wps(
        data_dir=data_dir,
        catalog_key=catalog_key,
        words_per_second=3.6,
        language="vi",
    )
    assert result["words_per_second"] == 3.6
    store = load_profiles(data_dir)
    assert store["profiles"]
    profile = next(iter(store["profiles"].values()))
    assert profile["words_per_second"] == 3.6
    assert profile["source"] == "manual"


def test_parse_catalog_key_invalid():
    with pytest.raises(ValueError):
        parse_catalog_key("invalid-key")


def test_auto_measure_uses_100_sample_calibration_and_outlier_pipeline(tmp_path: Path):
    synthesized_texts: list[str] = []

    def synthesize_sample(*, phrase: str, **_kwargs) -> Path:
        synthesized_texts.append(phrase)
        output_path = tmp_path / f"sample-{len(synthesized_texts)}.wav"
        sample_rate = 16000
        syllables = max(2, count_vietnamese_syllables(phrase))
        duration = syllables / 4.0
        frames = int(sample_rate * duration)
        samples = array.array("h", [9000] * frames)
        with wave.open(str(output_path), "w") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(samples.tobytes())
        return output_path

    with patch(
        "dv_backend.voice_wps_catalog._synthesize_measure_sample",
        side_effect=synthesize_sample,
    ):
        result = measure_voice_wps(
            data_dir=tmp_path,
            settings={},
            catalog_key=make_catalog_key("edge_tts", "preset", "vi-VN-HoaiMyNeural"),
            language="vi",
        )

    assert len(synthesized_texts) == 100
    assert result["sample_count_total"] == 100
    assert result["sample_count_accepted"] >= 90
    assert result["profile_source"] == "bootstrap_calibration"


def test_auto_measure_thai_uses_thai_100_sample_dataset(tmp_path: Path):
    synthesized_texts: list[str] = []

    def synthesize_sample(*, phrase: str, **_kwargs) -> Path:
        synthesized_texts.append(phrase)
        output_path = tmp_path / f"th-sample-{len(synthesized_texts)}.wav"
        sample_rate = 16000
        units = max(2, count_speech_units(phrase, "th"))
        duration = units / 4.0
        frames = int(sample_rate * duration)
        samples = array.array("h", [9000] * frames)
        with wave.open(str(output_path), "w") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(samples.tobytes())
        return output_path

    with patch(
        "dv_backend.voice_wps_catalog._synthesize_measure_sample",
        side_effect=synthesize_sample,
    ):
        result = measure_voice_wps(
            data_dir=tmp_path,
            settings={},
            catalog_key=make_catalog_key("edge_tts", "preset", "th-TH-PremwadeeNeural"),
            language="th",
        )

    assert len(synthesized_texts) == 100
    assert any("สวัสดี" in text or "ขอบคุณ" in text for text in synthesized_texts)
    assert result["sample_count_total"] == 100
    assert result["profile_source"] == "bootstrap_calibration"
