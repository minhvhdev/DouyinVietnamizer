import json
from pathlib import Path

import pytest

from dv_backend.database import Database
from dv_backend.settings import SettingsService


def service(tmp_path: Path) -> SettingsService:
    database = Database(tmp_path / "app.db")
    database.migrate()
    return SettingsService(database)


def test_defaults_use_free_pipeline(tmp_path: Path) -> None:
    settings = service(tmp_path)

    assert settings.get_all()["translation_backend"] == "google_free"
    assert settings.get_all()["tts_backend"] == "omnivoice"
    assert settings.get_all()["omnivoice_model"]
    assert settings.get_all()["omnivoice_device"] == "cuda:0"
    assert settings.get_all()["omnivoice_num_steps"] == 32
    assert settings.get_all()["mix_mode"] == "background_only"
    assert settings.get_all()["exact_timing_enabled"] is True
    assert settings.get_all()["exact_timing_tolerance_ms"] == 40
    assert settings.get_all()["exact_timing_max_stretch"] == 1.2
    assert settings.get_all()["subtitles_enabled"] is True
    assert settings.get_all()["subtitle_font_size"] == 48
    assert settings.get_all()["subtitle_position"] == "bottom"
    assert settings.get_all()["asr_backend"] == "qwen3_asr"
    assert settings.get_all()["qwen3_asr_model"] == "Qwen/Qwen3-ASR-1.7B"
    assert settings.get_all()["gemini_api_keys"] == []
    assert settings.get_all()["asr_alignment_mode"] == "accurate"
    assert settings.get_all()["sparse_asr_enabled"] is False
    assert settings.get_all()["sparse_asr_min_silence_ratio"] == 0.35
    assert settings.get_all()["sparse_asr_chunk_sec"] == 25
    assert settings.get_all()["sparse_asr_padding_ms"] == 200
    assert settings.get_all()["sparse_asr_merge_gap_sec"] == 0.25
    assert settings.get_all()["vad_engine"] == "silero"
    assert settings.get_all()["silero_vad_threshold"] == 0.5
    assert settings.get_all()["vad_false_positive_filter_enabled"] is True
    assert settings.get_all()["vad_energy_filter_enabled"] is True
    assert settings.get_all()["vad_energy_min_vocal_ratio"] == 1.15
    assert settings.get_all()["vietnamese_speaking_rate_wps"] == 3.2
    assert settings.get_all()["tts_session_reuse_enabled"] is True
    assert settings.get_all()["tts_micro_batch_enabled"] is True
    assert settings.get_all()["exact_timing_max_safe_stretch"] == 1.25
    assert settings.get_all()["short_tts_lengthen_min_gap_sec"] == 1.5
    assert settings.get_all()["short_tts_lengthen_max_ratio"] == 1.6
    assert settings.get_all()["tts_global_speed"] == 1.0
    assert settings.get_all()["vad_adaptive_enabled"] is False
    assert settings.get_all()["vad_neural_fallback_enabled"] is False
    assert settings.get_all()["gpu_model_idle_timeout_sec"] == 60
    assert settings.get_all()["gpu_keep_warm_enabled"] is True
    assert settings.get_all()["gpu_max_resident_models"] == 1
    assert settings.get_all()["tts_conversion_strategy"] == "lazy_mix"
    assert settings.get_all()["telemetry_max_file_mb"] == 16


def test_mix_mode_accepts_background_only_and_duck(tmp_path: Path) -> None:
    settings = service(tmp_path)

    settings.update({"mix_mode": "background_only"})
    assert settings.get_all()["mix_mode"] == "background_only"

    settings.update({"mix_mode": "duck"})
    assert settings.get_all()["mix_mode"] == "duck"

    settings.update({"mix_mode": "separate"})
    assert settings.get_all()["mix_mode"] == "background_only"

    with pytest.raises(ValueError, match="mix_mode"):
        settings.update({"mix_mode": "invalid"})


def test_pipeline_optimization_settings_are_validated(tmp_path: Path) -> None:
    settings = service(tmp_path)

    updated = settings.update({
        "asr_alignment_mode": "balanced",
        "sparse_asr_enabled": True,
        "sparse_asr_min_silence_ratio": 0.5,
        "sparse_asr_chunk_sec": 40,
        "sparse_asr_padding_ms": 300,
        "tts_session_reuse_enabled": False,
        "tts_micro_batch_enabled": True,
        "exact_timing_max_safe_stretch": 1.3,
        "vad_adaptive_enabled": True,
        "vad_neural_fallback_enabled": True,
        "vad_engine": "silencedetect",
        "silero_vad_threshold": 0.6,
        "sparse_asr_merge_gap_sec": 0.4,
        "short_tts_lengthen_min_gap_sec": 2.0,
        "short_tts_lengthen_max_ratio": 1.4,
        "tts_global_speed": 1.05,
    })

    assert updated["asr_alignment_mode"] == "balanced"
    assert updated["sparse_asr_enabled"] is True
    assert updated["sparse_asr_min_silence_ratio"] == 0.5
    assert updated["sparse_asr_chunk_sec"] == 40
    assert updated["sparse_asr_padding_ms"] == 300
    assert updated["tts_session_reuse_enabled"] is False
    assert updated["tts_micro_batch_enabled"] is True
    assert updated["exact_timing_max_safe_stretch"] == 1.3
    assert updated["vad_adaptive_enabled"] is True
    assert updated["vad_neural_fallback_enabled"] is True
    assert updated["vad_engine"] == "silencedetect"
    assert updated["silero_vad_threshold"] == 0.6
    assert updated["sparse_asr_merge_gap_sec"] == 0.4
    assert updated["short_tts_lengthen_min_gap_sec"] == 2.0
    assert updated["short_tts_lengthen_max_ratio"] == 1.4
    assert updated["tts_global_speed"] == 1.05

    with pytest.raises(ValueError, match="vad_engine"):
        settings.update({"vad_engine": "webrtc"})

    with pytest.raises(ValueError, match="asr_alignment_mode"):
        settings.update({"asr_alignment_mode": "maximum"})


def test_tts_global_speed_is_clamped_to_one_through_two(tmp_path: Path) -> None:
    settings = service(tmp_path)

    assert settings.update({"tts_global_speed": 0.85})["tts_global_speed"] == 1.0
    assert settings.update({"tts_global_speed": 2.5})["tts_global_speed"] == 2.5
    assert settings.update({"tts_global_speed": 3.0})["tts_global_speed"] == 2.5
    assert settings.update({"tts_global_speed": 1.75})["tts_global_speed"] == 1.75


def test_gpu_settings_are_normalized(tmp_path: Path) -> None:
    settings = service(tmp_path)
    updated = settings.update({
        "gpu_model_idle_timeout_sec": 120.5,
        "gpu_keep_warm_enabled": False,
        "gpu_max_resident_models": 3,
    })
    assert updated["gpu_model_idle_timeout_sec"] == 120.5
    assert updated["gpu_keep_warm_enabled"] is False
    assert updated["gpu_max_resident_models"] == 3

    with pytest.raises(ValueError, match="gpu_max_resident_models"):
        settings.update({"gpu_max_resident_models": "abc"})


def test_conversion_and_telemetry_settings_are_validated(tmp_path: Path) -> None:
    settings = service(tmp_path)
    updated = settings.update({
        "tts_conversion_strategy": "lazy_mix",
        "telemetry_max_file_mb": 64,
    })
    assert updated["tts_conversion_strategy"] == "lazy_mix"
    assert updated["telemetry_max_file_mb"] == 64

    with pytest.raises(ValueError, match="tts_conversion_strategy"):
        settings.update({"tts_conversion_strategy": "batch"})


def test_exact_timing_settings_are_normalized(tmp_path: Path) -> None:
    settings = service(tmp_path)

    updated = settings.update({
        "exact_timing_tolerance_ms": 120.5,
        "exact_timing_max_stretch": 4.2,
    })
    assert updated["exact_timing_tolerance_ms"] == 120.5
    assert updated["exact_timing_max_stretch"] == 1.2

    with pytest.raises(ValueError, match="exact_timing_tolerance_ms"):
        settings.update({"exact_timing_tolerance_ms": "abc"})


def test_subtitle_settings_accepts_supported_values(tmp_path: Path) -> None:
    settings = service(tmp_path)

    updated = settings.update({
        "subtitle_font_size": 64,
        "subtitle_font_color": "#ffcc00",
        "subtitle_background_color": "#112233",
        "subtitle_background_opacity": 55,
        "subtitle_position": "top",
        "subtitle_edge_margin": 24,
    })

    assert updated["subtitle_font_size"] == 64
    assert updated["subtitle_font_color"] == "#FFCC00"
    assert updated["subtitle_background_color"] == "#112233"
    assert updated["subtitle_background_opacity"] == 55
    assert updated["subtitle_position"] == "top"
    assert updated["subtitle_edge_margin"] == 24


def test_update_does_not_replace_unrelated_settings(tmp_path: Path) -> None:
    settings = service(tmp_path)
    settings.update({"omnivoice_ref_audio": "C:/voice.wav"})

    rows = settings.database.connection.execute(
        "SELECT key, value FROM settings WHERE key IN ('omnivoice_ref_audio', 'translation_backend')"
    ).fetchall()
    values = {row["key"]: json.loads(row["value"]) for row in rows}

    assert values == {
        "omnivoice_ref_audio": "C:/voice.wav",
        "translation_backend": "google_free",
    }


def test_gemini_key_pool_adds_masks_and_removes_without_exposing_keys(tmp_path: Path) -> None:
    settings = service(tmp_path)

    masked = settings.update({"gemini_api_key_add": "  AIzaSySecret1234567890  "})

    assert masked["gemini_api_keys"] == [
        {
            "id": masked["gemini_api_keys"][0]["id"],
            "label": "AIza...7890",
            "masked": "AIza...7890",
        }
    ]
    assert "Secret" not in json.dumps(masked)

    raw = settings.get_raw_all()
    assert raw["gemini_api_keys"][0]["key"] == "AIzaSySecret1234567890"

    removed = settings.update({"gemini_api_key_remove": masked["gemini_api_keys"][0]["id"]})

    assert removed["gemini_api_keys"] == []


def test_settings_update_ignores_masked_gemini_keys_from_ui(tmp_path: Path) -> None:
    settings = service(tmp_path)
    settings.update({"gemini_api_key_add": "AIzaSySecret1234567890"})

    settings.update({"gemini_api_keys": [{"id": "masked", "masked": "AIza...7890"}]})

    assert settings.get_raw_all()["gemini_api_keys"][0]["key"] == "AIzaSySecret1234567890"


def test_gemini_key_pool_updates_label_without_exposing_secret(tmp_path: Path) -> None:
    settings = service(tmp_path)
    added = settings.update({"gemini_api_key_add": "AIzaSySecret1234567890"})
    key_id = added["gemini_api_keys"][0]["id"]

    updated = settings.update({
        "gemini_api_key_update": {"id": key_id, "label": "studio quota 1"}
    })

    assert updated["gemini_api_keys"] == [
        {"id": key_id, "label": "studio quota 1", "masked": "AIza...7890"}
    ]
    assert "Secret" not in json.dumps(updated)
    assert settings.get_raw_all()["gemini_api_keys"][0]["key"] == "AIzaSySecret1234567890"


def test_legacy_pending_gemini_key_setting_is_migrated(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    database.migrate()
    with database.connection:
        database.connection.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("gemini_api_key_add", json.dumps("AIzaSyLegacySecret1234567890"), "now"),
        )

    settings = SettingsService(database)

    masked = settings.get_all()["gemini_api_keys"]
    assert len(masked) == 1
    assert masked[0]["masked"] == "AIza...7890"
    raw = settings.get_raw_all()
    assert raw["gemini_api_keys"][0]["key"] == "AIzaSyLegacySecret1234567890"
    assert "gemini_api_key_add" not in raw


def test_openai_api_key_is_masked_and_persisted(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    database.migrate()
    settings = SettingsService(database)

    masked = settings.update({"openai_api_key": "  sk-secret1234567890  "})
    assert masked["openai_api_key_configured"] is True
    assert masked["openai_api_key_masked"] == "sk-s...7890"
    assert "openai_api_key" not in masked

    raw = settings.get_raw_all()
    assert raw["openai_api_key"] == "sk-secret1234567890"
    assert raw["openai_api_base"] == "https://api.openai.com/v1"


def test_switching_to_thai_accepts_thai_google_voice(tmp_path: Path) -> None:
    settings = service(tmp_path)

    updated = settings.update(
        {
            "translation_target_language": "th",
            "google_tts_voice": "th-TH-Standard-A",
            "edge_tts_voice": "th-TH-PremwadeeNeural",
        }
    )

    assert updated["translation_target_language"] == "th"
    assert updated["google_tts_voice"] == "th-TH-Standard-A"
    assert updated["edge_tts_voice"] == "th-TH-PremwadeeNeural"


def test_switching_to_thai_migrates_vietnamese_google_voice(tmp_path: Path) -> None:
    settings = service(tmp_path)

    updated = settings.update(
        {
            "translation_target_language": "th",
            "google_tts_voice": "vi-VN-Standard-A",
            "edge_tts_voice": "vi-VN-HoaiMyNeural",
        }
    )

    assert updated["translation_target_language"] == "th"
    assert updated["google_tts_voice"] == "th-TH-Standard-A"
    assert updated["edge_tts_voice"] == "th-TH-PremwadeeNeural"
