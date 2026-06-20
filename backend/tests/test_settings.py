import json
from pathlib import Path

import pytest

from dv_backend.database import Database
from dv_backend.settings import SettingsService


def service(tmp_path: Path) -> SettingsService:
    database = Database(tmp_path / "app.db")
    database.migrate()
    return SettingsService(database)


def test_defaults_use_free_portable_pipeline(tmp_path: Path) -> None:
    settings = service(tmp_path)

    assert settings.get_all()["cookies_browser"] == "none"
    assert settings.get_all()["translation_backend"] == "google_free"
    assert settings.get_all()["tts_backend"] == "vieneu"
    assert settings.get_all()["vieneu_voice"] == "Xuân Vĩnh"
    assert settings.get_all()["vieneu_device"] == "cuda"
    assert settings.get_all()["mix_mode"] == "duck"
    assert settings.get_all()["speaker_diarization"] is False
    assert settings.get_all()["subtitles_enabled"] is True
    assert settings.get_all()["subtitle_font_size"] == 48
    assert settings.get_all()["subtitle_position"] == "bottom"
    assert settings.get_all()["asr_backend"] == "qwen3_asr"
    assert settings.get_all()["qwen3_asr_model"] == "Qwen/Qwen3-ASR-1.7B"
    assert settings.get_all()["gemini_api_keys"] == []


def test_cookie_browser_accepts_only_supported_values(tmp_path: Path) -> None:
    settings = service(tmp_path)

    settings.update({"cookies_browser": "edge"})
    assert settings.get_all()["cookies_browser"] == "edge"

    with pytest.raises(ValueError, match="cookies_browser"):
        settings.update({"cookies_browser": "opera"})


def test_mix_mode_accepts_supported_values(tmp_path: Path) -> None:
    settings = service(tmp_path)

    settings.update({"mix_mode": "separate"})
    assert settings.get_all()["mix_mode"] == "separate"

    with pytest.raises(ValueError, match="mix_mode"):
        settings.update({"mix_mode": "invalid"})


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


def test_legacy_invalid_vieneu_voice_is_migrated(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    database.migrate()
    with database.connection:
        database.connection.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("vieneu_voice", json.dumps("Phương Trang"), "now"),
        )

    settings = SettingsService(database)

    assert settings.get_all()["vieneu_voice"] == "Xuân Vĩnh"


def test_update_does_not_replace_unrelated_settings(tmp_path: Path) -> None:
    settings = service(tmp_path)
    settings.update({"vieneu_voice": "Ngọc Linh"})

    rows = settings.database.connection.execute(
        "SELECT key, value FROM settings WHERE key IN ('vieneu_voice', 'translation_backend')"
    ).fetchall()
    values = {row["key"]: json.loads(row["value"]) for row in rows}

    assert values == {
        "vieneu_voice": "Ngọc Linh",
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
