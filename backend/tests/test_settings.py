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
    assert settings.get_all()["tts_backend"] == "edge"
    assert settings.get_all()["asr_backend"] == "whisper_cpu"


def test_cookie_browser_accepts_only_supported_values(tmp_path: Path) -> None:
    settings = service(tmp_path)

    settings.update({"cookies_browser": "edge"})
    assert settings.get_all()["cookies_browser"] == "edge"

    with pytest.raises(ValueError, match="cookies_browser"):
        settings.update({"cookies_browser": "opera"})


def test_update_does_not_replace_unrelated_settings(tmp_path: Path) -> None:
    settings = service(tmp_path)
    settings.update({"edge_tts_voice": "vi-VN-NamMinhNeural"})

    rows = settings.database.connection.execute(
        "SELECT key, value FROM settings WHERE key IN ('edge_tts_voice', 'translation_backend')"
    ).fetchall()
    values = {row["key"]: json.loads(row["value"]) for row in rows}

    assert values == {
        "edge_tts_voice": "vi-VN-NamMinhNeural",
        "translation_backend": "google_free",
    }
