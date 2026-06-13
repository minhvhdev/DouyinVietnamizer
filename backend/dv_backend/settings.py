from datetime import datetime, timezone
import json
from typing import Any

from .database import Database


DEFAULT_SETTINGS: dict[str, Any] = {
    "cookies_browser": "none",
    "translation_backend": "google_free",
    "translation_source_language": "zh-CN",
    "translation_target_language": "vi",
    "tts_backend": "edge",
    "edge_tts_voice": "vi-VN-HoaiMyNeural",
    "edge_tts_rate": "+0%",
    "edge_tts_pitch": "+0Hz",
    "edge_tts_volume": "+0%",
    "asr_backend": "whisper_cpu",
    "whisper_model_path": "",
}

SUPPORTED_COOKIE_BROWSERS = {"none", "edge", "chrome", "firefox", "brave"}


class SettingsService:
    def __init__(self, database: Database) -> None:
        self.database = database
        self.ensure_defaults()

    def ensure_defaults(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.database.connection:
            for key, value in DEFAULT_SETTINGS.items():
                self.database.connection.execute(
                    "INSERT OR IGNORE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
                    (key, json.dumps(value), now),
                )

    def get_all(self) -> dict[str, Any]:
        rows = self.database.connection.execute("SELECT key, value FROM settings").fetchall()
        return {row["key"]: json.loads(row["value"]) for row in rows}

    def update(self, values: dict[str, Any]) -> dict[str, Any]:
        cookie_browser = values.get("cookies_browser")
        if cookie_browser is not None and cookie_browser not in SUPPORTED_COOKIE_BROWSERS:
            raise ValueError(
                "cookies_browser must be one of: "
                + ", ".join(sorted(SUPPORTED_COOKIE_BROWSERS))
            )

        now = datetime.now(timezone.utc).isoformat()
        with self.database.connection:
            for key, value in values.items():
                self.database.connection.execute(
                    """
                    INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                    """,
                    (key, json.dumps(value), now),
                )
        return self.get_all()
