from datetime import datetime, timezone
import json
from typing import Any
import uuid

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
    "gemini_api_keys": [],
    "gemini_key_cursor": 0,
    "gemini_translation_model": "gemini-2.5-flash",
    "gemini_tts_model": "gemini-2.5-flash-preview-tts",
    "gemini_tts_voice": "Zephyr",
    "asr_backend": "whisper_cpu",
    "whisper_model_path": "",
}

SUPPORTED_COOKIE_BROWSERS = {"none", "edge", "chrome", "firefox", "brave"}


def mask_api_key(api_key: str) -> str:
    if len(api_key) <= 8:
        return "****"
    return f"{api_key[:4]}...{api_key[-4:]}"


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

    def get_raw_all(self) -> dict[str, Any]:
        rows = self.database.connection.execute("SELECT key, value FROM settings").fetchall()
        return {row["key"]: json.loads(row["value"]) for row in rows}

    def get_all(self) -> dict[str, Any]:
        values = self.get_raw_all()
        values["gemini_api_keys"] = [
            {
                "id": item["id"],
                "label": item.get("label") or mask_api_key(item["key"]),
                "masked": mask_api_key(item["key"]),
            }
            for item in values.get("gemini_api_keys", [])
            if isinstance(item, dict) and item.get("key")
        ]
        return values

    def update(self, values: dict[str, Any]) -> dict[str, Any]:
        cookie_browser = values.get("cookies_browser")
        if cookie_browser is not None and cookie_browser not in SUPPORTED_COOKIE_BROWSERS:
            raise ValueError(
                "cookies_browser must be one of: "
                + ", ".join(sorted(SUPPORTED_COOKIE_BROWSERS))
            )

        values = dict(values)
        add_gemini_key = values.pop("gemini_api_key_add", None)
        remove_gemini_key = values.pop("gemini_api_key_remove", None)
        values.pop("gemini_api_keys", None)

        now = datetime.now(timezone.utc).isoformat()
        with self.database.connection:
            if add_gemini_key:
                raw = self.get_raw_all()
                key = str(add_gemini_key).strip()
                keys = [
                    item for item in raw.get("gemini_api_keys", [])
                    if isinstance(item, dict) and item.get("key") != key
                ]
                keys.append({
                    "id": uuid.uuid4().hex,
                    "key": key,
                    "label": mask_api_key(key),
                })
                values["gemini_api_keys"] = keys

            if remove_gemini_key:
                raw = self.get_raw_all()
                values["gemini_api_keys"] = [
                    item for item in raw.get("gemini_api_keys", [])
                    if isinstance(item, dict) and item.get("id") != remove_gemini_key
                ]

            for key, value in values.items():
                self.database.connection.execute(
                    """
                    INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                    """,
                    (key, json.dumps(value), now),
                )
        return self.get_all()
