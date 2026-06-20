from datetime import datetime, timezone
import json
from typing import Any
import uuid

from .adapters.separation import SUPPORTED_MIX_MODES
from .adapters.subtitles import (
    DEFAULT_SUBTITLE_BACKGROUND_COLOR,
    DEFAULT_SUBTITLE_BACKGROUND_OPACITY,
    DEFAULT_SUBTITLE_BACKGROUND_PADDING,
    DEFAULT_SUBTITLE_EDGE_MARGIN,
    DEFAULT_SUBTITLE_FONT_COLOR,
    DEFAULT_SUBTITLE_FONT_SIZE,
    DEFAULT_SUBTITLE_POSITION,
    SUPPORTED_SUBTITLE_POSITIONS,
    normalize_background_opacity,
    normalize_background_padding,
    normalize_edge_margin,
    normalize_font_size,
    normalize_hex_color,
    normalize_position,
)
from .adapters.tts import LEGACY_INVALID_PRESET_VOICES, VIENEU_PRESET_VOICES
from .database import Database


DEFAULT_SETTINGS: dict[str, Any] = {
    "cookies_browser": "none",
    "translation_backend": "google_free",
    "translation_source_language": "zh-CN",
    "translation_target_language": "vi",
    "tts_backend": "vieneu",
    "vieneu_voice": "Xuân Vĩnh",
    "vieneu_device": "cuda",
    "vieneu_model": "pnnbao-ump/VieNeu-TTS-v3-Turbo",
    "mix_mode": "duck",
    "gemini_api_keys": [],
    "gemini_key_cursor": 0,
    "gemini_translation_model": "gemini-2.5-flash",
    "asr_backend": "qwen3_asr",
    "qwen3_asr_model": "Qwen/Qwen3-ASR-1.7B",
    "qwen3_aligner_model": "Qwen/Qwen3-ForcedAligner-0.6B",
    "qwen3_device": "cuda:0",
    "speaker_diarization": False,
    "speaker_voices": {
        "0": "Xuân Vĩnh",
        "1": "Ngọc Linh",
        "2": "Gia Bảo",
    },
    "subtitles_enabled": True,
    "subtitle_font_size": DEFAULT_SUBTITLE_FONT_SIZE,
    "subtitle_font_color": DEFAULT_SUBTITLE_FONT_COLOR,
    "subtitle_background_color": DEFAULT_SUBTITLE_BACKGROUND_COLOR,
    "subtitle_background_opacity": DEFAULT_SUBTITLE_BACKGROUND_OPACITY,
    "subtitle_background_padding": DEFAULT_SUBTITLE_BACKGROUND_PADDING,
    "subtitle_edge_margin": DEFAULT_SUBTITLE_EDGE_MARGIN,
    "subtitle_position": DEFAULT_SUBTITLE_POSITION,
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
            self._migrate_legacy_pending_gemini_key(now)
            self._migrate_legacy_vieneu_voice(now)

    def _migrate_legacy_vieneu_voice(self, now: str) -> None:
        row = self.database.connection.execute(
            "SELECT value FROM settings WHERE key = 'vieneu_voice'"
        ).fetchone()
        if row is None:
            return
        try:
            voice = str(json.loads(row["value"])).strip()
        except (TypeError, json.JSONDecodeError):
            return
        if voice in LEGACY_INVALID_PRESET_VOICES or (
            not voice.lower().endswith(".wav") and voice not in VIENEU_PRESET_VOICES
        ):
            self.database.connection.execute(
                """
                INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                ("vieneu_voice", json.dumps(DEFAULT_SETTINGS["vieneu_voice"]), now),
            )

    def _migrate_legacy_pending_gemini_key(self, now: str) -> None:
        row = self.database.connection.execute(
            "SELECT value FROM settings WHERE key = 'gemini_api_key_add'"
        ).fetchone()
        if row is None:
            return
        try:
            pending_key = str(json.loads(row["value"])).strip()
        except (TypeError, json.JSONDecodeError):
            pending_key = ""
        raw_keys_row = self.database.connection.execute(
            "SELECT value FROM settings WHERE key = 'gemini_api_keys'"
        ).fetchone()
        keys = json.loads(raw_keys_row["value"]) if raw_keys_row else []
        if pending_key:
            keys = [
                item for item in keys
                if isinstance(item, dict) and item.get("key") != pending_key
            ]
            keys.append({
                "id": uuid.uuid4().hex,
                "key": pending_key,
                "label": mask_api_key(pending_key),
            })
            self.database.connection.execute(
                """
                INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                ("gemini_api_keys", json.dumps(keys), now),
            )
        self.database.connection.execute(
            "DELETE FROM settings WHERE key IN ('gemini_api_key_add', 'gemini_api_key_remove', 'gemini_api_key_update')"
        )

    def get_raw_all(self) -> dict[str, Any]:
        rows = self.database.connection.execute("SELECT key, value FROM settings").fetchall()
        return {row["key"]: json.loads(row["value"]) for row in rows}

    def get_all(self) -> dict[str, Any]:
        values = self.get_raw_all()
        values.pop("gemini_api_key_add", None)
        values.pop("gemini_api_key_remove", None)
        values.pop("gemini_api_key_update", None)
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

        mix_mode = values.get("mix_mode")
        if mix_mode is not None and mix_mode not in SUPPORTED_MIX_MODES:
            raise ValueError(
                "mix_mode must be one of: " + ", ".join(sorted(SUPPORTED_MIX_MODES))
            )

        subtitle_position = values.get("subtitle_position")
        if subtitle_position is not None:
            values["subtitle_position"] = normalize_position(subtitle_position)
            if values["subtitle_position"] not in SUPPORTED_SUBTITLE_POSITIONS:
                raise ValueError(
                    "subtitle_position must be one of: "
                    + ", ".join(sorted(SUPPORTED_SUBTITLE_POSITIONS))
                )

        if values.get("subtitle_font_size") is not None:
            values["subtitle_font_size"] = normalize_font_size(values["subtitle_font_size"])

        if values.get("subtitle_font_color") is not None:
            values["subtitle_font_color"] = normalize_hex_color(
                str(values["subtitle_font_color"]),
                fallback=DEFAULT_SUBTITLE_FONT_COLOR,
            )

        if values.get("subtitle_background_color") is not None:
            values["subtitle_background_color"] = normalize_hex_color(
                str(values["subtitle_background_color"]),
                fallback=DEFAULT_SUBTITLE_BACKGROUND_COLOR,
            )

        if values.get("subtitle_background_opacity") is not None:
            values["subtitle_background_opacity"] = normalize_background_opacity(
                values["subtitle_background_opacity"]
            )

        if values.get("subtitle_background_padding") is not None:
            values["subtitle_background_padding"] = normalize_background_padding(
                values["subtitle_background_padding"],
                font_size=int(values.get("subtitle_font_size") or DEFAULT_SUBTITLE_FONT_SIZE),
            )

        if values.get("subtitle_edge_margin") is not None:
            values["subtitle_edge_margin"] = normalize_edge_margin(
                values["subtitle_edge_margin"]
            )

        values = dict(values)
        add_gemini_key = values.pop("gemini_api_key_add", None)
        remove_gemini_key = values.pop("gemini_api_key_remove", None)
        update_gemini_key = values.pop("gemini_api_key_update", None)
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

            if update_gemini_key:
                raw = self.get_raw_all()
                update_id = str(update_gemini_key.get("id", ""))
                label = str(update_gemini_key.get("label", "")).strip()
                values["gemini_api_keys"] = [
                    {
                        **item,
                        "label": label or mask_api_key(item["key"]),
                    }
                    if isinstance(item, dict) and item.get("id") == update_id
                    else item
                    for item in raw.get("gemini_api_keys", [])
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
