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
from .adapters.tts import SUPPORTED_TTS_BACKENDS, VOXCPM_DEFAULT_MODEL
from .database import Database



DEFAULT_SETTINGS: dict[str, Any] = {
    "cookies_browser": "none",
    "translation_backend": "google_free",
    "translation_source_language": "zh-CN",
    "translation_target_language": "vi",
    "voxcpm_model": VOXCPM_DEFAULT_MODEL,
    "voxcpm_device": "cuda:0",
    "voxcpm_ref_audio": "",
    "voxcpm_instruct": "",
    "voxcpm_auto_voice": True,
    "voxcpm_num_steps": 10,
    "voxcpm_batch_size": 4,
    "voxcpm_batch_flush_ms": 150,
    "voxcpm_cache_enabled": True,
    "mix_mode": "duck",
    "gemini_api_keys": [],
    "gemini_key_cursor": 0,
    "gemini_translation_model": "gemini-2.5-flash",
    "asr_backend": "qwen3_asr",
    "qwen3_asr_model": "Qwen/Qwen3-ASR-1.7B",
    "qwen3_aligner_model": "Qwen/Qwen3-ForcedAligner-0.6B",
    "qwen3_device": "cuda:0",
    "exact_timing_enabled": True,
    "exact_timing_tolerance_ms": 40,
    "exact_timing_max_stretch": 1.8,
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

        if values.get("exact_timing_tolerance_ms") is not None:
            try:
                tolerance_ms = float(values["exact_timing_tolerance_ms"])
            except (TypeError, ValueError) as error:
                raise ValueError("exact_timing_tolerance_ms must be a number.") from error
            values["exact_timing_tolerance_ms"] = max(0.0, min(300.0, tolerance_ms))

        if values.get("exact_timing_max_stretch") is not None:
            try:
                max_stretch = float(values["exact_timing_max_stretch"])
            except (TypeError, ValueError) as error:
                raise ValueError("exact_timing_max_stretch must be a number.") from error
            values["exact_timing_max_stretch"] = max(1.0, min(3.0, max_stretch))

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

        diarization_backend = values.get("diarization_backend")
        if diarization_backend is not None and diarization_backend not in SUPPORTED_DIARIZATION_BACKENDS:
            raise ValueError(
                "diarization_backend must be one of: "
                + ", ".join(sorted(SUPPORTED_DIARIZATION_BACKENDS))
            )

        fallback_backend = values.get("diarization_fallback_backend")
        if fallback_backend is not None and fallback_backend not in SUPPORTED_DIARIZATION_FALLBACK_BACKENDS:
            raise ValueError(
                "diarization_fallback_backend must be one of: "
                + ", ".join(sorted(SUPPORTED_DIARIZATION_FALLBACK_BACKENDS))
            )

        demucs_mode = values.get("diarization_demucs_mode")
        if demucs_mode is not None and demucs_mode not in SUPPORTED_DIARIZATION_DEMUCS_MODES:
            raise ValueError(
                "diarization_demucs_mode must be one of: "
                + ", ".join(sorted(SUPPORTED_DIARIZATION_DEMUCS_MODES))
            )

        for float_key, low, high in (
            ("speaker_assignment_min_coverage", 0.0, 1.0),
            ("speaker_assignment_min_margin", 0.0, 1.0),
            ("speaker_overlap_flag_threshold", 0.0, 1.0),
            ("speaker_review_confidence_threshold", 0.0, 1.0),
        ):
            if values.get(float_key) is not None:
                try:
                    parsed = float(values[float_key])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{float_key} must be a number.") from error
                values[float_key] = max(low, min(high, parsed))

        if values.get("speaker_profile_min_seconds") is not None:
            try:
                min_seconds = float(values["speaker_profile_min_seconds"])
            except (TypeError, ValueError) as error:
                raise ValueError("speaker_profile_min_seconds must be a number.") from error
            values["speaker_profile_min_seconds"] = max(0.5, min(30.0, min_seconds))

        if values.get("speaker_merge_gap_sec") is not None:
            try:
                gap = float(values["speaker_merge_gap_sec"])
            except (TypeError, ValueError) as error:
                raise ValueError("speaker_merge_gap_sec must be a number.") from error
            values["speaker_merge_gap_sec"] = max(0.0, min(2.0, gap))

        for int_key, low, high in (
            ("diarization_min_speakers", 1, 12),
            ("diarization_max_speakers", 1, 12),
        ):
            if values.get(int_key) is not None:
                try:
                    parsed = int(values[int_key])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{int_key} must be an integer.") from error
                values[int_key] = max(low, min(high, parsed))

        min_spk = values.get("diarization_min_speakers")
        max_spk = values.get("diarization_max_speakers")
        if min_spk is not None and max_spk is not None and int(min_spk) > int(max_spk):
            raise ValueError("diarization_min_speakers cannot exceed diarization_max_speakers.")

        tts_backend = values.get("tts_backend")
        if tts_backend is not None and tts_backend not in SUPPORTED_TTS_BACKENDS:
            raise ValueError(
                "tts_backend must be one of: " + ", ".join(SUPPORTED_TTS_BACKENDS)
            )

        if values.get("voxcpm_num_steps") is not None:
            try:
                steps = int(values["voxcpm_num_steps"])
            except (TypeError, ValueError) as error:
                raise ValueError("voxcpm_num_steps must be an integer.") from error
            values["voxcpm_num_steps"] = max(4, min(64, steps))

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
