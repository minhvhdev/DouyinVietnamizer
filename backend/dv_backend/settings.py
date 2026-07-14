from datetime import datetime, timezone
import json
from typing import Any
import uuid

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
from .adapters.openai_compat import DEFAULT_OPENAI_API_BASE, normalize_openai_api_base
from .adapters.tts import SUPPORTED_TTS_BACKENDS, TTS_VOICE_INSTRUCT_PREFIX
from .omnivoice_env import OMNIVOICE_DEFAULT_MODEL
from .database import Database
from .dubbing_languages import SUPPORTED_DUB_LANGUAGES, default_speaking_rate_wps, dub_language_config, dub_language_from_settings, normalize_dub_language, voice_defaults_for_language


SUPPORTED_TRANSLATION_BACKENDS = {"google_free", "gemini", "openai"}


DEFAULT_SETTINGS: dict[str, Any] = {
    "translation_backend": "google_free",
    "translation_source_language": "zh-CN",
    "translation_target_language": "vi",
    "tts_backend": "omnivoice",
    "edge_tts_voice": "vi-VN-HoaiMyNeural",
    "google_tts_voice": "vi-VN-Standard-A",
    "google_tts_api_key": "",
    "google_tts_speaking_rate": 1.0,
    "gemini_tts_model": "gemini-2.5-flash-preview-tts",
    "gemini_tts_voice": "Zephyr",
    "omnivoice_model": OMNIVOICE_DEFAULT_MODEL,
    "omnivoice_device": "cuda:0",
    "omnivoice_ref_audio": "",
    "omnivoice_ref_text": "",
    "omnivoice_instruct": "",
    "omnivoice_auto_voice": True,
    "omnivoice_num_steps": 32,
    "omnivoice_language_id": "",
    "omnivoice_audio_chunk_threshold": 30.0,
    "omnivoice_audio_chunk_duration": 15.0,
    "omnivoice_external_chunking_enabled": False,
    "omnivoice_chunk_fidelity_fallback_full_segment": True,
    "omnivoice_chunk_retry_on_fidelity_failure": False,
    "omnivoice_chunk_target_chars": 180,
    "omnivoice_chunk_max_chars": 220,
    "omnivoice_chunk_min_chars": 40,
    "omnivoice_long_text_threshold": 240,
    "omnivoice_very_long_text_threshold": 500,
    "omnivoice_pause_comma_ms": 140,
    "omnivoice_pause_sentence_ms": 260,
    "omnivoice_pause_hard_ms": 50,
    "omnivoice_chunk_max_retries": 2,
    "omnivoice_chunk_retry_max_chars_1": 220,
    "omnivoice_chunk_retry_max_chars_2": 140,
    "omnivoice_chunk_retry_max_chars_3": 90,
    "omnivoice_fidelity_check_enabled": True,
    "tts_fidelity_retry_max_attempts": 1,
    "omnivoice_fidelity_check_min_chars": 240,
    "omnivoice_fidelity_check_all_segments": True,
    "omnivoice_fidelity_good_threshold": 0.85,
    "omnivoice_fidelity_review_threshold": 0.70,
    "omnivoice_fidelity_critical_threshold": 0.55,
    "mix_mode": "background_only",
    "gemini_api_keys": [],
    "gemini_key_cursor": 0,
    "gemini_translation_model": "gemini-2.5-flash",
    "openai_api_base": DEFAULT_OPENAI_API_BASE,
    "openai_api_key": "",
    "openai_translation_model": "",
    "asr_backend": "qwen3_asr",
    "qwen3_asr_model": "Qwen/Qwen3-ASR-1.7B",
    "qwen3_aligner_model": "Qwen/Qwen3-ForcedAligner-0.6B",
    "qwen3_device": "cuda:0",
    "exact_timing_enabled": True,
    "exact_timing_tolerance_ms": 40,
    "exact_timing_max_stretch": 1.2,
    "exact_timing_max_safe_stretch": 1.25,
    "edge_tts_overflow_speed_max": 1.15,
    "edge_tts_overflow_speed_hard_max": 1.2,
    "short_tts_lengthen_min_gap_sec": 1.5,
    "short_tts_lengthen_max_ratio": 1.6,
    "tts_global_speed": 1.0,
    "asr_alignment_mode": "accurate",
    "sparse_asr_enabled": False,
    "sparse_asr_min_silence_ratio": 0.35,
    "sparse_asr_chunk_sec": 25,
    "sparse_asr_padding_ms": 200,
    "sparse_asr_merge_gap_sec": 0.25,
    "vad_engine": "silero",
    "silero_vad_threshold": 0.5,
    "silero_vad_min_speech_duration_ms": 250,
    "silero_vad_min_silence_duration_ms": 300,
    "silero_vad_speech_pad_ms": 150,
    "silencedetect_noise_db": -30,
    "silencedetect_min_silence_sec": 0.5,
    "vad_false_positive_filter_enabled": True,
    "vad_energy_filter_enabled": True,
    "vad_energy_min_vocal_ratio": 1.15,
    "vietnamese_speaking_rate_wps": 3.2,
    "timing_safety_buffer_ms": 120,
    "timing_min_fill_ratio": 0.55,
    "duration_trim_max_ratio": 0.15,
    "duration_trim_max_ms": 600,
    "release_gate_blocking_enabled": True,
    "timing_candidate_translation_enabled": True,
    "timing_translation_candidate_count": 3,
    "timing_max_tts_attempts": 3,
    "timing_max_candidate_tts_attempts": 2,
    "timing_max_llm_rewrite_attempts": 1,
    "timing_preferred_tempo_min": 0.94,
    "timing_preferred_tempo_max": 1.08,
    "timing_warning_tempo_min": 0.90,
    "timing_warning_tempo_max": 1.12,
    "voice_duration_profile_enabled": True,
    "voice_auto_calibration_mode": "off",
    "tts_session_reuse_enabled": True,
    "tts_micro_batch_enabled": True,
    "vad_adaptive_enabled": False,
    "vad_neural_fallback_enabled": False,
    "gpu_model_idle_timeout_sec": 60,
    "gpu_keep_warm_enabled": True,
    "gpu_max_resident_models": 1,
    "tts_conversion_strategy": "lazy_mix",
    "telemetry_max_file_mb": 16,
    "subtitles_enabled": True,
    "subtitle_max_chars_per_line": 40,
    "subtitle_max_lines_per_cue": 2,
    "subtitle_min_cue_duration_ms": 700,
    "subtitle_max_cue_duration_ms": 5500,
    "subtitle_inter_cue_gap_ms": 50,
    "subtitle_font_size": DEFAULT_SUBTITLE_FONT_SIZE,
    "subtitle_font_color": DEFAULT_SUBTITLE_FONT_COLOR,
    "subtitle_background_color": DEFAULT_SUBTITLE_BACKGROUND_COLOR,
    "subtitle_background_opacity": DEFAULT_SUBTITLE_BACKGROUND_OPACITY,
    "subtitle_background_padding": DEFAULT_SUBTITLE_BACKGROUND_PADDING,
    "subtitle_edge_margin": DEFAULT_SUBTITLE_EDGE_MARGIN,
    "subtitle_position": DEFAULT_SUBTITLE_POSITION,
    "cookies_file": "",
}

SUPPORTED_MIX_MODES = {"background_only", "duck"}
SUPPORTED_ASR_ALIGNMENT_MODES = {"fast", "balanced", "accurate"}
SUPPORTED_VAD_ENGINES = {"silero", "silencedetect"}


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
            self._migrate_legacy_omnivoice_chunk_defaults(now)



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

    def _migrate_legacy_omnivoice_chunk_defaults(self, now: str) -> None:
        """Reset pre-OmniVoice chunk tuning values to official demo defaults."""
        legacy_pairs = {
            ("omnivoice_audio_chunk_threshold", 8.0): 30.0,
            ("omnivoice_audio_chunk_duration", 10.0): 15.0,
        }
        for (key, legacy_value), default_value in legacy_pairs.items():
            row = self.database.connection.execute(
                "SELECT value FROM settings WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                continue
            try:
                current = float(json.loads(row["value"]))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if abs(current - legacy_value) < 1e-6:
                self.database.connection.execute(
                    """
                    INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                    """,
                    (key, json.dumps(default_value), now),
                )

    def get_raw_all(self) -> dict[str, Any]:
        rows = self.database.connection.execute("SELECT key, value FROM settings").fetchall()
        return {row["key"]: json.loads(row["value"]) for row in rows}

    def get_all(self) -> dict[str, Any]:
        values = self.get_raw_all()
        values.pop("gemini_api_key_add", None)
        values.pop("gemini_api_key_remove", None)
        values.pop("gemini_api_key_update", None)
        values.pop("cookies_browser", None)
        values.pop("omnivoice_speed", None)
        values["gemini_api_keys"] = [
            {
                "id": item["id"],
                "label": item.get("label") or mask_api_key(item["key"]),
                "masked": mask_api_key(item["key"]),
            }
            for item in values.get("gemini_api_keys", [])
            if isinstance(item, dict) and item.get("key")
        ]
        raw_google_tts_key = str(values.get("google_tts_api_key") or "").strip()
        values["google_tts_api_key_configured"] = bool(raw_google_tts_key)
        values["google_tts_api_key_masked"] = mask_api_key(raw_google_tts_key) if raw_google_tts_key else ""
        values.pop("google_tts_api_key", None)
        raw_openai_key = str(values.get("openai_api_key") or "").strip()
        values["openai_api_key_configured"] = bool(raw_openai_key)
        values["openai_api_key_masked"] = mask_api_key(raw_openai_key) if raw_openai_key else ""
        values.pop("openai_api_key", None)
        values["openai_api_base"] = normalize_openai_api_base(str(values.get("openai_api_base") or DEFAULT_OPENAI_API_BASE))
        return values

    def update(self, values: dict[str, Any]) -> dict[str, Any]:
        values.pop("cookies_browser", None)
        values.pop("omnivoice_speed", None)

        translation_backend = values.get("translation_backend")
        if translation_backend is not None:
            backend = str(translation_backend).strip().lower()
            if backend not in SUPPORTED_TRANSLATION_BACKENDS:
                raise ValueError(
                    "translation_backend must be one of: "
                    + ", ".join(sorted(SUPPORTED_TRANSLATION_BACKENDS))
                )
            values["translation_backend"] = backend

        target_language = values.get("translation_target_language")
        if target_language is not None:
            normalized = normalize_dub_language(str(target_language))
            if normalized not in SUPPORTED_DUB_LANGUAGES:
                raise ValueError(
                    "translation_target_language must be one of: "
                    + ", ".join(sorted(SUPPORTED_DUB_LANGUAGES))
                )
            values["translation_target_language"] = normalized
            defaults = voice_defaults_for_language(normalized)
            merged_settings = {**self.get_raw_all(), **values}
            google_voice = str(merged_settings.get("google_tts_voice") or "").strip()
            google_locale = str(dub_language_config(normalized)["google_locale"]).lower()
            if not google_voice or not google_voice.lower().startswith(google_locale):
                values["google_tts_voice"] = defaults["google_tts_voice"]
            edge_voice = str(merged_settings.get("edge_tts_voice") or "").strip()
            edge_locale = str(dub_language_config(normalized)["edge_locale"]).lower()
            if not edge_voice or not edge_voice.lower().startswith(edge_locale):
                values["edge_tts_voice"] = defaults["edge_tts_voice"]

        if values.get("openai_api_base") is not None:
            values["openai_api_base"] = normalize_openai_api_base(str(values["openai_api_base"]))

        mix_mode = values.get("mix_mode")
        if mix_mode is not None:
            mix_mode = str(mix_mode).strip().lower()
            if mix_mode == "separate":
                mix_mode = "background_only"
            if mix_mode not in SUPPORTED_MIX_MODES:
                raise ValueError(
                    "mix_mode must be one of: " + ", ".join(sorted(SUPPORTED_MIX_MODES))
                )
            values["mix_mode"] = mix_mode

        alignment_mode = values.get("asr_alignment_mode")
        if alignment_mode is not None:
            mode = str(alignment_mode).strip().lower()
            if mode not in SUPPORTED_ASR_ALIGNMENT_MODES:
                raise ValueError(
                    "asr_alignment_mode must be one of: "
                    + ", ".join(sorted(SUPPORTED_ASR_ALIGNMENT_MODES))
                )
            values["asr_alignment_mode"] = mode

        for flag in (
            "sparse_asr_enabled",
            "tts_session_reuse_enabled",
            "tts_micro_batch_enabled",
            "vad_adaptive_enabled",
            "vad_neural_fallback_enabled",
            "vad_false_positive_filter_enabled",
            "vad_energy_filter_enabled",
        ):
            if values.get(flag) is not None:
                values[flag] = bool(values[flag])

        vad_engine = values.get("vad_engine")
        if vad_engine is not None:
            engine = str(vad_engine).strip().lower()
            if engine not in SUPPORTED_VAD_ENGINES:
                raise ValueError(
                    "vad_engine must be one of: " + ", ".join(sorted(SUPPORTED_VAD_ENGINES))
                )
            values["vad_engine"] = engine

        if values.get("silero_vad_threshold") is not None:
            try:
                threshold = float(values["silero_vad_threshold"])
            except (TypeError, ValueError) as error:
                raise ValueError("silero_vad_threshold must be a number.") from error
            values["silero_vad_threshold"] = max(0.0, min(1.0, threshold))

        for key, minimum, maximum in (
            ("silero_vad_min_speech_duration_ms", 0, 5000),
            ("silero_vad_min_silence_duration_ms", 0, 5000),
            ("silero_vad_speech_pad_ms", 0, 2000),
        ):
            if values.get(key) is not None:
                try:
                    parsed = int(values[key])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{key} must be an integer.") from error
                values[key] = max(minimum, min(maximum, parsed))

        if values.get("silencedetect_noise_db") is not None:
            try:
                noise_db = float(values["silencedetect_noise_db"])
            except (TypeError, ValueError) as error:
                raise ValueError("silencedetect_noise_db must be a number.") from error
            values["silencedetect_noise_db"] = max(-90.0, min(0.0, noise_db))

        if values.get("silencedetect_min_silence_sec") is not None:
            try:
                min_silence_sec = float(values["silencedetect_min_silence_sec"])
            except (TypeError, ValueError) as error:
                raise ValueError("silencedetect_min_silence_sec must be a number.") from error
            values["silencedetect_min_silence_sec"] = max(0.05, min(5.0, min_silence_sec))

        if values.get("sparse_asr_merge_gap_sec") is not None:
            try:
                merge_gap_sec = float(values["sparse_asr_merge_gap_sec"])
            except (TypeError, ValueError) as error:
                raise ValueError("sparse_asr_merge_gap_sec must be a number.") from error
            values["sparse_asr_merge_gap_sec"] = max(0.0, min(2.0, merge_gap_sec))

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
            values["exact_timing_max_stretch"] = max(1.0, min(1.2, max_stretch))

        if values.get("exact_timing_max_safe_stretch") is not None:
            try:
                safe_stretch = float(values["exact_timing_max_safe_stretch"])
            except (TypeError, ValueError) as error:
                raise ValueError("exact_timing_max_safe_stretch must be a number.") from error
            values["exact_timing_max_safe_stretch"] = max(1.0, min(1.35, safe_stretch))

        if values.get("short_tts_lengthen_min_gap_sec") is not None:
            try:
                min_gap = float(values["short_tts_lengthen_min_gap_sec"])
            except (TypeError, ValueError) as error:
                raise ValueError("short_tts_lengthen_min_gap_sec must be a number.") from error
            values["short_tts_lengthen_min_gap_sec"] = max(0.2, min(5.0, min_gap))

        if values.get("short_tts_lengthen_max_ratio") is not None:
            try:
                max_ratio = float(values["short_tts_lengthen_max_ratio"])
            except (TypeError, ValueError) as error:
                raise ValueError("short_tts_lengthen_max_ratio must be a number.") from error
            values["short_tts_lengthen_max_ratio"] = max(1.05, min(2.0, max_ratio))

        if values.get("tts_global_speed") is not None:
            try:
                speed = float(values["tts_global_speed"])
            except (TypeError, ValueError) as error:
                raise ValueError("tts_global_speed must be a number.") from error
            values["tts_global_speed"] = max(1.0, min(2.5, speed))

        if values.get("vietnamese_speaking_rate_wps") is not None:
            try:
                rate = float(values["vietnamese_speaking_rate_wps"])
            except (TypeError, ValueError) as error:
                raise ValueError("vietnamese_speaking_rate_wps must be a number.") from error
            values["vietnamese_speaking_rate_wps"] = max(2.0, min(5.0, rate))

        if values.get("timing_safety_buffer_ms") is not None:
            try:
                buffer_ms = float(values["timing_safety_buffer_ms"])
            except (TypeError, ValueError) as error:
                raise ValueError("timing_safety_buffer_ms must be a number.") from error
            values["timing_safety_buffer_ms"] = max(0.0, min(500.0, buffer_ms))

        if values.get("timing_min_fill_ratio") is not None:
            try:
                fill_ratio = float(values["timing_min_fill_ratio"])
            except (TypeError, ValueError) as error:
                raise ValueError("timing_min_fill_ratio must be a number.") from error
            values["timing_min_fill_ratio"] = max(0.0, min(1.0, fill_ratio))

        if values.get("vad_energy_min_vocal_ratio") is not None:
            try:
                ratio = float(values["vad_energy_min_vocal_ratio"])
            except (TypeError, ValueError) as error:
                raise ValueError("vad_energy_min_vocal_ratio must be a number.") from error
            values["vad_energy_min_vocal_ratio"] = max(0.8, min(3.0, ratio))

        if values.get("sparse_asr_min_silence_ratio") is not None:
            try:
                silence_ratio = float(values["sparse_asr_min_silence_ratio"])
            except (TypeError, ValueError) as error:
                raise ValueError("sparse_asr_min_silence_ratio must be a number.") from error
            values["sparse_asr_min_silence_ratio"] = max(0.0, min(0.95, silence_ratio))

        if values.get("sparse_asr_chunk_sec") is not None:
            try:
                chunk_sec = int(values["sparse_asr_chunk_sec"])
            except (TypeError, ValueError) as error:
                raise ValueError("sparse_asr_chunk_sec must be an integer.") from error
            values["sparse_asr_chunk_sec"] = max(5, min(120, chunk_sec))

        if values.get("sparse_asr_padding_ms") is not None:
            try:
                padding_ms = int(values["sparse_asr_padding_ms"])
            except (TypeError, ValueError) as error:
                raise ValueError("sparse_asr_padding_ms must be an integer.") from error
            values["sparse_asr_padding_ms"] = max(0, min(1000, padding_ms))

        if values.get("gpu_model_idle_timeout_sec") is not None:
            try:
                idle_sec = float(values["gpu_model_idle_timeout_sec"])
            except (TypeError, ValueError) as error:
                raise ValueError("gpu_model_idle_timeout_sec must be a number.") from error
            values["gpu_model_idle_timeout_sec"] = max(0.0, min(3600.0, idle_sec))

        for flag in ("gpu_keep_warm_enabled",):
            if values.get(flag) is not None:
                values[flag] = bool(values[flag])

        if values.get("gpu_max_resident_models") is not None:
            try:
                max_models = int(values["gpu_max_resident_models"])
            except (TypeError, ValueError) as error:
                raise ValueError("gpu_max_resident_models must be an integer.") from error
            values["gpu_max_resident_models"] = max(1, min(4, max_models))

        if values.get("tts_conversion_strategy") is not None:
            candidate = str(values["tts_conversion_strategy"]).strip().lower()
            if candidate not in {"per_segment", "lazy_mix"}:
                raise ValueError("tts_conversion_strategy must be one of: per_segment, lazy_mix")
            values["tts_conversion_strategy"] = candidate

        if values.get("telemetry_max_file_mb") is not None:
            try:
                max_mb = float(values["telemetry_max_file_mb"])
            except (TypeError, ValueError) as error:
                raise ValueError("telemetry_max_file_mb must be a number.") from error
            values["telemetry_max_file_mb"] = max(0.0, min(1024.0, max_mb))

        for key, minimum, maximum in (
            ("subtitle_max_chars_per_line", 12, 120),
            ("subtitle_max_lines_per_cue", 1, 4),
            ("subtitle_min_cue_duration_ms", 200, 4000),
            ("subtitle_max_cue_duration_ms", 1500, 15000),
            ("subtitle_inter_cue_gap_ms", 0, 1000),
        ):
            if values.get(key) is not None:
                try:
                    parsed = int(values[key])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{key} must be an integer.") from error
                values[key] = max(minimum, min(maximum, parsed))

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

        if values.get("google_tts_speaking_rate") is not None:
            try:
                rate = float(values["google_tts_speaking_rate"])
            except (TypeError, ValueError) as error:
                raise ValueError("google_tts_speaking_rate must be a number.") from error
            values["google_tts_speaking_rate"] = max(0.5, min(1.5, rate))

        if values.get("google_tts_voice") is not None:
            from .adapters.google_tts import GOOGLE_TTS_VOICE_IDS, DEFAULT_GOOGLE_TTS_VOICE

            voice = str(values["google_tts_voice"]).strip()
            if voice and voice not in GOOGLE_TTS_VOICE_IDS:
                merged_settings = {**self.get_raw_all(), **values}
                voice = voice_defaults_for_language(
                    dub_language_from_settings(merged_settings)
                )["google_tts_voice"]
            values["google_tts_voice"] = voice or DEFAULT_GOOGLE_TTS_VOICE

        if values.get("cookies_file") is not None:
            cookies_file = str(values.get("cookies_file") or "").strip()
            values["cookies_file"] = cookies_file

        tts_backend = values.get("tts_backend")
        if tts_backend is not None and tts_backend not in SUPPORTED_TTS_BACKENDS:
            raise ValueError(
                "tts_backend must be one of: " + ", ".join(SUPPORTED_TTS_BACKENDS)
            )

        if values.get("omnivoice_num_steps") is not None:
            try:
                steps = int(values["omnivoice_num_steps"])
            except (TypeError, ValueError) as error:
                raise ValueError("omnivoice_num_steps must be an integer.") from error
            values["omnivoice_num_steps"] = max(4, min(64, steps))

        if values.get("omnivoice_language_id") is not None:
            values["omnivoice_language_id"] = str(values.get("omnivoice_language_id") or "").strip()

        for chunk_key, minimum, maximum in (
            ("omnivoice_audio_chunk_threshold", 4.0, 60.0),
            ("omnivoice_audio_chunk_duration", 4.0, 30.0),
        ):
            if values.get(chunk_key) is not None:
                try:
                    chunk_value = float(values[chunk_key])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{chunk_key} must be a number.") from error
                values[chunk_key] = max(minimum, min(maximum, chunk_value))

        values = dict(values)
        add_gemini_key = values.pop("gemini_api_key_add", None)
        remove_gemini_key = values.pop("gemini_api_key_remove", None)
        update_gemini_key = values.pop("gemini_api_key_update", None)
        values.pop("gemini_api_keys", None)
        google_tts_api_key = values.pop("google_tts_api_key", None)
        values.pop("google_tts_api_key_masked", None)
        values.pop("google_tts_api_key_configured", None)
        openai_api_key = values.pop("openai_api_key", None)
        values.pop("openai_api_key_masked", None)
        values.pop("openai_api_key_configured", None)

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

            if google_tts_api_key is not None:
                key = str(google_tts_api_key).strip()
                if key:
                    values["google_tts_api_key"] = key

            if openai_api_key is not None:
                key = str(openai_api_key).strip()
                if key:
                    values["openai_api_key"] = key

            for key, value in values.items():
                self.database.connection.execute(
                    """
                    INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                    """,
                    (key, json.dumps(value), now),
                )
        return self.get_all()
