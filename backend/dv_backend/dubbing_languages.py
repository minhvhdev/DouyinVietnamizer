"""Supported dubbing target languages and per-language defaults."""

from __future__ import annotations

from typing import Any

SUPPORTED_DUB_LANGUAGES = frozenset({"vi", "th"})

DUB_LANGUAGE_CONFIG: dict[str, dict[str, Any]] = {
    "vi": {
        "id": "vi",
        "label": "Tiếng Việt",
        "label_en": "Vietnamese",
        "translation_target": "vi",
        "edge_locale": "vi",
        "google_locale": "vi-VN",
        "omnivoice_language_id": "vi",
        "speaking_rate_wps": 3.2,
        "default_edge_voice": "vi-VN-HoaiMyNeural",
        "default_google_voice": "vi-VN-Standard-A",
    },
    "th": {
        "id": "th",
        "label": "Tiếng Thái",
        "label_en": "Thai",
        "translation_target": "th",
        "edge_locale": "th",
        "google_locale": "th-TH",
        "omnivoice_language_id": "th",
        "speaking_rate_wps": 2.8,
        "default_edge_voice": "th-TH-PremwadeeNeural",
        "default_google_voice": "th-TH-Standard-A",
    },
}


def normalize_dub_language(language: str | None) -> str:
    value = str(language or "vi").strip().lower()
    if value in {"vietnamese", "vietnam", "viet nam"}:
        return "vi"
    if value in {"thai", "thailand"}:
        return "th"
    if value in SUPPORTED_DUB_LANGUAGES:
        return value
    return "vi"


def dub_language_config(language: str | None) -> dict[str, Any]:
    return DUB_LANGUAGE_CONFIG[normalize_dub_language(language)]


def dub_language_label(language: str | None, *, english: bool = False) -> str:
    config = dub_language_config(language)
    return str(config["label_en"] if english else config["label"])


def dub_language_from_settings(settings: dict[str, Any] | None) -> str:
    if not settings:
        return "vi"
    return normalize_dub_language(settings.get("translation_target_language"))


def default_speaking_rate_wps(language: str | None) -> float:
    return float(dub_language_config(language)["speaking_rate_wps"])


def list_dub_language_options() -> list[dict[str, str]]:
    return [
        {"id": config["id"], "label": config["label"], "label_en": config["label_en"]}
        for config in DUB_LANGUAGE_CONFIG.values()
    ]


def voice_defaults_for_language(language: str | None) -> dict[str, str]:
    config = dub_language_config(language)
    return {
        "edge_tts_voice": str(config["default_edge_voice"]),
        "google_tts_voice": str(config["default_google_voice"]),
        "omnivoice_language_id": str(config["omnivoice_language_id"]),
        "translation_target_language": str(config["translation_target"]),
    }
