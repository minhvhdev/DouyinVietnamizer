from dv_backend.adapters.omnivoice_infer import resolve_omnivoice_language
from dv_backend.dubbing_languages import (
    dub_language_config,
    dub_language_from_settings,
    normalize_dub_language,
    voice_defaults_for_language,
)


def test_normalize_dub_language() -> None:
    assert normalize_dub_language("vi") == "vi"
    assert normalize_dub_language("thai") == "th"
    assert normalize_dub_language("unknown") == "vi"


def test_thai_voice_defaults() -> None:
    defaults = voice_defaults_for_language("th")
    assert defaults["translation_target_language"] == "th"
    assert defaults["edge_tts_voice"] == "th-TH-PremwadeeNeural"
    assert defaults["google_tts_voice"] == "th-TH-Standard-A"
    assert defaults["omnivoice_language_id"] == "th"


def test_dub_language_from_settings() -> None:
    assert dub_language_from_settings({"translation_target_language": "th"}) == "th"
    config = dub_language_config("th")
    assert config["label"] == "Tiếng Thái"
    assert config["speaking_rate_wps"] == 2.8


def test_resolve_omnivoice_thai_aliases() -> None:
    assert resolve_omnivoice_language("thai") == "th"
    assert resolve_omnivoice_language("thailand") == "th"
