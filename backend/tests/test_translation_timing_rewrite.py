import re

import pytest

from dv_backend.errors import AppError
from dv_backend.translation_timing_rewrite import (
    invoke_timing_rewrite,
    lengthen_translation_for_timing,
    shorten_translation_for_timing,
)


def _estimate_word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def test_shorten_translation_uses_openai_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake_chat(api_base, api_key, model, messages, **kwargs):
        captured["api_base"] = api_base
        captured["api_key"] = api_key
        captured["model"] = model
        captured["prompt"] = messages[0]["content"]
        return {"choices": [{"message": {"content": "một hai ba bốn"}}]}

    monkeypatch.setattr(
        "dv_backend.translation_timing_rewrite.call_openai_chat",
        fake_chat,
    )

    settings = {
        "translation_backend": "openai",
        "openai_api_base": "https://api.openai.com/v1",
        "openai_api_key": "sk-test",
        "openai_translation_model": "gpt-4o",
    }

    shortened, target_words = shorten_translation_for_timing(
        settings,
        database=None,  # type: ignore[arg-type]
        text="một hai ba bốn năm sáu bảy tám chín mười",
        budget=7.5,
        current_duration=10.0,
        estimate_word_count=_estimate_word_count,
    )

    assert shortened == "một hai ba bốn"
    assert target_words == 8
    assert captured["model"] == "gpt-4o"
    assert "Target word count: approximately 8" in captured["prompt"]


def test_shorten_translation_skips_without_llm_keys() -> None:
    shortened, target_words = shorten_translation_for_timing(
        {"translation_backend": "gemini", "gemini_api_keys": []},
        database=None,  # type: ignore[arg-type]
        text="một hai ba bốn năm sáu bảy tám chín mười",
        budget=7.5,
        current_duration=10.0,
        estimate_word_count=_estimate_word_count,
    )

    assert shortened is None
    assert target_words == 10


def test_gemini_shortening_targets_mathematical_word_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def request(api_key, model, payload):
        captured["api_key"] = api_key
        captured["model"] = model
        captured["prompt"] = payload["contents"][0]["parts"][0]["text"]
        return {"candidates": [{"content": {"parts": [{"text": "một hai ba bốn năm sáu bảy tám"}]}}]}

    monkeypatch.setattr(
        "dv_backend.translation_timing_rewrite.default_request",
        request,
    )

    class FakeDatabase:
        connection = type(
            "Conn",
            (),
            {
                "execute": staticmethod(lambda *args, **kwargs: None),
                "__enter__": lambda self: self,
                "__exit__": lambda *args: None,
            },
        )()

    settings = {
        "translation_backend": "gemini",
        "gemini_api_keys": [{"id": "a", "key": "key-a"}, {"id": "b", "key": "key-b"}],
        "gemini_key_cursor": 0,
        "gemini_translation_model": "gemini-2.5-flash",
    }

    shortened, target_words = shorten_translation_for_timing(
        settings,
        FakeDatabase(),  # type: ignore[arg-type]
        text="một hai ba bốn năm sáu bảy tám chín mười",
        budget=7.5,
        current_duration=10.0,
        estimate_word_count=_estimate_word_count,
    )

    assert shortened == "một hai ba bốn năm sáu bảy tám"
    assert target_words == 8
    assert captured["api_key"] == "key-a"
    assert captured["model"] == "gemini-2.5-flash"
    assert "Current word count: approximately 10" in captured["prompt"]


def test_lengthen_translation_uses_openai_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dv_backend.translation_timing_rewrite.call_openai_chat",
        lambda *args, **kwargs: {
            "choices": [{"message": {"content": "Xin chào mọi người nhé."}}]
        },
    )

    lengthened, target_words = lengthen_translation_for_timing(
        {
            "translation_backend": "openai",
            "openai_api_base": "https://api.openai.com/v1",
            "openai_api_key": "sk-test",
            "openai_translation_model": "gpt-4o",
            "short_tts_lengthen_min_gap_sec": 1.5,
        },
        database=None,  # type: ignore[arg-type]
        text="Xin chào.",
        budget=4.0,
        current_duration=1.0,
        min_gap_sec=1.5,
        max_ratio=1.6,
        estimate_word_count=_estimate_word_count,
    )

    assert lengthened == "Xin chào mọi người nhé."
    assert target_words >= 2


def test_invoke_timing_rewrite_rejects_unknown_backend() -> None:
    with pytest.raises(AppError) as error:
        invoke_timing_rewrite(
            {"translation_backend": "unknown"},
            database=None,  # type: ignore[arg-type]
            prompt="test",
        )

    assert error.value.info.code == "UNSUPPORTED_TIMING_REWRITE_BACKEND"
