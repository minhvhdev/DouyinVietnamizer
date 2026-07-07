import json
from unittest.mock import patch

import pytest

from dv_backend.adapters.openai_compat import (
    OpenAiCompatTranslator,
    list_openai_models,
    normalize_openai_api_base,
    parse_json_array,
)
from dv_backend.errors import AppError


def test_normalize_openai_api_base_appends_v1() -> None:
    assert normalize_openai_api_base("https://example.com") == "https://example.com/v1"
    assert normalize_openai_api_base("https://example.com/v1/") == "https://example.com/v1"


def test_parse_json_array_accepts_plain_array() -> None:
    assert parse_json_array('["Xin chào", "Tạm biệt"]') == ["Xin chào", "Tạm biệt"]


def test_parse_json_array_accepts_wrapped_translations() -> None:
    payload = json.dumps({"translations": [{"index": 0, "translation": "Xin chào"}]})
    assert parse_json_array(payload) == ["Xin chào"]


def test_openai_translator_requires_api_key() -> None:
    translator = OpenAiCompatTranslator(api_base="https://api.openai.com/v1", api_key="", model="gpt-4o")
    with pytest.raises(AppError) as error:
        translator.translate(["你好"], source="zh-CN", target="vi")
    assert error.value.info.code == "MISSING_OPENAI_API_KEY"


def test_openai_translator_returns_translations() -> None:
    def request(_api_base: str, _api_key: str, _model: str, _messages: list, **kwargs) -> dict:
        return {
            "choices": [
                {"message": {"content": json.dumps(["Xin chào", "Tạm biệt"])}}
            ]
        }

    translator = OpenAiCompatTranslator(
        api_base="https://api.openai.com/v1",
        api_key="sk-test",
        model="gpt-4o",
        request=request,
    )
    translated = translator.translate(["你好", "再见"], source="zh-CN", target="vi")
    assert translated == ["Xin chào", "Tạm biệt"]


def test_list_openai_models_parses_response() -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "data": [
                        {"id": "gpt-4o"},
                        {"id": "gpt-4o-mini"},
                    ]
                }
            ).encode("utf-8")

    with patch("dv_backend.adapters.openai_compat.urllib.request.urlopen", return_value=FakeResponse()):
        models = list_openai_models("https://api.openai.com", "sk-test")
    assert models == [
        {"id": "gpt-4o", "name": "gpt-4o"},
        {"id": "gpt-4o-mini", "name": "gpt-4o-mini"},
    ]
