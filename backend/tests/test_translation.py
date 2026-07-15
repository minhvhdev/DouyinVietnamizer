"""Translation backend coverage after Google Translate Free removal."""

from __future__ import annotations

import pytest

from dv_backend.errors import AppError
from dv_backend.pipeline import _translate_texts


def test_translate_texts_rejects_unknown_backend() -> None:
    with pytest.raises(AppError) as error:
        _translate_texts(
            {"translation_backend": "unknown"},
            database=None,  # type: ignore[arg-type]
            texts=["你好"],
            source_lang="zh-CN",
            target_lang="vi",
        )

    assert error.value.info.code == "UNSUPPORTED_TRANSLATION_BACKEND"


def test_translate_texts_maps_legacy_google_free_to_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeTranslator:
        def __init__(self, key_pool, *, model: str) -> None:
            captured["model"] = model
            self.key_pool = key_pool

        def translate(self, texts, source, target, **kwargs):
            captured["texts"] = texts
            return ["Xin chào"]

    class FakePool:
        def __init__(self, keys, cursor=0) -> None:
            self.cursor = cursor

    monkeypatch.setattr("dv_backend.pipeline.GeminiKeyPool", FakePool)
    monkeypatch.setattr("dv_backend.pipeline.GeminiTranslator", FakeTranslator)
    monkeypatch.setattr("dv_backend.pipeline.save_setting", lambda *_args, **_kwargs: None)

    result = _translate_texts(
        {
            "translation_backend": "google_free",
            "gemini_api_keys": [{"id": "a", "key": "key-a"}],
            "gemini_translation_model": "gemini-2.5-flash",
        },
        database=None,  # type: ignore[arg-type]
        texts=["你好"],
        source_lang="zh-CN",
        target_lang="vi",
    )

    assert result == ["Xin chào"]
    assert captured["texts"] == ["你好"]
    assert captured["model"] == "gemini-2.5-flash"
