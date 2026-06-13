from unittest.mock import Mock

import pytest

from dv_backend.adapters.translation import GoogleFreeTranslator
from dv_backend.errors import AppError


def test_google_free_translator_preserves_segment_order() -> None:
    client = Mock()
    client.translate_batch.return_value = ["Xin chao", "Tam biet"]
    adapter = GoogleFreeTranslator(client_factory=lambda source, target: client, sleep=lambda _: None)

    translated = adapter.translate(["你好", "再见"], source="zh-CN", target="vi")

    assert translated == ["Xin chao", "Tam biet"]
    client.translate_batch.assert_called_once_with(["你好", "再见"])


def test_google_free_translator_retries_transient_failure() -> None:
    client = Mock()
    client.translate_batch.side_effect = [RuntimeError("rate limited"), ["Xin chao"]]
    sleeps: list[float] = []
    adapter = GoogleFreeTranslator(
        client_factory=lambda source, target: client,
        sleep=sleeps.append,
        max_attempts=2,
    )

    assert adapter.translate(["你好"], source="zh-CN", target="vi") == ["Xin chao"]
    assert sleeps == [1.0]


def test_google_free_translator_rejects_empty_output() -> None:
    client = Mock()
    client.translate_batch.return_value = [""]
    adapter = GoogleFreeTranslator(client_factory=lambda source, target: client, sleep=lambda _: None)

    with pytest.raises(AppError) as error:
        adapter.translate(["你好"], source="zh-CN", target="vi")

    assert error.value.info.code == "TRANSLATION_INVALID_OUTPUT"

