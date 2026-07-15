import base64
import json
import wave
from pathlib import Path

import pytest

from dv_backend.adapters.gemini import GeminiKeyPool, GeminiTranslator, GeminiTtsAdapter
from dv_backend.errors import AppError


def test_gemini_translator_rotates_api_keys_after_failure() -> None:
    calls: list[str] = []

    def request(api_key: str, _model: str, _payload: dict) -> dict:
        calls.append(api_key)
        if api_key == "key-a":
            raise RuntimeError("429 quota")
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": json.dumps(["Xin chao", "Tam biet"])}
                        ]
                    }
                }
            ]
        }

    pool = GeminiKeyPool([{"id": "a", "key": "key-a"}, {"id": "b", "key": "key-b"}])
    translated = GeminiTranslator(pool, request=request).translate(
        ["你好", "再见"],
        source="zh-CN",
        target="vi",
    )

    assert translated == ["Xin chao", "Tam biet"]
    assert calls == ["key-a", "key-b"]
    assert pool.cursor == 0


def test_gemini_translator_reports_model_unavailable() -> None:
    pool = GeminiKeyPool([{"id": "a", "key": "key-a"}])
    adapter = GeminiTranslator(
        pool,
        request=lambda *_args: (_ for _ in ()).throw(
            RuntimeError('Gemini HTTP 503: {"error":{"status":"UNAVAILABLE"}}')
        ),
    )

    with pytest.raises(AppError) as error:
        adapter.translate(["你好"], source="zh-CN", target="vi")

    assert error.value.info.code == "GEMINI_MODEL_UNAVAILABLE"


def test_gemini_translator_reports_all_keys_failed() -> None:
    pool = GeminiKeyPool([{"id": "a", "key": "key-a"}])
    adapter = GeminiTranslator(pool, request=lambda *_args: (_ for _ in ()).throw(RuntimeError("429")))

    with pytest.raises(AppError) as error:
        adapter.translate(["你好"], source="zh-CN", target="vi")

    assert error.value.info.code == "GEMINI_KEYS_EXHAUSTED"


def test_gemini_translator_includes_timing_guidance_in_prompt() -> None:
    captured: dict[str, dict] = {}

    def request(_api_key: str, _model: str, payload: dict) -> dict:
        captured["payload"] = payload
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": json.dumps(["Xin chao ban"])}
                        ]
                    }
                }
            ]
        }

    pool = GeminiKeyPool([{"id": "a", "key": "key-a"}])
    translated = GeminiTranslator(pool, request=request).translate(
        ["你好朋友"],
        source="zh-CN",
        target="vi",
        duration_budgets=[1.2],
        timing_guidance=[{
            "source_speech_units": 4,
            "target_vi_syllables": 4,
            "target_vi_syllable_range": [3, 5],
        }],
    )

    prompt = captured["payload"]["contents"][0]["parts"][0]["text"]
    assert translated == ["Xin chao ban"]
    assert "source_speech_units" in prompt
    assert "target_vi_syllables" in prompt
    assert "target_vi_syllable_range" in prompt
    assert "duration_budget_sec" in prompt
    assert "no fewer than" in prompt
    assert "no more than" in prompt
    assert "Priority order" in prompt
    assert "complete speakable thoughts" in prompt


def test_gemini_tts_writes_wave_from_inline_pcm(tmp_path: Path) -> None:
    pcm = (b"\x01\x00" * 2400)

    def request(_api_key: str, _model: str, _payload: dict) -> dict:
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "audio/L16;rate=24000",
                                    "data": base64.b64encode(pcm).decode("ascii"),
                                }
                            }
                        ]
                    }
                }
            ]
        }

    output = tmp_path / "tts.wav"
    pool = GeminiKeyPool([{"id": "a", "key": "key-a"}])

    GeminiTtsAdapter(pool, request=request).synthesize("Xin chao", output, voice="Zephyr")

    with wave.open(str(output), "rb") as wav:
        assert wav.getframerate() == 24000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.readframes(wav.getnframes()) == pcm


def test_gemini_tts_reports_quota_exhaustion() -> None:
    pool = GeminiKeyPool([{"id": "a", "key": "key-a"}])
    adapter = GeminiTtsAdapter(
        pool,
        request=lambda *_args: (_ for _ in ()).throw(
            RuntimeError("Gemini HTTP 429: RESOURCE_EXHAUSTED quota exceeded")
        ),
    )

    with pytest.raises(AppError) as error:
        adapter.synthesize("Xin chao", Path("unused.wav"), voice="Zephyr")

    assert error.value.info.code == "GEMINI_TTS_QUOTA_EXHAUSTED"
    assert error.value.info.retryable is True
