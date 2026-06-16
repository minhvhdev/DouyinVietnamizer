import base64
import json
import re
import urllib.error
import urllib.request
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..errors import AppError
from ..models import ErrorInfo


GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiKeyPool:
    def __init__(self, keys: list[dict[str, Any]], cursor: int = 0) -> None:
        self.keys = [item["key"] for item in keys if isinstance(item, dict) and item.get("key")]
        self.cursor = cursor % len(self.keys) if self.keys else 0

    def ordered_keys(self) -> list[tuple[int, str]]:
        if not self.keys:
            return []
        return [
            ((self.cursor + offset) % len(self.keys), self.keys[(self.cursor + offset) % len(self.keys)])
            for offset in range(len(self.keys))
        ]

    def mark_success(self, index: int) -> None:
        if self.keys:
            self.cursor = (index + 1) % len(self.keys)


def default_request(api_key: str, model: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{GEMINI_API_BASE}/models/{model}:generateContent",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as cause:
        detail = cause.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini HTTP {cause.code}: {detail}") from cause


def response_text(response: dict) -> str:
    try:
        parts = response["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as cause:
        raise ValueError("Gemini response did not contain text parts.") from cause
    text = "".join(part.get("text", "") for part in parts)
    if not text.strip():
        raise ValueError("Gemini returned empty text.")
    return text.strip()


def parse_json_array(text: str) -> list[str]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    data = json.loads(cleaned)
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError("Gemini translation response must be a JSON string array.")
    return data


class GeminiTranslator:
    def __init__(
        self,
        key_pool: GeminiKeyPool,
        *,
        model: str = "gemini-2.5-flash",
        request: Callable[[str, str, dict], dict] = default_request,
    ) -> None:
        self.key_pool = key_pool
        self.model = model
        self.request = request

    def translate(self, texts: list[str], source: str, target: str) -> list[str]:
        if not self.key_pool.keys:
            raise AppError(
                400,
                ErrorInfo(
                    code="MISSING_GEMINI_API_KEYS",
                    message="No Gemini API keys are configured.",
                    action="Add at least one Google AI Studio API key in Settings.",
                ),
            )

        prompt = (
            f"Translate this JSON array from {source} to {target}. "
            "Return only a JSON array of translated strings in the same order.\n"
            f"{json.dumps(texts, ensure_ascii=False)}"
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        }
        last_error: Exception | None = None
        for index, api_key in self.key_pool.ordered_keys():
            try:
                translated = parse_json_array(response_text(self.request(api_key, self.model, payload)))
                if len(translated) != len(texts) or any(not item.strip() for item in translated):
                    raise ValueError("Gemini returned incomplete translations.")
                self.key_pool.mark_success(index)
                return translated
            except Exception as cause:
                last_error = cause

        raise AppError(
            502,
            ErrorInfo(
                code="GEMINI_KEYS_EXHAUSTED",
                message="All Gemini API keys failed for translation.",
                action="Check key quota in Google AI Studio or add another key.",
                detail=str(last_error),
                retryable=True,
            ),
        )


def response_pcm(response: dict) -> tuple[bytes, int]:
    try:
        parts = response["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as cause:
        raise ValueError("Gemini response did not contain audio parts.") from cause
    for part in parts:
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and inline.get("data"):
            mime = inline.get("mimeType") or inline.get("mime_type") or ""
            match = re.search(r"rate=(\d+)", mime)
            sample_rate = int(match.group(1)) if match else 24000
            return base64.b64decode(inline["data"]), sample_rate
    raise ValueError("Gemini returned no inline PCM audio.")


class GeminiTtsAdapter:
    def __init__(
        self,
        key_pool: GeminiKeyPool,
        *,
        model: str = "gemini-2.5-flash-preview-tts",
        request: Callable[[str, str, dict], dict] = default_request,
    ) -> None:
        self.key_pool = key_pool
        self.model = model
        self.request = request

    def synthesize(self, text: str, output_path: Path, *, voice: str) -> None:
        if not self.key_pool.keys:
            raise AppError(
                400,
                ErrorInfo(
                    code="MISSING_GEMINI_API_KEYS",
                    message="No Gemini API keys are configured.",
                    action="Add at least one Google AI Studio API key in Settings.",
                ),
            )

        payload = {
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {"voiceName": voice}
                    }
                },
            },
        }
        last_error: Exception | None = None
        for index, api_key in self.key_pool.ordered_keys():
            try:
                pcm, sample_rate = response_pcm(self.request(api_key, self.model, payload))
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with wave.open(str(output_path), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(sample_rate)
                    wav.writeframes(pcm)
                self.key_pool.mark_success(index)
                return
            except Exception as cause:
                last_error = cause

        raise AppError(
            502,
            ErrorInfo(
                code="GEMINI_KEYS_EXHAUSTED",
                message="All Gemini API keys failed for TTS.",
                action="Check key quota in Google AI Studio or add another key.",
                detail=str(last_error),
                retryable=True,
            ),
        )
