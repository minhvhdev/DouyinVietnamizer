import base64
import json
import logging
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


def classify_gemini_failure(error: Exception) -> tuple[str, str, str]:
    message = str(error)
    if "HTTP 503" in message or "UNAVAILABLE" in message:
        return (
            "GEMINI_MODEL_UNAVAILABLE",
            "Gemini model is temporarily unavailable (server overload).",
            "Wait a few minutes and retry, or switch to gemini-2.5-flash in Settings.",
        )
    if "HTTP 404" in message or "NOT_FOUND" in message:
        return (
            "GEMINI_MODEL_NOT_FOUND",
            "The configured Gemini model name is not available on this API.",
            "Use a supported model such as gemini-2.5-flash or gemini-3.5-flash.",
        )
    if "HTTP 429" in message or "RESOURCE_EXHAUSTED" in message:
        return (
            "GEMINI_QUOTA_EXCEEDED",
            "Gemini API quota or rate limit was exceeded.",
            "Check usage in Google AI Studio or add another API key.",
        )
    if "HTTP 401" in message or "HTTP 403" in message or "API key not valid" in message:
        return (
            "GEMINI_KEY_INVALID",
            "One or more Gemini API keys were rejected.",
            "Verify your API keys in Google AI Studio.",
        )
    return (
        "GEMINI_KEYS_EXHAUSTED",
        "All Gemini API keys failed for translation.",
        "Check key quota in Google AI Studio or add another key.",
    )


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

    def translate(
        self,
        texts: list[str],
        source: str,
        target: str,
        *,
        duration_budgets: list[float] | None = None,
        timing_guidance: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        if not self.key_pool.keys:
            raise AppError(
                400,
                ErrorInfo(
                    code="MISSING_GEMINI_API_KEYS",
                    message="No Gemini API keys are configured.",
                    action="Add at least one Google AI Studio API key in Settings.",
                ),
            )

        if (
            (duration_budgets and len(duration_budgets) == len(texts))
            or (timing_guidance and len(timing_guidance) == len(texts))
        ):
            items = [
                {
                    "index": index,
                    "text": text,
                    **(
                        {"duration_budget_sec": round(float(duration_budgets[index]), 2)}
                        if duration_budgets and len(duration_budgets) == len(texts)
                        else {}
                    ),
                    **(
                        {
                            key: value
                            for key, value in (timing_guidance[index] or {}).items()
                            if value is not None
                        }
                        if timing_guidance and len(timing_guidance) == len(texts)
                        else {}
                    ),
                }
                for index, text in enumerate(texts)
            ]
            from ..translation_duration import timing_translate_prompt_rules

            prompt = (
                f"Translate these items from {source} to {target} for natural {target} dubbing. "
                "Return only a JSON array of translated strings in the same order. "
                f"{timing_translate_prompt_rules()}\n"
                f"{json.dumps(items, ensure_ascii=False)}"
            )
        else:
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
        saw_model_unavailable = False
        saw_model_not_found = False
        for index, api_key in self.key_pool.ordered_keys():
            try:
                translated = parse_json_array(response_text(self.request(api_key, self.model, payload)))
                if len(translated) != len(texts) or any(not item.strip() for item in translated):
                    raise ValueError("Gemini returned incomplete translations.")
                self.key_pool.mark_success(index)
                return translated
            except Exception as cause:
                last_error = cause
                code, _, _ = classify_gemini_failure(cause)
                if code == "GEMINI_MODEL_UNAVAILABLE":
                    saw_model_unavailable = True
                if code == "GEMINI_MODEL_NOT_FOUND":
                    saw_model_not_found = True

        if saw_model_unavailable and not saw_model_not_found:
            code, message, action = classify_gemini_failure(
                last_error or RuntimeError("Gemini model unavailable")
            )
        elif saw_model_not_found:
            code, message, action = classify_gemini_failure(
                last_error or RuntimeError("Gemini model not found")
            )
        else:
            code, message, action = classify_gemini_failure(
                last_error or RuntimeError("Gemini request failed")
            )

        raise AppError(
            502,
            ErrorInfo(
                code=code,
                message=message,
                action=action,
                detail=str(last_error),
                retryable=True,
            ),
        )

    def translate_candidates(
        self,
        segments: list[dict[str, Any]],
        texts: list[str],
        source: str,
        target: str,
        *,
        timing_profiles: list[dict[str, Any]] | None = None,
        speaking_rate_wps: float = 3.2,
        candidate_count: int = 3,
    ) -> list[list[dict[str, Any]]]:
        from ..translation_candidate_llm import (
            build_candidate_items,
            build_candidate_translation_prompt,
            parse_candidate_batches_failsoft,
        )

        if not self.key_pool.keys:
            raise AppError(
                400,
                ErrorInfo(
                    code="MISSING_GEMINI_API_KEYS",
                    message="No Gemini API keys are configured.",
                    action="Add at least one Google AI Studio API key in Settings.",
                ),
            )

        profiles = timing_profiles or [{} for _ in texts]
        items = build_candidate_items(
            segments,
            texts,
            timing_profiles=profiles,
            speaking_rate_wps=speaking_rate_wps,
        )
        prompt = build_candidate_translation_prompt(
            items,
            source=source,
            target=target,
            candidate_count=candidate_count,
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.3},
        }
        last_error: Exception | None = None
        for index, api_key in self.key_pool.ordered_keys():
            try:
                raw = response_text(self.request(api_key, self.model, payload))
                batches, parse_warning = parse_candidate_batches_failsoft(
                    raw,
                    expected_count=len(texts),
                    fallback_texts=texts,
                )
                if parse_warning:
                    logging.getLogger(__name__).warning("Gemini candidate parse fallback: %s", parse_warning)
                if not any(batch for batch in batches):
                    raise ValueError("Gemini returned empty candidate batches.")
                self.key_pool.mark_success(index)
                return batches
            except Exception as cause:
                last_error = cause

        code, message, action = classify_gemini_failure(last_error or RuntimeError("Gemini request failed"))
        raise AppError(
            502,
            ErrorInfo(
                code=code,
                message=message,
                action=action,
                detail=str(last_error),
                retryable=True,
            ),
        )

    def repair_fragment_translations(
        self,
        cluster_payloads: list[dict[str, Any]],
        *,
        source: str,
        target: str,
    ) -> Any:
        from ..translation_candidate_llm import (
            build_fragment_repair_prompt,
            parse_fragment_repair_response,
        )

        if not self.key_pool.keys:
            raise AppError(
                400,
                ErrorInfo(
                    code="MISSING_GEMINI_API_KEYS",
                    message="No Gemini API keys are configured.",
                    action="Add at least one Google AI Studio API key in Settings.",
                ),
            )
        if not cluster_payloads:
            return {"clusters": []}

        prompt = build_fragment_repair_prompt(cluster_payloads, source=source, target=target)
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2},
        }
        last_error: Exception | None = None
        for index, api_key in self.key_pool.ordered_keys():
            try:
                raw = response_text(self.request(api_key, self.model, payload))
                parsed = parse_fragment_repair_response(raw)
                self.key_pool.mark_success(index)
                return parsed
            except Exception as cause:
                last_error = cause

        code, message, action = classify_gemini_failure(last_error or RuntimeError("Gemini request failed"))
        raise AppError(
            502,
            ErrorInfo(
                code=code,
                message=message,
                action=action,
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

    def synthesize(
        self,
        text: str,
        output_path: Path,
        *,
        voice: str,
        ref_text: str | None = None,
        anchor_text: str | None = None,
        clone: bool = False,
        clone_mode: str | None = None,
        **kwargs,
    ) -> None:
        del ref_text, anchor_text, clone, clone_mode, kwargs
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

        if last_error and ("429" in str(last_error) or "RESOURCE_EXHAUSTED" in str(last_error)):
            raise AppError(
                429,
                ErrorInfo(
                    code="GEMINI_TTS_QUOTA_EXHAUSTED",
                    message="Gemini TTS quota exceeded.",
                    action="Check key quota in Google AI Studio or add another key.",
                    detail=str(last_error),
                    retryable=True,
                ),
            )

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

