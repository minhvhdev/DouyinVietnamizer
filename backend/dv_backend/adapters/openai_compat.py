import json
import logging
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..errors import AppError
from ..models import ErrorInfo

DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_CHAT_TIMEOUT_SEC = 600


def normalize_openai_api_base(api_base: str) -> str:
    base = str(api_base or "").strip().rstrip("/")
    if not base:
        return DEFAULT_OPENAI_API_BASE
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def call_openai_chat(
    api_base: str,
    api_key: str,
    model: str,
    messages: list,
    *,
    json_mode: bool = False,
) -> dict:
    base = normalize_openai_api_base(api_base)
    url = f"{base}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=OPENAI_CHAT_TIMEOUT_SEC) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as cause:
        body = cause.read().decode("utf-8", errors="replace")
        try:
            err_data = json.loads(body)
            err_msg = err_data.get("error", {}).get("message", body)
        except Exception:
            err_msg = body
        raise AppError(
            502,
            ErrorInfo(
                code="OPENAI_CHAT_ERROR",
                message=f"Translation API error ({cause.code}).",
                action="Verify your API key, base URL, and network connection.",
                detail=err_msg,
                retryable=cause.code in {408, 429, 500, 502, 503, 504},
            ),
        ) from cause
    except AppError:
        raise
    except Exception as cause:
        raise AppError(
            502,
            ErrorInfo(
                code="OPENAI_CHAT_FAILED",
                message="Failed to connect to the OpenAPI-compatible translation service.",
                action="Check settings and try again.",
                detail=str(cause),
                retryable=True,
            ),
        ) from cause


def call_openai_tts(
    api_base: str,
    api_key: str,
    model: str,
    voice: str,
    text: str,
    output_path: Path,
) -> None:
    base = normalize_openai_api_base(api_base)
    url = f"{base}/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "input": text,
        "voice": voice,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as handle:
                handle.write(response.read())
    except urllib.error.HTTPError as cause:
        body = cause.read().decode("utf-8", errors="replace")
        raise AppError(
            502,
            ErrorInfo(
                code="OPENAI_TTS_ERROR",
                message=f"TTS API error ({cause.code}).",
                action="Verify settings and billing status.",
                detail=body,
            ),
        ) from cause
    except Exception as cause:
        raise AppError(
            502,
            ErrorInfo(
                code="OPENAI_TTS_FAILED",
                message="Failed to connect to TTS API.",
                action="Check your internet connection and API config.",
                detail=str(cause),
            ),
        ) from cause


def list_openai_models(api_base: str, api_key: str) -> list[dict[str, str]]:
    base = normalize_openai_api_base(api_base)
    url = f"{base}/models"
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as cause:
        body = cause.read().decode("utf-8", errors="replace")
        try:
            err_data = json.loads(body)
            err_msg = err_data.get("error", {}).get("message", body)
        except Exception:
            err_msg = body
        raise AppError(
            502,
            ErrorInfo(
                code="OPENAI_MODELS_ERROR",
                message=f"Could not list models ({cause.code}).",
                action="Verify the base URL and API key, then retry.",
                detail=err_msg,
            ),
        ) from cause
    except Exception as cause:
        raise AppError(
            502,
            ErrorInfo(
                code="OPENAI_MODELS_FAILED",
                message="Failed to fetch models from the OpenAPI-compatible service.",
                action="Check settings and network connection.",
                detail=str(cause),
            ),
        ) from cause

    models: list[dict[str, str]] = []
    data = payload.get("data")
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id") or "").strip()
            if not model_id:
                continue
            models.append({"id": model_id, "name": model_id})
    models.sort(key=lambda item: item["id"].lower())
    if not models:
        raise AppError(
            502,
            ErrorInfo(
                code="OPENAI_MODELS_EMPTY",
                message="The models endpoint returned no usable model IDs.",
                action="Confirm the provider supports GET /v1/models.",
            ),
        )
    return models


def parse_json_array(text: str) -> list[str]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    data = json.loads(cleaned)
    if isinstance(data, dict):
        for key in ("translations", "items", "results"):
            nested = data.get(key)
            if isinstance(nested, list):
                extracted: list[str] = []
                for item in nested:
                    if isinstance(item, str):
                        extracted.append(item)
                    elif isinstance(item, dict):
                        value = item.get("translation") or item.get("text")
                        if isinstance(value, str):
                            extracted.append(value)
                if extracted:
                    return extracted
        raise ValueError("OpenAPI translation response JSON did not contain a string array.")
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError("OpenAPI translation response must be a JSON string array.")
    return data


def response_text(response: dict) -> str:
    try:
        message = response["choices"][0]["message"]
        content = message.get("content", "")
    except (KeyError, IndexError, TypeError) as cause:
        raise ValueError("OpenAPI response did not contain message content.") from cause
    if isinstance(content, list):
        text = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    else:
        text = str(content or "")
    if not text.strip():
        raise ValueError("OpenAPI returned empty text.")
    return text.strip()


def classify_openai_failure(error: Exception) -> tuple[str, str, str]:
    message = str(error)
    if isinstance(error, AppError):
        return error.info.code, error.info.message, error.info.action or ""
    if "HTTP 401" in message or "HTTP 403" in message or "invalid api key" in message.lower():
        return (
            "OPENAI_KEY_INVALID",
            "The OpenAPI-compatible API key was rejected.",
            "Verify the API key in Settings → Dịch thuật.",
        )
    if "HTTP 404" in message or "model" in message.lower() and "not found" in message.lower():
        return (
            "OPENAI_MODEL_NOT_FOUND",
            "The configured translation model is not available on this API.",
            "Choose another model from the dropdown in Settings.",
        )
    if "HTTP 429" in message:
        return (
            "OPENAI_RATE_LIMITED",
            "The translation API rate limit was exceeded.",
            "Wait briefly and retry, or switch translation backend.",
        )
    return (
        "OPENAI_TRANSLATION_FAILED",
        "OpenAPI-compatible translation failed.",
        "Check base URL, API key, and model selection in Settings.",
    )


class OpenAiCompatTranslator:
    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        request: Callable[..., dict] = call_openai_chat,
    ) -> None:
        self.api_base = normalize_openai_api_base(api_base)
        self.api_key = (api_key or "").strip()
        self.model = (model or "").strip()
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
        if not self.api_key:
            raise AppError(
                400,
                ErrorInfo(
                    code="MISSING_OPENAI_API_KEY",
                    message="No OpenAPI-compatible API key is configured.",
                    action="Add an API key in Settings → Dịch thuật.",
                ),
            )
        if not self.model:
            raise AppError(
                400,
                ErrorInfo(
                    code="MISSING_OPENAI_TRANSLATION_MODEL",
                    message="No translation model is selected.",
                    action="Choose a model from the dropdown in Settings.",
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

        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.request(
                self.api_base,
                self.api_key,
                self.model,
                messages,
                json_mode=False,
            )
            translated = parse_json_array(response_text(response))
        except AppError:
            raise
        except Exception as cause:
            code, message, action = classify_openai_failure(cause)
            raise AppError(
                502,
                ErrorInfo(
                    code=code,
                    message=message,
                    action=action,
                    detail=str(cause),
                    retryable=True,
                ),
            ) from cause

        if len(translated) != len(texts) or any(not item.strip() for item in translated):
            raise AppError(
                502,
                ErrorInfo(
                    code="OPENAI_TRANSLATION_INCOMPLETE",
                    message="OpenAPI-compatible translation returned incomplete output.",
                    action="Retry translation or choose another model.",
                    retryable=True,
                ),
            )
        return translated

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

        if not self.api_key or not self.model:
            raise AppError(
                400,
                ErrorInfo(
                    code="MISSING_OPENAI_TRANSLATION_MODEL",
                    message="OpenAPI translation is not configured.",
                    action="Add API key and model in Settings.",
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
        try:
            response = self.request(
                self.api_base,
                self.api_key,
                self.model,
                [{"role": "user", "content": prompt}],
                json_mode=True,
            )
            batches, parse_warning = parse_candidate_batches_failsoft(
                response_text(response),
                expected_count=len(texts),
                fallback_texts=texts,
            )
            if parse_warning:
                logging.getLogger(__name__).warning("OpenAI candidate parse fallback: %s", parse_warning)
            if not any(batch for batch in batches):
                raise ValueError("OpenAPI returned empty candidate batches.")
            return batches
        except AppError:
            raise
        except Exception as cause:
            code, message, action = classify_openai_failure(cause)
            raise AppError(
                502,
                ErrorInfo(
                    code=code,
                    message=message,
                    action=action,
                    detail=str(cause),
                    retryable=True,
                ),
            ) from cause

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

        if not self.api_key or not self.model:
            raise AppError(
                400,
                ErrorInfo(
                    code="MISSING_OPENAI_TRANSLATION_MODEL",
                    message="OpenAPI translation is not configured.",
                    action="Add API key and model in Settings.",
                ),
            )
        if not cluster_payloads:
            return {"clusters": []}

        prompt = build_fragment_repair_prompt(cluster_payloads, source=source, target=target)
        try:
            response = self.request(
                self.api_base,
                self.api_key,
                self.model,
                [{"role": "user", "content": prompt}],
                json_mode=True,
            )
            return parse_fragment_repair_response(response_text(response))
        except AppError:
            raise
        except Exception as cause:
            code, message, action = classify_openai_failure(cause)
            raise AppError(
                502,
                ErrorInfo(
                    code=code,
                    message=message,
                    action=action,
                    detail=str(cause),
                    retryable=True,
                ),
            ) from cause
