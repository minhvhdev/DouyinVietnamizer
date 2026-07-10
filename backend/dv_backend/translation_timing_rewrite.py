"""Rewrite translated lines during duration repair using the configured LLM backend."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from .adapters.gemini import (
    GeminiKeyPool,
    classify_gemini_failure,
    default_request,
    response_text as gemini_response_text,
)
from .adapters.openai_compat import (
    call_openai_chat,
    classify_openai_failure,
    response_text as openai_response_text,
)
from .errors import AppError
from .models import ErrorInfo

if TYPE_CHECKING:
    from .database import Database


def _save_setting(database: Database, key: str, value) -> None:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with database.connection:
        database.connection.execute(
            """
            INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, json.dumps(value), now),
        )


def translation_backend(settings: dict) -> str:
    return str(settings.get("translation_backend", "google_free") or "google_free")


def llm_timing_rewrite_available(settings: dict) -> bool:
    backend = translation_backend(settings)
    if backend == "gemini":
        return bool(settings.get("gemini_api_keys"))
    if backend == "openai":
        return bool(str(settings.get("openai_api_key") or "").strip()) and bool(
            str(settings.get("openai_translation_model") or "").strip()
        )
    return False


def _build_shorten_prompt(
    *,
    text: str,
    budget: float,
    current_duration: float,
    current_words: int,
    target_words: int,
    target_ratio: float,
    language_label: str = "Vietnamese",
) -> str:
    overrun_pct = max(0.0, ((current_duration / budget) - 1.0) * 100.0)
    return (
        f"Rewrite this {language_label} dubbing line so it stays natural but fits the target timing.\n"
        f"Current line: {text}\n"
        f"Current duration: {current_duration:.2f}s\n"
        f"Target duration budget: {budget:.2f}s\n"
        f"Current word count: approximately {current_words}\n"
        f"Target word count: approximately {target_words} (timing ratio {target_ratio:.3f})\n"
        f"Current line overruns the timing by {overrun_pct:.1f}%.\n"
        "Remove filler words and redundant phrasing first. Preserve names, numbers, core meaning, and causal relationships. "
        f"Return only the rewritten {language_label} line with no quotes, notes, or formatting."
    )


def _build_lengthen_prompt(
    *,
    text: str,
    budget: float,
    current_duration: float,
    current_words: int,
    target_words: int,
    target_ratio: float,
    language_label: str = "Vietnamese",
) -> str:
    underrun_pct = max(0.0, ((budget / current_duration) - 1.0) * 100.0)
    return (
        f"Expand this {language_label} dubbing line so it sounds natural when spoken aloud and better fills the target timing.\n"
        f"Current line: {text}\n"
        f"Current duration: {current_duration:.2f}s\n"
        f"Target duration budget: {budget:.2f}s\n"
        f"Current word count: approximately {current_words}\n"
        f"Target word count: approximately {target_words} (timing ratio {target_ratio:.3f})\n"
        f"The line is about {underrun_pct:.1f}% shorter than the timing budget.\n"
        "Add natural filler words, light discourse markers, or slightly fuller phrasing only when needed. "
        "Do not change core meaning, facts, names, numbers, or causal relationships. "
        f"Return only the expanded {language_label} line with no quotes, notes, or formatting."
    )


def _invoke_gemini_rewrite(settings: dict, database: Database, prompt: str) -> str:
    key_pool = GeminiKeyPool(
        settings.get("gemini_api_keys", []),
        cursor=int(settings.get("gemini_key_cursor", 0)),
    )
    if not key_pool.keys:
        raise AppError(
            400,
            ErrorInfo(
                code="MISSING_GEMINI_API_KEY",
                message="No Gemini API keys are configured.",
                action="Add Gemini keys in Settings → Dịch thuật.",
            ),
        )

    model = str(settings.get("gemini_translation_model", "gemini-2.5-flash") or "gemini-2.5-flash")
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2},
    }

    last_error: Exception | None = None
    saw_model_unavailable = False
    saw_model_not_found = False
    for index, api_key in key_pool.ordered_keys():
        try:
            rewritten = gemini_response_text(default_request(api_key, model, payload)).strip().strip("\"'")
            if rewritten:
                key_pool.mark_success(index)
                _save_setting(database, "gemini_key_cursor", key_pool.cursor)
                return rewritten
            raise ValueError("Gemini returned an empty rewrite result.")
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


def _invoke_openai_rewrite(settings: dict, prompt: str) -> str:
    api_key = str(settings.get("openai_api_key") or "").strip()
    model = str(settings.get("openai_translation_model") or "").strip()
    api_base = str(settings.get("openai_api_base") or "")
    if not api_key:
        raise AppError(
            400,
            ErrorInfo(
                code="MISSING_OPENAI_API_KEY",
                message="No OpenAPI-compatible API key is configured.",
                action="Add an API key in Settings → Dịch thuật.",
            ),
        )
    if not model:
        raise AppError(
            400,
            ErrorInfo(
                code="MISSING_OPENAI_TRANSLATION_MODEL",
                message="No translation model is selected.",
                action="Choose a model from the dropdown in Settings.",
            ),
        )

    try:
        response = call_openai_chat(
            api_base,
            api_key,
            model,
            [{"role": "user", "content": prompt}],
        )
        rewritten = openai_response_text(response).strip().strip("\"'")
        if rewritten:
            return rewritten
        raise ValueError("OpenAPI-compatible service returned an empty rewrite result.")
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


def invoke_timing_rewrite(settings: dict, database: Database, prompt: str) -> str:
    backend = translation_backend(settings)
    if backend == "gemini":
        return _invoke_gemini_rewrite(settings, database, prompt)
    if backend == "openai":
        return _invoke_openai_rewrite(settings, prompt)
    raise AppError(
        400,
        ErrorInfo(
            code="UNSUPPORTED_TIMING_REWRITE_BACKEND",
            message="Duration repair text rewrite requires Gemini or OpenAPI translation backend.",
            action="Choose Gemini or OpenAPI in Settings → Dịch thuật.",
        ),
    )


def shorten_translation_for_timing(
    settings: dict,
    database: Database,
    *,
    text: str,
    budget: float,
    current_duration: float,
    estimate_word_count,
    language_label: str = "Vietnamese",
) -> tuple[str | None, int]:
    current_words = estimate_word_count(text)
    if current_words < 2 or budget <= 0 or current_duration <= budget:
        return None, current_words
    if not llm_timing_rewrite_available(settings):
        return None, current_words

    target_ratio = max(0.1, min(1.0, float(budget) / float(current_duration)))
    target_words = max(1, round(current_words * target_ratio))
    if target_words >= current_words:
        return None, target_words

    prompt = _build_shorten_prompt(
        text=text,
        budget=budget,
        current_duration=current_duration,
        current_words=current_words,
        target_words=target_words,
        target_ratio=target_ratio,
        language_label=language_label,
    )
    shortened = invoke_timing_rewrite(settings, database, prompt)
    return shortened, target_words


def lengthen_translation_for_timing(
    settings: dict,
    database: Database,
    *,
    text: str,
    budget: float,
    current_duration: float,
    min_gap_sec: float,
    max_ratio: float,
    estimate_word_count,
    language_label: str = "Vietnamese",
) -> tuple[str | None, int]:
    cleaned = text.strip()
    current_words = estimate_word_count(cleaned)
    if current_words < 1 or budget <= 0 or current_duration >= budget:
        return None, current_words

    gap = budget - current_duration
    if gap <= min_gap_sec:
        return None, current_words
    if not llm_timing_rewrite_available(settings):
        return None, current_words

    target_ratio = min(max_ratio, float(budget) / max(0.05, float(current_duration)))
    target_words = max(current_words + 1, min(current_words + 2, round(current_words * target_ratio)))
    prompt = _build_lengthen_prompt(
        text=cleaned,
        budget=budget,
        current_duration=current_duration,
        current_words=current_words,
        target_words=target_words,
        target_ratio=target_ratio,
        language_label=language_label,
    )
    lengthened = invoke_timing_rewrite(settings, database, prompt)
    if lengthened == cleaned:
        return None, target_words
    return lengthened, target_words
