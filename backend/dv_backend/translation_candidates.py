"""Translation candidate generation orchestration."""

from __future__ import annotations

import time
from typing import Any, Callable

from .translation_candidate_ranking import rank_translation_candidates
from .voice_duration_profile import resolve_voice_profile


CandidateBatchResult = list[list[dict[str, Any]]]


def _single_candidate(text: str, *, style: str = "natural", source: str = "legacy") -> list[dict[str, Any]]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    return [{"text": cleaned, "style": style, "meaning_notes": [], "candidate_source": source}]


def google_free_candidates(translated_text: str) -> tuple[list[dict[str, Any]], str]:
    return _single_candidate(translated_text, source="google_free_single"), "google_free_single"


def apply_candidate_selection(
    segment: dict[str, Any],
    *,
    candidates: list[dict[str, Any]],
    source_text: str,
    language: str,
    settings: dict[str, Any],
    data_dir,
    ranking_started: float | None = None,
) -> None:
    voice_profile = resolve_voice_profile(settings, language=language, data_dir=data_dir)
    timing_profile = segment.get("timing_profile") or {}
    ranking = rank_translation_candidates(
        candidates,
        timing_profile=timing_profile,
        source_text=source_text,
        language=language,
        voice_profile=voice_profile,
        reference_text=str(segment.get("translation") or "") or None,
        settings=settings,
    )
    selected_index = ranking["selected_candidate_index"]
    if selected_index < 0 or selected_index >= len(candidates):
        return

    selected = candidates[selected_index]
    segment["translation"] = str(selected.get("text") or "").strip()
    segment["translation_candidates"] = candidates
    segment["selected_candidate_index"] = selected_index
    segment["selected_candidate_reason"] = ranking.get("selected_candidate_reason")
    segment["selected_candidate_style"] = ranking.get("selected_candidate_style")
    segment["candidate_rankings"] = ranking.get("rankings")
    segment["translation_duration_prediction"] = ranking.get("rankings", [{}])[selected_index].get("prediction")
    segment["predicted_duration"] = ranking.get("predicted_duration")
    if ranking.get("duration_error_ms") is not None:
        segment["duration_error_ms"] = ranking["duration_error_ms"]
    if ranking_started is not None:
        segment["candidate_ranking_wall_time_ms"] = round((time.perf_counter() - ranking_started) * 1000)


def wrap_legacy_translations_as_candidates(
    segments: list[dict[str, Any]],
    translated: list[str],
    *,
    source: str = "single_translation",
) -> None:
    for segment, text in zip(segments, translated, strict=True):
        candidates = _single_candidate(text, source=source)
        segment["translation_candidates"] = candidates
        segment["selected_candidate_index"] = 0 if candidates else -1
        segment["translation_candidate_source"] = source


def translate_segments_with_candidates(
    settings: dict[str, Any],
    database,
    segments: list[dict[str, Any]],
    *,
    source_lang: str,
    target_lang: str,
    translate_fn: Callable[..., list[str]],
    translate_candidates_fn: Callable[..., CandidateBatchResult] | None,
    save_setting_fn: Callable[[Any, str, Any], None] | None = None,
    data_dir=None,
) -> None:
    texts = [segment["text"] for segment in segments]
    timing_profiles = [segment.get("timing_profile") or {} for segment in segments]
    speaking_rate = float(settings.get("vietnamese_speaking_rate_wps") or 3.2)

    enabled = bool(settings.get("timing_candidate_translation_enabled", False))
    backend = str(settings.get("translation_backend") or "google_free")
    use_candidates = enabled and backend in {"gemini", "openai"} and translate_candidates_fn is not None

    gen_started = time.perf_counter()
    if use_candidates:
        try:
            candidate_batches = translate_candidates_fn(
                segments,
                texts,
                source=source_lang,
                target=target_lang,
                timing_profiles=timing_profiles,
                settings=settings,
                database=database,
                speaking_rate_wps=speaking_rate,
            )
        except Exception as error:
            gen_ms = round((time.perf_counter() - gen_started) * 1000)
            translated = translate_fn(
                settings,
                database,
                texts,
                source_lang=source_lang,
                target_lang=target_lang,
                duration_budgets=[
                    float(profile.get("speech_target_duration") or segment.get("duration_budget") or 0.0)
                    for profile, segment in zip(timing_profiles, segments, strict=True)
                ],
                timing_guidance=[segment.get("timing_guidance") or {} for segment in segments],
            )
            for segment, text in zip(segments, translated, strict=True):
                segment["candidate_generation_wall_time_ms"] = gen_ms
                segment["translation_candidate_source"] = f"{backend}_fallback_single"
                segment["candidate_generation_warning"] = f"candidate_generation_failed:{type(error).__name__}"
                candidates = _single_candidate(text, source=f"{backend}_fallback_single")
                segment["translation_candidates"] = candidates
                segment["selected_candidate_index"] = 0 if candidates else -1
                segment["translation"] = text
                if candidates:
                    apply_candidate_selection(
                        segment,
                        candidates=candidates,
                        source_text=str(segment.get("text") or ""),
                        language=target_lang,
                        settings=settings,
                        data_dir=data_dir,
                    )
            return

        gen_ms = round((time.perf_counter() - gen_started) * 1000)
        for segment, batch in zip(segments, candidate_batches, strict=True):
            segment["candidate_generation_wall_time_ms"] = gen_ms
            segment["translation_candidate_source"] = backend
            if not batch:
                continue
            apply_candidate_selection(
                segment,
                candidates=batch,
                source_text=str(segment.get("text") or ""),
                language=target_lang,
                settings=settings,
                data_dir=data_dir,
                ranking_started=time.perf_counter(),
            )
        return

    duration_budgets = [
        float(profile.get("speech_target_duration") or segment.get("duration_budget") or 0.0)
        for profile, segment in zip(timing_profiles, segments, strict=True)
    ]
    timing_guidance = [
        segment.get("timing_guidance") or {}
        for segment in segments
    ]
    translated = translate_fn(
        settings,
        database,
        texts,
        source_lang=source_lang,
        target_lang=target_lang,
        duration_budgets=duration_budgets,
        timing_guidance=timing_guidance,
    )
    gen_ms = round((time.perf_counter() - gen_started) * 1000)
    source_tag = "google_free_single" if backend == "google_free" else f"{backend}_single"
    for segment, text in zip(segments, translated, strict=True):
        segment["candidate_generation_wall_time_ms"] = gen_ms
        candidates = _single_candidate(text, source=source_tag)
        segment["translation_candidates"] = candidates
        segment["translation_candidate_source"] = source_tag
        segment["selected_candidate_index"] = 0 if candidates else -1
        segment["translation"] = text
        if candidates:
            apply_candidate_selection(
                segment,
                candidates=candidates,
                source_text=str(segment.get("text") or ""),
                language=target_lang,
                settings=settings,
                data_dir=data_dir,
            )
