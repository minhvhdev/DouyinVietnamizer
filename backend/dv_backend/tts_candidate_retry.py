"""TTS synthesis retry strategy for ranked translation candidates."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Callable

from .duration_fit_policy import (
    acceptable_duration_fit,
    classify_duration_fit,
    duration_fit_decision_trace,
    policy_from_settings,
)
from .duration_predictor import predict_spoken_duration
from .semantic_safeguards import candidate_passes_semantic_guards
from .timing_profile import timing_profile_from_segment
from .translation_timing_rewrite import (
    lengthen_translation_for_timing,
    shorten_translation_for_timing,
)
from .tts_attempt_budget import TtsAttemptBudget, budget_from_settings
from .tts_speech_analysis import attach_speech_metrics, measure_speech_envelope
from .voice_duration_profile import resolve_voice_profile, update_voice_profile_from_sample


def timing_attempt_limits(settings: dict[str, Any]) -> dict[str, int]:
    budget = budget_from_settings(settings)
    return {
        "max_total": budget.max_total_syntheses,
        "max_candidate": budget.max_candidate_attempts,
        "max_rewrite": budget.max_rewrite_attempts,
    }


def _candidate_order(segment: dict[str, Any]) -> list[int]:
    candidates = segment.get("translation_candidates") or []
    selected = int(segment.get("selected_candidate_index") if segment.get("selected_candidate_index") is not None else 0)
    order: list[int] = []
    if 0 <= selected < len(candidates):
        order.append(selected)
    for index in range(len(candidates)):
        if index not in order:
            order.append(index)
    return order


def synthesize_with_candidate_retry(
    segment: dict[str, Any],
    *,
    settings: dict[str, Any],
    data_dir,
    language: str,
    session,
    synthesize_one: Callable[[str, Path], None],
    wav_path: Path,
    measure_envelope: Callable[[Path], Any] | None = None,
    database=None,
    estimate_word_count: Callable[[str], int] | None = None,
    dub_lang_label: str = "Vietnamese",
    attempt_budget: TtsAttemptBudget | None = None,
) -> dict[str, Any]:
    budget = attempt_budget or budget_from_settings(settings)
    policy = policy_from_settings(settings)
    profile = timing_profile_from_segment(segment)
    voice_profile = resolve_voice_profile(settings, language=language, data_dir=data_dir)
    candidates = segment.get("translation_candidates") or [
        {"text": segment.get("translation"), "style": "natural"},
    ]
    order = _candidate_order(segment)
    attempts: list[dict[str, Any]] = []
    accepted = False
    accepted_text: str | None = None
    rewrite_count = 0
    source_text = str(segment.get("text") or "")

    measure = measure_envelope or (lambda path: measure_speech_envelope(path))

    for candidate_index in order:
        if not budget.can_try_candidate():
            break
        candidate = candidates[candidate_index] if candidate_index < len(candidates) else {}
        text = str(candidate.get("text") or segment.get("translation") or "").strip()
        if not text:
            attempts.append({"candidate_index": candidate_index, "skipped": True, "reason": "rejected_empty", "source": "candidate"})
            continue
        if not candidate_passes_semantic_guards(text, source_text=source_text, reference_text=str(segment.get("translation") or "")):
            attempts.append({"candidate_index": candidate_index, "skipped": True, "reason": "rejected_semantic", "source": "candidate"})
            continue

        prediction = predict_spoken_duration(text, language, voice_profile=voice_profile)
        predicted = float(prediction["predicted_seconds"])
        started = time.perf_counter()
        synthesize_one(text, wav_path)
        budget.record_candidate()
        synth_ms = round((time.perf_counter() - started) * 1000)
        envelope = measure(wav_path)
        attach_speech_metrics(segment, envelope)
        speech_duration = float(envelope.speech_duration or envelope.raw_wav_duration)
        fit = classify_duration_fit(speech_duration, profile, policy=policy)
        attempt = {
            "candidate_index": candidate_index,
            "text": text,
            "predicted_duration": predicted,
            "actual_duration": round(envelope.raw_wav_duration, 3),
            "speech_duration": round(speech_duration, 3),
            "leading_silence": envelope.leading_silence,
            "trailing_silence": envelope.trailing_silence,
            "accepted": acceptable_duration_fit(fit),
            "reason": fit,
            "source": "candidate",
            "synthesize_ms": synth_ms,
            "duration_fit_trace": duration_fit_decision_trace(speech_duration, profile, policy=policy),
            "attempt_budget": budget.to_dict(),
        }
        attempts.append(attempt)

        if acceptable_duration_fit(fit):
            accepted = True
            accepted_text = text
            segment["tts_spoken_text"] = text
            segment["selected_candidate_index"] = candidate_index
            update_voice_profile_from_sample(
                settings,
                text=text,
                speech_duration_sec=speech_duration,
                data_dir=data_dir,
                language=language,
                measurement_confidence=float(getattr(envelope, "measurement_confidence", 1.0)),
            )
            break

    measured_attempts = [item for item in attempts if not item.get("skipped")]
    if (
        not accepted
        # Rewriting requires a real synthesized take to measure. Candidate guards may
        # append skipped attempts after a measured take, so select only measured entries.
        and measured_attempts
        and budget.can_rewrite()
        and database is not None
        and estimate_word_count is not None
    ):
        last = measured_attempts[-1]
        speech_duration = float(last.get("speech_duration") or 0.0)
        text = str(last.get("text") or segment.get("translation") or "")
        target = float(profile.get("speech_target_duration") or 0.0)
        hard_max = float(profile.get("hard_max_duration") or target)
        rewrite_text: str | None = None
        if speech_duration > hard_max and target > 0:
            rewrite_text, _ = shorten_translation_for_timing(
                settings,
                database,
                text=text,
                budget=target,
                current_duration=speech_duration,
                estimate_word_count=estimate_word_count,
                language_label=dub_lang_label,
            )
        elif speech_duration < float(profile.get("soft_min_duration") or 0.0):
            rewrite_text, _ = lengthen_translation_for_timing(
                settings,
                database,
                text=text,
                budget=target,
                current_duration=speech_duration,
                min_gap_sec=float(settings.get("short_tts_lengthen_min_gap_sec", 1.5) or 1.5),
                max_ratio=float(settings.get("short_tts_lengthen_max_ratio", 1.6) or 1.6),
                estimate_word_count=estimate_word_count,
                language_label=dub_lang_label,
            )
        if rewrite_text and rewrite_text.strip() and rewrite_text.strip() != text.strip():
            rewrite_count += 1
            prediction = predict_spoken_duration(rewrite_text, language, voice_profile=voice_profile)
            backup_path = wav_path.with_suffix(".pre_rewrite.wav")
            if wav_path.is_file():
                shutil.copy2(wav_path, backup_path)
            synthesize_one(rewrite_text.strip(), wav_path)
            budget.record_rewrite()
            envelope = measure(wav_path)
            attach_speech_metrics(segment, envelope)
            speech_duration = float(envelope.speech_duration or envelope.raw_wav_duration)
            fit = classify_duration_fit(speech_duration, profile, policy=policy)
            rewrite_accepted = acceptable_duration_fit(fit) or fit in {"slightly_long", "slightly_short"}
            attempts.append(
                {
                    "candidate_index": -1,
                    "text": rewrite_text.strip(),
                    "predicted_duration": prediction["predicted_seconds"],
                    "actual_duration": round(envelope.raw_wav_duration, 3),
                    "speech_duration": round(speech_duration, 3),
                    "accepted": rewrite_accepted,
                    "reason": f"rewrite:{fit}",
                    "source": "rewrite",
                    "attempt_budget": budget.to_dict(),
                }
            )
            if rewrite_accepted:
                accepted = True
                accepted_text = rewrite_text.strip()
            elif backup_path.is_file():
                shutil.copy2(backup_path, wav_path)
                envelope = measure(wav_path)
                attach_speech_metrics(segment, envelope)
            backup_path.unlink(missing_ok=True)

    if not accepted and measured_attempts:
        fallback = measured_attempts[-1]
        fallback_text = str(fallback.get("text") or "").strip()
        if fallback_text:
            accepted = True
            accepted_text = fallback_text
            segment["tts_spoken_text_source"] = "fallback_last_measured"

    if accepted_text:
        segment["translation"] = accepted_text
        segment["tts_spoken_text"] = accepted_text
    segment["tts_attempts"] = attempts
    segment["tts_attempt_count"] = len([item for item in attempts if not item.get("skipped")])
    segment["tts_attempt_budget"] = budget.to_dict()
    segment["accepted_without_repair"] = bool(
        accepted and all(item.get("source") != "rewrite" for item in attempts if item.get("accepted"))
    )
    return {
        "attempts": attempts,
        "accepted": accepted,
        "rewrite_count": rewrite_count,
        "attempt_budget": budget.to_dict(),
    }
