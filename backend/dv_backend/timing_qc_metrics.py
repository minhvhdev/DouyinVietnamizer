"""Shared QC metric definitions for Phase 2 timing evaluation."""

from __future__ import annotations

import statistics
from typing import Any


def _segments_with_tts(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        segment
        for segment in segments
        if str(segment.get("translation") or "").strip()
        and (segment.get("tts_duration") or segment.get("tts_speech_duration") or segment.get("tts_path") or segment.get("tts_raw_path"))
    ]


def compute_timing_qc_metrics(segments: list[dict[str, Any]], *, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute timing QC metrics with consistent denominators."""
    del settings  # reserved for future phase-2-disabled detection
    tts_segments = _segments_with_tts(segments)
    denom = max(1, len(tts_segments))

    errors: list[float] = []
    tempos: list[float] = []
    placement_shifts: list[float] = []
    first_attempt = 0
    retries = 0
    rewrites = 0
    warning_stretch = 0
    danger_stretch = 0
    speech_trim = 0
    alignment_fallback = 0
    subtitle_overlap = 0
    tts_synthesis_count = 0
    candidate_api_calls = 0
    rewrite_api_calls = 0
    cache_avoided = 0
    semantic_critical = 0
    chunked_segments = 0
    total_chunks = 0
    chunk_retries = 0
    fidelity_checked = 0
    fidelity_good = 0
    fidelity_poor = 0
    fidelity_failed = 0
    fidelity_scores: list[float] = []
    very_long_count = 0

    for segment in tts_segments:
        predicted = segment.get("predicted_duration") or (segment.get("translation_duration_prediction") or {}).get("predicted_seconds")
        actual = segment.get("tts_speech_duration") or segment.get("tts_duration")
        if predicted is not None and actual is not None:
            errors.append(abs(float(actual) - float(predicted)) * 1000.0)

        attempts = segment.get("tts_attempts") or []
        attempt_count = int(segment.get("tts_attempt_count") or len(attempts) or 0)
        synth_count = sum(1 for item in attempts if item.get("source") in {"candidate", "rewrite"})
        tts_synthesis_count += synth_count if synth_count else (0 if segment.get("tts_cache_hit") else 1)

        if segment.get("tts_cache_hit"):
            cache_avoided += 1

        if attempt_count <= 1 and segment.get("accepted_without_repair"):
            first_attempt += 1
        if attempt_count > 1:
            retries += 1
        if any(item.get("source") == "rewrite" for item in attempts):
            rewrites += 1

        user_speed = float(segment.get("user_requested_speed") or 1.0)
        auto_tempo = float(segment.get("automatic_tempo_factor") or segment.get("time_stretch_factor") or 1.0)
        effective = float(segment.get("effective_speed") or user_speed * auto_tempo)
        tempos.append(effective)

        if segment.get("duration_repair_risk") == "warning" or (auto_tempo > 1.12 or auto_tempo < 0.9):
            warning_stretch += 1
        if segment.get("duration_repair_risk") == "danger" or auto_tempo > 1.25 or auto_tempo < 0.85:
            danger_stretch += 1
        if segment.get("speech_trimmed"):
            speech_trim += 1

        if segment.get("dub_alignment_status") in {"fallback", "skipped", "error"}:
            alignment_fallback += 1
        if segment.get("subtitle_overlap"):
            subtitle_overlap += 1

        if segment.get("candidate_generation_wall_time_ms") is not None and segment.get("translation_candidate_source") in {"gemini", "openai"}:
            candidate_api_calls += 1
        if any("shorten" in str(item.get("reason") or "") or "lengthen" in str(item.get("reason") or "") for item in attempts):
            rewrite_api_calls += 1

        selected_rankings = [
            ranking
            for ranking in (segment.get("candidate_rankings") or [])
            if ranking.get("selected") or ranking.get("index") == segment.get("selected_candidate_index")
        ]
        rankings_to_check = selected_rankings or []
        if any(ranking.get("critical_violation") for ranking in rankings_to_check):
            semantic_critical += 1

        if segment.get("tts_chunking_used"):
            chunked_segments += 1
        total_chunks += int(segment.get("tts_chunk_count") or 1)
        chunk_retries += int(segment.get("tts_chunk_retry_count") or 0)
        if "very_long_text_segment" in (segment.get("segment_diagnostics") or []):
            very_long_count += 1
        fidelity_status = str(segment.get("tts_fidelity_status") or "not_checked")
        if fidelity_status != "not_checked":
            fidelity_checked += 1
        if fidelity_status == "good":
            fidelity_good += 1
        elif fidelity_status in {"poor", "review"}:
            fidelity_poor += 1
        elif fidelity_status == "failed":
            fidelity_failed += 1
        score = segment.get("tts_text_similarity")
        if isinstance(score, (int, float)):
            fidelity_scores.append(float(score))

    sorted_errors = sorted(errors)
    sorted_tempos = sorted(tempos)
    p90_error = sorted_errors[int(len(sorted_errors) * 0.9)] if sorted_errors else None
    p90_tempo = sorted_tempos[int(len(sorted_tempos) * 0.9)] if sorted_tempos else None

    return {
        "segment_count": len(segments),
        "tts_segment_count": len(tts_segments),
        "first_attempt_acceptance_rate": round(first_attempt / denom, 4),
        "candidate_retry_rate": round(retries / denom, 4),
        "rewrite_rate": round(rewrites / denom, 4),
        "mean_prediction_error_ms": round(statistics.mean(errors), 1) if errors else None,
        "median_prediction_error_ms": round(statistics.median(errors), 1) if errors else None,
        "p90_prediction_error_ms": round(p90_error, 1) if p90_error is not None else None,
        "warning_stretch_count": warning_stretch,
        "danger_stretch_count": danger_stretch,
        "warning_stretch_rate": round(warning_stretch / denom, 4),
        "danger_stretch_rate": round(danger_stretch / denom, 4),
        "speech_trim_count": speech_trim,
        "median_effective_tempo": round(statistics.median(tempos), 4) if tempos else None,
        "p90_effective_tempo": round(p90_tempo, 4) if p90_tempo is not None else None,
        "placement_shift_median_ms": round(statistics.median(placement_shifts), 1) if placement_shifts else None,
        "alignment_fallback_count": alignment_fallback,
        "subtitle_overlap_count": subtitle_overlap,
        "tts_synthesis_call_count": tts_synthesis_count,
        "candidate_api_call_count": candidate_api_calls,
        "rewrite_api_call_count": rewrite_api_calls,
        "cache_avoided_tts_calls": cache_avoided,
        "semantic_safeguard_critical_violations": semantic_critical,
        "tts_chunked_segment_count": chunked_segments,
        "tts_total_chunk_count": total_chunks,
        "tts_chunk_retry_count": chunk_retries,
        "tts_fidelity_checked_count": fidelity_checked,
        "tts_fidelity_good_count": fidelity_good,
        "tts_fidelity_poor_count": fidelity_poor,
        "tts_fidelity_failed_count": fidelity_failed,
        "tts_text_similarity_mean": round(statistics.mean(fidelity_scores), 4) if fidelity_scores else None,
        "tts_text_similarity_median": round(statistics.median(fidelity_scores), 4) if fidelity_scores else None,
        "tts_text_similarity_p10": round(sorted(fidelity_scores)[max(0, int(len(fidelity_scores) * 0.1) - 1)], 4)
        if fidelity_scores
        else None,
        "very_long_segment_count": very_long_count,
    }


def compare_timing_metrics(baseline: dict[str, Any], phase2: dict[str, Any]) -> list[dict[str, Any]]:
    keys = [
        "first_attempt_acceptance_rate",
        "candidate_retry_rate",
        "rewrite_rate",
        "mean_prediction_error_ms",
        "median_prediction_error_ms",
        "p90_prediction_error_ms",
        "median_effective_tempo",
        "p90_effective_tempo",
        "warning_stretch_count",
        "danger_stretch_count",
        "speech_trim_count",
        "alignment_fallback_count",
        "subtitle_overlap_count",
        "tts_synthesis_call_count",
        "candidate_api_call_count",
    ]
    rows: list[dict[str, Any]] = []
    for key in keys:
        base_val = baseline.get(key)
        phase_val = phase2.get(key)
        delta = None
        if isinstance(base_val, (int, float)) and isinstance(phase_val, (int, float)):
            delta = round(float(phase_val) - float(base_val), 4)
        rows.append({"metric": key, "baseline": base_val, "phase2": phase_val, "delta": delta})
    return rows
