"""Evaluation helpers for OmniVoice num_steps benchmark."""
from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Any

from ..semantic_safeguards import _NUMBER, _count

QUALITY_GATE = {
    "max_mean_fidelity_drop": 0.01,
    "max_p5_fidelity_drop": 0.02,
    "max_retry_rate_increase_pct": 2.0,
    "max_listening_text_drop": 0.2,
    "max_listening_voice_drop": 0.2,
}


def extract_segment_row(
    *,
    num_steps: int,
    run_index: int,
    segment_index: int,
    group: str,
    target_text: str,
    output_path: Path,
    segment: dict[str, Any] | None,
    audio_duration_sec: float | None = None,
) -> dict[str, Any]:
    segment = segment or {}
    heard = str(segment.get("tts_asr_text") or "")
    status = str(segment.get("tts_fidelity_status") or "not_checked")
    similarity = segment.get("tts_text_similarity")
    deletion_span = segment.get("tts_max_contiguous_deletion")
    checked = status not in {"not_checked", "None", ""}
    return {
        "num_steps": num_steps,
        "run_index": run_index,
        "segment_index": segment_index,
        "group": group,
        "target_text": target_text,
        "output_path": str(output_path),
        "audio_duration_sec": round(audio_duration_sec, 4) if audio_duration_sec is not None else None,
        "fidelity_similarity": float(similarity) if isinstance(similarity, (int, float)) else None,
        "fidelity_status": status,
        "fidelity_retry_count": int(segment.get("tts_chunk_retry_count") or 0),
        "asr_transcript": heard or None,
        "content_coverage": segment.get("tts_content_coverage"),
        "max_contiguous_deletion": deletion_span,
        "missing_number": detect_missing_numbers(target_text, heard) if checked else False,
        "truncated_ending": detect_truncated_ending(target_text, heard, deletion_span) if checked else False,
        "failure": status == "failed",
        "missing_output": not output_path.is_file() or output_path.stat().st_size == 0,
    }


def detect_missing_numbers(expected: str, heard: str) -> bool:
    expected_count = _count(_NUMBER, expected)
    if expected_count <= 0:
        return False
    heard_count = _count(_NUMBER, heard)
    return heard_count < expected_count


def detect_truncated_ending(
    expected: str,
    heard: str,
    deletion_span: Any = None,
) -> bool:
    if deletion_span is not None:
        try:
            if int(deletion_span) >= 12:
                return True
        except (TypeError, ValueError):
            pass
    expected_clean = re.sub(r"\s+", " ", (expected or "").strip())
    heard_clean = re.sub(r"\s+", " ", (heard or "").strip())
    if not expected_clean or not heard_clean:
        return False
    if expected_clean[-1] in ".?!…" and heard_clean[-1] not in ".?!…":
        return len(heard_clean) < len(expected_clean) * 0.85
    return len(heard_clean) < len(expected_clean) * 0.75


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def summarize_by_group(segment_results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in segment_results:
        grouped.setdefault(str(row.get("group") or "unknown"), []).append(row)
    summary: dict[str, Any] = {}
    for group, rows in grouped.items():
        similarities = [
            float(row["fidelity_similarity"])
            for row in rows
            if isinstance(row.get("fidelity_similarity"), (int, float))
        ]
        summary[group] = {
            "count": len(rows),
            "mean_similarity": round(statistics.mean(similarities), 4) if similarities else None,
            "p5_similarity": round(percentile(similarities, 5), 4) if similarities else None,
            "retry_rate": round(
                sum(int(row.get("fidelity_retry_count") or 0) for row in rows) / max(1, len(rows)),
                4,
            ),
            "failure_rate": round(
                sum(1 for row in rows if row.get("failure")) / max(1, len(rows)),
                4,
            ),
            "missing_number_count": sum(1 for row in rows if row.get("missing_number")),
            "truncated_ending_count": sum(1 for row in rows if row.get("truncated_ending")),
        }
    return summary


def evaluate_quality_gate(
    *,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    baseline_issues: dict[str, int] | None = None,
) -> dict[str, Any]:
    baseline_issues = baseline_issues or {}
    candidate_issues = {
        "missing_number": int(candidate.get("missing_number_total") or 0),
        "truncated_ending": int(candidate.get("truncated_ending_total") or 0),
    }
    mean_drop = None
    p5_drop = None
    if candidate.get("mean_fidelity_similarity") is not None and baseline.get("mean_fidelity_similarity") is not None:
        mean_drop = round(float(baseline["mean_fidelity_similarity"]) - float(candidate["mean_fidelity_similarity"]), 4)
    if candidate.get("p5_fidelity_similarity") is not None and baseline.get("p5_fidelity_similarity") is not None:
        p5_drop = round(float(baseline["p5_fidelity_similarity"]) - float(candidate["p5_fidelity_similarity"]), 4)
    retry_increase = None
    if candidate.get("retry_count_mean") is not None and baseline.get("retry_count_mean") is not None:
        retry_increase = round(float(candidate["retry_count_mean"]) - float(baseline["retry_count_mean"]), 4)
    violations: list[str] = []
    if candidate.get("failure_count_total", 0) > baseline.get("failure_count_total", 0):
        violations.append("failure_count_increased")
    if retry_increase is not None and retry_increase > QUALITY_GATE["max_retry_rate_increase_pct"]:
        violations.append("retry_rate_increased")
    if mean_drop is not None and mean_drop > QUALITY_GATE["max_mean_fidelity_drop"]:
        violations.append("mean_fidelity_drop")
    if p5_drop is not None and p5_drop > QUALITY_GATE["max_p5_fidelity_drop"]:
        violations.append("p5_fidelity_drop")
    for key, count in candidate_issues.items():
        if count > baseline_issues.get(key, 0):
            violations.append(f"new_{key}")
    passed = not violations
    return {
        "passed": passed,
        "violations": violations,
        "mean_fidelity_drop": mean_drop,
        "p5_fidelity_drop": p5_drop,
        "retry_increase": retry_increase,
    }


def recommend_steps(cases: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next((case for case in cases if case.get("num_steps") == 32), None)
    if baseline is None:
        return {"recommendation": 32, "reason": "baseline_missing"}
    ordered = sorted(
        (case for case in cases if case.get("quality_gate", {}).get("passed")),
        key=lambda item: (
            -float(item.get("speedup_vs_32_pct") or 0.0),
            float(item.get("retry_count_mean") or 0.0),
            -float(item.get("p5_fidelity_similarity") or 0.0),
        ),
    )
    if not ordered:
        return {"recommendation": 32, "reason": "no_candidate_passed_quality_gate"}
    best = ordered[0]
    return {
        "recommendation": int(best["num_steps"]),
        "reason": "passed_quality_gate_best_speed",
        "speedup_vs_32_pct": best.get("speedup_vs_32_pct"),
    }
