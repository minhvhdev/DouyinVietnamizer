"""Tests for OmniVoice num_steps benchmark tooling."""
from __future__ import annotations

from pathlib import Path

from dv_backend.omnivoice_steps_eval import (
    detect_missing_numbers,
    detect_truncated_ending,
    evaluate_quality_gate,
    recommend_steps,
    summarize_by_group,
)
from scripts.benchmark_omnivoice_steps import QUALITY_CORPUS, build_blind_manifest


def test_quality_corpus_has_minimum_segments_and_groups() -> None:
    assert len(QUALITY_CORPUS) >= 40
    groups = {entry["group"] for entry in QUALITY_CORPUS}
    assert groups >= {"short", "medium", "long"}


def test_blind_manifest_does_not_expose_num_steps(tmp_path: Path) -> None:
    manifest_path, key_path, entries = build_blind_manifest(
        out_dir=tmp_path,
        steps_levels=[32, 16],
        corpus=QUALITY_CORPUS[:4],
        seed=1,
    )
    manifest_text = manifest_path.read_text(encoding="utf-8")
    key_text = key_path.read_text(encoding="utf-8")
    assert len(entries) == 8
    assert "num_steps" not in manifest_text
    assert "32" in key_text
    assert "16" in key_text
    assert (tmp_path / "listening_scores_template.csv").is_file()


def test_summarize_by_group_computes_percentiles() -> None:
    rows = [
        {"group": "short", "fidelity_similarity": 0.95, "fidelity_retry_count": 0, "failure": False, "missing_number": False, "truncated_ending": False},
        {"group": "short", "fidelity_similarity": 0.90, "fidelity_retry_count": 0, "failure": False, "missing_number": False, "truncated_ending": False},
        {"group": "medium", "fidelity_similarity": 0.88, "fidelity_retry_count": 1, "failure": False, "missing_number": False, "truncated_ending": False},
    ]
    summary = summarize_by_group(rows)
    assert summary["short"]["count"] == 2
    assert summary["short"]["mean_similarity"] == 0.925
    assert summary["medium"]["p5_similarity"] <= summary["medium"]["mean_similarity"]


def test_detect_missing_numbers() -> None:
    assert detect_missing_numbers("Tỷ lệ 95,7% năm 2026.", "Tỷ lệ phần trăm năm.") is True
    assert detect_missing_numbers("Xin chào.", "Xin chào.") is False


def test_detect_truncated_ending() -> None:
    assert detect_truncated_ending("Câu dài kết thúc trọn vẹn.", "Câu dài kết thúc") is True
    assert detect_truncated_ending("Ngắn.", "Ngắn.", deletion_span=4) is False


def test_quality_gate_rejects_large_fidelity_drop() -> None:
    baseline = {
        "num_steps": 32,
        "failure_count_total": 0,
        "retry_count_mean": 0.0,
        "mean_fidelity_similarity": 0.95,
        "p5_fidelity_similarity": 0.90,
        "missing_number_total": 0,
        "truncated_ending_total": 0,
    }
    candidate = {
        "num_steps": 16,
        "failure_count_total": 0,
        "retry_count_mean": 0.0,
        "mean_fidelity_similarity": 0.90,
        "p5_fidelity_similarity": 0.80,
        "missing_number_total": 1,
        "truncated_ending_total": 0,
    }
    gate = evaluate_quality_gate(candidate=candidate, baseline=baseline, baseline_issues={"missing_number": 0, "truncated_ending": 0})
    assert gate["passed"] is False
    assert "mean_fidelity_drop" in gate["violations"]
    assert "new_missing_number" in gate["violations"]


def test_recommend_steps_prefers_fastest_passing_candidate() -> None:
    cases = [
        {"num_steps": 32, "speedup_vs_32_pct": 0.0, "quality_gate": {"passed": True}, "retry_count_mean": 0.0, "p5_fidelity_similarity": 0.9},
        {"num_steps": 16, "speedup_vs_32_pct": 25.0, "quality_gate": {"passed": True}, "retry_count_mean": 0.0, "p5_fidelity_similarity": 0.88},
        {"num_steps": 20, "speedup_vs_32_pct": 10.0, "quality_gate": {"passed": False}, "retry_count_mean": 0.0, "p5_fidelity_similarity": 0.89},
    ]
    rec = recommend_steps(cases)
    assert rec["recommendation"] == 16
