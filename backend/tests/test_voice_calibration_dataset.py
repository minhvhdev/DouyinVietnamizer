"""Tests for voice calibration dataset selection."""

from __future__ import annotations

from collections import Counter

from dv_backend.voice_calibration_dataset import (
    CALIBRATION_MODES,
    load_calibration_dataset,
    select_calibration_samples,
    validate_dataset,
)


def test_dataset_load_valid() -> None:
    dataset = load_calibration_dataset()
    assert dataset["version"] == "vi_duration_v1"
    assert len(dataset["samples"]) >= 100


def test_sample_ids_unique() -> None:
    dataset = load_calibration_dataset()
    ids = [entry["id"] for entry in dataset["samples"]]
    assert len(ids) == len(set(ids))


def test_categories_present() -> None:
    issues = validate_dataset(load_calibration_dataset())
    assert not any(issue.startswith("missing_categories") for issue in issues)


def test_quick_selection_balanced() -> None:
    dataset = load_calibration_dataset()
    selected = select_calibration_samples(dataset, "quick")
    assert len(selected) == CALIBRATION_MODES["quick"]
    categories = Counter(sample.category for sample in selected)
    assert len(categories) >= 8


def test_standard_selection_balanced() -> None:
    dataset = load_calibration_dataset()
    selected = select_calibration_samples(dataset, "standard")
    assert len(selected) == CALIBRATION_MODES["standard"]
    categories = Counter(sample.category for sample in selected)
    assert len(categories) >= 12


def test_full_uses_all_enabled() -> None:
    dataset = load_calibration_dataset()
    selected = select_calibration_samples(dataset, "full")
    enabled = [entry for entry in dataset["samples"] if entry.get("enabled", True)]
    assert len(selected) == len(enabled)


def test_selection_deterministic() -> None:
    dataset = load_calibration_dataset()
    first = [sample.id for sample in select_calibration_samples(dataset, "standard")]
    second = [sample.id for sample in select_calibration_samples(dataset, "standard")]
    assert first == second
