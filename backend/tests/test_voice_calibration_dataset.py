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


def test_thai_dataset_load_valid() -> None:
    dataset = load_calibration_dataset(language="th")
    assert dataset["version"] == "th_duration_v1"
    assert dataset.get("language") == "th"
    assert len(dataset["samples"]) >= 100
    assert len(select_calibration_samples(dataset, "full")) == 100
    assert not any(issue.startswith("missing_categories") for issue in validate_dataset(dataset))


def test_sample_ids_unique() -> None:
    dataset = load_calibration_dataset()
    ids = [entry["id"] for entry in dataset["samples"]]
    assert len(ids) == len(set(ids))


def test_categories_present() -> None:
    issues = validate_dataset(load_calibration_dataset())
    assert not any(issue.startswith("missing_categories") for issue in issues)


def test_only_full_calibration_mode_is_available() -> None:
    assert CALIBRATION_MODES == {"full": 100}


def test_full_uses_exactly_100_balanced_samples() -> None:
    dataset = load_calibration_dataset()
    selected = select_calibration_samples(dataset, "full")
    assert len(selected) == 100
    categories = Counter(sample.category for sample in selected)
    assert len(categories) >= 12


def test_selection_deterministic() -> None:
    dataset = load_calibration_dataset()
    first = [sample.id for sample in select_calibration_samples(dataset, "full")]
    second = [sample.id for sample in select_calibration_samples(dataset, "full")]
    assert first == second
