"""Voice duration calibration dataset loading and balanced sample selection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .dubbing_languages import normalize_dub_language

DATASET_VERSION = "vi_duration_v1"
DATASET_FILENAME = "voice_duration_calibration_vi_v1.json"

DATASET_BY_LANGUAGE: dict[str, tuple[str, str]] = {
    "vi": ("vi_duration_v1", "voice_duration_calibration_vi_v1.json"),
    "th": ("th_duration_v1", "voice_duration_calibration_th_v1.json"),
}

CALIBRATION_MODES = {"full": 100}


@dataclass(frozen=True)
class CalibrationSample:
    id: str
    text: str
    category: str
    difficulty: str = "normal"
    enabled: bool = True
    tags: tuple[str, ...] = ()


def dataset_path(language: str | None = None) -> Path:
    lang = normalize_dub_language(language)
    filename = DATASET_BY_LANGUAGE.get(lang, DATASET_BY_LANGUAGE["vi"])[1]
    return Path(__file__).resolve().parent / "data" / filename


def dataset_version_for_language(language: str | None = None) -> str:
    lang = normalize_dub_language(language)
    return DATASET_BY_LANGUAGE.get(lang, DATASET_BY_LANGUAGE["vi"])[0]


@lru_cache(maxsize=4)
def load_calibration_dataset(path: Path | None = None, language: str | None = None) -> dict[str, Any]:
    if path is None:
        lang = normalize_dub_language(language)
        target = dataset_path(lang)
        default_version = dataset_version_for_language(lang)
    else:
        target = path
        default_version = DATASET_VERSION
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Calibration dataset must be a JSON object.")
    payload.setdefault("version", default_version)
    samples = payload.get("samples") or []
    if not isinstance(samples, list):
        raise ValueError("Calibration dataset samples must be a list.")
    return payload


def enabled_samples(dataset: dict[str, Any]) -> list[CalibrationSample]:
    result: list[CalibrationSample] = []
    for entry in dataset.get("samples") or []:
        if not isinstance(entry, dict):
            continue
        if not entry.get("enabled", True):
            continue
        result.append(
            CalibrationSample(
                id=str(entry["id"]),
                text=str(entry["text"]),
                category=str(entry.get("category") or "normal_sentence"),
                difficulty=str(entry.get("difficulty") or "normal"),
                enabled=True,
                tags=tuple(entry.get("tags") or ()),
            )
        )
    return result


def _selection_seed(dataset_version: str, mode: str) -> int:
    digest = hashlib.sha256(f"{dataset_version}:{mode}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _round_robin_by_category(samples: list[CalibrationSample]) -> list[CalibrationSample]:
    by_category: dict[str, list[CalibrationSample]] = {}
    for sample in samples:
        by_category.setdefault(sample.category, []).append(sample)
    categories = sorted(by_category.keys())
    ordered: list[CalibrationSample] = []
    index = 0
    while True:
        added = False
        for category in categories:
            bucket = by_category[category]
            if index < len(bucket):
                ordered.append(bucket[index])
                added = True
        if not added:
            break
        index += 1
    return ordered


def select_calibration_samples(
    dataset: dict[str, Any],
    mode: str,
    dataset_version: str | None = None,
) -> list[CalibrationSample]:
    mode_key = (mode or "full").strip().lower()
    if mode_key not in CALIBRATION_MODES:
        raise ValueError(f"Unsupported calibration mode: {mode}")
    version = dataset_version or str(dataset.get("version") or DATASET_VERSION)
    all_enabled = enabled_samples(dataset)
    if not all_enabled:
        return []
    ordered = _round_robin_by_category(all_enabled)
    limit = CALIBRATION_MODES[mode_key]
    if limit is None:
        return ordered
    return ordered[: min(limit, len(ordered))]


def dataset_content_fingerprint(dataset: dict[str, Any]) -> str:
    samples = []
    for entry in dataset.get("samples") or []:
        if not isinstance(entry, dict):
            continue
        samples.append({"id": entry.get("id"), "text": entry.get("text"), "enabled": entry.get("enabled", True)})
    payload = json.dumps({"version": dataset.get("version"), "samples": samples}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def validate_dataset(dataset: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    samples = dataset.get("samples") or []
    ids = [str(entry.get("id")) for entry in samples if isinstance(entry, dict)]
    if len(ids) != len(set(ids)):
        issues.append("duplicate_sample_ids")
    categories = {str(entry.get("category")) for entry in samples if isinstance(entry, dict) and entry.get("enabled", True)}
    required = {
        "short_utterance",
        "normal_sentence",
        "long_sentence",
        "comma_pause",
        "question",
        "exclamation",
        "numbers",
        "decimal_numbers",
        "percentages",
        "currency",
        "dates",
        "acronyms",
        "latin_words",
        "product_models",
        "proper_names",
        "parentheses",
        "dash",
        "ellipsis",
        "mixed_punctuation",
    }
    missing = sorted(required - categories)
    if missing:
        issues.append(f"missing_categories:{','.join(missing)}")
    return issues
