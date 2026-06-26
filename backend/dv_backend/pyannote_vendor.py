"""Local Pyannote Community-1 vendor cache helpers."""

from __future__ import annotations

import os
from pathlib import Path

PYANNOTE_MODEL_ID = "pyannote/speaker-diarization-community-1"
PYANNOTE_MODEL_DIRNAME = "speaker-diarization-community-1"
PYANNOTE_REQUIRED_MARKERS = ("config.yaml",)
PYANNOTE_REQUIRED_WEIGHTS = (
    "segmentation/pytorch_model.bin",
    "embedding/pytorch_model.bin",
    "plda/plda.npz",
    "plda/xvec_transform.npz",
)


def pyannote_cache_dir(vendor_dir: Path) -> Path:
    return vendor_dir / "pyannote"


def pyannote_model_dir(vendor_dir: Path) -> Path:
    return pyannote_cache_dir(vendor_dir) / PYANNOTE_MODEL_DIRNAME


def huggingface_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    return token.strip() if token and token.strip() else None


def validate_pyannote_model_dir(model_dir: Path) -> str | None:
    if not model_dir.is_dir():
        return f"Directory does not exist: {model_dir}"
    if not any(model_dir.iterdir()):
        return f"Directory is empty: {model_dir}"
    missing = [name for name in PYANNOTE_REQUIRED_MARKERS if not (model_dir / name).is_file()]
    missing_weights = [
        rel for rel in PYANNOTE_REQUIRED_WEIGHTS if not (model_dir / rel).is_file()
    ]
    if missing_weights:
        missing.extend(missing_weights)
    if missing:
        return f"Missing required files: {', '.join(missing)}"
    return None


def pyannote_bootstrap_action(*, token_present: bool) -> str:
    if not token_present:
        return (
            "Accept the Hugging Face license for pyannote/speaker-diarization-community-1, "
            "set HF_TOKEN in your environment, then open Môi trường and run bootstrap."
        )
    return (
        "Open Môi trường and run bootstrap to download "
        "vendor/pyannote/speaker-diarization-community-1/."
    )
