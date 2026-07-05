"""Resolve the VoxCPM2 GGUF runtime (voxcpm2-cli + model weights)."""

from __future__ import annotations

from pathlib import Path

from .voxcpm_gguf import (
    is_gguf_runtime_ready,
    resolve_voxcpm_cli,
    resolve_voxcpm_gguf_paths,
    resolve_worker_python,
)


def voxcpm_venv_root() -> Path:
    """Legacy name kept for portable-runtime env wiring.

    Returns the default GGUF model directory under ``DV_MODELS_DIR`` when set,
    otherwise ``backend/models/voxcpm2``.
    """
    import os

    override = os.environ.get("DV_MODELS_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve() / "voxcpm2"
    return Path(__file__).resolve().parents[1] / "models" / "voxcpm2"


def resolve_voxcpm_python() -> Path:
    return resolve_worker_python()


def is_voxcpm_available() -> bool:
    return is_gguf_runtime_ready()


__all__ = [
    "is_voxcpm_available",
    "resolve_voxcpm_cli",
    "resolve_voxcpm_gguf_paths",
    "resolve_voxcpm_python",
    "voxcpm_venv_root",
]
