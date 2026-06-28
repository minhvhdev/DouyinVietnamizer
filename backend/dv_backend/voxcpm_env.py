"""Resolve the isolated VoxCPM Python environment."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def voxcpm_venv_root() -> Path:
    override = os.environ.get("DV_VOXCPM_VENV", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / ".venv-voxcpm"


def resolve_voxcpm_python() -> Path:
    env_override = os.environ.get("DV_VOXCPM_PYTHON", "").strip()
    if env_override:
        path = Path(env_override).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"DV_VOXCPM_PYTHON does not exist: {path}")

    venv_root = voxcpm_venv_root()
    if sys.platform == "win32":
        candidates = (
            venv_root / "Scripts" / "python.exe",
            venv_root / "Scripts" / "python",
        )
    else:
        candidates = (
            venv_root / "bin" / "python3",
            venv_root / "bin" / "python",
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "VoxCPM environment was not found. "
        f"Expected virtualenv at {venv_root}. "
        "Run: python scripts/setup_voxcpm.py"
    )


def is_voxcpm_available() -> bool:
    try:
        python = resolve_voxcpm_python()
    except FileNotFoundError:
        return False
    try:
        completed = subprocess.run(
            [str(python), "-c", "import voxcpm"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0
