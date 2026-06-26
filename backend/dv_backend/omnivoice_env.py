"""Resolve the isolated OmniVoice Python environment."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def omnivoice_venv_root() -> Path:
    override = os.environ.get("DV_OMNIVOICE_VENV", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / ".venv-omnivoice"


def resolve_omnivoice_python() -> Path:
    env_override = os.environ.get("DV_OMNIVOICE_PYTHON", "").strip()
    if env_override:
        path = Path(env_override).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"DV_OMNIVOICE_PYTHON does not exist: {path}")

    venv_root = omnivoice_venv_root()
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
        "OmniVoice environment was not found. "
        f"Expected virtualenv at {venv_root}. "
        "Run: python scripts/setup_omnivoice.py"
    )


def is_omnivoice_available() -> bool:
    try:
        python = resolve_omnivoice_python()
    except FileNotFoundError:
        return False
    try:
        completed = subprocess.run(
            [str(python), "-c", "import omnivoice"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0
