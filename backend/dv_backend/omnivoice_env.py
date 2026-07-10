"""Resolve the isolated OmniVoice Python runtime."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

OMNIVOICE_DEFAULT_MODEL = "k2-fsa/OmniVoice"
OMNIVOICE_DEFAULT_SAMPLE_RATE = 24_000


def omnivoice_venv_root() -> Path:
    override = os.environ.get("DV_OMNIVOICE_VENV", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "venvs" / "omnivoice"


def _read_pyvenv_home(venv_root: Path) -> Path | None:
    cfg = venv_root / "pyvenv.cfg"
    if not cfg.is_file():
        return None
    try:
        for line in cfg.read_text(encoding="utf-8").splitlines():
            if not line.startswith("home = "):
                continue
            home = Path(line.split("=", 1)[1].strip())
            if sys.platform == "win32":
                candidate = home / "python.exe"
            else:
                candidate = home / "bin" / "python3"
            if candidate.is_file():
                return candidate
    except OSError:
        return None
    return None


def build_omnivoice_subprocess_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    venv_root = omnivoice_venv_root()
    env["VIRTUAL_ENV"] = str(venv_root)
    scripts = venv_root / ("Scripts" if sys.platform == "win32" else "bin")
    env["PATH"] = str(scripts) + os.pathsep + env.get("PATH", "")
    env.setdefault("PYTHONUNBUFFERED", "1")

    pythonpath_parts: list[str] = []
    if sys.platform == "win32":
        site_packages = venv_root / "Lib" / "site-packages"
    else:
        pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        site_packages = venv_root / "lib" / pyver / "site-packages"
    if site_packages.is_dir():
        pythonpath_parts.append(str(site_packages))

    existing = env.get("PYTHONPATH", "").strip()
    if existing:
        pythonpath_parts.append(existing)
    if pythonpath_parts:
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return env


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

    real_python = _read_pyvenv_home(venv_root)
    if real_python is not None:
        return real_python

    return Path(sys.executable).resolve()


def is_omnivoice_available() -> bool:
    python = resolve_omnivoice_python()
    env = build_omnivoice_subprocess_env()
    try:
        completed = subprocess.run(
            [str(python), "-c", "import omnivoice"],
            capture_output=True,
            text=True,
            timeout=30.0,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


__all__ = [
    "OMNIVOICE_DEFAULT_MODEL",
    "OMNIVOICE_DEFAULT_SAMPLE_RATE",
    "build_omnivoice_subprocess_env",
    "is_omnivoice_available",
    "omnivoice_venv_root",
    "resolve_omnivoice_python",
]
