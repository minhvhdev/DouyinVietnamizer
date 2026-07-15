#!/usr/bin/env python3
"""Create an isolated virtualenv with OmniVoice and verify the runtime."""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def omnivoice_runtime_packages(platform_name: str, machine: str) -> list[str]:
    if platform_name == "darwin":
        if machine.lower() not in {"arm64", "aarch64"}:
            raise ValueError("OmniVoice MPS setup supports only Apple Silicon.")
        return [
            "torch==2.8.0",
            "torchaudio==2.8.0",
            "soundfile",
            "omnivoice==0.2.0",
        ]
    return ["torch", "torchaudio", "soundfile", "omnivoice==0.2.0"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--venv-dir",
        default="",
        help="Target virtualenv directory (default: backend/venvs/omnivoice)",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Only verify that omnivoice can be imported in the existing venv",
    )
    args = parser.parse_args()

    backend_dir = Path(__file__).resolve().parents[1]
    venv_dir = (
        Path(args.venv_dir).expanduser().resolve()
        if args.venv_dir
        else backend_dir / "venvs" / "omnivoice"
    )
    if sys.platform == "win32":
        python = venv_dir / "Scripts" / "python.exe"
    else:
        python = venv_dir / "bin" / "python3"

    if not args.skip_install:
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        if not python.is_file():
            _run([sys.executable, "-m", "venv", str(venv_dir)])
        _run([str(python), "-m", "pip", "install", "--upgrade", "pip", "wheel"])
        try:
            runtime_packages = omnivoice_runtime_packages(sys.platform, platform.machine())
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        _run(
            [str(python), "-m", "pip", "install", *runtime_packages]
        )

    if not python.is_file():
        print(f"OmniVoice venv python not found: {python}", file=sys.stderr)
        return 1

    completed = subprocess.run(
        [str(python), "-m", "dv_backend.adapters.omnivoice_worker", "--health-check"],
        capture_output=True,
        text=True,
        cwd=str(backend_dir),
        env={
            **dict(__import__("os").environ),
            "PYTHONPATH": str(backend_dir),
        },
    )
    if completed.returncode != 0:
        print(completed.stdout, file=sys.stderr)
        print(completed.stderr, file=sys.stderr)
        print("OmniVoice health check failed.", file=sys.stderr)
        return 1

    print(f"OmniVoice venv: {venv_dir}")
    print(f"Python: {python}")
    print("OmniVoice runtime is ready.")
    print(f"Set DV_OMNIVOICE_VENV={venv_dir} if auto-detection does not find it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
