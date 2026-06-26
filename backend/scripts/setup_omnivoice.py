#!/usr/bin/env python3
"""Create an isolated OmniVoice virtualenv for TTS inference."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--venv",
        default=str(Path(__file__).resolve().parents[1] / ".venv-omnivoice"),
        help="Target virtualenv path",
    )
    parser.add_argument(
        "--skip-torch",
        action="store_true",
        help="Skip installing PyTorch (use when torch is already present)",
    )
    args = parser.parse_args()

    backend_dir = Path(__file__).resolve().parents[1]
    venv_path = Path(args.venv).resolve()

    _run(["uv", "venv", str(venv_path), "--python", "3.12"], cwd=backend_dir)
    python = venv_path / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

    if not args.skip_torch:
        _run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "torch",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cu128",
            ],
            cwd=backend_dir,
        )

    _run(
        ["uv", "pip", "install", "--python", str(python), "omnivoice"],
        cwd=backend_dir,
    )

    _run([str(python), "-c", "import omnivoice; print('omnivoice', omnivoice.__version__)"], cwd=backend_dir)
    print(f"OmniVoice ready at {venv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
