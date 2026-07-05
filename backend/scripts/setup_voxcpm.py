#!/usr/bin/env python3
"""Download VoxCPM2 GGUF weights and verify the voxcpm2-cli runtime."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _download_gguf(*, dest: Path, repo: str, files: list[str]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install it in the backend environment: uv pip install huggingface_hub"
        ) from exc

    for filename in files:
        target = dest / filename
        if target.is_file() and target.stat().st_size > 0:
            print(f"Already present: {target}")
            continue
        print(f"Downloading {repo}/{filename} -> {target}")
        downloaded = hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=str(dest),
        )
        print(f"Saved {downloaded}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        default="",
        help="Directory for GGUF files (default: backend/models/voxcpm2 or DV_MODELS_DIR/voxcpm2)",
    )
    parser.add_argument(
        "--repo",
        default="DennisHuang648/VoxCPM2-GGUF",
        help="Hugging Face repo id for GGUF weights",
    )
    parser.add_argument(
        "--quant",
        choices=("q8", "f16"),
        default="q8",
        help="BaseLM quantization to download (acoustic stack stays F16)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only verify voxcpm2-cli and existing GGUF files",
    )
    args = parser.parse_args()

    backend_dir = Path(__file__).resolve().parents[1]
    if args.models_dir:
        models_dest = Path(args.models_dir).expanduser().resolve()
    else:
        models_dest = backend_dir / "models" / "voxcpm2"

    baselm = (
        "VoxCPM2-BaseLM-Q8_0.gguf"
        if args.quant == "q8"
        else "VoxCPM2-BaseLM-F16.gguf"
    )
    acoustic = "VoxCPM2-Acoustic-F16.gguf"

    if not args.skip_download:
        _download_gguf(dest=models_dest, repo=args.repo, files=[baselm, acoustic])

    sys.path.insert(0, str(backend_dir))
    from dv_backend.voxcpm_gguf import is_gguf_runtime_ready, resolve_voxcpm_cli, resolve_voxcpm_gguf_paths

    if not is_gguf_runtime_ready():
        print(
            "VoxCPM2 GGUF runtime is incomplete.\n"
            f"  models: {models_dest}\n"
            "  cli: build llama.cpp-omni target voxcpm2-cli and place it under vendor/voxcpm2/\n"
            "       or set DV_VOXCPM_CLI to the binary path.",
            file=sys.stderr,
        )
        return 1

    cli = resolve_voxcpm_cli()
    baselm_path, acoustic_path = resolve_voxcpm_gguf_paths("gguf-q8" if args.quant == "q8" else "gguf-f16")
    print(f"voxcpm2-cli: {cli}")
    print(f"BaseLM: {baselm_path}")
    print(f"Acoustic: {acoustic_path}")
    print("VoxCPM2 GGUF runtime is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
