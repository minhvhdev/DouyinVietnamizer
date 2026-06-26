"""Download Pyannote Community-1 into vendor/pyannote/. Requires HF_TOKEN in .env or environment."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dv_backend.local_env import load_repo_dotenv
from dv_backend.pyannote_vendor import pyannote_model_dir, validate_pyannote_model_dir


def main() -> int:
    load_repo_dotenv()
    import os

    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")):
        print("HF_TOKEN is not set. Add it to .env or your environment.", file=sys.stderr)
        return 1

    from huggingface_hub import snapshot_download

    project_root = Path(__file__).resolve().parents[2]
    vendor_dir = project_root / "vendor"
    dest = pyannote_model_dir(vendor_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)

    issue = validate_pyannote_model_dir(dest)
    if issue is None:
        print(f"Already installed: {dest}")
        return 0

    if dest.is_dir() and any(dest.iterdir()):
        print(f"Removing incomplete install: {dest}")
        import shutil

        shutil.rmtree(dest, ignore_errors=True)

    print(f"Downloading pyannote/speaker-diarization-community-1 -> {dest}")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    try:
        snapshot_download(
            "pyannote/speaker-diarization-community-1",
            local_dir=str(dest),
            token=token,
        )
    except Exception as error:
        message = str(error)
        if "GatedRepoError" in type(error).__name__ or "gated" in message.lower() or "403" in message:
            print(
                "Access denied. Log into Hugging Face with the same account as HF_TOKEN, "
                "open https://huggingface.co/pyannote/speaker-diarization-community-1 "
                "and accept the user conditions, then rerun this script.",
                file=sys.stderr,
            )
        raise
    remaining = validate_pyannote_model_dir(dest)
    if remaining:
        print(f"Download incomplete: {remaining}", file=sys.stderr)
        return 1
    print("Pyannote Community-1 ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
