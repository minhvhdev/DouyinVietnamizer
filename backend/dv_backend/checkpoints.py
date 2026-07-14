import json
import os
import tempfile
from pathlib import Path


PIPELINE_STEPS = (
    "resolve",
    "download",
    "extract_audio",
    "vad",
    "asr",
    "normalize_segments",
    "translate",
    "tts",
    "duration_repair",
    "align_final_dub",
    "mix",
    "render",
    "qc",
)


def checkpoint_path(data_dir: Path, job_id: str, step_name: str) -> Path:
    return data_dir / "jobs" / job_id / "checkpoints" / f"{step_name}.json"


def save_checkpoint(data_dir: Path, job_id: str, step_name: str, data: dict) -> None:
    path = checkpoint_path(data_dir, job_id, step_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = path.parent
    fd, temp_path = tempfile.mkstemp(dir=str(temp_dir), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, str(path))
    except Exception:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass
        raise


def load_checkpoint(data_dir: Path, job_id: str, step_name: str) -> dict | None:
    path = checkpoint_path(data_dir, job_id, step_name)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

