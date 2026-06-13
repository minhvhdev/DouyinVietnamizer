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
    "mix",
    "render",
    "qc",
)


def checkpoint_path(data_dir: Path, job_id: str, step_name: str) -> Path:
    return data_dir / "jobs" / job_id / "checkpoints" / f"{step_name}.json"

