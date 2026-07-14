from __future__ import annotations

import json
import wave
from pathlib import Path

from dv_backend.checkpoints import save_checkpoint
from dv_backend.config import AppConfig


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x01" * 8000)


import importlib.util
import sys
from pathlib import Path


def test_validate_final_dub_script_no_model(tmp_path: Path, monkeypatch) -> None:
    from dv_backend.checkpoints import save_checkpoint
    from dv_backend.config import AppConfig

    config = AppConfig(tmp_path)
    config.ensure_directories()
    job_id = "job-validate"
    job_dir = tmp_path / "jobs" / job_id
    wav = job_dir / "artifacts" / "tts" / "tts_repaired_0.wav"
    _write_wav(wav)
    segments = [
        {
            "index": 0,
            "translation": "Xin chào.",
            "placement_start": 2.5,
            "repaired_duration": 0.5,
            "dub_alignment_status": "aligned",
            "dub_alignment_method": "qwen_forced_aligner_words",
            "dub_words": [
                {
                    "text": "Xin",
                    "start": 0.0,
                    "end": 0.2,
                    "absolute_start": 2.5,
                    "absolute_end": 2.7,
                    "alignment": "exact",
                },
                {
                    "text": "chào.",
                    "start": 0.2,
                    "end": 0.45,
                    "absolute_start": 2.7,
                    "absolute_end": 2.95,
                    "alignment": "exact",
                },
            ],
        }
    ]
    save_checkpoint(tmp_path, job_id, "align_final_dub", {"segments": segments})
    monkeypatch.setenv("DV_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate", job_id, "--no-model", "--data-dir", str(tmp_path)],
    )
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_final_dub_alignment.py"
    spec = importlib.util.spec_from_file_location("validate_final_dub_alignment", script_path)
    assert spec and spec.loader
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    assert validator.main() == 0
    payload = json.loads((job_dir / "artifacts" / "final_dub_validation.json").read_text(encoding="utf-8"))
    assert payload["segments"][0]["absolute_timeline_valid"] is True
