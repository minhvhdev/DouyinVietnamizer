"""Verify speaker diarization on a job's audio_16k.wav."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dv_backend.adapters.asr import reset_model_cache, transcribe_audio  # noqa: E402


def main() -> int:
    job_id = sys.argv[1] if len(sys.argv) > 1 else "f0a00d36-6f88-47c7-a8ac-f800b2d68dfb"
    data_dir = Path(os.environ.get("LOCALAPPDATA", "")) / "DouyinVietnamizer"
    artifacts = data_dir / "jobs" / job_id / "artifacts"
    audio_path = artifacts / "vocals_16k.wav"
    if not audio_path.is_file():
        audio_path = artifacts / "audio_16k.wav"
    vendor_dir = ROOT.parent / "vendor"

    if not audio_path.is_file():
        print("missing audio:", audio_path)
        return 1

    print("audio:", audio_path)
    print("size_mb:", round(audio_path.stat().st_size / (1024 * 1024), 2))
    reset_model_cache()

    segments = transcribe_audio(
        audio_path,
        vendor_dir=vendor_dir,
        speaker_diarization=True,
    )

    speakers = Counter(str(seg.get("speaker_id")) for seg in segments if seg.get("speaker_id") is not None)
    summary = {
        "segment_count": len(segments),
        "speaker_ids": dict(speakers),
        "sample": [
            {
                "start": seg.get("start"),
                "end": seg.get("end"),
                "speaker_id": seg.get("speaker_id"),
                "text_preview": str(seg.get("text") or "")[:80],
            }
            for seg in segments[:8]
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if len(segments) < 2 or not speakers:
        return 1
    if len(speakers) < 2:
        print("WARN: only one speaker detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
