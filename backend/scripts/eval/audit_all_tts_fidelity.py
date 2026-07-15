#!/usr/bin/env python3
"""Run ASR-back fidelity on every TTS segment in a job (forced, no min_chars gate)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dv_backend.checkpoints import load_checkpoint, save_checkpoint  # noqa: E402
from dv_backend.config import AppConfig  # noqa: E402
from dv_backend.settings import DEFAULT_SETTINGS  # noqa: E402
from dv_backend.tts_fidelity import evaluate_tts_fidelity, transcribe_wav_to_text  # noqa: E402


def _vendor_dir() -> Path:
    env = os.environ.get("DV_VENDOR_DIR", "").strip()
    if env:
        return Path(env)
    return ROOT.parent / "vendor"


def _segment_wav(data_dir: Path, job_id: str, segment: dict) -> Path | None:
    raw = segment.get("tts_raw_path")
    if raw:
        p = Path(str(raw))
        if p.is_file():
            return p
    idx = segment.get("index")
    if idx is None:
        return None
    candidate = data_dir / "jobs" / job_id / "artifacts" / "tts" / f"tts_raw_{idx}.wav"
    return candidate if candidate.is_file() else None


def audit_job(*, data_dir: Path, job_id: str, write_checkpoint: bool) -> dict:
    for step in ("align_final_dub", "duration_repair", "tts"):
        cp = load_checkpoint(data_dir, job_id, step)
        if cp and cp.get("segments"):
            segments = list(cp["segments"])
            source_step = step
            break
    else:
        raise SystemExit(f"No segments found for job {job_id}")

    settings = dict(DEFAULT_SETTINGS)
    vendor = _vendor_dir()
    rows: list[dict] = []
    updated: list[dict] = []

    for segment in segments:
        idx = segment.get("index")
        translation = str(segment.get("translation") or "").strip()
        wav = _segment_wav(data_dir, job_id, segment)
        row = {
            "index": idx,
            "start": segment.get("start"),
            "end": segment.get("end"),
            "translation": translation,
            "tts_speech_duration": segment.get("tts_speech_duration"),
            "tts_duration": segment.get("tts_duration"),
            "wav_path": str(wav) if wav else None,
        }
        if not wav or not translation:
            fidelity = {
                "tts_text_similarity": None,
                "tts_content_coverage": None,
                "tts_fidelity_status": "not_checked",
                "tts_fidelity_warnings": ["missing_wav_or_translation"],
                "tts_asr_text": None,
            }
        else:
            heard = transcribe_wav_to_text(wav, vendor_dir=vendor, language="Vietnamese")
            fidelity = evaluate_tts_fidelity(
                expected_text=translation,
                heard_text=heard,
                settings=settings,
            )
        row.update(fidelity)
        rows.append(row)
        seg = dict(segment)
        seg.update(fidelity)
        updated.append(seg)

    summary = {
        "job_id": job_id,
        "source_checkpoint": source_step,
        "segment_count": len(rows),
        "checked": sum(1 for r in rows if r.get("tts_fidelity_status") != "not_checked"),
        "good": sum(1 for r in rows if r.get("tts_fidelity_status") == "good"),
        "review": sum(1 for r in rows if r.get("tts_fidelity_status") == "review"),
        "poor": sum(1 for r in rows if r.get("tts_fidelity_status") == "poor"),
        "failed": sum(1 for r in rows if r.get("tts_fidelity_status") == "failed"),
        "not_checked": sum(1 for r in rows if r.get("tts_fidelity_status") == "not_checked"),
    }

    out_dir = data_dir / "jobs" / job_id / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "tts_fidelity_full_audit.json"
    report_path.write_text(
        json.dumps({"summary": summary, "segments": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if write_checkpoint and source_step == "align_final_dub":
        cp = load_checkpoint(data_dir, job_id, source_step) or {}
        cp["segments"] = updated
        save_checkpoint(data_dir, job_id, source_step, cp)

    return {"summary": summary, "report_path": str(report_path), "segments": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--write-checkpoint", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    config = AppConfig(args.data_dir) if args.data_dir else AppConfig.from_env()
    result = audit_job(data_dir=config.data_dir, job_id=args.job_id, write_checkpoint=args.write_checkpoint)
    if args.json:
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        print(f"report: {result['report_path']}")


if __name__ == "__main__":
    main()
