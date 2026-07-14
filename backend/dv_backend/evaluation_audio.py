"""Export per-segment evaluation audio previews for A/B review."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

DEFAULT_PADDING_SEC = 0.3


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_clip(
    ffmpeg_path: str,
    input_path: Path,
    output_path: Path,
    *,
    start: float,
    duration: float,
) -> bool:
    if not input_path.is_file():
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_path,
        "-y",
        "-ss",
        f"{max(0.0, start):.3f}",
        "-t",
        f"{max(0.05, duration):.3f}",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path.is_file() and output_path.stat().st_size > 44
    except (subprocess.CalledProcessError, OSError):
        return False


def _resolve_dub_wav(job_dir: Path, segment: dict[str, Any]) -> Path | None:
    idx = segment.get("index")
    for candidate in (
        segment.get("tts_repaired_path"),
        segment.get("tts_path"),
        segment.get("tts_raw_path"),
    ):
        if candidate:
            path = Path(str(candidate))
            if path.is_file():
                return path
    if idx is not None:
        tts_dir = job_dir / "artifacts" / "tts"
        for name in (f"tts_repaired_{idx}.wav", f"tts_{idx}.wav", f"tts_raw_{idx}.wav"):
            path = tts_dir / name
            if path.is_file():
                return path
    return None


def _source_audio_path(job_dir: Path) -> Path | None:
    artifacts = job_dir / "artifacts"
    for name in ("vocals_16k.wav", "vocals.wav", "audio_16k.wav", "original_48k.wav"):
        path = artifacts / name
        if path.is_file():
            return path
    return None


def export_evaluation_audio(
    data_dir: Path,
    job_id: str,
    segments: list[dict[str, Any]],
    *,
    output_dir: Path | None = None,
    ffmpeg_path: str = "ffmpeg",
    padding_sec: float = DEFAULT_PADDING_SEC,
    label: str = "job",
) -> dict[str, Any]:
    job_dir = data_dir / "jobs" / job_id
    out_dir = output_dir or (job_dir / "artifacts" / "evaluation_audio" / label)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_audio = _source_audio_path(job_dir)
    exported: list[dict[str, Any]] = []
    fingerprint_parts: list[str] = []

    for segment in segments:
        idx = int(segment.get("index", 0))
        start = max(0.0, float(segment.get("start") or 0.0) - padding_sec)
        end = float(segment.get("end") or start)
        duration = max(0.1, (end - float(segment.get("start") or 0.0)) + padding_sec * 2)

        row: dict[str, Any] = {"index": idx, "files": {}}
        source_out = out_dir / f"segment_{idx:03d}_source.wav"
        dub_out = out_dir / f"segment_{idx:03d}_{label}.wav"

        meta_path = out_dir / f"segment_{idx:03d}.meta.json"
        dub_wav = _resolve_dub_wav(job_dir, segment)
        inputs = {
            "source_audio": str(source_audio) if source_audio else None,
            "dub_wav": str(dub_wav) if dub_wav else None,
            "start": segment.get("start"),
            "end": segment.get("end"),
            "padding_sec": padding_sec,
        }
        expected_fp = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode("utf-8")).hexdigest()[:24]
        if meta_path.is_file():
            try:
                cached = json.loads(meta_path.read_text(encoding="utf-8"))
                if cached.get("fingerprint") == expected_fp and source_out.is_file() and dub_out.is_file():
                    row["files"] = cached.get("files", {})
                    row["cached"] = True
                    exported.append(row)
                    continue
            except (OSError, json.JSONDecodeError):
                pass

        if source_audio:
            if _extract_clip(ffmpeg_path, source_audio, source_out, start=start, duration=duration):
                row["files"]["source"] = str(source_out.relative_to(job_dir))
                fingerprint_parts.append(_sha256_file(source_out))

        if dub_wav and dub_wav.is_file():
            shutil.copy2(dub_wav, dub_out)
            row["files"][label] = str(dub_out.relative_to(job_dir))
            fingerprint_parts.append(_sha256_file(dub_out))

        meta_path.write_text(
            json.dumps({"fingerprint": expected_fp, "files": row["files"], "inputs": inputs}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        exported.append(row)

    bundle_fp = hashlib.sha256("".join(fingerprint_parts).encode()).hexdigest()[:24] if fingerprint_parts else None
    return {"output_dir": str(out_dir), "segments": exported, "bundle_fingerprint": bundle_fp}
