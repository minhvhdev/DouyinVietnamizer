"""Analyze whether subtitle ASR is used or overridden for a job."""
from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

from dv_backend.subtitle_timing import (
    SUBTITLE_MIN_CHUNKS_FOR_TTS_ASR,
    SUBTITLE_MIN_DURATION_FOR_TTS_ASR,
    annotate_subtitle_playback_windows,
    build_segment_subtitle_cues,
    enforce_monotonic_cues,
    map_chunks_to_asr_timeline,
    segment_subtitle_end,
    segment_subtitle_start,
    split_for_subtitle_display,
    transcribe_tts_clip_for_subtitles,
    _asr_cues_are_usable,
    allocate_proportional_cues,
    _resolve_tts_wav,
)
from dv_backend.pipeline import transcribe_audio


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def main(job_id: str) -> None:
    job = Path.home() / "AppData/Local/DouyinVietnamizer/jobs" / job_id
    repair = json.loads((job / "checkpoints/duration_repair.json").read_text(encoding="utf-8"))
    segments = repair["segments"]
    annotate_subtitle_playback_windows(segments)

    project_root = Path(__file__).resolve().parents[2]
    vendor_dir = project_root / "vendor"
    ffmpeg = project_root / "tools" / "ffmpeg" / "ffmpeg.exe"
    if not ffmpeg.is_file():
        ffmpeg = Path("ffmpeg")

    rows = {}
    database_settings = {}
    try:
        from dv_backend.config import AppConfig
        from dv_backend.database import Database

        config = AppConfig.load()
        database = Database(config.database_path)
        database_settings = {
            row["key"]: json.loads(row["value"])
            for row in database.connection.execute("SELECT key, value FROM settings").fetchall()
        }
    except Exception:
        database_settings = {"qwen3_device": "cuda:0"}

    asr_ok = asr_reject = asr_skip = 0
    print(f"job={job_id}\n")
    print("idx | chunks | window | wav | speech_est | ASR verdict")
    print("-" * 72)

    for segment in segments:
        index = segment.get("index")
        chunks = split_for_subtitle_display(str(segment.get("translation") or ""))
        if not chunks:
            continue

        window_start = segment_subtitle_start(segment)
        window_end = segment_subtitle_end(segment)
        window_duration = window_end - window_start
        wav_path = _resolve_tts_wav(job, segment)
        wav_d = wav_duration(wav_path) if wav_path and wav_path.is_file() else None

        eligible = (
            len(chunks) >= SUBTITLE_MIN_CHUNKS_FOR_TTS_ASR
            and window_duration >= SUBTITLE_MIN_DURATION_FOR_TTS_ASR
            and wav_path is not None
        )
        if not eligible:
            asr_skip += 1
            print(
                f"{index:3} | {len(chunks):6} | {window_duration:6.2f}s | "
                f"{wav_d or 0:5.2f}s | skip (single chunk or short window)"
            )
            continue

        try:
            from dv_backend.dubbing_languages import dub_language_config, dub_language_from_settings

            language = dub_language_config(dub_language_from_settings(database_settings))["label_en"]
            units = transcribe_tts_clip_for_subtitles(
                wav_path,
                vendor_dir=vendor_dir,
                settings=database_settings,
                language=language,
                ffmpeg_path=ffmpeg,
                cache_dir=job / "artifacts" / "subtitle_asr",
                transcribe_fn=transcribe_audio,
            )
            aligned = map_chunks_to_asr_timeline(
                chunks,
                units,
                window_start=window_start,
                window_duration=window_duration,
            )
            if not aligned:
                verdict = "reject (map failed) -> proportional"
                asr_reject += 1
            else:
                normalized = enforce_monotonic_cues(
                    aligned,
                    window_start=window_start,
                    window_end=window_end,
                )
                if _asr_cues_are_usable(
                    normalized,
                    window_duration=window_duration,
                    chunk_count=len(chunks),
                ):
                    verdict = "USE ASR"
                    asr_ok += 1
                else:
                    verdict = "reject (low quality) -> proportional"
                    asr_reject += 1
                    rows[index] = {
                        "aligned_first": aligned[0],
                        "aligned_last": aligned[-1],
                        "normalized_last": normalized[-1],
                        "min_dur": min(float(c["end"]) - float(c["start"]) for c in normalized),
                        "units": len(units),
                    }
        except Exception as exc:
            verdict = f"reject (error: {exc}) -> proportional"
            asr_reject += 1

        speech_est = "?"
        if units:
            try:
                speech_est = f"{float(units[-1]['end']):.2f}s"
            except Exception:
                pass

        print(
            f"{index:3} | {len(chunks):6} | {window_duration:6.2f}s | "
            f"{wav_d or 0:5.2f}s | {speech_est:>10} | {verdict}"
        )

    print("-" * 72)
    print(f"ASR used: {asr_ok} | rejected/fallback: {asr_reject} | skipped: {asr_skip}")

    if rows.get(7):
        print("\nsegment 7 detail:")
        for key, value in rows[7].items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ebef2dac-3e2e-4dc2-9f8b-575d7e50342d")
