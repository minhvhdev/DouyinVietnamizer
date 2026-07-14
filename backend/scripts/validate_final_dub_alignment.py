#!/usr/bin/env python3
"""Validate final dub alignment checkpoints without requiring the desktop UI."""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path


def _repo_backend_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _bootstrap() -> None:
    backend_dir = _repo_backend_dir()
    repo_root = backend_dir.parent
    sys.path.insert(0, str(backend_dir))
    os.environ.setdefault("DV_VENDOR_DIR", str(repo_root / "vendor"))


def _load_job_context(job_id: str, data_dir: Path):
    from dv_backend.checkpoints import load_checkpoint
    from dv_backend.config import AppConfig

    config = AppConfig.from_env()
    if data_dir:
        config = AppConfig(data_dir)
    job_dir = config.data_dir / "jobs" / job_id
    align_cp = load_checkpoint(config.data_dir, job_id, "align_final_dub")
    repair_cp = load_checkpoint(config.data_dir, job_id, "duration_repair")
    segments = []
    source = "none"
    if align_cp and isinstance(align_cp.get("segments"), list):
        segments = list(align_cp["segments"])
        source = "align_final_dub"
    elif repair_cp and isinstance(repair_cp.get("segments"), list):
        segments = list(repair_cp["segments"])
        source = "duration_repair"
    return config, job_dir, segments, source, align_cp, repair_cp


def validate_job(
    job_id: str,
    *,
    data_dir: Path,
    segment_filter: int | None,
    force_realign: bool,
    no_model: bool,
) -> dict:
    _bootstrap()
    from dv_backend.adapters.subtitles import build_subtitle_cues
    from dv_backend.config import AppConfig
    from dv_backend.database import Database
    from dv_backend.final_dub_alignment import (
        build_alignment_cache_identity,
        reconstruct_target_text_from_dub_words,
        refresh_all_segment_dub_word_timestamps,
        segment_has_usable_dub_words,
        segment_placement_start,
        segment_target_text,
        validate_dub_words_timeline,
    )
    from dv_backend.pipeline import align_final_dub_step
    from dv_backend.subtitle_timing import resolve_ass_quantized_cues, resolve_overlapping_cues

    config = AppConfig(data_dir)
    job_dir = config.data_dir / "jobs" / job_id
    _config, job_dir, segments, source, _align_cp, _repair_cp = _load_job_context(job_id, data_dir)
    if not segments:
        raise SystemExit(f"No segments found for job {job_id}")

    if force_realign and not no_model:
        database = Database(config.database_path)
        database.migrate()
        from dv_backend.runner import JobRunner

        align_final_dub_step(job_id, config, database, JobRunner(config, database))
        _config, job_dir, segments, source, _align_cp, _repair_cp = _load_job_context(job_id, data_dir)

    refresh_all_segment_dub_word_timestamps(segments)
    report_segments: list[dict] = []
    total_overlap = 0
    total_oob = 0

    for segment in segments:
        index = segment.get("index")
        if segment_filter is not None and int(index or -1) != segment_filter:
            continue
        target_text = segment_target_text(segment)
        placement = segment_placement_start(segment)
        audio_rel = segment.get("dub_alignment_audio_path")
        audio_path = job_dir / str(audio_rel) if audio_rel else None
        if audio_path is None or not audio_path.is_file():
            repaired = job_dir / "artifacts" / "tts" / f"tts_repaired_{index}.wav"
            audio_path = repaired if repaired.is_file() else None
        audio_duration = float(segment.get("repaired_duration") or 0.0)
        dub_words = list(segment.get("dub_words") or [])
        timeline = validate_dub_words_timeline(
            dub_words,
            placement_start=placement,
            max_duration=audio_duration,
        )
        reconstructed = reconstruct_target_text_from_dub_words(dub_words)
        text_valid = reconstructed.replace(" ", "") == target_text.replace(" ", "") or bool(dub_words)
        cues = build_subtitle_cues([segment]) if segment_has_usable_dub_words(segment) else []
        ass_cues = resolve_ass_quantized_cues(resolve_overlapping_cues(cues))
        overlap = 0
        for idx in range(1, len(ass_cues)):
            if float(ass_cues[idx]["start"]) < float(ass_cues[idx - 1]["end"]) - 0.001:
                overlap += 1
        oob = sum(
            1
            for cue in ass_cues
            if float(cue["start"]) < placement - 0.05
            or float(cue["end"]) > placement + audio_duration + 0.2
        )
        total_overlap += overlap
        total_oob += oob
        cache_identity = None
        if audio_path and audio_path.is_file() and target_text:
            cache_identity = build_alignment_cache_identity(
                audio_path=audio_path,
                target_text=target_text,
                target_language=str(segment.get("dub_alignment_language") or "Vietnamese"),
                asr_model=str(segment.get("dub_asr_model") or ""),
                aligner_model=str(segment.get("dub_aligner_model") or ""),
            )
        report_segments.append(
            {
                "index": index,
                "audio_path": str(audio_path) if audio_path else None,
                "audio_duration": audio_duration,
                "placement_start": placement,
                "target_text": target_text,
                "alignment_status": segment.get("dub_alignment_status"),
                "alignment_method": segment.get("dub_alignment_method"),
                "relative_timeline_valid": timeline["relative_timeline_valid"],
                "absolute_timeline_valid": timeline["absolute_timeline_valid"],
                "text_reconstruction_valid": text_valid,
                "cue_count": len(ass_cues),
                "cue_overlap_count": overlap,
                "out_of_bounds_count": oob,
                "cache_identity": cache_identity,
                "warnings": timeline["warnings"],
            }
        )

    payload = {
        "job_id": job_id,
        "checkpoint_source": source,
        "segment_count": len(report_segments),
        "subtitle_cue_overlap_count": total_overlap,
        "subtitle_out_of_bounds_count": total_oob,
        "segments": report_segments,
        "no_model": no_model,
        "force_realign": force_realign,
    }
    out_json = job_dir / "artifacts" / "final_dub_validation.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _write_html(payload: dict, path: Path) -> None:
    rows = []
    for segment in payload.get("segments") or []:
        warnings = ", ".join(segment.get("warnings") or []) or "-"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(segment.get('index')))}</td>"
            f"<td>{html.escape(str(segment.get('alignment_status')))}</td>"
            f"<td>{html.escape(str(segment.get('alignment_method')))}</td>"
            f"<td>{segment.get('relative_timeline_valid')}</td>"
            f"<td>{segment.get('absolute_timeline_valid')}</td>"
            f"<td>{segment.get('text_reconstruction_valid')}</td>"
            f"<td>{segment.get('cue_count')}</td>"
            f"<td>{html.escape(warnings)}</td>"
            "</tr>"
        )
    content = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Final Dub Validation</title></head><body>"
        f"<h1>Final Dub Validation — {html.escape(str(payload.get('job_id')))}</h1>"
        f"<p>Checkpoint source: {html.escape(str(payload.get('checkpoint_source')))}</p>"
        "<table border='1' cellpadding='4'>"
        "<tr><th>Index</th><th>Status</th><th>Method</th><th>Rel</th><th>Abs</th>"
        "<th>Text</th><th>Cues</th><th>Warnings</th></tr>"
        f"{''.join(rows)}"
        "</table></body></html>"
    )
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    parser.add_argument("--segment", type=int, default=None)
    parser.add_argument("--export-html", action="store_true")
    parser.add_argument("--force-realign", action="store_true")
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.force_realign and args.no_model:
        print("--force-realign cannot be combined with --no-model", file=sys.stderr)
        return 2

    _bootstrap()
    from dv_backend.config import AppConfig

    config = AppConfig(args.data_dir) if args.data_dir else AppConfig.from_env()

    payload = validate_job(
        args.job_id.strip(),
        data_dir=config.data_dir,
        segment_filter=args.segment,
        force_realign=args.force_realign,
        no_model=args.no_model,
    )
    job_dir = config.data_dir / "jobs" / args.job_id.strip()
    json_path = job_dir / "artifacts" / "final_dub_validation.json"
    print(json_path)
    if args.export_html:
        html_path = job_dir / "artifacts" / "final_dub_validation.html"
        _write_html(payload, html_path)
        print(html_path)
    invalid = [
        segment
        for segment in payload.get("segments") or []
        if not segment.get("relative_timeline_valid")
        or not segment.get("absolute_timeline_valid")
        or not segment.get("text_reconstruction_valid")
    ]
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
