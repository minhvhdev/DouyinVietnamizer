#!/usr/bin/env python3
"""Targeted TTS re-synthesis for specific job segments, then optional pipeline rerun."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dv_backend.adapters.tts import TtsSession, create_tts_adapter, prepare_spoken_text_for_tts
from dv_backend.checkpoints import PIPELINE_STEPS, load_checkpoint, save_checkpoint
from dv_backend.config import AppConfig
from dv_backend.database import Database
from dv_backend.jobs import JobService
from dv_backend.omnivoice_wav_concat import concat_omnivoice_chunks
from dv_backend.runner import JobRunner
from dv_backend.tts_fidelity import (
    compact_fidelity_text,
    evaluate_tts_fidelity,
    transcribe_wav_to_text,
)
from dv_backend.tts_speech_analysis import attach_speech_metrics, measure_speech_envelope
from dv_backend.pipeline import get_wav_duration


def _vendor_dir() -> Path:
    env = os.environ.get("DV_VENDOR_DIR", "").strip()
    if env:
        return Path(env)
    return ROOT.parent / "vendor"


def _segment_index(segment: dict) -> int:
    return int(segment.get("index", 0))


def _score_candidate(expected: str, heard: str, fidelity: dict, *, critical_phrases: list[str] | None = None) -> float:
    compact_heard = compact_fidelity_text(heard)
    phrase_bonus = 0.0
    for phrase in critical_phrases or []:
        compact_phrase = compact_fidelity_text(phrase)
        if compact_phrase and compact_phrase in compact_heard:
            phrase_bonus += 0.12
    sim = float(fidelity.get("tts_text_similarity") or 0.0)
    cov = float(fidelity.get("tts_content_coverage") or 0.0)
    status = str(fidelity.get("tts_fidelity_status") or "")
    status_bonus = {"good": 0.15, "review": 0.05}.get(status, 0.0)
    del_penalty = min(0.2, float(fidelity.get("tts_max_contiguous_deletion") or 0) / 100.0)
    return sim * 0.45 + cov * 0.4 + phrase_bonus + status_bonus - del_penalty


def _critical_phrases_for_text(text: str) -> list[str]:
    phrases: list[str] = []
    if "súc sinh" in text.lower():
        phrases.append("Súc sinh vẫn là súc sinh")
    if "145" in text or "mười bốn" in text.lower():
        phrases.extend(["145 năm", "59 năm", "Cẩu yêu khai trí"])
    if "Thẩm đại nhân" in text:
        phrases.extend(["Bảo vệ Thẩm đại nhân", "xiêu xiêu vẹo vẹo", "gia nô"])
    return phrases


def _clause_chunks(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _evaluate_candidate(
    *,
    expected: str,
    wav_path: Path,
    settings: dict,
    vendor: Path,
    critical_phrases: list[str],
) -> dict:
    heard = transcribe_wav_to_text(wav_path, vendor_dir=vendor, language="Vietnamese")
    fidelity = evaluate_tts_fidelity(expected_text=expected, heard_text=heard, settings=settings)
    score = _score_candidate(expected, heard, fidelity, critical_phrases=critical_phrases)
    return {"heard": heard, "score": score, **fidelity}


def _synth_per_clause_assembly(
    *,
    clauses: list[str],
    synthesize_fn,
    tts_dir: Path,
    idx: int,
    assembly_attempt: int,
    pause_ms: int,
    clause_seeds: int,
    settings: dict,
    vendor: Path,
) -> tuple[Path, list[Path]]:
    chunk_paths: list[Path] = []
    for clause_i, clause in enumerate(clauses):
        best_path: Path | None = None
        best_score = -1.0
        for seed in range(clause_seeds):
            clause_path = tts_dir / f"tts_raw_{idx}.clause_{clause_i}.a{assembly_attempt}.s{seed}.wav"
            synthesize_fn(clause, clause_path)
            heard = transcribe_wav_to_text(clause_path, vendor_dir=vendor, language="Vietnamese")
            fidelity = evaluate_tts_fidelity(expected_text=clause, heard_text=heard, settings=settings)
            score = float(fidelity.get("tts_content_coverage") or 0.0) + float(fidelity.get("tts_text_similarity") or 0.0)
            if score > best_score:
                best_score = score
                best_path = clause_path
        if best_path is None:
            raise RuntimeError(f"Failed to synthesize clause {clause_i} for segment {idx}")
        chunk_paths.append(best_path)
    output_path = tts_dir / f"tts_raw_{idx}.per_clause_a{assembly_attempt}.wav"
    concat_omnivoice_chunks(
        chunk_paths,
        pause_ms_list=[pause_ms] * max(0, len(chunk_paths) - 1),
        output_path=output_path,
    )
    return output_path, chunk_paths


def resynth_segment(
    *,
    segment: dict,
    settings: dict,
    config: AppConfig,
    runner,
    job_id: str,
    seeds: int,
    allow_chunk_fallback: bool,
    per_clause: bool,
    clause_pause_ms: int,
    clause_seeds: int,
) -> dict:
    vendor = _vendor_dir()
    idx = _segment_index(segment)
    text = prepare_spoken_text_for_tts(
        str(segment.get("translation") or ""),
        speech_duration=float(segment.get("original_duration") or 0.0),
    )
    tts_dir = config.data_dir / "jobs" / job_id / "artifacts" / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)
    raw_tts = tts_dir / f"tts_raw_{idx}.wav"
    candidates: list[dict] = []
    critical_phrases = _critical_phrases_for_text(text)

    with TtsSession(settings, data_dir=config.data_dir, runner=runner, adapter_factory=create_tts_adapter) as session:
        def synthesize_fn(candidate_text: str, output_path: Path) -> None:
            session.synthesize(candidate_text, output_path, segment=segment)

        if per_clause:
            clauses = _clause_chunks(text)
            if len(clauses) > 1:
                for attempt in range(seeds):
                    assembly_path, _chunk_paths = _synth_per_clause_assembly(
                        clauses=clauses,
                        synthesize_fn=synthesize_fn,
                        tts_dir=tts_dir,
                        idx=idx,
                        assembly_attempt=attempt,
                        pause_ms=clause_pause_ms,
                        clause_seeds=clause_seeds,
                        settings=settings,
                        vendor=vendor,
                    )
                    row = _evaluate_candidate(
                        expected=text,
                        wav_path=assembly_path,
                        settings=settings,
                        vendor=vendor,
                        critical_phrases=critical_phrases,
                    )
                    candidates.append({"attempt": f"per_clause_{attempt}", "path": str(assembly_path), **row})

        for attempt in range(seeds):
            candidate_path = tts_dir / f"tts_raw_{idx}.candidate_{attempt}.wav"
            synthesize_fn(text, candidate_path)
            row = _evaluate_candidate(
                expected=text,
                wav_path=candidate_path,
                settings=settings,
                vendor=vendor,
                critical_phrases=critical_phrases,
            )
            candidates.append({"attempt": attempt, "path": str(candidate_path), **row})

        best = max(candidates, key=lambda row: float(row["score"]))
        chosen_path = Path(str(best["path"]))
        used_chunk_fallback = False

        if allow_chunk_fallback and str(best.get("tts_fidelity_status")) in {"failed", "poor", "review"}:
            from dv_backend.omnivoice_chunk_synthesis import synthesize_omnivoice_with_chunking

            chunk_path = tts_dir / f"tts_raw_{idx}.chunked.wav"
            synthesize_omnivoice_with_chunking(
                text=text,
                output_path=chunk_path,
                synthesize_fn=synthesize_fn,
                settings={**settings, "omnivoice_external_chunking_enabled": True},
                segment=segment,
                language="vi",
                transcribe_fn=lambda path: transcribe_wav_to_text(path, vendor_dir=vendor, language="Vietnamese"),
                vendor_dir=vendor,
            )
            row = _evaluate_candidate(
                expected=text,
                wav_path=chunk_path,
                settings=settings,
                vendor=vendor,
                critical_phrases=critical_phrases,
            )
            chunk_row = {"attempt": "chunked", "path": str(chunk_path), **row}
            candidates.append(chunk_row)
            if float(chunk_row["score"]) >= float(best["score"]):
                best = chunk_row
                chosen_path = chunk_path
                used_chunk_fallback = True

        shutil.copy2(chosen_path, raw_tts)
        envelope = measure_speech_envelope(raw_tts)
        attach_speech_metrics(segment, envelope)
        segment["tts_spoken_text"] = text
        segment["tts_raw_path"] = str(raw_tts)
        segment["tts_duration"] = round(get_wav_duration(raw_tts), 2)
        segment["tts_fidelity_status"] = best.get("tts_fidelity_status")
        segment["tts_text_similarity"] = best.get("tts_text_similarity")
        segment["tts_content_coverage"] = best.get("tts_content_coverage")
        segment["tts_asr_text"] = best.get("heard")
        segment["tts_targeted_resynth"] = True
        segment["tts_resynth_candidates"] = candidates
        segment["tts_chunking_used"] = used_chunk_fallback or str(best.get("attempt", "")).startswith("per_clause")
        return {
            "index": idx,
            "chosen": best,
            "used_chunk_fallback": used_chunk_fallback,
            "candidates": candidates,
        }


def _update_checkpoint_segments(data_dir: Path, job_id: str, step: str, indices: set[int], segments_by_index: dict[int, dict]) -> None:
    cp = load_checkpoint(data_dir, job_id, step)
    if not cp or not cp.get("segments"):
        return
    updated = []
    for segment in cp["segments"]:
        idx = _segment_index(segment)
        if idx in indices and idx in segments_by_index:
            merged = dict(segment)
            merged.update(segments_by_index[idx])
            updated.append(merged)
        else:
            updated.append(segment)
    cp["segments"] = updated
    save_checkpoint(data_dir, job_id, step, cp)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    parser.add_argument("segments", nargs="+", type=int, help="1-based segment numbers")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--chunk-fallback", action="store_true")
    parser.add_argument("--per-clause", action="store_true", help="Also try per-clause synth with pause concat.")
    parser.add_argument("--clause-pause-ms", type=int, default=120)
    parser.add_argument("--clause-seeds", type=int, default=2, help="Seeds per clause inside each per-clause assembly.")
    parser.add_argument("--rerun-from", default="duration_repair", choices=list(PIPELINE_STEPS))
    parser.add_argument("--no-rerun", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("DV_VENDOR_DIR", str(_vendor_dir()))
    config = AppConfig.from_env()
    database = Database(config.database_path)
    database.migrate()
    rows = database.connection.execute("SELECT key, value FROM settings").fetchall()
    settings = {r["key"]: json.loads(r["value"]) for r in rows}

    indices = {int(value) - 1 for value in args.segments}
    source = None
    segments: list[dict] = []
    for step in ("tts", "translate"):
        cp = load_checkpoint(config.data_dir, args.job_id, step)
        if cp and cp.get("segments"):
            source = step
            segments = [dict(segment) for segment in cp["segments"]]
            break
    if not segments:
        raise SystemExit(f"No segments found for job {args.job_id}")

    runner = JobRunner(config, database)
    results: list[dict] = []
    segments_by_index: dict[int, dict] = {}
    for segment in segments:
        if _segment_index(segment) not in indices:
            continue
        result = resynth_segment(
            segment=segment,
            settings=settings,
            config=config,
            runner=runner,
            job_id=args.job_id,
            seeds=max(1, args.seeds),
            allow_chunk_fallback=args.chunk_fallback,
            per_clause=args.per_clause,
            clause_pause_ms=max(80, args.clause_pause_ms),
            clause_seeds=max(1, args.clause_seeds),
        )
        results.append(result)
        segments_by_index[_segment_index(segment)] = segment

    for step in ("translate", "tts"):
        _update_checkpoint_segments(config.data_dir, args.job_id, step, indices, segments_by_index)

    report_path = config.data_dir / "jobs" / args.job_id / "artifacts" / "targeted_resynth_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"source": source, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))

    if args.no_rerun:
        return 0

    jobs = JobService(database, config.data_dir)
    keep = list(PIPELINE_STEPS[: PIPELINE_STEPS.index(args.rerun_from)])
    jobs.rerun(args.job_id, keep)
    print(f"Rerun from {args.rerun_from}, kept through {keep[-1] if keep else 'none'}", flush=True)
    runner.start_job(args.job_id)
    deadline = time.time() + 6 * 3600
    while time.time() < deadline:
        hydrated = jobs.get(args.job_id)
        if hydrated.status in {"completed", "failed", "cancelled", "interrupted"}:
            print("FINAL", hydrated.status, hydrated.last_error_code, hydrated.last_error_message, flush=True)
            return 0 if hydrated.status == "completed" else 1
        time.sleep(15)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
