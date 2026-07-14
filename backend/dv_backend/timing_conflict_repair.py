"""Conflict-cluster timing repair: merge clause fragments and fit within hard anchors."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .subtitle_timing import split_translation_sentences
from .timing_placement import (
    BOUNDARY_MARGIN_SEC,
    HARD_ANCHOR_SILENCE_SEC,
    HARD_MAX_DRIFT_SEC,
    _clip_duration,
    compute_placement_starts,
    enforce_zero_overlap_placements,
    schedule_soft_placements,
    segments_with_voiced_overlap,
)

logger = logging.getLogger(__name__)

MIN_DIALOGUE_GAP_SEC = 0.05
PREFERRED_MERGED_MIN_SEC = 2.5
PREFERRED_MERGED_MAX_SEC = 7.0
MAX_CLUSTER_SEGMENTS = 5
MAX_CLUSTER_SPAN_SEC = 14.0


def _audible_end(segment: dict[str, Any]) -> float:
    start = float(segment.get("placement_start") or segment.get("start") or 0.0)
    return start + _clip_duration(segment)


def find_conflict_clusters(
    segments: list[dict[str, Any]],
    *,
    expand: int = 1,
    hard_anchor_silence_sec: float = HARD_ANCHOR_SILENCE_SEC,
) -> list[list[int]]:
    """Return lists of segment indices that form overflow/overlap conflict clusters."""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda item: int(item.get("index", 0) or 0))
    conflict: set[int] = set()

    for i, seg in enumerate(ordered):
        if float(seg.get("timing_overflow_sec") or 0.0) > 0.15:
            conflict.add(i)
        if i + 1 < len(ordered):
            end = _audible_end(seg)
            nxt = float(ordered[i + 1].get("placement_start") or ordered[i + 1].get("start") or 0.0)
            if end + MIN_DIALOGUE_GAP_SEC > nxt + 0.02:
                conflict.add(i)
                conflict.add(i + 1)

    if not conflict:
        return []

    expanded: set[int] = set()
    for i in sorted(conflict):
        lo = max(0, i - expand)
        hi = min(len(ordered) - 1, i + expand)
        for j in range(lo, hi + 1):
            if j == i:
                expanded.add(j)
                continue
            left = ordered[min(i, j)]
            right = ordered[max(i, j)]
            gap = float(right.get("start") or 0.0) - float(left.get("end") or 0.0)
            if gap >= hard_anchor_silence_sec:
                continue
            expanded.add(j)

    runs: list[list[int]] = []
    current: list[int] = []
    for i in sorted(expanded):
        if not current or i == current[-1] + 1:
            if current:
                left = ordered[current[-1]]
                right = ordered[i]
                gap = float(right.get("start") or 0.0) - float(left.get("end") or 0.0)
                if gap >= hard_anchor_silence_sec:
                    runs.append(current)
                    current = [i]
                    continue
            current.append(i)
        else:
            runs.append(current)
            current = [i]
    if current:
        runs.append(current)

    # Split long runs so re-merge stays local (ChatGPT: cluster, not whole-scene mash).
    capped: list[list[int]] = []
    for run in runs:
        chunk: list[int] = []
        chunk_start = 0.0
        for pos in run:
            seg = ordered[pos]
            idx = int(seg.get("index", 0) or 0)
            if not chunk:
                chunk = [idx]
                chunk_start = float(seg.get("start") or 0.0)
                continue
            span = float(seg.get("end") or 0.0) - chunk_start
            if len(chunk) >= MAX_CLUSTER_SEGMENTS or span > MAX_CLUSTER_SPAN_SEC:
                capped.append(chunk)
                chunk = [idx]
                chunk_start = float(seg.get("start") or 0.0)
            else:
                chunk.append(idx)
        if chunk:
            capped.append(chunk)
    return capped


def _join_vietnamese(parts: list[str]) -> str:
    cleaned = [str(part or "").strip().lstrip(".") for part in parts if str(part or "").strip()]
    if not cleaned:
        return ""
    out = cleaned[0]
    for part in cleaned[1:]:
        if out.endswith((".", "!", "?", "…", "。", "！", "？")):
            out = f"{out} {part}"
        else:
            out = f"{out} {part}"
    return " ".join(out.split())


def _pack_sentences_into_units(
    sentences: list[str],
    *,
    total_budget: float,
) -> list[str]:
    """Pack sentence list into roughly preferred-duration text units."""
    if not sentences:
        return []
    if total_budget <= PREFERRED_MERGED_MAX_SEC * 1.2 or len(sentences) == 1:
        return [" ".join(sentences)]

    target_chars = max(
        24,
        int(sum(len(s) for s in sentences) * (PREFERRED_MERGED_MIN_SEC / max(total_budget, 0.1))),
    )
    units: list[str] = []
    buf = ""
    for sentence in sentences:
        candidate = f"{buf} {sentence}".strip() if buf else sentence
        if buf and len(candidate) > max(40, target_chars * 2):
            units.append(buf)
            buf = sentence
        else:
            buf = candidate
    if buf:
        units.append(buf)
    return units


def repair_conflict_clusters(
    segments: list[dict[str, Any]],
    *,
    settings: dict[str, Any],
    ffmpeg_path: Path,
    tts_dir: Path,
    job_id: str,
    runner: Any,
    session: Any,
    get_wav_duration: Callable[[Path], float],
    build_atempo_chain: Callable[[float], str],
    run_ffmpeg_audio_filter: Callable[..., None],
    video_duration: float | None = None,
) -> list[dict[str, Any]]:
    """Merge conflict clusters into clause-complete units and re-TTS to fit anchors."""
    del settings  # reserved for future compact/rewrite policy
    clusters = find_conflict_clusters(segments)
    if not clusters or session is None:
        return segments

    by_index = {int(seg.get("index", 0) or 0): dict(seg) for seg in segments}
    ordered_indices = sorted(by_index)
    consumed: set[int] = set()
    rebuilt: list[dict[str, Any]] = []
    next_new_index = 0
    cluster_map = {idx: cluster for cluster in clusters for idx in cluster}

    def _propose_cluster_speed(seg: dict[str, Any], duration: float, alloc: float) -> float:
        """Record needed speed only; uniform apply happens later in duration_repair."""
        proposed = 1.0
        if duration > alloc + 0.15:
            proposed = min(1.2, duration / max(alloc, 0.2))
        prev = float(seg.get("proposed_speed_factor") or 1.0)
        seg["proposed_speed_factor"] = round(max(prev, proposed), 4)
        return duration

    i = 0
    while i < len(ordered_indices):
        idx = ordered_indices[i]
        if idx in consumed:
            i += 1
            continue
        if idx not in cluster_map:
            seg = dict(by_index[idx])
            seg["index"] = next_new_index
            next_new_index += 1
            rebuilt.append(seg)
            i += 1
            continue

        cluster = cluster_map[idx]
        members = [by_index[m] for m in cluster if m in by_index]
        for m in cluster:
            consumed.add(m)

        cluster_start = float(members[0].get("start") or 0.0)
        after_idx = None
        for candidate in ordered_indices:
            if candidate > cluster[-1]:
                after_idx = candidate
                break
        if after_idx is not None:
            budget_end = float(by_index[after_idx].get("start") or cluster_start)
        elif video_duration:
            budget_end = float(video_duration)
        else:
            budget_end = float(members[-1].get("end") or cluster_start) + 2.0
        cluster_span = max(1.0, budget_end - cluster_start)
        # Leave a small soft-drift buffer for the next anchor; keep most of the span usable.
        cluster_budget = max(0.8, cluster_span - min(0.35, HARD_MAX_DRIFT_SEC * 0.25))

        cn = "".join(str(m.get("text") or "") for m in members)
        vi = _join_vietnamese(
            [str(m.get("tts_spoken_text") or m.get("translation") or "") for m in members]
        )
        units_text = _pack_sentences_into_units(
            split_translation_sentences(vi) or ([vi] if vi else []),
            total_budget=cluster_budget,
        )
        if not units_text and vi:
            units_text = [vi]
        # Prefer a single fitted unit when the span is short.
        if cluster_budget <= PREFERRED_MERGED_MAX_SEC and len(units_text) > 1:
            units_text = [" ".join(units_text)]
        if not units_text:
            for member in members:
                seg = dict(member)
                seg["index"] = next_new_index
                next_new_index += 1
                rebuilt.append(seg)
            i += len(cluster)
            continue

        synthesized: list[dict[str, Any]] = []
        for unit_i, text in enumerate(units_text):
            out_path = tts_dir / f"tts_cluster_{cluster[0]}_{unit_i}.wav"
            repaired_path = tts_dir / f"tts_repaired_cluster_{cluster[0]}_{unit_i}.wav"
            out_path.unlink(missing_ok=True)
            repaired_path.unlink(missing_ok=True)
            try:
                session.synthesize(text, out_path, segment=members[0])
            except Exception:
                logger.exception("Cluster resynth failed cluster=%s unit=%s", cluster, unit_i)
                continue
            if not out_path.is_file():
                continue
            shutil.copyfile(out_path, repaired_path)
            duration = get_wav_duration(repaired_path)
            synthesized.append(
                {
                    "text": text,
                    "out_path": out_path,
                    "repaired_path": repaired_path,
                    "duration": duration,
                }
            )

        if not synthesized:
            for member in members:
                seg = dict(member)
                seg["index"] = next_new_index
                next_new_index += 1
                rebuilt.append(seg)
            i += len(cluster)
            continue

        # Keep raw cluster TTS durations. Propose needed speeds only; duration_repair
        # applies one uniform max later so reading pace stays even across the job.
        gaps = MIN_DIALOGUE_GAP_SEC * max(0, len(synthesized) - 1)
        usable = max(0.6, cluster_budget - gaps)
        total = sum(float(item["duration"]) for item in synthesized)
        if total > usable + 0.15:
            scale_alloc = usable / max(len(synthesized), 1)
            for item in synthesized:
                scratch = {"proposed_speed_factor": 1.0}
                _propose_cluster_speed(scratch, float(item["duration"]), max(0.4, scale_alloc))
                item["proposed_speed_factor"] = scratch["proposed_speed_factor"]
            total = sum(float(item["duration"]) for item in synthesized)
            if total > usable + 0.15:
                for item in synthesized:
                    share = usable * (float(item["duration"]) / max(total, 0.01))
                    scratch = {
                        "proposed_speed_factor": float(item.get("proposed_speed_factor") or 1.0)
                    }
                    _propose_cluster_speed(scratch, float(item["duration"]), max(0.35, share))
                    item["proposed_speed_factor"] = scratch["proposed_speed_factor"]

        # Place units with proportional Chinese windows across the cluster span.
        weights = [max(0.35, float(item["duration"])) for item in synthesized]
        weight_sum = sum(weights)
        cursor = cluster_start
        for unit_i, item in enumerate(synthesized):
            share = cluster_span * (weights[unit_i] / weight_sum)
            win_start = cursor
            win_end = budget_end if unit_i + 1 == len(synthesized) else min(budget_end, cursor + share)
            if win_end <= win_start + 0.2:
                win_end = min(budget_end, win_start + max(0.35, float(item["duration"])))
            duration = float(item["duration"])
            alloc = max(0.35, win_end - win_start)
            proposed = {"proposed_speed_factor": float(item.get("proposed_speed_factor") or 1.0)}
            if duration > alloc + 0.15:
                _propose_cluster_speed(proposed, duration, alloc)
            timing_status = "CLUSTER_REPAIRED"
            if duration > alloc + 0.15:
                timing_status = "UNRESOLVED_TIMING"
            rebuilt.append(
                {
                    "index": next_new_index,
                    "start": round(win_start, 2),
                    "end": round(win_end, 2),
                    "text": cn if unit_i == 0 else "",
                    "translation": item["text"],
                    "tts_spoken_text": item["text"],
                    "original_duration": round(alloc, 2),
                    "duration_budget": round(alloc, 2),
                    "tts_duration": round(duration, 2),
                    "repaired_duration": round(duration, 2),
                    "tts_path": str(item["repaired_path"]),
                    "tts_raw_path": str(item["out_path"]),
                    "repaired_method": "conflict_cluster_merge",
                    "placement_start": round(win_start, 3),
                    "preferred_placement_start": round(win_start, 3),
                    "timing_status": timing_status,
                    "proposed_speed_factor": proposed["proposed_speed_factor"],
                    "soft_speed_factor": 1.0,
                    "cluster_source_indices": list(cluster),
                }
            )
            next_new_index += 1
            cursor = win_end

        i += len(cluster)

    for new_i, seg in enumerate(rebuilt):
        seg["index"] = new_i
        # Canonicalize onto tts_repaired_{index}.wav so mix/API/align never prefer stale files.
        src = Path(str(seg.get("tts_path") or ""))
        canonical = tts_dir / f"tts_repaired_{new_i}.wav"
        if src.is_file():
            if src.resolve() != canonical.resolve():
                shutil.copyfile(src, canonical)
            seg["tts_path"] = str(canonical)
    compute_placement_starts(rebuilt)
    schedule_soft_placements(rebuilt)
    enforce_zero_overlap_placements(rebuilt)

    # Clear sticky UNRESOLVED when placement is healthy after reschedule.
    for seg in rebuilt:
        overflow = float(seg.get("timing_overflow_sec") or 0.0)
        drift = abs(float(seg.get("placement_drift_sec") or 0.0))
        if overflow <= 0.15 and drift <= HARD_MAX_DRIFT_SEC + 1e-6:
            if seg.get("timing_status") == "UNRESOLVED_TIMING":
                seg["timing_status"] = "SHIFTED" if drift > 0.02 else "CLUSTER_REPAIRED"

    overlaps = segments_with_voiced_overlap(rebuilt, margin_sec=BOUNDARY_MARGIN_SEC)
    overflow = sum(1 for s in rebuilt if float(s.get("timing_overflow_sec") or 0) > 0.15)
    logger.info(
        "Conflict cluster repair job=%s clusters=%s segments %s→%s overlaps=%s overflow=%s",
        job_id,
        len(clusters),
        len(segments),
        len(rebuilt),
        len(overlaps),
        overflow,
    )
    return rebuilt
