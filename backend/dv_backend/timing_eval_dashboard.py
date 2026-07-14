"""Production HTML dashboard for timing evaluation and A/B review."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .dubbing_quality_score import score_segment_quality
from .experiment_comparability import validate_experiment_comparability
from .release_quality_gate import evaluate_release_gate
from .timing_qc_metrics import compare_timing_metrics

METRIC_VERDICT: dict[str, str] = {
    "speech_trim_count": "lower_better",
    "danger_stretch_count": "lower_better",
    "warning_stretch_count": "lower_better",
    "first_attempt_acceptance_rate": "higher_better",
    "candidate_retry_rate": "lower_better",
    "rewrite_rate": "lower_better",
    "mean_prediction_error_ms": "lower_better",
    "median_prediction_error_ms": "lower_better",
    "p90_prediction_error_ms": "lower_better",
    "alignment_fallback_count": "lower_better",
    "subtitle_overlap_count": "lower_better",
    "tts_synthesis_call_count": "lower_better",
    "candidate_api_call_count": "lower_better",
}


def _verdict(metric: str, delta: float | None) -> str:
    if delta is None or delta == 0:
        return "neutral"
    direction = METRIC_VERDICT.get(metric, "neutral")
    if direction == "lower_better":
        return "better" if delta < 0 else "worse"
    if direction == "higher_better":
        return "better" if delta > 0 else "worse"
    return "neutral"


def enrich_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        base = row.get("baseline")
        exp = row.get("phase2")
        delta = row.get("delta")
        pct = None
        if isinstance(base, (int, float)) and base not in (0, 0.0) and isinstance(delta, (int, float)):
            pct = round(float(delta) / float(base) * 100.0, 2)
        enriched.append(
            {
                **row,
                "percent_delta": pct,
                "verdict": _verdict(str(row.get("metric")), delta if isinstance(delta, (int, float)) else None),
            }
        )
    return enriched


def build_dashboard_payload(
    data_dir: Path,
    experiment_job_id: str,
    *,
    baseline_job_id: str | None = None,
    baseline_segments: list[dict] | None = None,
    experiment_segments: list[dict] | None = None,
    baseline_summary: dict | None = None,
    experiment_summary: dict | None = None,
    baseline_settings: dict | None = None,
    experiment_settings: dict | None = None,
    experiment_id: str | None = None,
    include_audio: bool = False,
    ffmpeg_path: str = "ffmpeg",
) -> dict[str, Any]:
    from .checkpoints import load_checkpoint
    from .evaluation_audio import export_evaluation_audio

    exp_segments = experiment_segments
    if exp_segments is None:
        for step in ("duration_repair", "tts", "translate"):
            cp = load_checkpoint(data_dir, experiment_job_id, step)
            if cp and cp.get("segments"):
                exp_segments = list(cp["segments"])
                break
    exp_segments = exp_segments or []

    base_segments = baseline_segments
    base_summary = baseline_summary
    if baseline_job_id:
        if base_segments is None:
            for step in ("duration_repair", "tts", "translate"):
                cp = load_checkpoint(data_dir, baseline_job_id, step)
                if cp and cp.get("segments"):
                    base_segments = list(cp["segments"])
                    break
        base_segments = base_segments or []

    from .timing_qc_metrics import compute_timing_qc_metrics

    exp_summary = experiment_summary or compute_timing_qc_metrics(exp_segments)
    base_summary = base_summary or (compute_timing_qc_metrics(base_segments) if base_segments else {})

    comparison = None
    comparability = None
    if baseline_job_id and baseline_settings and experiment_settings:
        comparability = validate_experiment_comparability(
            data_dir,
            baseline_job_id,
            experiment_job_id,
            baseline_settings=baseline_settings,
            experiment_settings=experiment_settings,
        )
    if base_summary:
        comparison = enrich_comparison_rows(compare_timing_metrics(base_summary, exp_summary))

    audio_exports: dict[str, Any] = {}
    if include_audio and baseline_job_id:
        audio_exports["baseline"] = export_evaluation_audio(
            data_dir, baseline_job_id, base_segments, ffmpeg_path=ffmpeg_path, label="baseline"
        )
        audio_exports["experiment"] = export_evaluation_audio(
            data_dir, experiment_job_id, exp_segments, ffmpeg_path=ffmpeg_path, label="experiment"
        )

    base_by_index = {int(s.get("index", i)): s for i, s in enumerate(base_segments or [])}
    per_segment = []
    for segment in exp_segments:
        idx = int(segment.get("index", 0))
        base = base_by_index.get(idx, {})
        quality = score_segment_quality(segment)
        row = {
            "index": idx,
            "source_start": segment.get("start"),
            "source_end": segment.get("end"),
            "source_text": segment.get("text"),
            "baseline_translation": base.get("translation"),
            "experiment_translation": segment.get("translation"),
            "selected_candidate_style": segment.get("selected_candidate_style"),
            "semantic_warnings": [
                p for r in (segment.get("candidate_rankings") or []) for p in (r.get("penalties") or [])
            ],
            "predicted_duration": segment.get("predicted_duration"),
            "actual_speech_duration": segment.get("tts_speech_duration") or segment.get("tts_duration"),
            "baseline_tempo": base.get("automatic_tempo_factor") or base.get("time_stretch_factor"),
            "experiment_tempo": segment.get("automatic_tempo_factor") or segment.get("time_stretch_factor"),
            "placement_shift": segment.get("placement_start"),
            "repair_method": segment.get("repaired_method"),
            "alignment_status": segment.get("dub_alignment_status"),
            "subtitle_cue_count": len(segment.get("subtitle_cues") or []),
            "quality_severity": quality["quality_severity"],
            "quality_score": quality["quality_score"],
        }
        if include_audio:
            row["audio_baseline"] = f"evaluation_audio/baseline/segment_{idx:03d}_baseline.wav"
            row["audio_experiment"] = f"evaluation_audio/experiment/segment_{idx:03d}_experiment.wav"
            row["audio_source"] = f"evaluation_audio/experiment/segment_{idx:03d}_source.wav"
        per_segment.append(row)

    gate = evaluate_release_gate(exp_segments, metrics=exp_summary, comparison=comparability)

    return {
        "experiment_id": experiment_id,
        "baseline_job_id": baseline_job_id,
        "experiment_job_id": experiment_job_id,
        "summary": exp_summary,
        "baseline_summary": base_summary,
        "comparison": comparison,
        "comparability": comparability,
        "release_gate": gate,
        "segments": per_segment,
        "audio_exports": audio_exports,
    }


REVIEW_JS = """
const STORAGE_KEY = 'dv_timing_review_' + (window.EXPERIMENT_ID || 'default');
function loadReview() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}'); } catch { return {}; }
}
function saveReviewField(index, field, value) {
  const data = loadReview();
  data.segments = data.segments || {};
  data.segments[index] = data.segments[index] || {};
  data.segments[index][field] = value;
  data.experiment_id = window.EXPERIMENT_ID;
  data.updated_at = new Date().toISOString();
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
}
function exportReviewJson() {
  const blob = new Blob([JSON.stringify(loadReview(), null, 2)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = (window.EXPERIMENT_ID || 'timing_review') + '.json';
  a.click();
}
document.querySelectorAll('[data-review-field]').forEach(el => {
  el.addEventListener('change', () => {
    saveReviewField(el.dataset.index, el.dataset.reviewField, el.type === 'checkbox' ? el.checked : el.value);
  });
});
"""


def export_dashboard_html(path: Path, payload: dict[str, Any]) -> None:
    exp_id = payload.get("experiment_id") or payload.get("experiment_job_id") or "timing"
    comparability = payload.get("comparability") or {}
    comparison = payload.get("comparison") or []
    segments = payload.get("segments") or []

    compare_rows = "".join(
        f"<tr><td>{html.escape(str(r.get('metric')))}</td>"
        f"<td>{r.get('baseline')}</td><td>{r.get('phase2')}</td>"
        f"<td>{r.get('delta')}</td><td>{r.get('percent_delta')}</td>"
        f"<td>{r.get('verdict')}</td></tr>"
        for r in comparison
    )

    seg_rows = []
    for s in segments:
        idx = s.get("index")
        audio_cells = ""
        if s.get("audio_source"):
            audio_cells = (
                f"<td><audio controls src='{html.escape(s['audio_source'])}'></audio></td>"
                f"<td><audio controls src='{html.escape(s.get('audio_baseline',''))}'></audio></td>"
                f"<td><audio controls src='{html.escape(s.get('audio_experiment',''))}'></audio></td>"
            )
        else:
            audio_cells = "<td></td><td></td><td></td>"
        review_cells = (
            f"<td><input data-index='{idx}' data-review-field='naturalness' type='number' min='1' max='5'></td>"
            f"<td><input data-index='{idx}' data-review-field='timing' type='number' min='1' max='5'></td>"
            f"<td><input data-index='{idx}' data-review-field='meaning_preserved' type='checkbox'></td>"
            f"<td><input data-index='{idx}' data-review-field='subtitle_sync' type='number' min='1' max='5'></td>"
            f"<td><select data-index='{idx}' data-review-field='preferred'>"
            f"<option value=''>-</option><option>baseline</option><option>experiment</option><option>tie</option></select></td>"
            f"<td><input data-index='{idx}' data-review-field='notes' type='text' style='width:180px'></td>"
        )
        seg_rows.append(
            f"<tr><td>{idx}</td><td>{html.escape(str(s.get('source_text') or '')[:80])}</td>"
            f"<td>{html.escape(str(s.get('baseline_translation') or '')[:60])}</td>"
            f"<td>{html.escape(str(s.get('experiment_translation') or '')[:60])}</td>"
            f"<td>{s.get('quality_severity')}</td><td>{s.get('quality_score')}</td>"
            f"{audio_cells}{review_cells}</tr>"
        )

    html_doc = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Timing Production Dashboard</title>
<style>body{{font-family:system-ui;margin:20px}} table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #ccc;padding:6px;font-size:13px}}</style>
<script>window.EXPERIMENT_ID={json.dumps(exp_id)};</script></head><body>
<h1>Timing Production Dashboard</h1>
<p><b>Baseline job:</b> {html.escape(str(payload.get('baseline_job_id') or ''))}<br>
<b>Experiment job:</b> {html.escape(str(payload.get('experiment_job_id') or ''))}<br>
<b>Comparison valid:</b> {comparability.get('comparison_valid')}</p>
<h2>Summary</h2><pre>{html.escape(json.dumps(payload.get('summary') or {{}}, ensure_ascii=False, indent=2))}</pre>
<h2>Release gate</h2><pre>{html.escape(json.dumps(payload.get('release_gate') or {{}}, ensure_ascii=False, indent=2))}</pre>
<h2>Delta table</h2>
<table><tr><th>Metric</th><th>Baseline</th><th>Experiment</th><th>Delta</th><th>%</th><th>Verdict</th></tr>{compare_rows or '<tr><td colspan=6>No comparison</td></tr>'}</table>
<h2>Per-segment review <button onclick='exportReviewJson()'>Export review JSON</button></h2>
<table><tr><th>Idx</th><th>Source</th><th>Baseline tr</th><th>Experiment tr</th><th>QC</th><th>Score</th>
<th>Source audio</th><th>Baseline audio</th><th>Experiment audio</th>
<th>Natural 1-5</th><th>Timing 1-5</th><th>Meaning</th><th>Subtitle 1-5</th><th>Preferred</th><th>Notes</th></tr>
{''.join(seg_rows)}</table>
<script>{REVIEW_JS}</script>
</body></html>"""
    path.write_text(html_doc, encoding="utf-8")
