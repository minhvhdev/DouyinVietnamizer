# Evaluation and benchmark scripts

Offline tooling for timing A/B experiments, dubbing QC dashboards, and performance benchmarks.
Not used by the Tauri app or production pipeline.

Run from the `backend/` directory:

```bash
uv run python scripts/eval/<script>.py [args]
```

## Key scripts

| Script | Purpose |
|--------|---------|
| `run_timing_experiment.py` | A/B timing experiment orchestration |
| `evaluate_dubbing_timing.py` | Timing QC dashboard (JSON/HTML) |
| `evaluate_production_batch.py` | Batch release gate scoring |
| `recommend_timing_settings.py` | Suggest settings from experiment manifests |
| `benchmark_vad_asr_tts.py` | Dubbing optimization layer benchmark |
| `benchmark_omnivoice_steps.py` | OmniVoice step-level benchmark |
| `benchmark_omnivoice_queued_batch.py` | Queued vs sequential TTS batching |
| `audit_all_tts_fidelity.py` | TTS fidelity audit across jobs |
| `audit_translation_semantics.py` | Translation semantic safeguards audit |
| `analyze_job_segments.py` | ASR/subtitle segment analysis |
| `analyze_subtitle_asr.py` | Subtitle vs ASR comparison |
| `evaluate_voice_profile.py` | Voice profile convergence report |
| `report_soft_placement.py` | Soft placement drift report |

## Python modules

Evaluation logic lives in `dv_backend/eval/` (import as `dv_backend.eval.*`).

Production pipeline modules (`timing_qc_metrics`, `release_quality_gate`, `production_thresholds`) remain in `dv_backend/`.
