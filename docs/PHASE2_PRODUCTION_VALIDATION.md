# Phase 2 Production Validation Protocol

See also Phase 3 tooling: `run_timing_experiment.py`, `preflight_release.py`, production dashboard via `evaluate_dubbing_timing.py --compare --include-audio` (all under `scripts/eval/`).

This document defines how to run controlled A/B validation for timing-aware translation and natural TTS duration control in DouyinVietnamizer.

## Why Phase 2 defaults are conservative

- `timing_candidate_translation_enabled = false` by default because multi-candidate translation requires Gemini or OpenAI keys and adds cloud cost per job.
- `voice_duration_profile_enabled = true` by default because it learns locally from accepted raw TTS and does not add cloud calls.
- Legacy jobs without timing profiles or candidates remain compatible: existing translation is treated as a single natural candidate.

## Baseline A (control)

```json
{
  "timing_candidate_translation_enabled": false,
  "voice_duration_profile_enabled": false
}
```

## Phase 2 B (experiment)

```json
{
  "timing_candidate_translation_enabled": true,
  "timing_translation_candidate_count": 3,
  "timing_max_candidate_tts_attempts": 2,
  "timing_max_tts_attempts": 3,
  "voice_duration_profile_enabled": true
}
```

Keep constant across A and B:

- Translation backend
- TTS backend and voice/reference audio
- Global TTS speed
- ASR/VAD settings
- Mix/render settings

## Test matrix

| Video | Focus |
| --- | --- |
| A — fast continuous speech | compact candidates, overlap risk |
| B — many pauses | speech envelope, no unnecessary lengthen |
| C — numbers/proper nouns | semantic safeguards |
| D — long translations | retry/rewrite/light stretch |
| E — very short lines | tolerance and subtitle timing |

Run each video twice (baseline job + Phase 2 job). Do not overwrite artifacts between runs.

## Evaluation commands

```bash
cd backend
uv run python scripts/eval/evaluate_dubbing_timing.py <baseline_job_id> --json
uv run python scripts/eval/evaluate_dubbing_timing.py <phase2_job_id> --compare <baseline_job_id> --export-html
```

## Initial acceptance thresholds (targets, not guarantees)

| Metric | Target |
| --- | --- |
| speech_trim_count | 0 |
| danger_stretch_rate | 0 |
| warning_stretch_rate | < 10% |
| median effective tempo | 0.95–1.07 |
| P90 effective tempo | 0.90–1.12 |
| first_attempt_acceptance_rate | > 70% |
| rewrite_rate | < 15% |
| candidate_retry_rate | < 30% |
| median prediction error | < 300 ms |
| P90 prediction error | < 700 ms |
| semantic safeguard critical violations | 0 |
| subtitle overlap count | 0 |

If a threshold is missed, inspect per-segment rows in `timing_eval.html` and segment checkpoints (`translate`, `tts`, `duration_repair`).

## Settings that must be enabled together

- Multi-candidate translation: `translation_backend` must be `gemini` or `openai` with valid keys.
- Candidate TTS retry: requires `timing_candidate_translation_enabled = true` and more than one candidate per segment.
- Voice profile learning: requires `voice_duration_profile_enabled = true` and accepted raw TTS (not repaired/stretch/rejected samples).

## Cloud cost worst case (per segment)

- Translation candidates: 1 LLM call per batch (not per segment) when batch architecture is used.
- TTS syntheses: capped by `timing_max_tts_attempts` including duration repair resynthesis.
- Rewrite: capped by `timing_max_llm_rewrite_attempts` (default 1).

Telemetry fields: `candidate_api_call_count`, `rewrite_api_call_count`, `tts_synthesis_call_count`, `cache_avoided_tts_calls`.

## Checkpoint invalidation

| Rerun from | Invalidates |
| --- | --- |
| translate | tts, duration_repair, align_final_dub, mix, render, qc |
| tts | duration_repair, align_final_dub, mix, render, qc |
| duration_repair | align_final_dub, mix, render, qc |
| align_final_dub | mix, render, qc |

## Parts requiring cloud/GPU for full verification

- Gemini/OpenAI candidate generation and rewrite
- OmniVoice TTS synthesis
- CUDA/MPS alignment when enabled in settings

Unit tests cover deterministic ranking, semantic safeguards, attempt budget, cache identity, and timing profile math without cloud access.
