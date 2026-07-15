# Archived scripts

One-off diagnostic, P0 incident, and ad-hoc debug scripts moved here during repo cleanup.
They are kept for historical reference but are not part of the supported operator workflow.

Run from the `backend/` directory:

```bash
uv run python scripts/archive/<script>.py [args]
```

## Contents

| Script | Purpose |
|--------|---------|
| `diagnose_omnivoice_*.py` | Controlled OmniVoice clone/failure experiments |
| `debug_gemini.py` | One-off Gemini API debugger |
| `debug_subtitle_job.py` | Subtitle timing inspection for a job |
| `test_omnivoice_reftrim*.py` | Ref-text + trim experiments |
| `p0_*.py`, `targeted_p0_repair.py`, `rollback_narrow_p0.py` | Narrow P0 job recovery (hardcoded job IDs) |
| `_inspect_user_segments.py`, `show_problem_segments.py`, `inspect_segment_cuts.py` | Segment inspection helpers |
| `dry_run_pause_split.py` | Pause-split dry run on a fixed job |
| `measure_tts_spill.py`, `measure_vad_false_positives.py` | Timing/VAD measurement utilities |
| `sweep_omnivoice_ref_prefix.py` | Ref-prefix sweep experiment |

## Active scripts (parent `scripts/`)

Production and operator tools remain in `backend/scripts/`, including:

- `setup_omnivoice.py`, `smoke_omnivoice_mps.py`
- `resume_job.py`, `preflight_release.py`, `verify_omnivoice_tts.py`
- `benchmark_*.py`, `evaluate_*.py`, `audit_*.py` → see `scripts/eval/`
