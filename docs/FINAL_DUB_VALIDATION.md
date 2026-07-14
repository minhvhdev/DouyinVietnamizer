# Final Dub Alignment — Production Validation

This guide validates Phase 1 `align_final_dub` on real jobs without the desktop UI.

## Quick validation (no GPU)

Uses existing checkpoints/cache only:

```bash
cd backend
uv run python scripts/validate_final_dub_alignment.py <job_id> --no-model --export-html
```

Outputs:

- `jobs/<job_id>/artifacts/final_dub_validation.json`
- `jobs/<job_id>/artifacts/final_dub_validation.html` (with `--export-html`)

## Force realignment (GPU required)

```bash
uv run python scripts/validate_final_dub_alignment.py <job_id> --force-realign --export-html
```

## Rerun pipeline from alignment

```bash
uv run python scripts/rerun_from_duration_repair.py <job_id> --from-step align_final_dub
```

## Smoke clip types

### Clip A — continuous speech (2–5s)

Check:

- `dub_words` monotonic inside `repaired_duration`
- Subtitle cues track word clusters
- `absolute_start = placement_start + start`

### Clip B — pauses inside sentence

Check:

- Multiple subtitle cues per segment
- Pause split threshold (~300ms) creates separate cues

### Clip C — stretched TTS + shifted placement

Check:

- `placement_start != source start`
- Subtitle timing follows repaired audio, not source ASR
- Validator `absolute_timeline_valid=true`

## Per-segment fields to inspect

```json
{
  "dub_alignment_status": "aligned | fallback_interpolated | no_speech | failed | skipped",
  "dub_alignment_method": "qwen_forced_aligner_words | qwen_asr_segment_mapping | weighted_interpolation:...",
  "dub_words": [{"text": "...", "start": 0.1, "end": 0.3, "absolute_start": 31.57, "absolute_end": 31.77}],
  "subtitle_cue_count": 2
}
```

## Cache behavior

Alignment cache stores **relative** word timestamps only.

Changing `placement_start` after cache hit must **not** rerun Qwen; absolute timestamps are recomputed at load/render time.

Cache identity includes:

- SHA-256 of repaired WAV bytes
- target text
- target language
- ASR + aligner model ids
- cache + token normalization version
