# OmniVoice TTS Pipeline

The OmniVoice backend now runs as a **long-lived worker process** with
batched GPU inference and an on-disk result cache. This replaces the
previous one-subprocess-per-segment design and dramatically reduces
end-to-end dubbing time for long videos.

## Architecture

```
backend (main process)                   .venv-omnivoice (worker process)
-------------------------                -----------------------------------
OmniVoiceTtsAdapter                      dv_backend.adapters.omnivoice_worker
   |                                          |
   v                                          v
OmniVoiceWorkerClient  ---stdin/stdout---> OmniVoiceEngine (keeps model in VRAM)
   |                                          |
   v                                          v
OmniVoiceCache (sha256 -> .wav)          batched inference per voice signature
```

### Key components

- `dv_backend/adapters/omnivoice_worker.py` — worker script that runs inside
  the isolated `.venv-omnivoice` virtualenv, loads the OmniVoice model
  once, and processes JSONL requests from stdin.
- `dv_backend/adapters/omnivoice_client.py` — backend-side client that
  spawns the worker, batches requests by `(model, device, ref_audio,
  ref_text, instruct, num_step)`, and surfaces cancel via
  `JobRunner.register_process`.
- `dv_backend/adapters/omnivoice_cache.py` — content-addressed cache
  stored under `<data_dir>/cache/omnivoice/`. Disabled with
  `DV_OMNIVOICE_CACHE_DISABLED=1` or `omnivoice_cache_enabled=False` in
  settings.
- `dv_backend/adapters/tts.py` — `OmniVoiceTtsAdapter` no longer spawns
  `python -m omnivoice.cli.infer` per call. It routes through the worker
  client and consults the cache first.

## Performance

For a 60-segment Vietnamese dubbing job on a single GPU the
single-subprocess-per-segment design paid the model load + Python
interpreter startup cost for every segment (often 5-15 s each). With
the new design:

- The OmniVoice model is loaded once and stays resident in VRAM.
- Requests with the same voice signature are batched together
  (`--max-batch`, default 4) and flushed every `--flush-ms` ms
  (default 150 ms) so a single diffusion call amortises across many
  segments.
- Repeated work (resume after crash, dubbing the same video twice)
  becomes instant thanks to the on-disk cache.

The exact speedup depends on VRAM size, segment length, and voice
variety, but a typical 60-segment job should drop from several minutes
to under a minute on the same hardware.

## Settings

| Setting                        | Default | Description                                            |
| ------------------------------ | ------- | ------------------------------------------------------ |
| `omnivoice_batch_size`         | 4       | Maximum segments per batched inference call.          |
| `omnivoice_batch_flush_ms`     | 150     | Flush window for incomplete batches.                   |
| `omnivoice_cache_enabled`      | true    | Disable to skip the on-disk result cache.              |
| `omnivoice_num_steps`          | 32      | Diffusion steps. **Lower = faster but lower quality.** |

## Cancellation

When the user cancels a job the `JobRunner` kills the worker Popen.
The reader thread detects the broken pipe, fails any in-flight
requests, and the next call transparently respawns the worker. No
adapter code change is required to take advantage of this.

## Smoke test

`backend/scripts/run_omnivoice_smoke.py` exercises the full TTS
pipeline end-to-end with OmniVoice. Use it after upgrading the
worker script to confirm nothing regressed:

```powershell
cd backend
python scripts/run_omnivoice_smoke.py
```

## Troubleshooting

- **`OMNIVOICE_TTS_FAILED` with `code = OMNIVOICE_WORKER_DIED`** —
  the worker crashed (typically OOM). Lower `omnivoice_batch_size` to
  reduce peak VRAM usage.
- **Slow first segment, fast subsequent ones** — expected; the worker
  is loading the model on the first request. Cache hits bypass the
  worker entirely.
- **No batching benefit** — every segment uses a different voice
  (e.g. a per-speaker reference audio). The cache still helps when
  segments repeat, but the per-batch win is limited.
