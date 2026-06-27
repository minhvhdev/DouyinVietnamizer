# VoxCPM2 TTS Pipeline

The VoxCPM2 backend runs as a **long-lived worker process** with an
on-disk result cache. This replaces OmniVoice as the only supported
TTS engine and dramatically reduces end-to-end dubbing time for
long videos.

## Architecture

```
backend (main process)                   .venv-voxcpm (worker process)
-------------------------                -----------------------------------
VoxCPMTtsAdapter                         dv_backend.adapters.voxcpm_worker
   |                                          |
   v                                          v
VoxCPMWorkerClient  ---stdin/stdout---> VoxCPMEngine (keeps model in VRAM)
   |                                          |
   v                                          v
VoxCPMCache (sha256 -> .wav)             inference per voice signature
```

### Key components

- `dv_backend/adapters/voxcpm_worker.py` — worker script that runs inside
  the isolated `.venv-voxcpm` virtualenv, loads the VoxCPM2 model once,
  and processes JSONL requests from stdin.
- `dv_backend/adapters/voxcpm_client.py` — backend-side client that
  spawns the worker, groups requests by `(model, device, ref_audio,
  ref_text, voice_design, num_step, cfg_value)`, and surfaces cancel
  via `JobRunner.register_process`.
- `dv_backend/adapters/voxcpm_cache.py` — content-addressed cache
  stored under `<data_dir>/cache/voxcpm/`. Disabled with
  `DV_VOXCPM_CACHE_DISABLED=1` or `voxcpm_cache_enabled=False` in
  settings.
- `dv_backend/adapters/tts.py` — `VoxCPMTtsAdapter` routes through
  the worker client and consults the cache first.

### Voice inputs

- `auto` (or empty) — use VoxCPM2's built-in default voice.
- `instruct:<description>` — emit a `(<description>)<text>` prefix
  that VoxCPM2 reads as a voice-design prompt. Example:
  `instruct:female, low pitch`.
- `/path/to/ref.wav` — voice clone using the supplied reference audio
  (and the segment's source text as `prompt_text`).

## Performance

VoxCPM2's `generate()` is single-text; we call it per request so the
model stays hot in VRAM across segments. Repeated work (resume after
crash, dubbing the same video twice) becomes instant thanks to the
on-disk cache.

## Settings

| Setting                       | Default                | Description                                              |
| ----------------------------- | ---------------------- | -------------------------------------------------------- |
| `voxcpm_model`                | `openbmb/VoxCPM2`      | Hugging Face model id (or local path).                   |
| `voxcpm_device`               | `cuda:0`               | Torch device (e.g. `cpu` for the smoke test).            |
| `voxcpm_ref_audio`            | empty                  | Path to a `.wav` for voice cloning.                      |
| `voxcpm_instruct`             | empty                  | Free-form voice description (e.g. `female, low pitch`).  |
| `voxcpm_auto_voice`           | true                   | Allow the engine to pick a voice when no ref is given.   |
| `voxcpm_num_steps`            | 10                     | Diffusion steps. **Lower = faster but lower quality.**   |
| `voxcpm_cache_enabled`         | true                   | Disable to skip the on-disk result cache.                |

## Cancellation

When the user cancels a job the `JobRunner` kills the worker Popen.
The reader thread detects the broken pipe, fails any in-flight
requests, and the next call transparently respawns the worker. No
adapter code change is required to take advantage of this.

## Smoke test

`backend/scripts/run_voxcpm_smoke.py` exercises the full TTS
pipeline end-to-end with VoxCPM2. Use it after upgrading the worker
script to confirm nothing regressed:

```powershell
cd backend
python scripts/run_voxcpm_smoke.py
```

## Troubleshooting

- **`VOXCPM_TTS_FAILED` with `code = VOXCPM_WORKER_DIED`** —
  the worker crashed (typically OOM). Reduce `voxcpm_num_steps` or
  segment length to lower peak VRAM usage.
- **`VOXCPM_NOT_INSTALLED`** — the isolated `.venv-voxcpm` is missing.
  Run `python scripts/setup_voxcpm.py` in the backend folder.
- **Slow first segment, fast subsequent ones** — expected; the worker
  is loading the model on the first request. Cache hits bypass the
  worker entirely.
