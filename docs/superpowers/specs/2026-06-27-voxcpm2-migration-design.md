# VoxCPM2 TTS Migration Design

**Date:** 2026-06-27
**Status:** Draft (pending user review)
**Scope:** Replace OmniVoice TTS engine with VoxCPM2 across the DouyinVietnamizer stack.

## 1. Goals & non-goals

**Goals**
- Replace OmniVoice with VoxCPM2 (`openbmb/VoxCPM2`) as the sole TTS engine.
- Preserve the long-lived worker + on-disk cache architecture (model stays in VRAM between segments, repeated jobs hit cache).
- Preserve all public surface area: `pipeline.py`, `api.py`, settings keys, frontend bindings, error codes — only the underlying engine changes.
- Keep voice cloning (ref audio) and voice design (`(description)text`) features working.
- Drop all legacy OmniVoice files; the old code paths become unreachable.

**Non-goals**
- Keep OmniVoice as a selectable backend. `SUPPORTED_TTS_BACKENDS = ("voxcpm",)` — single backend.
- Migrate user settings automatically. Old `omnivoice_*` keys in SQLite are ignored; users reconfigure via UI.
- Add streaming TTS output to the HTTP API. The existing segment-by-segment WAV pipeline is preserved.
- Switch ASR, translation, separation, or mix steps.

## 2. Architecture

```
backend (main process)                   .venv-voxcpm (worker process)
-------------------------                -----------------------------------
VoxCPMTtsAdapter                         dv_backend.adapters.voxcpm_worker
   |                                          |
   v                                          v
VoxCPMWorkerClient  ---stdin/stdout---> VoxCPMEngine (keeps model in VRAM)
   |                                          |
   v                                          v
VoxCPMCache (sha256 -> .wav)             per-batch inference via VoxCPM2.generate()
```

Same shape as the OmniVoice pipeline (see `docs/OMNIVOICE.md` for the pattern that VoxCPM.md replaces). The only architectural difference is inside the worker: VoxCPM2's `generate()` is called per-segment, not via a diffusion loop with `num_step`.

### Voice flow

1. UI binds `voxcpm_ref_audio` (a .wav path or empty for auto) and `voxcpm_instruct` (free-form voice description, e.g. `female, low pitch`).
2. `pipeline._default_tts_voice()` returns `instruct:...` or the ref-audio path or `auto`.
3. `VoxCPMTtsAdapter.synthesize()` parses the voice string into `(prompt_wav_path, prompt_text, voice_design)`; if `voice_design` is set, the adapter prefixes the text with `(voice_design)text` before sending to the worker.
4. Worker calls `model.generate(text=..., prompt_wav_path=..., prompt_text=...)` and writes a 16-bit mono WAV at the model's reported sample rate.
5. Adapter materialises the result to the segment output path; on cache hit the worker is bypassed entirely.

## 3. Renaming map (settings, env vars, errors, files)

### Settings keys (DB column `settings.key`)

| Old | New | Default |
| --- | --- | --- |
| `omnivoice_model` | `voxcpm_model` | `openbmb/VoxCPM2` |
| `omnivoice_device` | `voxcpm_device` | `cuda:0` |
| `omnivoice_ref_audio` | `voxcpm_ref_audio` | `""` |
| `omnivoice_instruct` | `voxcpm_instruct` | `""` |
| `omnivoice_auto_voice` | `voxcpm_auto_voice` | `True` |
| `omnivoice_num_steps` | `voxcpm_num_steps` | `10` (was 32 — VoxCPM2 default timesteps is 10) |
| `omnivoice_batch_size` | `voxcpm_batch_size` | `4` |
| `omnivoice_batch_flush_ms` | `voxcpm_batch_flush_ms` | `150` |
| `omnivoice_cache_enabled` | `voxcpm_cache_enabled` | `True` |

### Environment variables

| Old | New |
| --- | --- |
| `DV_OMNIVOICE_VENV` | `DV_VOXCPM_VENV` |
| `DV_OMNIVOICE_PYTHON` | `DV_VOXCPM_PYTHON` |
| `DV_OMNIVOICE_CACHE_DISABLED` | `DV_VOXCPM_CACHE_DISABLED` |
| `DV_OMNIVOICE_CACHE_DIR` | `DV_VOXCPM_CACHE_DIR` |

### Error codes

| Old | New |
| --- | --- |
| `OMNIVOICE_NOT_INSTALLED` | `VOXCPM_NOT_INSTALLED` |
| `OMNIVOICE_TTS_FAILED` | `VOXCPM_TTS_FAILED` |
| `OMNIVOICE_TIMEOUT` | `VOXCPM_TIMEOUT` |
| `OMNIVOICE_WORKER_DIED` | `VOXCPM_WORKER_DIED` |
| `OMNIVOICE_INFERENCE_FAILED` | `VOXCPM_INFERENCE_FAILED` |
| `OMNIVOICE_WRITE_FAILED` | `VOXCPM_WRITE_FAILED` |
| `OMNIVOICE_SYNTHESIZE_FAILED` | `VOXCPM_SYNTHESIZE_FAILED` |
| `OMNIVOICE_GPU_OOM` | `VOXCPM_GPU_OOM` |

User-facing labels switch from "OmniVoice" to "VoxCPM2".

### Files

| Old path | New path | Action |
| --- | --- | --- |
| `backend/dv_backend/omnivoice_env.py` | `backend/dv_backend/voxcpm_env.py` | Rename; rewrite body for `voxcpm` package. |
| `backend/dv_backend/adapters/omnivoice_client.py` | `backend/dv_backend/adapters/voxcpm_client.py` | Rename; rewrite body to call new worker script. |
| `backend/dv_backend/adapters/omnivoice_worker.py` | `backend/dv_backend/adapters/voxcpm_worker.py` | Rename; rewrite `_generate()` to use `VoxCPM.generate()`. |
| `backend/dv_backend/adapters/omnivoice_cache.py` | `backend/dv_backend/adapters/voxcpm_cache.py` | Rename; bump `VOXCPM_CACHE_VERSION = "v1"` to invalidate old entries. |
| `backend/dv_backend/adapters/tts.py` | (same) | Replace `OmniVoiceTtsAdapter` with `VoxCPMTtsAdapter`; update `SUPPORTED_TTS_BACKENDS`, `OMNIVOICE_*` constants, `parse_omnivoice_voice`, `_is_omnivoice_voice_clone`, `create_tts_adapter`. |
| `backend/dv_backend/runtime.py` | (same) | Replace `_check_omnivoice` with `_check_voxcpm`. |
| `backend/dv_backend/api.py` | (same) | Replace all `omnivoice_*` refs with `voxcpm_*`; update `output_suffix` to `"voxcpm"`. |
| `backend/dv_backend/pipeline.py` | (same) | Replace `omnivoice_*` keys in `_default_tts_voice()` and `device` lookup. |
| `backend/dv_backend/settings.py` | (same) | Replace `DEFAULT_SETTINGS` keys; rename `OMNIVOICE_DEFAULT_MODEL` → `VOXCPM_DEFAULT_MODEL`. |
| `backend/scripts/setup_omnivoice.py` | `backend/scripts/setup_voxcpm.py` | Rename; install `voxcpm` package instead of `omnivoice`. |
| `backend/scripts/run_omnivoice_smoke.py` | `backend/scripts/run_voxcpm_smoke.py` | Rename; switch to `voxcpm_*` env. |
| `backend/tests/test_omnivoice_tts.py` | `backend/tests/test_voxcpm_tts.py` | Rename; rewrite worker-API regression test. |
| `backend/tests/test_settings.py` | (same) | Update assertions to use `voxcpm_*` keys. |
| `frontend/src/renderer/App.tsx` | (same) | Rename card title; rebind settings keys. |
| `frontend/tests/App.test.tsx` | (same) | Update `tts_backend: "voxcpm"` and card title. |
| `docs/OMNIVOICE.md` | `docs/VOXCPM.md` | Rewrite body; keep same section structure. |
| `docs/DIARIZATION.md` | (same) | One-line mention updated. |
| `README.md` | (same) | Update all OmniVoice references. |

### Files deleted (no replacement)

- `backend/.venv-omnivoice/` (user may delete manually to free disk; the migration does not touch it).
- `<data_dir>/cache/omnivoice/` (old cache entries become orphaned; user may delete manually).

## 4. Component design

### 4.1 Worker (`voxcpm_worker.py`)

Lazy model load on first request. On `op="synthesize"`:

```python
engine_obj = engine.get(model=model, device=device)
# _run_batch processes one request at a time (VoxCPM2 generate is not natively multi-text)
audio = engine_obj.generate(
    text=request["text"],
    prompt_wav_path=request.get("prompt_wav_path"),
    prompt_text=request.get("prompt_text"),
    cfg_value=request.get("cfg_value", 2.0),
    inference_timesteps=int(request.get("inference_timesteps", 10)),
)
sample_rate = engine_obj.tts_model.sample_rate
duration = _write_wav(request["output_path"], audio, sample_rate)
```

`_write_wav()` writes a 16-bit mono PCM WAV at the model's actual sample rate (not the hardcoded 24000 used by OmniVoice).

Batching: requests sharing `(model, device, prompt_wav_path, prompt_text, voice_design_prefix, inference_timesteps, cfg_value)` are coalesced. The worker iterates each request and emits a response per request, with timing logged.

Worker ops:
- `synthesize` — generate audio and write WAV.
- `ping` — return `{"ok": True, "pong": True}` for liveness.
- `shutdown` — exit cleanly.
- Unknown op → `{"ok": False, "code": "UNKNOWN_OP", ...}`.

Health check (`--health-check` flag): imports `voxcpm` and exits 0 / 1, used by the runtime smoke test.

### 4.2 Client (`voxcpm_client.py`)

Structural clone of `OmniVoiceWorkerClient`:

- `WORKER_SCRIPT = "dv_backend.adapters.voxcpm_worker"`
- `acquire_client(*, data_dir, model, device, num_steps)` keys clients by `(model, device, num_steps)` and reuses one worker per key.
- `synthesize(*, text, output_path, prompt_wav_path, prompt_text, voice_design, cfg_value, inference_timesteps, cache_key)` writes a JSONL request, awaits a single response from the correlation queue.
- `register_with_runner(runner)` registers the Popen with `JobRunner` so cancellation kills the worker; the next call respawns.
- `_keep_alive()` pings the worker every 60 s when idle.
- Idle shutdown timer set by `--idle-timeout-sec` (default 0 = no auto-shutdown).
- `close()` and `release_all_clients()` mirror the OmniVoice version.

### 4.3 Adapter (`tts.py`)

`VoxCPMTtsAdapter`:

- `__init__(*, model, device, num_steps, data_dir, runner, max_batch, flush_ms, enable_cache, _client, _cache)` — same test seams as `OmniVoiceTtsAdapter`.
- `synthesize(text, output_path, *, voice, ref_text=None)`:
  1. Sanitise and chunk text with `split_tts_text` (unchanged helper).
  2. For each chunk call `_synthesize_single(chunk, ...)`.
  3. If multiple chunks, concat WAVs via `_concat_wav_files` (unchanged).
  4. Clean up `.part*.wav` artefacts in `finally`.
- `_synthesize_single`:
  1. `parse_voxcpm_voice(voice)` returns `(prompt_wav_path, prompt_text, voice_design)`.
  2. If `voice_design`, build effective text `f"({voice_design}){text}"`.
  3. Check cache: `cache_key(voice_id, text, model, num_step, voice_design)`. If hit, `materialize` to output_path and return.
  4. Otherwise call `client.synthesize(...)`. On success, `cache.put()` the result.

`parse_voxcpm_voice(voice)`:
- `"auto"` / `""` → `(None, None, None)`.
- `"instruct:<desc>"` → `(None, None, "<desc>")`.
- Path to existing `.wav` → `(str(path), None, None)`.
- Otherwise treat as voice name string (handled by caller / ignored — auto).

`create_tts_adapter(settings, *, data_dir, runner)`:
- Reads `voxcpm_model`, `voxcpm_device`, `voxcpm_num_steps`, `voxcpm_batch_size`, `voxcpm_batch_flush_ms`, `voxcpm_cache_enabled`.
- Returns `VoxCPMTtsAdapter(...)`.

`SUPPORTED_TTS_BACKENDS = ("voxcpm",)`.

### 4.4 Environment (`voxcpm_env.py`)

- `voxcpm_venv_root()` → `backend/.venv-voxcpm` (override via `DV_VOXCPM_VENV`).
- `resolve_voxcpm_python()` → `.venv-voxcpm/Scripts/python.exe` (Windows) or `.venv-voxcpm/bin/python3` (POSIX).
- `is_voxcpm_available()` → `python -c "import voxcpm"` returns 0.

### 4.5 Cache (`voxcpm_cache.py`)

Same hashing scheme as `omnivoice_cache.py`, with `VOXCPM_CACHE_VERSION = "v1"` (bumped from `"v2-lang-vi"` to invalidate legacy entries). The cache key incorporates `voice_design` and `cfg_value` so different voice prompts / guidance scales don't collide.

### 4.6 Setup script (`setup_voxcpm.py`)

```text
uv venv .venv-voxcpm --python 3.12
uv pip install --python <python> torch torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install --python <python> voxcpm
<python> -c "import voxcpm; print(voxcpm.__version__)"
```

Accepts `--venv` and `--skip-torch` (same flags as the old script).

### 4.7 Runtime check (`runtime.py`)

`_check_voxcpm()`:

1. If `not is_voxcpm_available()` → `RuntimeCheck(id="voxcpm", display_name="VoxCPM2", status="blocked", required=True, action="Run 'python scripts/setup_voxcpm.py' in the backend folder.")`.
2. Try `acquire_client(data_dir, model="openbmb/VoxCPM2", device="cpu", num_steps=10)` and call `_ensure_alive()`. If it fails, return a blocked `RuntimeCheck` with the exception detail.
3. Otherwise return a ready `RuntimeCheck` with `resolved_path=str(voxcpm_venv_root())`.

`release_all_clients()` is called in `finally` to avoid leaking the smoke-test worker.

### 4.8 Frontend (`App.tsx`)

The TTS settings card:

- Title: `Lồng tiếng VoxCPM2`.
- Description: `VoxCPM2 là engine TTS. Chọn audio tham chiếu .wav, nhập voice design, hoặc để auto voice.`
- Field bindings: `settings.voxcpm_ref_audio`, `settings.voxcpm_instruct`, `settings.voxcpm_auto_voice` (all `??` default to the same as before).
- Cloned-voice select options unchanged.

Tests in `App.test.tsx` update `tts_backend: "voxcpm"` and the `Lồng tiếng VoxCPM2` assertion.

## 5. Data flow (end-to-end TTS step)

1. `tts_step()` reads the `translate` checkpoint, gathers segments.
2. For each segment, `_synthesize_segment_tts()` calls `create_tts_adapter().synthesize(text, output_path, voice=_default_tts_voice(settings), ref_text=original_text)`.
3. Adapter:
   - `parse_voxcpm_voice(voice)` → `(prompt_wav_path, prompt_text, voice_design)`.
   - Effective text = `f"({voice_design}){text}"` if `voice_design` else `text`.
   - Cache lookup. Hit → done.
   - Miss → `client.synthesize(text=effective_text, output_path, prompt_wav_path, prompt_text, voice_design, cfg_value=2.0, inference_timesteps=num_steps, cache_key=...)`.
4. Client writes a JSONL line to worker stdin, awaits response.
5. Worker batches compatible requests, runs `model.generate()` per request, writes WAV, returns duration + sample_rate.
6. Adapter writes the cache entry and the segment WAV is later mixed into the dubbed output.

## 6. Error handling

- Empty text → `AppError(422, code="EMPTY_TTS_TEXT")` (unchanged).
- Worker dies / pipe broken → `VOXCPM_WORKER_DIED`; next call respawns transparently.
- Worker startup timeout (30 s) → `VOXCPM_TTS_FAILED`, retryable.
- Synthesis timeout (default 120 s) → `VOXCPM_TIMEOUT`, retryable.
- voxcpm package missing → `VOXCPM_NOT_INSTALLED` (HTTP 400), not retryable; user must run setup script.
- Inference failure → `VOXCPM_INFERENCE_FAILED`, retryable.
- Voice preview / clone test endpoints surface `VOXCPM_SYNTHESIZE_FAILED` with action hint pointing at the setup script.

## 7. Testing plan

Unit / regression tests (no GPU required, run on CI):

- `test_voxcpm_tts.py` (replaces `test_omnivoice_tts.py`):
  - `test_parse_voxcpm_voice_modes` — auto / instruct: / .wav path.
  - `test_parse_voxcpm_voice_with_ref_audio` — file detection.
  - `test_create_tts_adapter_always_selects_voxcpm` — `create_tts_adapter({"tts_backend": "other", ...})` returns `VoxCPMTtsAdapter`.
  - `test_adapter_routes_to_client` — `FakeClient` is called with the right kwargs (prompt_wav_path, prompt_text, voice_design, etc.).
  - `test_adapter_clone_uses_ref_audio_and_ref_text`.
  - `test_adapter_chunks_long_text` — same chunking as OmniVoice.
  - `test_adapter_rejects_empty_text`.
  - `test_adapter_propagates_client_error`.
  - `test_adapter_uses_cache` — second call with same input is a cache hit.
  - `test_adapter_cache_disabled_always_calls_client`.
  - `test_cache_key_is_stable` and `test_cache_key_differs_on_inputs` — including new `voice_design` axis.
  - `test_cache_put_and_materialize` and `test_cache_miss_returns_false`.
  - `test_worker_generate_uses_real_voxcpm_api` — regression: worker calls `model.generate(text=..., prompt_wav_path=..., ...)` with the actual VoxCPM2 kwarg names, not OmniVoice kwargs.

- `test_settings.py`: update assertions to read `voxcpm_*` keys; ensure old `omnivoice_*` keys are not required.

- `App.test.tsx`: assert the new card title and the `tts_backend: "voxcpm"` payload round-trip.

Manual smoke test (requires GPU):

- `python scripts/setup_voxcpm.py` → creates `.venv-voxcpm`, installs deps.
- `python scripts/run_voxcpm_smoke.py` → end-to-end pipeline on a synthetic Vietnamese job; verify WAV is non-empty, voice design is applied.

Runtime smoke test (no real audio):

- `POST /api/runtime/smoke-test` → `_check_voxcpm()` returns `ready` when `.venv-voxcpm` exists with `voxcpm` importable, `blocked` otherwise.

## 8. Open questions / future work

- Whether to add streaming TTS over WebSocket for long jobs. Out of scope for this migration.
- Whether to allow per-segment voice overrides (e.g. multiple cloned voices in one job). Out of scope; the existing single-voice design is preserved.
- Bumping the default `voxcpm_num_steps` to 16+ for higher fidelity. Default kept at 10 to match the VoxCPM2 recommended setting; user can adjust in settings.
