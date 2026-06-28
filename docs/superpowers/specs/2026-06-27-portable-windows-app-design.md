# Portable Windows App — Design Spec

**Date:** 2026-06-27
**Status:** Approved
**Target:** Ship DouyinVietnamizer as a Windows x64 portable folder that runs with full features immediately, without first-run setup, while preserving fast `pnpm tauri:dev` iteration.

## Goal

Change the current Tauri desktop app from a first-run setup model into a portable runtime model:

1. The release artifact is a copyable folder containing `Douyin Vietnamizer.exe` and bundled runtime resources.
2. The app runs full pipeline features immediately on Windows x64 machines with a compatible NVIDIA CUDA setup.
3. No user-facing setup wizard is required for Python, backend dependencies, FFmpeg, yt-dlp, Qwen3-ASR, or VoxCPM2 models.
4. `pnpm tauri:dev` uses the same prepared portable runtime, but keeps frontend HMR and backend reload for fast code changes.

## Non-Goals

- Single-file `.exe` packaging.
- CPU/non-CUDA fallback.
- Runtime/model downloading inside the app.
- MSI-first installer flow.
- Code signing, auto-update, or cloud distribution.
- Supporting machines without the required NVIDIA/CUDA driver stack; the app reports a clear error instead.

## Architecture

### Runtime topology

```
┌────────────────────────────────────────────────────────────┐
│  Portable folder                                            │
│  ├─ Douyin Vietnamizer.exe                                  │
│  └─ resources/portable-runtime/                             │
│     ├─ python/ or .venv/                                     │
│     ├─ backend/                                              │
│     ├─ tools/                                                │
│     │  ├─ ffmpeg/...                                         │
│     │  └─ yt-dlp/...                                         │
│     └─ models/                                               │
│        ├─ qwen3-asr/...                                      │
│        └─ voxcpm2/...                                        │
└────────────────────────────────────────────────────────────┘

Dev equivalent:
repo/
├─ backend/dv_backend/              # live source, reloads in dev
├─ frontend/src/                    # Vite HMR
└─ vendor/portable-runtime/         # prepared runtime used by tauri dev
```

Rust owns runtime discovery and process spawning. Python owns runtime checks and pipeline execution. React only waits for backend readiness and displays errors.

### Runtime discovery

Resolution order:

1. `DV_PORTABLE_RUNTIME_DIR`, if set. Used for debugging and local packaging checks.
2. Dev profile: `<repo>/vendor/portable-runtime`.
3. Release profile: Tauri resource path `portable-runtime` beside/in app resources.

The resolver validates required paths before spawning Python:

- Python executable or venv script.
- Backend entrypoint.
- `tools/ffmpeg` and `tools/yt-dlp`.
- Required model directories.

If validation fails, Rust returns `PortableRuntimeMissing { path, missing_items }` to the frontend.

### Backend spawn

Dev spawn:

- Python executable from `vendor/portable-runtime`.
- `current_dir` points at repo `backend`.
- `PYTHONPATH`/env points at repo backend source.
- `DV_RELOAD=1`.
- `DV_PORTABLE_RUNTIME_DIR` points at `vendor/portable-runtime`.
- `PATH` is prepended with `runtime/tools`.

Release spawn:

- Python executable from packaged `portable-runtime`.
- `current_dir` points at packaged backend directory.
- `DV_RELOAD=0`.
- `DV_PORTABLE_RUNTIME_DIR` points at packaged runtime.
- `PATH` is prepended with packaged tools.

This keeps runtime immutable in normal use and avoids rebuilding Python dependencies/models when source files change during development.

### Tauri bundle

`tauri.conf.json` changes from MSI-only to portable-friendly output. The exact Tauri target should use the smallest working Windows folder artifact available in Tauri 2 for this repo. If Tauri requires a bundle target, prefer NSIS only as a secondary artifact and document the folder under `src-tauri/target/release/` as the portable output.

The build includes `vendor/portable-runtime/**` as app resources. The packaging step does not generate or download runtime files; it only copies a prepared runtime.

## Components

### `src-tauri/src/backend.rs`

Replace venv/PATH detection with portable runtime detection.

Public surface becomes centered on:

- `PortableRuntime` — resolved paths for Python, backend, tools, models.
- `PortableRuntimeStatus` — `Ready(PortableRuntime)` or `Missing { root, items }`.
- `resolve_portable_runtime(dev_profile: bool) -> PortableRuntimeStatus`.
- `spawn_uvicorn(runtime: &PortableRuntime, dev_profile: bool) -> Result<Child, BackendStartError>`.

Existing health polling and stderr truncation stay.

### `src-tauri/src/state.rs`

Store the resolved runtime path in `BackendState` so commands use one consistent runtime per app session.

### `src-tauri/src/commands.rs`

`get_backend_status` changes from setup-oriented status to portable-oriented status:

- `portable_missing { root, missing_items }`.
- `starting`.
- `ready { base_url }`.
- `crashed { stderr }`.
- `already_running`.

`run_first_time_setup_cmd` is no longer part of the normal portable flow. Keep it only if tests or old UI still reference it; otherwise remove the command and frontend call together.

### `frontend/src/lib/tauri-bridge.ts`

Update `BackendStatus` types to match portable statuses. `waitForBackend()` still polls `get_backend_status`, but rejects with `portable_missing` instead of `setup_required`.

### `frontend/src/renderer/App.tsx`

Replace setup wizard routing for Tauri portable mode:

- `portable_missing`: show a concise fatal screen listing missing packaged files and the runtime path.
- `crashed`: show existing crash UI with stderr.
- `ready`: render full app.

The in-app runtime wizard for downloading vendor/model assets is not the default path in portable mode. It can be deleted if no longer reachable, or left unused for non-portable/browser dev if removing it would create unnecessary churn.

### Backend runtime/tool resolution

Backend code that finds FFmpeg, yt-dlp, and models should prefer `DV_PORTABLE_RUNTIME_DIR`:

- Tools: `DV_PORTABLE_RUNTIME_DIR/tools` before existing `vendor/` and PATH logic.
- Models: `DV_PORTABLE_RUNTIME_DIR/models` before default model cache paths when the pipeline expects bundled models.
- `DV_ALLOW_PATH_TOOLS` remains useful for tests, but release portable should not depend on PATH.

## Data Flow

### Release cold start

1. User double-clicks `Douyin Vietnamizer.exe` from the portable folder.
2. Tauri resolves packaged `portable-runtime` from app resources.
3. Rust validates Python, backend, tools, and model paths.
4. Rust spawns backend with packaged Python and portable env.
5. Rust polls `GET /health` on `127.0.0.1:8765`.
6. React `waitForBackend()` receives `ready`.
7. App renders normally and all existing feature API calls work.

### Dev cold start

1. Developer runs `pnpm tauri:dev`.
2. Tauri starts Vite through existing `beforeDevCommand`.
3. Rust resolves `vendor/portable-runtime`.
4. Rust spawns packaged Python against live `backend/` source with `DV_RELOAD=1`.
5. Frontend edits reload through Vite HMR.
6. Backend edits reload through uvicorn without rebuilding runtime.
7. Rust edits still trigger Cargo rebuild, as expected.

### Missing runtime

1. Runtime validation fails before backend spawn.
2. Rust returns `portable_missing` with exact missing items.
3. Frontend shows a fatal package error.
4. User/developer fixes the portable folder; no in-app downloader runs.

### Missing CUDA/GPU driver

1. Runtime exists and backend starts.
2. Backend runtime status detects missing or incompatible NVIDIA/CUDA setup.
3. Frontend shows blocked runtime status with a clear driver/GPU message.
4. No CPU fallback is attempted.

## Error Handling

- Missing runtime root: show path and `DV_PORTABLE_RUNTIME_DIR` hint.
- Missing packaged file: show exact relative path.
- Backend spawn failure: show command context and stderr excerpt.
- Backend timeout: kill child and show timeout plus runtime path.
- Backend crash: preserve existing stderr UI, add runtime root context.
- CUDA/driver missing: backend runtime status reports blocked; frontend does not offer setup/download.

## Testing

### Rust unit tests

Add focused tests around pure resolver logic:

- Env override wins.
- Dev profile resolves `vendor/portable-runtime`.
- Release-style root validates when required files exist.
- Missing root reports root missing.
- Missing required items are all listed.
- Spawn env builder prepends tool paths and sets `DV_PORTABLE_RUNTIME_DIR`.

No new test framework. Use `tempfile`, already present.

### Backend tests

Update existing runtime/vendor tests so `DV_PORTABLE_RUNTIME_DIR` is covered:

- Tools resolve from portable runtime before PATH.
- Missing portable tool produces blocked/fail status.
- Model path checks can use tiny temp directories; do not require real model files in tests.

### Frontend tests

Update Tauri bridge/App tests:

- `portable_missing` renders fatal package error, not setup wizard.
- `ready` renders app shell.
- `crashed` still renders crash/setup error path or its replacement.

### Manual verification

1. Prepare `vendor/portable-runtime` on the dev machine.
2. Run `pnpm tauri:dev`.
3. Confirm backend starts without `uv`, Python PATH, or setup wizard dependency.
4. Edit `frontend/src/renderer/App.tsx`; Vite HMR refreshes.
5. Edit `backend/dv_backend/api.py`; backend reloads without rebuilding runtime.
6. Build release.
7. Copy the output folder to a clean Windows x64 NVIDIA machine.
8. Run `Douyin Vietnamizer.exe`; app reaches the main UI and runtime status passes.

## Files Expected to Change

- `src-tauri/tauri.conf.json` — portable resources and bundle target.
- `src-tauri/src/backend.rs` — portable runtime resolver and spawn command.
- `src-tauri/src/state.rs` — store runtime root/resolved config.
- `src-tauri/src/commands.rs` — status enum and command flow.
- `frontend/src/lib/tauri-bridge.ts` — status types.
- `frontend/src/renderer/App.tsx` — portable missing/crash handling.
- `frontend/src/renderer/SetupWizard.tsx` — remove from portable path or delete if unused.
- `backend/dv_backend/vendor.py` / `runtime.py` / related probes — prefer portable runtime env paths.
- `backend/tests/**`, `frontend/tests/**`, `src-tauri` tests — update coverage.
- `README.md` — document portable runtime layout, dev workflow, and release copy instructions.
- `package.json` — add helper scripts only if needed; keep existing `pnpm tauri:dev` as the main dev command.

## Build Artifact Contract

A valid portable release folder contains:

```
Douyin Vietnamizer.exe
resources/
└─ portable-runtime/
   ├─ python/ or .venv/
   ├─ backend/
   ├─ tools/
   └─ models/
```

The app may create user data under `%LOCALAPPDATA%\DouyinVietnamizer` unless `DV_DATA_DIR` is set. Runtime files under the portable folder are treated as read-only.

## Open Questions

None. The user selected folder-based portable packaging, fully bundled runtime/models, Windows x64 NVIDIA/CUDA-only support, and `pnpm tauri:dev` using the portable runtime.