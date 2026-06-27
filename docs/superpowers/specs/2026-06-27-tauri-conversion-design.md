# Tauri 2 Conversion — Design Spec

**Date:** 2026-06-27
**Status:** Approved
**Target:** Convert the existing DouyinVietnamizer (Vite+React frontend + Python FastAPI backend) into a Tauri 2 desktop app for Windows, with macOS/Linux builds as follow-up.

## Goal

Replace the browser-based dev workflow (`pnpm dev` opens a browser tab pointing at `http://localhost:5173` while a separate uvicorn process serves the API on `127.0.0.1:8765`) with a single Tauri 2 desktop application that:

1. Bundles the React frontend into a native window (WebView2 on Windows).
2. Spawns the Python backend as a managed child process from Rust.
3. Auto-detects whether the Python environment is ready and runs a first-run setup wizard if not.
4. Preserves the existing developer experience: `pnpm dev` continues to work for frontend-only or backend-only iteration.

## Non-Goals

- macOS/Linux production builds in this iteration (Windows only). Code paths stay cross-platform; CI for other OSes is follow-up.
- Code signing, notarization, auto-update (deferred; Tauri config leaves hooks for `tauri-plugin-updater` later).
- Replacing the HTTP loopback with a Tauri command-driven backend (backend stays a FastAPI server, frontend calls it via fetch).
- Bundling Torch/pyannote into the installer (per hybrid decision: user keeps `uv`-managed venv, only Rust + frontend ship in the bundle).

## Architecture

### Runtime topology

```
┌─────────────────────────────────────────────────────┐
│  Tauri window (WebView2 on Windows)                │
│  ┌───────────────────────────────────────────────┐  │
│  │  React app (frontend/dist or Vite dev)        │  │
│  │  - baseApi → http://127.0.0.1:8765/*          │  │
│  │  - Tauri commands:                            │  │
│  │      getBackendStatus()                       │  │
│  │      runFirstTimeSetup()                      │  │
│  │      restartBackend()                         │  │
│  │      openDevtools()                           │  │
│  └────────────────┬──────────────────────────────┘  │
│                   │ invoke                          │
│  ┌────────────────▼──────────────────────────────┐  │
│  │  Rust (src-tauri)                             │  │
│  │  - BackendState: Mutex<Option<Child>>         │  │
│  │  - On window ready:                           │  │
│  │      1. detect_venv()                         │  │
│  │      2. spawn_uvicorn() if Ready              │  │
│  │      3. wait_for_ready() polls /health 5s     │  │
│  │  - Commands expose status to frontend         │  │
│  └────────────────┬──────────────────────────────┘  │
└───────────────────┼─────────────────────────────────┘
                    │ spawn + stdio pipes
        ┌───────────▼────────────┐
        │  Python uvicorn        │
        │  child process         │
        │  DV_RELOAD=1 in dev    │
        └────────────────────────┘
```

### Layout

```
src-tauri/
├── Cargo.toml
├── tauri.conf.json
├── build.rs
├── capabilities/
│   └── default.json
├── icons/
└── src/
    ├── main.rs
    ├── backend.rs
    ├── setup.rs
    └── commands.rs

frontend/
└── src/lib/
    ├── api.ts                 (unchanged)
    └── tauri-bridge.ts        (new)
```

The frontend directory stays where it is. `src-tauri/` is added at the repository root. The `tauri.conf.json` `frontendDist` points at `../frontend/dist` and `devUrl` points at `http://localhost:5173`.

## Components

### `src-tauri/src/main.rs`

Entry point. Calls `tauri::Builder::default().setup(...)` which on window ready:
1. Reads `BackendState` from managed state.
2. Does **not** block window creation on backend readiness — the window shows a splash component while frontend awaits `getBackendStatus()`.
3. Forwards child process stderr/stdout to the same log file as Rust.

### `src-tauri/src/backend.rs`

Knows: how to spawn a process, how to poll HTTP. Does not know: UI, frontend, setup wizard logic.

Public API:
- `enum VenvStatus { Ready(PathBuf), MissingPython, MissingUv, MissingVenv }`
- `fn detect_venv(backend_dir: &Path) -> VenvStatus`
- `fn spawn_uvicorn(backend_dir: &Path) -> Result<Child, BackendStartError>`
- `async fn wait_for_ready(base_url: &str, timeout: Duration) -> Result<(), BackendStartError>`
- `enum BackendStartError { Spawn(io::Error), Timeout, Crashed{code: Option<i32>, stderr: String} }`

`detect_venv` checks, in order:
1. `uv --version` succeeds (`MissingUv` otherwise).
2. `python --version` reports 3.12.x (`MissingPython` otherwise).
3. `backend/.venv/pyvenv.cfg` exists (`MissingVenv` otherwise).

`spawn_uvicorn` runs `uv run python -m dv_backend.main` with `current_dir = backend_dir` and the inherited environment. Sets `DV_RELOAD=1` when the Tauri build profile is `dev`. Stderr is piped to a `BufReader` consumed by a background task that writes to the log file.

`wait_for_ready` polls `GET /health` on the configured base URL with 100ms interval up to 5s total. On timeout, kills the child and returns `BackendStartError::Timeout`. On child exit during polling, drains stderr pipe and returns `BackendStartError::Crashed`.

### `src-tauri/src/setup.rs`

Knows: how to invoke `uv` subcommands and stream progress. Does not know: HTTP, uvicorn, frontend.

Public API:
- `async fn run_first_time_setup(backend_dir: &Path, on_progress: F) -> Result<(), SetupError>`
- `enum SetupError { UvNotInstalled, PythonInstallFailed(String), SyncFailed(String) }`

Steps:
1. `uv python install 3.12` — streams `setup://progress {stage: "python", pct}`.
2. `uv sync --group dev` in `backend_dir` — streams `setup://progress {stage: "sync", pct}`.

Idempotent: re-running on a partially-completed setup is safe (uv handles partial state).

### `src-tauri/src/commands.rs`

Knows: how to map a Tauri command name to a function. Does not know: implementation details.

```rust
#[tauri::command]
async fn get_backend_status(state: State<'_, BackendState>) -> BackendStatus;

#[tauri::command]
async fn run_first_time_setup(
    state: State<'_, BackendState>,
    app: AppHandle,
) -> Result<(), SetupError>;

#[tauri::command]
async fn restart_backend(state: State<'_, BackendState>) -> Result<(), BackendStartError>;

#[tauri::command]
fn open_devtools(window: Window);
```

`BackendStatus` is a serializable enum: `SetupRequired(VenvStatus)`, `Starting`, `Ready(String /* baseUrl */)`, `Crashed{stderr: String}`, `AlreadyRunning`.

### `frontend/src/lib/tauri-bridge.ts`

New file. Wraps `invoke()` and adds:
- A `waitForBackend()` function that polls `getBackendStatus` every 200ms up to 30s, resolves with the `Ready` baseUrl, rejects with the latest status.
- A `subscribeBackendEvents()` function that registers listeners for `backend://ready` and `backend://crashed` and returns an unsubscribe handle.
- Re-exports the `baseApi` from `api.ts` unchanged.

The existing `App.tsx` is modified at exactly one site: the `useEffect` that currently probes backend health on mount now calls `waitForBackend()` first. If it returns `SetupRequired`, the app routes to `/setup`. If `Crashed`, it shows a banner with a Restart button.

## Data Flow

### Cold start, venv ready

1. `tauri::Builder.setup()` runs in parallel with window creation.
2. Window mounts, React renders `<Splash />`.
3. `useEffect` calls `tauri-bridge.waitForBackend()`.
4. Rust side: `detect_venv` returns `Ready`; `spawn_uvicorn` starts child; `wait_for_ready` polls `/health`.
5. Within 5s, poll succeeds → Rust stores `baseUrl` in `BackendState`, returns `Ready`.
6. Frontend unmounts `<Splash />`, mounts `<App />`.
7. Frontend starts making real `baseApi` calls to `http://127.0.0.1:8765`.

### Cold start, venv missing

1. `detect_venv` returns `MissingVenv` (or `MissingUv` / `MissingPython`).
2. `getBackendStatus` returns `SetupRequired(status)`.
3. Frontend routes to `<SetupWizard status={...} />`.
4. User clicks "Setup now" → `run_first_time_setup` invoked.
5. Rust streams `setup://progress` events; wizard shows progress bar.
6. On success, wizard re-invokes `getBackendStatus` → `Ready` → normal flow.
7. On failure, wizard shows error and a "Retry" button.

### Dev mode (`pnpm tauri:dev`)

1. `tauri.conf.json` `devUrl: "http://localhost:5173"`.
2. `tauri-cli` spawns Vite via `pnpm --filter frontend dev` in parallel with Cargo.
3. WebView loads `http://localhost:5173` instead of `frontend/dist`.
4. React HMR: edit `.tsx` → Vite refreshes.
5. Python reload: `DV_RELOAD=1` is set in the child env, uvicorn watches `backend/dv_backend/`.
6. Rust reload: edit `.rs` → Cargo rebuilds only changed crates, WebView reloads.

### Backend crash runtime

1. Child exits with non-zero code.
2. Rust's background stdout/stderr task detects exit, drains pipe.
3. Rust emits `backend://crashed {stderr}` event.
4. Frontend listener shows banner with stderr excerpt and a "Restart" button.
5. User clicks → `restartBackend` command → Rust spawns fresh child, waits ready, emits `backend://ready`.

## Error Handling

### Setup errors

`VenvStatus::{MissingUv, MissingPython, MissingVenv}` maps to three wizard screens with copy explaining how to fix and links to docs (`https://docs.astral.sh/uv/`). `SetupError` is shown as a red banner with the underlying message and a "Retry" button. No rollback on partial completion (idempotent re-run is preferred).

### Backend startup errors

`BackendStartError::{Spawn, Timeout, Crashed}` is shown as a modal dialog with:
- The error variant name
- The stderr excerpt (last 4KB) when available
- Buttons: "Open backend folder" (uses `tauri-plugin-shell` `open`), "Copy error" (uses `tauri-plugin-clipboard-manager`), "Retry"

No auto-retry on startup failure — the user inspects the log first.

### Runtime errors

The HTTP layer in `tauri-bridge.ts` wraps every `baseApi` call:
- `TypeError` from `fetch` (network failure) → one automatic retry after 500ms.
- Persistent failure → emits a `BackendDisconnected` event the App component listens to; shows a reconnecting indicator.
- HTTP 4xx/5xx from the backend pass through to the calling component unchanged (the existing error handling in the React app continues to work because the backend behavior is identical to the browser-based dev workflow).

### Idempotency guards

- `BackendState` is `Mutex<Option<Child>>`. `spawn_uvicorn` checks the option first; if `Some`, returns `AlreadyRunning`.
- `setup.rs` uses an `AtomicBool` `SETUP_IN_PROGRESS` so concurrent `run_first_time_setup` calls return `SetupInProgress` rather than racing.
- `detect_venv` is pure (filesystem read only).

## Testing

### Existing (unchanged)

- `backend/tests/` — pytest suite, invoked via `pnpm test:backend`.
- `frontend/tests/` — vitest suite, invoked via `pnpm test:frontend`.
- `pnpm test` runs both.

### New: `src-tauri/src/backend.rs` unit tests

A `#[cfg(test)] mod tests` block at the bottom of `backend.rs` with:
- `detect_venv` tested against `tempfile::tempdir()`:
  - With a stub `pyvenv.cfg` → `Ready`.
  - Without one → `MissingVenv`.
  - With `uv` removed from PATH (via env mutation in test) → `MissingUv`.
- `parse_uvicorn_stderr` (a small pure function extracted from the spawn helper) tested with sample Python tracebacks.
- `is_health_ok` tested against an in-process `tokio::net::TcpListener` that returns 200 OK; assert both success and timeout cases.

No test framework beyond `cargo test`. The `tempfile` crate is the only new dev-dependency beyond Tauri itself.

### Manual E2E checklist (in README)

1. `pnpm tauri:dev` opens a Tauri window, backend log appears in the spawning terminal.
2. Edit `frontend/src/renderer/App.tsx` — Vite HMR refreshes the window.
3. Edit `backend/dv_backend/api.py` — uvicorn `--reload` kicks in.
4. Edit `src-tauri/src/backend.rs` — Cargo rebuilds, window refreshes.
5. `pnpm tauri:build` produces `src-tauri/target/release/bundle/msi/*.msi`. Install in a clean VM, double-click, the app starts, the setup wizard runs (or skips if venv is already present from a prior dev run).

## Files Changed / Added

### Added

- `src-tauri/Cargo.toml`
- `src-tauri/tauri.conf.json`
- `src-tauri/build.rs`
- `src-tauri/capabilities/default.json`
- `src-tauri/src/main.rs`
- `src-tauri/src/backend.rs`
- `src-tauri/src/setup.rs`
- `src-tauri/src/commands.rs`
- `src-tauri/icons/*` (placeholder PNG/ICO files)
- `frontend/src/lib/tauri-bridge.ts`
- `src-tauri/.gitignore` (target/)

### Modified

- `package.json` — add `tauri:dev` and `tauri:build` scripts. Existing `dev`, `test`, `setup` stay.
- `frontend/src/renderer/App.tsx` — one site change: the mount-time backend health probe is replaced with `waitForBackend()`. New route `/setup` for the wizard.
- `frontend/src/renderer/SetupWizard.tsx` (new file in renderer dir) — first-run setup UI.
- `frontend/package.json` — add `@tauri-apps/api` ^2 as a dependency.
- `.gitignore` — exclude `src-tauri/target/`.
- `README.md` — add "Tauri desktop app" section with the manual E2E checklist.

### Unchanged

- All backend code (`backend/dv_backend/**`).
- All backend tests.
- All existing frontend code except the one site in `App.tsx` and the new `tauri-bridge.ts`.
- `docs/**`, `scripts/**`, `vendor/**`, `release/**`.

## Scripts

After this change, the top-level `package.json` `scripts` block becomes:

```json
{
  "tauri:dev": "tauri dev",
  "tauri:build": "tauri build",
  "dev": "pnpm dlx kill-port 8765 && concurrently -k -n backend,ui -c blue,magenta \"pnpm run dev:backend\" \"pnpm run dev:frontend\"",
  "dev:backend": "cd backend && uv run python -m dv_backend.main",
  "dev:frontend": "pnpm --filter frontend dev",
  "setup": "pnpm install && pnpm run setup:backend",
  "setup:backend": "cd backend && uv sync --group dev",
  "test": "pnpm run test:backend && pnpm run test:frontend",
  "test:backend": "cd backend && uv run pytest -v",
  "test:frontend": "pnpm --filter frontend test",
  "build": "pnpm --filter frontend build"
}
```

`tauri` CLI is added to `devDependencies` in the root `package.json` (the latest 2.x release at the time of implementation).

## Open Questions

None at design time. macOS/Linux production builds, code signing, and auto-update are explicitly deferred and out of scope for this iteration.
