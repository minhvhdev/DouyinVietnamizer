# Portable Backend Error-Only Logging

## Goal

Capture backend errors to a durable file in the portable Windows build so users (and
us) can debug what went wrong after a crash, without adding an info-level log stream.

## Scope

- Backend Python process only. Not the Tauri shell, not the frontend.
- Errors only (`logger.error` and unhandled exceptions). INFO/WARN/DEBUG stay out of
  the file but continue flowing to stderr as today.
- Applies to both dev (`python -m dv_backend.main`) and portable (Tauri-spawned)
  invocations, since they share the same `create_app` entry point.

## Non-Goals

- No log shipping, no remote reporting.
- No per-job log files — one rotating file is enough for triage.
- No new env vars or settings.

## File Layout

- Path: `%LOCALAPPDATA%\DouyinVietnamizer\logs\backend-error.log`
  - Reuses the existing `AppConfig` data dir convention. Already writable in both dev
    and installed contexts.
- Rotation: `RotatingFileHandler(maxBytes=1MB, backupCount=3)`.
  - File plus `.1`, `.2`, `.3`. ~4MB total ceiling.
- Format: `%(asctime)s %(levelname)s %(name)s: %(message)s`
  - One line per record, with module name to make scanning fast.
- Uvicorn's own access/error logs keep going to stderr — the Tauri `parse_uvicorn_stderr`
  crash banner keeps working.

## Components

### New: `backend/dv_backend/error_logging.py`

Single function:

```
def configure_error_logging(config: AppConfig) -> None
```

- Adds a `RotatingFileHandler` to the root logger at `ERROR` level.
- Installs a `sys.excepthook` that logs the unhandled traceback to the same handler
  and re-raises (does not swallow the exception — uvicorn still gets to crash).
- Idempotent: if called twice, replaces the existing handler to avoid duplicate lines.

### Updated: `backend/dv_backend/config.py`

Add `error_log_path` property returning `data_dir / "logs" / "backend-error.log"`.
`ensure_directories` already creates the `logs` subdir.

### Updated: `backend/dv_backend/api.py`

Call `configure_error_logging(config)` at the top of `create_app`, before app or
services are constructed. This catches `logger.error` calls that happen during
imports of pipeline/adapters as well as runtime errors.

## Data Flow

1. App starts → `create_app` → `configure_error_logging(config)` attaches the handler.
2. `logger.error("...")` in pipeline/adapters → root logger dispatches to file.
3. Unhandled exception in a request → FastAPI default handler returns 500, the
   original `sys.excepthook` (now wrapped) writes the traceback to file, then re-raises
   so uvicorn's behavior is unchanged.
4. Startup crash (e.g. import error before `create_app` finishes) — not captured here;
   the Tauri side already shows the last 4KB of stderr in the crash banner, which is
   the right surface for that case.

## Error Handling

- If the log directory is not writable, fall back to `tempfile.gettempdir()` and emit
  a single stderr warning. Logging must never prevent the app from starting.
- `sys.excepthook` failure must not crash the app — wrap the whole hook body in a
  broad try/except that ultimately delegates to the previous hook.

## Testing

New file: `backend/tests/test_error_logging.py`

- `test_error_log_writes_to_configured_path(tmp_path)` — point AppConfig at `tmp_path`,
  call `configure_error_logging`, log an error, assert file exists and contains the
  message.
- `test_exception_hook_writes_traceback(tmp_path)` — install the hook in a thread,
  raise, assert file contains `Traceback` and the exception message.
- `test_rotate_replaces_handler_on_repeat_call(tmp_path)` — call twice, assert only
  one handler attached to root logger.
- `test_falls_back_to_tempdir_when_dir_unwritable(tmp_path, monkeypatch)` — make
  `mkdir` raise, assert handler falls back to tempdir and app does not crash.

## Touchpoints

1. New: `backend/dv_backend/error_logging.py`
2. Update: `backend/dv_backend/config.py` (add `error_log_path`)
3. Update: `backend/dv_backend/api.py` (call `configure_error_logging` in `create_app`)
4. New: `backend/tests/test_error_logging.py`
