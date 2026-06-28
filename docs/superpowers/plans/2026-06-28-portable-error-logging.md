# Portable Backend Error-Only Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist Python backend errors to `%LOCALAPPDATA%\DouyinVietnamizer\logs\backend-error.log` (rotated) so users can diagnose crashes after the fact.

**Architecture:** One new module `backend/dv_backend/error_logging.py` exposes `configure_error_logging(config)`, which attaches a `RotatingFileHandler(level=ERROR)` to the root logger and installs a `sys.excepthook` that writes unhandled tracebacks to the same file. `create_app` calls it once at startup. Stderr behaviour is unchanged so Tauri crash banner keeps working.

**Tech Stack:** Python stdlib `logging.handlers.RotatingFileHandler`, `sys.excepthook`, pytest.

## Global Constraints

- Error log file: `data_dir / "logs" / "backend-error.log"`, where `data_dir` is `AppConfig.data_dir` (already defaults to `%LOCALAPPDATA%\DouyinVietnamizer`).
- Rotation: `maxBytes=1MB`, `backupCount=3`.
- Log format: `%(asctime)s %(levelname)s %(name)s: %(message)s`.
- Handler level: `ERROR`. INFO/WARN/DEBUG are NOT captured in the file.
- Unhandled exceptions: hook must `logging.error(...)` the traceback, then re-raise / delegate to previous hook so behaviour is unchanged.
- Idempotent: calling `configure_error_logging` twice must not attach duplicate handlers.
- Graceful fallback: if the logs dir is not writable, fall back to `tempfile.gettempdir()` and emit one stderr warning. Never block app startup.
- Touchpoints limited to: `backend/dv_backend/error_logging.py` (new), `backend/dv_backend/config.py` (add property), `backend/dv_backend/api.py` (call at top of `create_app`), `backend/tests/test_error_logging.py` (new).

---

## File Structure

- **`backend/dv_backend/config.py`** — add `error_log_path` property. Everything else stays the same.
- **`backend/dv_backend/error_logging.py`** (new) — `configure_error_logging(config)`, plus a private `_setup_handler` helper for testability.
- **`backend/dv_backend/api.py`** — one new import + one new line at the top of `create_app`.
- **`backend/tests/test_error_logging.py`** (new) — covers happy path, exception hook, idempotence, fallback.

---

### Task 1: Add `error_log_path` property to AppConfig

**Files:**
- Modify: `backend/dv_backend/config.py:9-28`
- Test: `backend/tests/test_config.py`

**Interfaces:**
- Consumes: nothing (uses existing `self.data_dir`)
- Produces: `AppConfig.error_log_path -> Path` (returns `data_dir / "logs" / "backend-error.log"`)

- [ ] **Step 1: Add failing test to `backend/tests/test_config.py`**

Append to the existing test file (keep the existing test intact):

```python
def test_error_log_path_uses_logs_subdir(tmp_path: Path) -> None:
    config = AppConfig(tmp_path)

    assert config.error_log_path == tmp_path / "logs" / "backend-error.log"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd backend && python -m pytest tests/test_config.py::test_error_log_path_uses_logs_subdir -v`
Expected: FAIL with `AttributeError: 'AppConfig' object has no attribute 'error_log_path'`

- [ ] **Step 3: Add the property to `AppConfig`**

In `backend/dv_backend/config.py`, add a new property below `log_path` (after line 15, before `from_env`):

```python
    @property
    def error_log_path(self) -> Path:
        return self.data_dir / "logs" / "backend-error.log"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd backend && python -m pytest tests/test_config.py -v`
Expected: PASS (both `test_data_dir_override_is_used` and the new one).

- [ ] **Step 5: Commit**

```bash
git add backend/dv_backend/config.py backend/tests/test_config.py
git commit -m "feat(config): add error_log_path property"
```

---

### Task 2: Create `error_logging.py` with `configure_error_logging`

**Files:**
- Create: `backend/dv_backend/error_logging.py`
- Test: `backend/tests/test_error_logging.py`

**Interfaces:**
- Consumes: `AppConfig` (Task 1) — uses `error_log_path` and `ensure_directories`.
- Produces: `configure_error_logging(config: AppConfig) -> logging.Handler` (returns the attached handler so tests can inspect it).

- [ ] **Step 1: Write the failing test file `backend/tests/test_error_logging.py`**

```python
import logging
import sys
import tempfile
from pathlib import Path

import pytest

from dv_backend.config import AppConfig
from dv_backend.error_logging import configure_error_logging


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    return AppConfig(tmp_path)


def _drain_root_handlers() -> None:
    for handler in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(handler)


@pytest.fixture(autouse=True)
def reset_root_logger() -> None:
    _drain_root_handlers()
    yield
    _drain_root_handlers()
    sys.excepthook = sys.__excepthook__


def test_configure_attaches_rotating_handler(config: AppConfig) -> None:
    handler = configure_error_logging(config)

    assert isinstance(handler, logging.handlers.RotatingFileHandler)
    assert handler.baseFilename == str(config.error_log_path)
    assert handler.level == logging.ERROR
    assert handler.maxBytes == 1 * 1024 * 1024
    assert handler.backupCount == 3
    assert handler in logging.getLogger().handlers


def test_logger_error_writes_to_file(config: AppConfig) -> None:
    configure_error_logging(config)

    logging.getLogger("dv_backend.test").error("boom %s", "boom")

    contents = config.error_log_path.read_text(encoding="utf-8")
    assert "ERROR" in contents
    assert "dv_backend.test" in contents
    assert "boom boom" in contents


def test_repeat_call_replaces_existing_handler(config: AppConfig) -> None:
    configure_error_logging(config)
    configure_error_logging(config)

    handlers = [
        h for h in logging.getLogger().handlers
        if isinstance(h, logging.handlers.RotatingFileHandler)
    ]
    assert len(handlers) == 1


def test_unhandled_exception_is_logged(config: AppConfig) -> None:
    configure_error_logging(config)

    # Invoke the installed excepthook directly so we don't need a real crash.
    hook = sys.excepthook
    try:
        raise RuntimeError("explode")
    except RuntimeError as exc:
        hook(exc.__class__, exc, exc.__traceback__)

    contents = config.error_log_path.read_text(encoding="utf-8")
    assert "RuntimeError" in contents
    assert "explode" in contents
    assert "Traceback" in contents


def test_falls_back_to_tempdir_when_logs_dir_unwritable(
    config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(_self) -> None:
        raise PermissionError("read-only volume")

    monkeypatch.setattr(Path, "mkdir", boom)

    handler = configure_error_logging(config)

    # Handler should attach successfully, but pointing somewhere writable.
    assert handler.baseFilename != str(config.error_log_path)
    assert tempfile.gettempdir() in handler.baseFilename

    logging.getLogger("dv_backend.test").error("fallback works")
    Path(handler.baseFilename).read_text(encoding="utf-8")  # should not raise
```

- [ ] **Step 2: Run the test to verify it fails (collection error expected)**

Run: `cd backend && python -m pytest tests/test_error_logging.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'dv_backend.error_logging'`

- [ ] **Step 3: Implement `backend/dv_backend/error_logging.py`**

```python
"""Error-only file logging for the backend.

Attaches a rotating file handler to the root logger at ERROR level and installs
a ``sys.excepthook`` that writes unhandled tracebacks to the same file. The
hook re-raises to the previous hook so existing behaviour is preserved.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import tempfile
import traceback
from pathlib import Path

from .config import AppConfig

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_MARKER = "dv_backend.error_logging"


def _resolve_handler_path(config: AppConfig) -> Path:
    try:
        config.ensure_directories()
        return config.error_log_path
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "douyin-vietnamizer-backend-error.log"
        print(
            f"[{_MARKER}] logs dir not writable, falling back to {fallback}",
            file=sys.stderr,
        )
        fallback.parent.mkdir(parents=True, exist_ok=True)
        return fallback


def _build_handler(path: Path) -> logging.handlers.RotatingFileHandler:
    handler = logging.handlers.RotatingFileHandler(
        path,
        maxBytes=1 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setLevel(logging.ERROR)
    handler.setFormatter(logging.Formatter(_FORMAT))
    return handler


def _excepthook(handler: logging.Handler, previous) -> None:
    def hook(exc_type, exc_value, exc_tb) -> None:
        try:
            logging.getLogger(_MARKER).error(
                "Unhandled exception:\n%s",
                "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
            )
        except Exception:
            pass
        previous(exc_type, exc_value, exc_tb)

    return hook


def configure_error_logging(config: AppConfig) -> logging.Handler:
    """Attach an error-only rotating file handler to the root logger.

    Replaces any handler previously installed by this module so the function is
    safe to call multiple times. Also installs a ``sys.excepthook`` that logs
    unhandled exceptions to the same file.
    """

    path = _resolve_handler_path(config)
    handler = _build_handler(path)

    root = logging.getLogger()
    for existing in list(root.handlers):
        if getattr(existing, "_dv_error_handler", False):
            root.removeHandler(existing)
            existing.close()
    handler._dv_error_handler = True  # type: ignore[attr-defined]
    root.addHandler(handler)

    sys.excepthook = _excepthook(handler, sys.excepthook)
    return handler
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_error_logging.py -v`
Expected: PASS for all 5 tests.

If `test_falls_back_to_tempdir_when_logs_dir_unwritable` fails because the fallback file already exists, delete the leftover at `tempfile.gettempdir()/douyin-vietnamizer-backend-error.log` and re-run.

- [ ] **Step 5: Commit**

```bash
git add backend/dv_backend/error_logging.py backend/tests/test_error_logging.py
git commit -m "feat(logging): add error-only rotating file handler"
```

---

### Task 3: Wire `configure_error_logging` into `create_app`

**Files:**
- Modify: `backend/dv_backend/api.py:1-25` (imports) and `backend/dv_backend/api.py:229-232` (call site)
- Test: `backend/tests/test_api.py` (existing) — no new tests needed; existing health-check test will exercise the wired path.

**Interfaces:**
- Consumes: `configure_error_logging` from Task 2.
- Produces: side effect — `create_app` now attaches the handler at the start.

- [ ] **Step 1: Run the existing API tests to confirm baseline is green**

Run: `cd backend && python -m pytest tests/test_api.py -v`
Expected: PASS. If any fail, fix forward and re-run.

- [ ] **Step 2: Add the import**

In `backend/dv_backend/api.py`, add the new import in alphabetical order among the existing `from ....` block (after `from .database import Database`, before `from .errors import ...`):

```python
from .error_logging import configure_error_logging
```

- [ ] **Step 3: Call `configure_error_logging` in `create_app`**

In `backend/dv_backend/api.py`, modify the start of `create_app` (line 229-232) to call the logger right after `ensure_directories`. The function body should become:

```python
def create_app(config: AppConfig | None = None) -> FastAPI:
    load_repo_dotenv()
    config = config or AppConfig.from_env()
    config.ensure_directories()
    configure_error_logging(config)
    database = Database(config.database_path)
    database.migrate()
```

- [ ] **Step 4: Run the full backend test suite**

Run: `cd backend && python -m pytest -q`
Expected: PASS. If the new `test_error_logging` tests fail because `create_app` was called by another test and left handlers behind, the `reset_root_logger` autouse fixture in Task 2 will already clean them up — no extra work needed.

- [ ] **Step 5: Commit**

```bash
git add backend/dv_backend/api.py
git commit -m "feat(api): wire error-only logging into create_app"
```

---

### Task 4: Manual smoke verification

**Files:** none (no code change).

This task is a one-shot check that the file shows up at runtime; no automated test because the path is platform-dependent (`%LOCALAPPDATA%`).

- [ ] **Step 1: Start the backend, trigger an error, inspect the log**

```bash
cd backend
DV_DATA_DIR=$PWD/.tmp-data python -c "
from dv_backend.api import create_app
from dv_backend.config import AppConfig
config = AppConfig.from_env()
config.ensure_directories()
create_app(config)
import logging
logging.getLogger('smoke').error('hello from smoke test')
"
```

Then:

```bash
ls .tmp-data/logs/
cat .tmp-data/logs/backend-error.log
```

Expected:
- `backend-error.log` exists.
- File contains a line with `ERROR smoke: hello from smoke test`.

- [ ] **Step 2: Commit any incidental cleanup (usually nothing)**

If no changes, skip this step.

---

## Self-Review Notes

- **Spec coverage:** Goal + file layout → Tasks 1+2+3. Rotation/format/level → Task 2 (constants in `_FORMAT` and `_build_handler`). Unhandled exceptions → Task 2 (`_excepthook`). Graceful fallback → Task 2 (`_resolve_handler_path`). Idempotence → Task 2 (`_dv_error_handler` marker). Testing → Tasks 1, 2, 4. Touchpoints → all four files listed.
- **Type consistency:** `error_log_path` is referenced consistently as `config.error_log_path` in Tasks 1, 2, 4. `configure_error_logging(config)` is the single public name; tests and `create_app` both use the same signature.
- **No placeholders:** all code blocks are complete; no "TBD" / "add appropriate handling" anywhere.
