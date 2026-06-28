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
    def boom(self, *args, **kwargs) -> None:
        # Only fail mkdir for paths under the configured data_dir so the
        # fallback path inside tempfile.gettempdir() can still be created.
        if str(self).startswith(str(config.data_dir)):
            raise PermissionError("read-only volume")

    monkeypatch.setattr(Path, "mkdir", boom)

    handler = configure_error_logging(config)

    # Handler should attach successfully, but pointing somewhere writable.
    assert handler.baseFilename != str(config.error_log_path)
    assert tempfile.gettempdir() in handler.baseFilename

    logging.getLogger("dv_backend.test").error("fallback works")
    Path(handler.baseFilename).read_text(encoding="utf-8")  # should not raise
