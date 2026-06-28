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


def _excepthook(previous) -> None:
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

    sys.excepthook = _excepthook(sys.excepthook)
    return handler
