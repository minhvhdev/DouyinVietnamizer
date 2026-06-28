"""Global GPU lease to serialize heavyweight inference across job threads."""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)

_lease_lock = threading.Lock()
_lease_holder: str | None = None
_lease_condition = threading.Condition(_lease_lock)


@contextmanager
def gpu_lease(owner: str) -> Iterator[None]:
    """Acquire exclusive GPU access for *owner* (e.g. ``job-abc:asr``)."""
    global _lease_holder
    with _lease_condition:
        while _lease_holder is not None and _lease_holder != owner:
            logger.debug("GPU lease busy (%s); %s waiting", _lease_holder, owner)
            _lease_condition.wait(timeout=30.0)
        _lease_holder = owner
        logger.debug("GPU lease acquired by %s", owner)

    try:
        yield
    finally:
        with _lease_condition:
            if _lease_holder == owner:
                _lease_holder = None
                _lease_condition.notify_all()
                logger.debug("GPU lease released by %s", owner)


def gpu_lease_holder() -> str | None:
    with _lease_lock:
        return _lease_holder


def reset_gpu_lease_for_tests() -> None:
    global _lease_holder
    with _lease_condition:
        _lease_holder = None
        _lease_condition.notify_all()
