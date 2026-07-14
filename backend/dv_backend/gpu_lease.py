"""Process-wide GPU lease for heavyweight inference."""

from __future__ import annotations

from contextlib import contextmanager
import logging
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import BinaryIO, Iterator

logger = logging.getLogger(__name__)

_lease_lock = threading.Lock()
_lease_holders: dict[str, str] = {}
_lease_condition = threading.Condition(_lease_lock)


def _device_key(device: str | None) -> str:
    return (device or "cuda:0").strip() or "cuda:0"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _default_lock_dir() -> Path:
    configured = os.environ.get("DV_GPU_LOCK_DIR", "").strip()
    if configured:
        return Path(configured)
    return Path(tempfile.gettempdir()) / "douyin-vietnamizer-gpu-locks"


def _lock_file(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_file(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def gpu_lease(owner: str, *, device: str | None = None, lock_dir: Path | None = None) -> Iterator[None]:
    """Acquire exclusive GPU access for one device across threads and processes."""
    device_key = _device_key(device)
    lock_root = Path(lock_dir) if lock_dir is not None else _default_lock_dir()
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f"gpu-{_safe_name(device_key)}.lock"
    lock_path.touch(exist_ok=True)
    handle = lock_path.open("r+b")
    acquired_file = False
    try:
        with _lease_condition:
            while _lease_holders.get(device_key) is not None and _lease_holders.get(device_key) != owner:
                logger.debug("GPU lease busy (%s); %s waiting", _lease_holders.get(device_key), owner)
                _lease_condition.wait(timeout=30.0)
            _lease_holders[device_key] = owner
        _lock_file(handle)
        acquired_file = True
        logger.debug("GPU lease acquired by %s on %s", owner, device_key)
        yield
    finally:
        try:
            if acquired_file:
                _unlock_file(handle)
        finally:
            handle.close()
            with _lease_condition:
                if _lease_holders.get(device_key) == owner:
                    _lease_holders.pop(device_key, None)
                    _lease_condition.notify_all()
                    logger.debug("GPU lease released by %s on %s", owner, device_key)


def gpu_lease_holder(device: str | None = None) -> str | None:
    with _lease_lock:
        return _lease_holders.get(_device_key(device))


def clear_gpu_lease_state(*, reason: str = "manual") -> list[str]:
    """Drop in-process GPU lease holders (e.g. after cancel/interrupt without __exit__)."""
    with _lease_condition:
        previous = [f"{device}: {owner}" for device, owner in sorted(_lease_holders.items())]
        _lease_holders.clear()
        _lease_condition.notify_all()
    if previous:
        logger.info("Cleared stale GPU lease holders (%s): %s", reason, ", ".join(previous))
    return previous


def reset_gpu_lease_for_tests() -> None:
    clear_gpu_lease_state(reason="test_reset")
