"""Delayed, debounced VRAM release for OmniVoice preview synthesis."""

from __future__ import annotations

import logging
import threading
import wave
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

PREVIEW_RELEASE_GRACE_SEC = 20.0


def _release_preview_clients() -> None:
    from .adapters.omnivoice_client import release_clients

    release_clients("preview")


def wav_duration_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            frame_rate = handle.getframerate()
            if frame_rate <= 0:
                return 0.0
            return max(0.0, handle.getnframes() / frame_rate)
    except (OSError, wave.Error):
        return 0.0


class PreviewVramReleaseScheduler:
    def __init__(
        self,
        *,
        release_callback: Callable[[], None] = _release_preview_clients,
        timer_factory=threading.Timer,
    ) -> None:
        self._release_callback = release_callback
        self._timer_factory = timer_factory
        self._lock = threading.RLock()
        self._generation = 0
        self._timer = None

    def begin_preview(self) -> int:
        with self._lock:
            self._generation += 1
            token = self._generation
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            return token

    def complete_preview(self, token: int, *, audio_duration_sec: float) -> float | None:
        delay = max(0.0, float(audio_duration_sec)) + PREVIEW_RELEASE_GRACE_SEC
        with self._lock:
            if token != self._generation:
                return None
            if self._timer is not None:
                self._timer.cancel()
            timer = self._timer_factory(delay, lambda: self._release_if_current(token))
            timer.daemon = True
            self._timer = timer
            timer.start()
        return delay

    def abort_preview(self, token: int) -> float | None:
        return self.complete_preview(token, audio_duration_sec=0.0)

    def _release_if_current(self, token: int) -> None:
        with self._lock:
            if token != self._generation:
                return
            self._timer = None
        try:
            self._release_callback()
        except Exception:
            logger.exception("Failed to release OmniVoice preview VRAM.")


_SCHEDULER = PreviewVramReleaseScheduler()


def begin_omnivoice_preview() -> int:
    return _SCHEDULER.begin_preview()


def complete_omnivoice_preview(token: int, output_wav: Path) -> float | None:
    return _SCHEDULER.complete_preview(
        token,
        audio_duration_sec=wav_duration_seconds(output_wav),
    )


def abort_omnivoice_preview(token: int) -> float | None:
    return _SCHEDULER.abort_preview(token)
