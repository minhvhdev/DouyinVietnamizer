from __future__ import annotations

import importlib
import importlib.util


class _FakeTimer:
    instances: list["_FakeTimer"] = []

    def __init__(self, delay: float, callback) -> None:
        self.delay = delay
        self.callback = callback
        self.cancelled = False
        self.daemon = False
        self.started = False
        self.instances.append(self)

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        self.callback()


def _release_module():
    spec = importlib.util.find_spec("dv_backend.preview_vram_release")
    assert spec is not None, "preview VRAM scheduler module must exist"
    return importlib.import_module("dv_backend.preview_vram_release")


def test_preview_release_delay_matches_audio_duration_plus_20_seconds() -> None:
    module = _release_module()
    _FakeTimer.instances.clear()
    released: list[str] = []
    scheduler = module.PreviewVramReleaseScheduler(
        release_callback=lambda: released.append("released"),
        timer_factory=_FakeTimer,
    )

    token = scheduler.begin_preview()
    delay = scheduler.complete_preview(token, audio_duration_sec=7.25)

    assert delay == 27.25
    timer = _FakeTimer.instances[-1]
    assert timer.delay == 27.25
    assert timer.started is True
    assert timer.daemon is True
    timer.fire()
    assert released == ["released"]


def test_new_preview_cancels_and_replaces_pending_release() -> None:
    module = _release_module()
    _FakeTimer.instances.clear()
    released: list[str] = []
    scheduler = module.PreviewVramReleaseScheduler(
        release_callback=lambda: released.append("released"),
        timer_factory=_FakeTimer,
    )

    first_token = scheduler.begin_preview()
    scheduler.complete_preview(first_token, audio_duration_sec=5.0)
    first_timer = _FakeTimer.instances[-1]

    second_token = scheduler.begin_preview()
    assert first_timer.cancelled is True
    scheduler.complete_preview(second_token, audio_duration_sec=9.0)
    second_timer = _FakeTimer.instances[-1]

    first_timer.fire()
    assert released == []
    second_timer.fire()
    assert released == ["released"]
