"""Unit tests for OmniVoice worker explicit batch collection."""
from __future__ import annotations

import queue
import time

from dv_backend.adapters.omnivoice_worker import _collect_synthesize_batch


def _synth(*, batch_id: str | None = None, batch_size: int | None = None, batch_index: int | None = None) -> dict:
    payload: dict = {"op": "synthesize", "id": f"req-{batch_id}-{batch_index}", "text": "hello"}
    if batch_id is not None:
        payload["batch_id"] = batch_id
    if batch_size is not None:
        payload["batch_size"] = batch_size
    if batch_index is not None:
        payload["batch_index"] = batch_index
    return payload


def test_explicit_batch_size_one_flushes_immediately() -> None:
    messages: queue.Queue = queue.Queue()
    batch, reason, deferred = _collect_synthesize_batch(
        messages,
        _synth(batch_id="a", batch_size=1, batch_index=0),
        max_batch=4,
        flush_sec=0.15,
    )
    assert len(batch) == 1
    assert reason == "explicit_batch_complete"
    assert deferred is None


def test_explicit_batch_size_four_collects_without_timeout() -> None:
    messages: queue.Queue = queue.Queue()
    for index in range(1, 4):
        messages.put(_synth(batch_id="block", batch_size=4, batch_index=index))
    started = time.perf_counter()
    batch, reason, deferred = _collect_synthesize_batch(
        messages,
        _synth(batch_id="block", batch_size=4, batch_index=0),
        max_batch=4,
        flush_sec=0.15,
    )
    elapsed = time.perf_counter() - started
    assert len(batch) == 4
    assert reason == "explicit_batch_complete"
    assert deferred is None
    assert elapsed < 0.1


def test_explicit_batch_size_two_does_not_wait_flush_timeout() -> None:
    messages: queue.Queue = queue.Queue()
    messages.put(_synth(batch_id="pair", batch_size=2, batch_index=1))
    started = time.perf_counter()
    batch, reason, deferred = _collect_synthesize_batch(
        messages,
        _synth(batch_id="pair", batch_size=2, batch_index=0),
        max_batch=4,
        flush_sec=0.15,
    )
    elapsed = time.perf_counter() - started
    assert len(batch) == 2
    assert reason == "explicit_batch_complete"
    assert deferred is None
    assert elapsed < 0.1


def test_interleaved_batch_ids_are_not_mixed() -> None:
    messages: queue.Queue = queue.Queue()
    messages.put(_synth(batch_id="other", batch_size=2, batch_index=0))
    batch, reason, deferred = _collect_synthesize_batch(
        messages,
        _synth(batch_id="first", batch_size=2, batch_index=0),
        max_batch=4,
        flush_sec=0.15,
    )
    assert len(batch) == 1
    assert reason == "explicit_batch_complete"
    assert deferred is not None
    assert deferred["batch_id"] == "other"


def test_legacy_batch_waits_for_flush_timeout() -> None:
    messages: queue.Queue = queue.Queue()
    started = time.perf_counter()
    batch, reason, deferred = _collect_synthesize_batch(
        messages,
        _synth(),
        max_batch=4,
        flush_sec=0.05,
    )
    elapsed = time.perf_counter() - started
    assert len(batch) == 1
    assert reason == "flush_timeout"
    assert deferred is None
    assert elapsed >= 0.04


def test_legacy_batch_reaches_max_batch_without_flush_timeout() -> None:
    messages: queue.Queue = queue.Queue()
    for _ in range(3):
        messages.put(_synth())
    started = time.perf_counter()
    batch, reason, deferred = _collect_synthesize_batch(
        messages,
        _synth(),
        max_batch=4,
        flush_sec=0.15,
    )
    elapsed = time.perf_counter() - started
    assert len(batch) == 4
    assert reason == "max_batch_reached"
    assert deferred is None
    assert elapsed < 0.1
