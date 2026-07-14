"""Lifecycle tests for OmniVoiceWorkerClient queued batching."""
from __future__ import annotations

import queue
from pathlib import Path

import pytest

from dv_backend.adapters.omnivoice_client import OmniVoiceWorkerClient
from dv_backend.errors import AppError


def _client_stub() -> OmniVoiceWorkerClient:
    return OmniVoiceWorkerClient(
        data_dir=Path("."),
        model="k2-fsa/OmniVoice",
        device="cpu",
        num_step=32,
        speed=1.0,
        language_id="vi",
        audio_chunk_threshold=30.0,
        audio_chunk_duration=15.0,
    )


def test_wait_many_preserves_input_order_with_out_of_order_responses() -> None:
    client = _client_stub()
    q1: queue.Queue = queue.Queue(maxsize=1)
    q2: queue.Queue = queue.Queue(maxsize=1)
    client._pending = {"req-1": q1, "req-2": q2}
    q2.put_nowait({"ok": True, "id": "req-2", "output_path": "/tmp/2.wav"})
    q1.put_nowait({"ok": True, "id": "req-1", "output_path": "/tmp/1.wav"})
    results = client.wait_many(["req-1", "req-2"], timeout_sec=1.0)
    assert [item["id"] for item in results] == ["req-1", "req-2"]
    assert client.pending_count == 0


def test_timeout_clears_pending_request() -> None:
    client = _client_stub()
    client._pending = {"req-1": queue.Queue(maxsize=1)}
    with pytest.raises(AppError) as exc:
        client.wait_result("req-1", timeout_sec=0.01)
    assert exc.value.info.code == "OMNIVOICE_TIMEOUT"
    assert client.pending_count == 0


def test_late_response_after_timeout_is_ignored_safely() -> None:
    client = _client_stub()
    late_q: queue.Queue = queue.Queue(maxsize=1)
    client._pending = {"req-1": late_q}
    with pytest.raises(AppError):
        client.wait_result("req-1", timeout_sec=0.01)
    assert client.pending_count == 0
    late_q.put_nowait({"ok": True, "id": "req-1"})
    with client._pending_lock:
        assert client._pending.get("req-1") is None


def test_fail_pending_clears_all_requests() -> None:
    client = _client_stub()
    q1: queue.Queue = queue.Queue(maxsize=1)
    q2: queue.Queue = queue.Queue(maxsize=1)
    client._pending = {"req-1": q1, "req-2": q2}
    client._fail_pending("worker died", retryable=True)
    assert client.pending_count == 0
    assert q1.get_nowait()["code"] == "OMNIVOICE_WORKER_DIED"
    assert q2.get_nowait()["code"] == "OMNIVOICE_WORKER_DIED"


def test_wait_result_returns_failed_response_without_raising() -> None:
    client = _client_stub()
    q: queue.Queue = queue.Queue(maxsize=1)
    client._pending = {"req-1": q}
    q.put_nowait({"ok": False, "id": "req-1", "code": "OMNIVOICE_INFERENCE_FAILED", "message": "boom"})
    response = client.wait_result("req-1", timeout_sec=1.0)
    assert response["ok"] is False
    assert client.pending_count == 0


def test_wait_raises_on_failed_response() -> None:
    client = _client_stub()
    q: queue.Queue = queue.Queue(maxsize=1)
    client._pending = {"req-1": q}
    q.put_nowait({"ok": False, "id": "req-1", "code": "OMNIVOICE_INFERENCE_FAILED", "message": "boom"})
    with pytest.raises(AppError):
        client.wait("req-1", timeout_sec=1.0)
