"""Tests for OmniVoice queued batch client and adapter."""
from __future__ import annotations

from pathlib import Path
import array
import wave

import pytest

from dv_backend.adapters.omnivoice_client import OmniVoiceWorkerClient
from dv_backend.adapters.omnivoice_tts import OmniVoiceTtsAdapter


def _write_tone_wav(path: Path, *, duration_sec: float = 0.2, rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(rate * duration_sec)
    samples = array.array("h", [8000 if (index // 100) % 2 == 0 else -8000 for index in range(frames)])
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())


class _TrackingClient:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self._counter = 0

    def submit(
        self,
        *,
        text: str,
        output_path: Path,
        ref_audio: str | None,
        ref_text: str | None,
        anchor_text: str | None = None,
        instruct: str | None,
        include_perf: bool = False,
        batch_id: str | None = None,
        batch_index: int | None = None,
        batch_size: int | None = None,
    ) -> str:
        self._counter += 1
        request_id = f"req-{self._counter}"
        self.events.append(("submit", request_id))
        _ = output_path, ref_audio, ref_text, anchor_text, instruct, text, include_perf
        _ = batch_id, batch_index, batch_size
        return request_id

    def wait(self, request_id: str, *, timeout_sec: float = 600.0) -> dict:
        self.events.append(("wait", request_id))
        return {"ok": True, "id": request_id}

    def wait_result(self, request_id: str, *, timeout_sec: float = 600.0) -> dict:
        self.events.append(("wait", request_id))
        return {"ok": True, "id": request_id}

    def wait_many(self, request_ids: list[str], *, timeout_sec: float = 600.0) -> list[dict]:
        return [self.wait_result(request_id, timeout_sec=timeout_sec) for request_id in request_ids]

    def synthesize_many(self, requests: list[dict], *, timeout_sec: float = 600.0) -> list[dict]:
        request_ids = [
            self.submit(
                text=str(req["text"]),
                output_path=Path(req["output_path"]),
                ref_audio=req.get("ref_audio"),
                ref_text=req.get("ref_text"),
                anchor_text=req.get("anchor_text"),
                instruct=req.get("instruct"),
                include_perf=bool(req.get("include_perf")),
            )
            for req in requests
        ]
        return self.wait_many(request_ids, timeout_sec=timeout_sec)


def test_synthesize_many_submits_all_before_first_wait() -> None:
    client = _TrackingClient()
    requests = [
        {"text": f"segment {index}", "output_path": Path(f"/tmp/out_{index}.wav")}
        for index in range(4)
    ]
    responses = client.synthesize_many(requests)
    assert len(responses) == 4
    submit_indices = [index for index, event in enumerate(client.events) if event[0] == "submit"]
    wait_indices = [index for index, event in enumerate(client.events) if event[0] == "wait"]
    assert len(submit_indices) == 4
    assert len(wait_indices) == 4
    assert max(submit_indices) < min(wait_indices)


def test_omnivoice_adapter_synthesize_batch_uses_synthesize_many(tmp_path: Path) -> None:
    client = _TrackingClient()
    batch_calls: list[int] = []

    def _synthesize_many(requests: list[dict], *, timeout_sec: float = 600.0) -> list[dict]:
        batch_calls.append(len(requests))
        for req in requests:
            _write_tone_wav(Path(req["output_path"]))
        return OmniVoiceWorkerClient.synthesize_many(client, requests, timeout_sec=timeout_sec)

    client.synthesize_many = _synthesize_many  # type: ignore[method-assign]

    adapter = OmniVoiceTtsAdapter(
        model="k2-fsa/OmniVoice",
        _client=client,
        settings={"omnivoice_fidelity_check_enabled": False, "tts_micro_batch_size": 4},
    )
    items = [
        {
            "text": f"Câu ngắn số {index}.",
            "output_path": tmp_path / f"tts_{index}.wav",
            "voice": "instruct:female, low pitch",
            "segment": {"index": index},
        }
        for index in range(4)
    ]
    adapter.synthesize_batch(items)

    assert batch_calls == [4]
    submit_indices = [index for index, event in enumerate(client.events) if event[0] == "submit"]
    wait_indices = [index for index, event in enumerate(client.events) if event[0] == "wait"]
    assert max(submit_indices) < min(wait_indices)


def test_omnivoice_adapter_synthesize_batch_preserves_order_with_chunked_segment(
    tmp_path: Path,
) -> None:
    client = _TrackingClient()
    synthesize_calls: list[str] = []
    batch_sizes: list[int] = []

    def _synthesize_many(requests: list[dict], *, timeout_sec: float = 600.0) -> list[dict]:
        batch_sizes.append(len(requests))
        for req in requests:
            _write_tone_wav(Path(req["output_path"]))
        return OmniVoiceWorkerClient.synthesize_many(client, requests, timeout_sec=timeout_sec)

    def _synthesize(**kwargs) -> dict:
        synthesize_calls.append(str(kwargs["text"])[:20])
        _write_tone_wav(Path(kwargs["output_path"]))
        return {"ok": True}

    client.synthesize_many = _synthesize_many  # type: ignore[method-assign]
    client.synthesize = _synthesize  # type: ignore[method-assign]

    long_text = " ".join(
        ["Đây là một câu tiếng Việt đủ dài để vượt ngưỡng external chunk khi bật chunking."]
        * 12
    )
    adapter = OmniVoiceTtsAdapter(
        model="k2-fsa/OmniVoice",
        _client=client,
        settings={
            "omnivoice_fidelity_check_enabled": False,
            "omnivoice_external_chunking_enabled": True,
            "omnivoice_long_text_threshold": 120,
        },
    )
    items = [
        {
            "text": "Câu một.",
            "output_path": tmp_path / "tts_1.wav",
            "voice": "instruct:female, low pitch",
            "segment": {"index": 1},
        },
        {
            "text": long_text,
            "output_path": tmp_path / "tts_2.wav",
            "voice": "instruct:female, low pitch",
            "segment": {"index": 2},
        },
        {
            "text": "Câu ba.",
            "output_path": tmp_path / "tts_3.wav",
            "voice": "instruct:female, low pitch",
            "segment": {"index": 3},
        },
    ]
    adapter.synthesize_batch(items)

    assert batch_sizes == [1, 1]
    assert len(synthesize_calls) >= 1
    assert (tmp_path / "tts_1.wav").is_file()
    assert (tmp_path / "tts_2.wav").is_file()
    assert (tmp_path / "tts_3.wav").is_file()


def test_omnivoice_adapter_reports_worker_batch_size(tmp_path: Path) -> None:
    client = _TrackingClient()
    batch_sizes: list[int] = []

    def _synthesize_many(requests: list[dict], *, timeout_sec: float = 600.0) -> list[dict]:
        batch_sizes.append(len(requests))
        responses = []
        for index, req in enumerate(requests):
            _write_tone_wav(Path(req["output_path"]))
            responses.append(
                {
                    "ok": True,
                    "id": f"req-{index + 1}",
                    "perf": {"worker_batch_size": len(requests), "model_synthesis_ms": 10.0},
                }
            )
        return responses

    client.synthesize_many = _synthesize_many  # type: ignore[method-assign]

    adapter = OmniVoiceTtsAdapter(
        model="k2-fsa/OmniVoice",
        _client=client,
        settings={"omnivoice_fidelity_check_enabled": False, "omnivoice_tts_include_perf": True},
    )
    items = [
        {
            "text": f"Câu {index}.",
            "output_path": tmp_path / f"tts_{index}.wav",
            "voice": "instruct:female, low pitch",
            "segment": {"index": index},
        }
        for index in range(4)
    ]
    adapter.synthesize_batch(items)
    diagnostics = adapter.last_batch_diagnostics
    assert diagnostics["submitted_block_size"] == 4
    assert diagnostics["max_worker_batch_size"] == 4
    assert diagnostics["mode"] == "omnivoice_queued_batch"


def test_omnivoice_adapter_splits_direct_block_by_configured_batch_size(tmp_path: Path) -> None:
    client = _TrackingClient()
    batch_calls: list[int] = []

    def _synthesize_many(requests: list[dict], *, timeout_sec: float = 600.0) -> list[dict]:
        batch_calls.append(len(requests))
        for req in requests:
            _write_tone_wav(Path(req["output_path"]))
        return OmniVoiceWorkerClient.synthesize_many(client, requests, timeout_sec=timeout_sec)

    client.synthesize_many = _synthesize_many  # type: ignore[method-assign]

    adapter = OmniVoiceTtsAdapter(
        model="k2-fsa/OmniVoice",
        _client=client,
        settings={
            "omnivoice_fidelity_check_enabled": False,
            "tts_micro_batch_size": 4,
        },
    )
    items = [
        {
            "text": f"Câu {index}.",
            "output_path": tmp_path / f"tts_{index}.wav",
            "voice": "instruct:female, low pitch",
            "segment": {"index": index},
        }
        for index in range(25)
    ]
    adapter.synthesize_batch(items)
    assert batch_calls == [4, 4, 4, 4, 4, 4, 1]
    diagnostics = adapter.last_batch_diagnostics
    assert diagnostics["configured_batch_size"] == 4
    assert diagnostics["explicit_batch_sizes"] == [4, 4, 4, 4, 4, 4, 1]
    assert diagnostics["max_worker_batch_size"] <= 4


def test_omnivoice_adapter_has_synthesize_batch() -> None:
    adapter = OmniVoiceTtsAdapter(model="k2-fsa/OmniVoice", settings={})
    assert callable(getattr(adapter, "synthesize_batch", None))
