"""Long-lived OmniVoice inference worker.

This script runs *inside* the isolated ``.venv-omnivoice`` virtualenv so the
heavy ``torch`` / ``omnivoice`` stack stays isolated from the main backend
environment. The backend process spawns this worker as a child process and
exchanges newline-delimited JSON messages on stdin/stdout.

Request (backend -> worker)::

    {"id": "req-1", "op": "synthesize", "text": "Xin chào",
     "ref_audio": "/abs/path/ref.wav", "ref_text": "hello",
     "instruct": null, "num_step": 32, "model": "k2-fsa/OmniVoice",
     "device": "cuda:0", "output_path": "/abs/path/out.wav"}

Response (worker -> backend)::

    {"id": "req-1", "ok": true, "output_path": "/abs/path/out.wav",
     "duration_sec": 1.42, "sample_rate": 24000}
    {"id": "req-1", "ok": false, "code": "...", "message": "...",
     "retryable": true}

Batching: requests sharing ``(model, device, ref_audio, ref_text, instruct,
num_step)`` are coalesced into a single batched inference call. The first
request in the group carries the actual reference audio / instruct used
for sampling; the remaining items reuse the same prosody profile but feed
their own text.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "k2-fsa/OmniVoice"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_NUM_STEP = 32
DEFAULT_FLUSH_MS = 150
DEFAULT_MAX_BATCH = 4
DEFAULT_SAMPLE_RATE = 24000


def _log(message: str) -> None:
    sys.stderr.write(f"[omnivoice-worker] {message}\n")
    sys.stderr.flush()


def _emit(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


class OmniVoiceEngine:
    """Lazy wrapper around the ``omnivoice`` package.

    The actual import is deferred until first use so that ``--health-check``
    style invocations do not require torch to be available.
    """

    def __init__(self) -> None:
        self._model_id: str | None = None
        self._device: str | None = None
        self._engine: Any = None

    def get(self, *, model: str, device: str) -> Any:
        if self._engine is not None and self._model_id == model and self._device == device:
            return self._engine
        self._engine = None
        self._model_id = model
        self._device = device
        from omnivoice import OmniVoice  # type: ignore

        _log(f"Loading OmniVoice model={model} device={device}")
        with contextlib.redirect_stdout(sys.stderr):
            engine = OmniVoice.from_pretrained(model)
            if device:
                engine = engine.to(device)
            engine.eval()
        _log("OmniVoice model ready")
        self._engine = engine
        return self._engine

    def release(self) -> None:
        self._engine = None
        self._model_id = None
        self._device = None
        try:
            import gc
            import torch  # type: ignore
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _coalesce(requests: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[tuple, list[dict[str, Any]]] = {}
    order: list[tuple] = []
    for req in requests:
        key = (
            req.get("model") or DEFAULT_MODEL,
            req.get("device") or DEFAULT_DEVICE,
            req.get("ref_audio") or "",
            req.get("ref_text") or "",
            req.get("instruct") or "",
            int(req.get("num_step") or DEFAULT_NUM_STEP),
        )
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(req)
    return [groups[key] for key in order]


def _write_wav(output_path: str, audio: Any, sample_rate: int) -> float:
    import numpy as np
    import wave

    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    arr = np.asarray(audio).reshape(-1)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 1.0:
        arr = arr / peak
    pcm = np.clip(arr, -1.0, 1.0)
    pcm_int = (pcm * 32767.0).astype(np.int16)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm_int.tobytes())
    return float(arr.shape[0]) / float(sample_rate) if arr.size else 0.0


def _resolve_voice(req: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    return (
        req.get("ref_audio") or None,
        req.get("ref_text") or None,
        req.get("instruct") or None,
    )


def _generate(
    engine_obj: Any,
    texts: list[str],
    *,
    ref_audio,
    ref_text,
    instruct,
    num_step,
) -> list[Any]:
    from omnivoice import OmniVoiceGenerationConfig  # type: ignore

    texts = [
        "".join(
            character
            for character in str(text or "")
            if not (0xD800 <= ord(character) <= 0xDFFF)
        )
        for text in texts
    ]
    generation_config = OmniVoiceGenerationConfig(num_step=int(num_step))
    kwargs: dict[str, Any] = {
        "generation_config": generation_config,
        "language": "vi",
    }
    if ref_audio:
        kwargs["ref_audio"] = ref_audio
        if ref_text:
            kwargs["ref_text"] = ref_text
    if instruct:
        kwargs["instruct"] = instruct
    _log(f"Generating {len(texts)} segment(s), num_step={num_step}")
    with contextlib.redirect_stdout(sys.stderr):
        audios = engine_obj.generate(texts, **kwargs)
    _log(f"Generated {len(audios)} segment(s)")
    return audios


def _run_batch(engine_obj: Any, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    leader = batch[0]
    ref_audio, ref_text, instruct = _resolve_voice(leader)
    num_step = int(leader.get("num_step") or DEFAULT_NUM_STEP)

    audios = _generate(
        engine_obj,
        [r["text"] for r in batch],
        ref_audio=ref_audio,
        ref_text=ref_text,
        instruct=instruct,
        num_step=num_step,
    )

    for req, audio in zip(batch, audios):
        try:
            duration = _write_wav(req["output_path"], audio, DEFAULT_SAMPLE_RATE)
        except Exception as exc:  # noqa: BLE001
            responses.append(
                {"id": req["id"], "ok": False, "code": "OMNIVOICE_WRITE_FAILED", "message": f"Failed to write WAV: {exc}", "retryable": True}
            )
            continue
        responses.append(
            {
                "id": req["id"],
                "ok": True,
                "output_path": req["output_path"],
                "duration_sec": round(duration, 3),
                "sample_rate": DEFAULT_SAMPLE_RATE,
            }
        )
    return responses


def _read_request(line: str) -> dict[str, Any] | None:
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        _emit({"id": None, "ok": False, "code": "BAD_REQUEST", "message": str(exc)})
        return None


def serve(*, max_batch: int = DEFAULT_MAX_BATCH, flush_ms: int = DEFAULT_FLUSH_MS, idle_timeout_sec: float = 0.0) -> int:
    _ = max_batch, flush_ms, idle_timeout_sec
    engine = OmniVoiceEngine()

    def _handle_synthesize(request: dict[str, Any]) -> None:
        try:
            model = request.get("model") or DEFAULT_MODEL
            device = request.get("device") or DEFAULT_DEVICE
            request_text = "".join(
                character
                for character in str(request.get("text") or "")
                if not (0xD800 <= ord(character) <= 0xDFFF)
            )
            request["text"] = request_text
            _log(
                f"Handling synthesize id={request.get('id')} model={model} device={device} "
                f"text_len={len(request_text)} "
                f"ref_audio={bool(request.get('ref_audio'))} ref_text_len={len(str(request.get('ref_text') or ''))} "
                f"instruct={bool(request.get('instruct'))}"
            )
            engine_obj = engine.get(model=model, device=device)
            for response in _run_batch(engine_obj, [request]):
                _emit(response)
        except Exception as exc:  # noqa: BLE001
            _log(f"Synthesize failed: {exc!r}")
            traceback.print_exc(file=sys.stderr)
            _emit(
                {
                    "id": request.get("id"),
                    "ok": False,
                    "code": "OMNIVOICE_INFERENCE_FAILED",
                    "message": str(exc),
                    "retryable": True,
                }
            )

    _log("Worker ready for JSONL requests")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = _read_request(line)
        if request is None:
            continue
        op = request.get("op") or "synthesize"
        if op == "shutdown":
            _log("Worker shutdown requested")
            return 0
        if op == "ping":
            _emit({"id": request.get("id"), "ok": True, "pong": True})
            continue
        if op != "synthesize":
            _emit({"id": request.get("id"), "ok": False, "code": "UNKNOWN_OP", "message": f"Unknown op: {op}"})
            continue
        _handle_synthesize(request)
    _log("Worker stdin closed")
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-batch", type=int, default=DEFAULT_MAX_BATCH)
    parser.add_argument("--flush-ms", type=int, default=DEFAULT_FLUSH_MS)
    parser.add_argument("--idle-timeout-sec", type=float, default=0.0)
    parser.add_argument("--health-check", action="store_true", help="Import omnivoice and exit; used by the runtime smoke test.")
    args = parser.parse_args()

    if args.health_check:
        try:
            import omnivoice  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            _emit({"ok": False, "code": "OMNIVOICE_NOT_INSTALLED", "message": str(exc)})
            return 1
        _emit({"ok": True, "code": "READY"})
        return 0

    return serve(max_batch=args.max_batch, flush_ms=args.flush_ms, idle_timeout_sec=args.idle_timeout_sec)


if __name__ == "__main__":
    raise SystemExit(main())
