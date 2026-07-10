"""Long-lived OmniVoice inference worker.

Runs in an isolated virtualenv with the ``omnivoice`` package installed.
The backend spawns this worker and exchanges newline-delimited JSON on stdin/stdout.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import traceback
import wave
from pathlib import Path
from typing import Any

from dv_backend.adapters.omnivoice_infer import plan_official_omnivoice_call
from dv_backend.omnivoice_env import OMNIVOICE_DEFAULT_MODEL, OMNIVOICE_DEFAULT_SAMPLE_RATE

DEFAULT_DEVICE = "cuda:0"
DEFAULT_NUM_STEP = 32
DEFAULT_SPEED = 1.0
DEFAULT_FLUSH_MS = 150
DEFAULT_MAX_BATCH = 4
DEFAULT_SYNTHESIS_TIMEOUT_SEC = 300.0


def _log(message: str) -> None:
    sys.stderr.write(f"[omnivoice-worker] {message}\n")
    sys.stderr.flush()


def _emit(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def _sanitize_text(text: str) -> str:
    return "".join(
        character
        for character in str(text or "")
        if not (0xD800 <= ord(character) <= 0xDFFF)
    )


def _wav_duration(output_path: str) -> tuple[float, int]:
    with wave.open(str(output_path), "rb") as wav_file:
        sample_rate = int(wav_file.getframerate() or OMNIVOICE_DEFAULT_SAMPLE_RATE)
        frames = int(wav_file.getnframes())
        if sample_rate <= 0 or frames <= 0:
            raise ValueError(f"Generated WAV is empty: {output_path}")
        return frames / float(sample_rate), sample_rate


class OmniVoiceEngine:
    """Keeps a hot OmniVoice model resident across segments."""

    def __init__(self) -> None:
        self._model = None
        self._model_id: str | None = None
        self._device: str | None = None
        self._clone_prompt_cache: dict[tuple[str, str], object] = {}

    def get(self, *, model: str, device: str) -> "OmniVoiceEngine":
        model_id = (model or OMNIVOICE_DEFAULT_MODEL).strip() or OMNIVOICE_DEFAULT_MODEL
        device_key = (device or DEFAULT_DEVICE).strip() or DEFAULT_DEVICE
        if self._model is not None and self._model_id == model_id and self._device == device_key:
            return self

        self.release()
        import torch
        from omnivoice import OmniVoice

        dtype = torch.float16
        if device_key in {"mps", "cpu"}:
            dtype = torch.float32
        _log(f"Loading OmniVoice model={model_id} device={device_key} dtype={dtype}")
        self._model = OmniVoice.from_pretrained(
            model_id,
            device_map=device_key,
            dtype=dtype,
        )
        self._model_id = model_id
        self._device = device_key
        self._clone_prompt_cache = {}
        _log("OmniVoice model ready")
        return self

    def _clone_prompt(self, *, ref_audio: str, ref_text: str, preprocess_prompt: bool):
        cache_key = (ref_audio, ref_text)
        cached = self._clone_prompt_cache.get(cache_key)
        if cached is not None:
            return cached
        assert self._model is not None
        prompt = self._model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            preprocess_prompt=preprocess_prompt,
        )
        self._clone_prompt_cache[cache_key] = prompt
        return prompt

    def synthesize(
        self,
        text: str,
        output_path: str,
        *,
        ref_audio: str | None,
        ref_text: str | None,
        anchor_text: str | None,
        instruct: str | None,
        num_step: int,
        speed: float,
        language_id: str | None,
        audio_chunk_threshold: float = 30.0,
        audio_chunk_duration: float = 15.0,
    ) -> tuple[float, int]:
        if self._model is None:
            raise RuntimeError("OmniVoice engine is not initialized.")

        import soundfile as sf
        from omnivoice import OmniVoiceGenerationConfig

        _ = ref_text
        plan = plan_official_omnivoice_call(
            text=text,
            speed=speed,
            num_step=num_step,
            language_id=language_id,
            ref_audio=ref_audio,
            anchor_text=anchor_text,
            instruct=instruct,
            audio_chunk_threshold=audio_chunk_threshold,
            audio_chunk_duration=audio_chunk_duration,
        )
        generation_config = OmniVoiceGenerationConfig(**dict(plan.pop("generation_config")))
        generate_kwargs: dict[str, Any] = {
            "text": plan.pop("text"),
            "generation_config": generation_config,
        }
        if "language" in plan:
            generate_kwargs["language"] = plan.pop("language")
        if "speed" in plan:
            generate_kwargs["speed"] = plan.pop("speed")
        if "instruct" in plan:
            generate_kwargs["instruct"] = plan.pop("instruct")

        clone_ref_audio = plan.pop("ref_audio", None)
        clone_ref_text = plan.pop("ref_text", None)
        if clone_ref_audio:
            generate_kwargs["voice_clone_prompt"] = self._clone_prompt(
                ref_audio=clone_ref_audio,
                ref_text=clone_ref_text or "",
                preprocess_prompt=generation_config.preprocess_prompt,
            )

        _log(
            f"Synthesize clone={bool(clone_ref_audio)} design={bool(generate_kwargs.get('instruct'))} "
            f"steps={generation_config.num_step} speed={generate_kwargs.get('speed', 1.0)}"
        )
        samples = self._model.generate(**generate_kwargs)[0]
        if samples is None or len(samples) == 0:
            raise RuntimeError("OmniVoice returned no audio.")
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), samples, OMNIVOICE_DEFAULT_SAMPLE_RATE)
        return _wav_duration(output_path)

    def release(self) -> None:
        self._model = None
        self._model_id = None
        self._device = None
        self._clone_prompt_cache = {}
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _iter_batch(engine: OmniVoiceEngine, batch: list[dict[str, Any]]):
    for req in batch:
        try:
            duration, sample_rate = engine.synthesize(
                _sanitize_text(req["text"]),
                req["output_path"],
                ref_audio=req.get("ref_audio"),
                ref_text=req.get("ref_text"),
                anchor_text=req.get("anchor_text"),
                instruct=req.get("instruct"),
                num_step=int(req.get("num_step") or DEFAULT_NUM_STEP),
                speed=float(req.get("speed") or DEFAULT_SPEED),
                language_id=req.get("language_id"),
                audio_chunk_threshold=float(req.get("audio_chunk_threshold") or 30.0),
                audio_chunk_duration=float(req.get("audio_chunk_duration") or 15.0),
            )
        except ValueError as exc:
            _log(f"Synthesize rejected: {exc!r}")
            yield {
                "id": req.get("id"),
                "ok": False,
                "code": "OMNIVOICE_BAD_REQUEST",
                "message": str(exc),
                "retryable": False,
            }
            continue
        except Exception as exc:  # noqa: BLE001
            _log(f"Synthesize failed: {exc!r}")
            traceback.print_exc(file=sys.stderr)
            yield {
                "id": req.get("id"),
                "ok": False,
                "code": "OMNIVOICE_INFERENCE_FAILED",
                "message": str(exc),
                "retryable": True,
            }
            continue
        yield {
            "id": req.get("id"),
            "ok": True,
            "output_path": req["output_path"],
            "duration_sec": round(duration, 3),
            "sample_rate": sample_rate,
        }


def _read_request(line: str) -> dict[str, Any] | None:
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        _emit({"id": None, "ok": False, "code": "BAD_REQUEST", "message": str(exc)})
        return None


def _batch_key(request: dict[str, Any]) -> tuple[str, str]:
    return (
        str(request.get("model") or OMNIVOICE_DEFAULT_MODEL),
        str(request.get("device") or DEFAULT_DEVICE),
    )


def _sanitize_synthesize_request(request: dict[str, Any]) -> dict[str, Any]:
    request["model"] = request.get("model") or OMNIVOICE_DEFAULT_MODEL
    request["device"] = request.get("device") or DEFAULT_DEVICE
    request["text"] = _sanitize_text(str(request.get("text") or ""))
    return request


def _run_group(engine: OmniVoiceEngine, batch: list[dict[str, Any]]) -> None:
    if not batch:
        return
    model, device = _batch_key(batch[0])
    _log(f"Handling synthesize batch size={len(batch)} model={model} device={device}")
    engine.get(model=model, device=device)
    for response in _iter_batch(engine, batch):
        _emit(response)


def _run_synthesize_batch(engine: OmniVoiceEngine, batch: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for request in batch:
        request = _sanitize_synthesize_request(request)
        groups.setdefault(_batch_key(request), []).append(request)
    for group in groups.values():
        _run_group(engine, group)


def serve(*, max_batch: int = DEFAULT_MAX_BATCH, flush_ms: int = DEFAULT_FLUSH_MS, idle_timeout_sec: float = 0.0) -> int:
    _ = idle_timeout_sec
    max_batch = max(1, int(max_batch or DEFAULT_MAX_BATCH))
    flush_sec = max(0.0, float(flush_ms or 0) / 1000.0)
    # Import CUDA/torch before starting stdin reader threads. On Windows,
    # initializing torch while another thread is active can deadlock.
    _log("Preloading torch and omnivoice runtime")
    import torch  # noqa: F401
    from omnivoice import OmniVoice  # noqa: F401

    engine = OmniVoiceEngine()
    messages: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def _reader() -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            request = _read_request(line)
            if request is not None:
                messages.put(request)
        messages.put(None)

    def _emit_unknown(request: dict[str, Any], op: str) -> None:
        _emit({"id": request.get("id"), "ok": False, "code": "UNKNOWN_OP", "message": f"Unknown op: {op}"})

    try:
        _log("Worker ready for JSONL requests")
        threading.Thread(target=_reader, name="omnivoice-worker-stdin-reader", daemon=True).start()
        while True:
            try:
                request = messages.get(timeout=float(idle_timeout_sec)) if idle_timeout_sec > 0 else messages.get()
            except queue.Empty:
                _log("Worker idle timeout reached")
                return 0
            if request is None:
                _log("Worker stdin closed")
                return 0
            op = request.get("op") or "synthesize"
            if op == "shutdown":
                _log("Worker shutdown requested")
                return 0
            if op == "ping":
                _emit({"id": request.get("id"), "ok": True, "pong": True})
                continue
            if op != "synthesize":
                _emit_unknown(request, op)
                continue

            batch = [request]
            deadline = time.perf_counter() + flush_sec
            while len(batch) < max_batch:
                timeout = max(0.0, deadline - time.perf_counter())
                if flush_sec <= 0.0 and batch:
                    break
                try:
                    next_request = messages.get(timeout=timeout)
                except queue.Empty:
                    break
                if next_request is None:
                    _run_synthesize_batch(engine, batch)
                    _log("Worker stdin closed")
                    return 0
                next_op = next_request.get("op") or "synthesize"
                if next_op == "synthesize":
                    batch.append(next_request)
                    continue
                _run_synthesize_batch(engine, batch)
                batch = []
                if next_op == "shutdown":
                    _log("Worker shutdown requested")
                    return 0
                if next_op == "ping":
                    _emit({"id": next_request.get("id"), "ok": True, "pong": True})
                else:
                    _emit_unknown(next_request, next_op)
                break
            if batch:
                _run_synthesize_batch(engine, batch)
    finally:
        engine.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-batch", type=int, default=DEFAULT_MAX_BATCH)
    parser.add_argument("--flush-ms", type=int, default=DEFAULT_FLUSH_MS)
    parser.add_argument("--idle-timeout-sec", type=float, default=0.0)
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Verify omnivoice import; used by the runtime smoke test.",
    )
    args = parser.parse_args()

    if args.health_check:
        try:
            import omnivoice  # noqa: F401
        except ImportError as exc:
            _emit(
                {
                    "ok": False,
                    "code": "OMNIVOICE_NOT_INSTALLED",
                    "message": f"omnivoice package is missing: {exc}",
                }
            )
            return 1
        _emit({"ok": True, "code": "READY"})
        return 0

    return serve(max_batch=args.max_batch, flush_ms=args.flush_ms, idle_timeout_sec=args.idle_timeout_sec)


if __name__ == "__main__":
    raise SystemExit(main())
