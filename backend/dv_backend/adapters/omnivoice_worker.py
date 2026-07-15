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
from dv_backend.audio_probe import diagnostics_enabled, probe_wav_path, probe_waveform
from dv_backend.omnivoice_env import OMNIVOICE_DEFAULT_MODEL, OMNIVOICE_DEFAULT_SAMPLE_RATE

DEFAULT_DEVICE = "cuda:0"
DEFAULT_NUM_STEP = 32
DEFAULT_SPEED = 1.0
DEFAULT_FLUSH_MS = 150
DEFAULT_MAX_BATCH = 4
DEFAULT_SYNTHESIS_TIMEOUT_SEC = 300.0
DEFAULT_INCOMPLETE_BATCH_TIMEOUT_SEC = 30.0


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

    def _clone_prompt(self, *, ref_audio: str, ref_text: str, preprocess_prompt: bool, request_id: str | None = None):
        cache_key = (ref_audio, ref_text)
        cached = self._clone_prompt_cache.get(cache_key)
        cache_hit = cached is not None
        if diagnostics_enabled():
            from dv_backend.omnivoice_diagnostics import describe_ref_conditioning, log_event

            log_event(
                "W0_ref_input",
                {
                    "request_id": request_id,
                    **describe_ref_conditioning(
                        ref_audio=ref_audio,
                        ref_text=ref_text,
                        cache_hit=cache_hit,
                    ),
                    "preprocess_prompt": bool(preprocess_prompt),
                },
            )
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
        include_perf: bool = False,
        request_id: str | None = None,
    ) -> tuple[float, int, dict[str, Any] | None]:
        if self._model is None:
            raise RuntimeError("OmniVoice engine is not initialized.")

        import soundfile as sf
        from omnivoice import OmniVoiceGenerationConfig

        _ = ref_text
        started = time.perf_counter()
        perf: dict[str, Any] = {}
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
        mode = "clone" if clone_ref_audio else ("instruct" if generate_kwargs.get("instruct") else "auto")
        corr_id = request_id or Path(output_path).stem
        clone_started = time.perf_counter()
        if clone_ref_audio:
            generate_kwargs["voice_clone_prompt"] = self._clone_prompt(
                ref_audio=clone_ref_audio,
                ref_text=clone_ref_text or "",
                preprocess_prompt=generation_config.preprocess_prompt,
                request_id=corr_id,
            )
        if include_perf:
            perf["clone_prompt_ms"] = round((time.perf_counter() - clone_started) * 1000, 2)

        _log(
            f"Synthesize clone={bool(clone_ref_audio)} design={bool(generate_kwargs.get('instruct'))} "
            f"steps={generation_config.num_step} speed={generate_kwargs.get('speed', 1.0)}"
        )
        if diagnostics_enabled():
            from dv_backend.omnivoice_diagnostics import describe_target_conditioning, log_event

            log_event(
                "W0_target_before_generate",
                {
                    "request_id": corr_id,
                    **describe_target_conditioning(
                        text=str(generate_kwargs.get("text") or ""),
                        mode=mode,
                    ),
                    "language": generate_kwargs.get("language"),
                    "has_clone_prompt": bool(generate_kwargs.get("voice_clone_prompt")),
                    "has_instruct": bool(generate_kwargs.get("instruct")),
                    "num_step": generation_config.num_step,
                    "speed": generate_kwargs.get("speed", 1.0),
                },
            )
        model_started = time.perf_counter()
        samples = self._model.generate(**generate_kwargs)[0]
        if include_perf:
            perf["model_synthesis_ms"] = round((time.perf_counter() - model_started) * 1000, 2)
        if samples is None or len(samples) == 0:
            raise RuntimeError("OmniVoice returned no audio.")

        if diagnostics_enabled():
            from dv_backend.omnivoice_diagnostics import log_event, write_waveform_artifact

            w1 = probe_waveform(samples, sample_rate=OMNIVOICE_DEFAULT_SAMPLE_RATE)
            log_event(
                "W1_generate_output",
                {
                    "request_id": corr_id,
                    "mode": mode,
                    "postprocess_output": bool(generation_config.postprocess_output),
                    "probe": w1,
                },
            )
            write_waveform_artifact(
                samples,
                sample_rate=OMNIVOICE_DEFAULT_SAMPLE_RATE,
                request_id=corr_id,
                label=f"{mode}_raw",
            )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        encode_started = time.perf_counter()
        sf.write(str(out), samples, OMNIVOICE_DEFAULT_SAMPLE_RATE)
        if include_perf:
            perf["encode_ms"] = round((time.perf_counter() - encode_started) * 1000, 2)
            perf["total_worker_ms"] = round((time.perf_counter() - started) * 1000, 2)
        duration, sample_rate = _wav_duration(output_path)

        if diagnostics_enabled():
            from dv_backend.audio_probe import diagnostics_dir
            from dv_backend.omnivoice_diagnostics import (
                copy_artifact,
                log_event,
                maybe_capture_clone_failure_bundle,
            )

            w2_probe = probe_wav_path(out)
            log_event(
                "W2_written_output",
                {
                    "request_id": corr_id,
                    "mode": mode,
                    "worker_duration_sec": duration,
                    "probe": w2_probe,
                },
            )
            copy_artifact(out, request_id=corr_id, label=f"{mode}_post")
            generate_artifact = diagnostics_dir() / f"{corr_id}_{mode}_raw.wav"
            w1_probe = probe_wav_path(generate_artifact) if generate_artifact.is_file() else None
            failure_stage = "generate_output"
            if w1_probe and w1_probe.get("speech_detected"):
                failure_stage = "written_output"
            maybe_capture_clone_failure_bundle(
                request_id=corr_id,
                mode=mode,
                ref_audio=clone_ref_audio,
                ref_text=clone_ref_text,
                target_text=generate_kwargs.get("text"),
                generate_output_path=generate_artifact if generate_artifact.is_file() else None,
                written_output_path=out,
                generate_probe=w1_probe,
                written_probe=w2_probe,
                failure_stage=failure_stage,
                language=generate_kwargs.get("language"),
                generation_config={
                    "num_step": generation_config.num_step,
                    "postprocess_output": bool(generation_config.postprocess_output),
                    "preprocess_prompt": bool(generation_config.preprocess_prompt),
                },
                model_identity={"model_id": self._model_id, "device": self._device},
            )

        return duration, sample_rate, perf if include_perf else None

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
    batch_size = len(batch)
    for req in batch:
        try:
            include_perf = bool(req.get("include_perf"))
            duration, sample_rate, perf = engine.synthesize(
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
                include_perf=include_perf,
                request_id=str(req.get("id") or "") or None,
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
        response = {
            "id": req.get("id"),
            "ok": True,
            "output_path": req["output_path"],
            "duration_sec": round(duration, 3),
            "sample_rate": sample_rate,
        }
        if include_perf and perf is not None:
            response["perf"] = {
                **perf,
                "worker_batch_size": batch_size,
                "flush_reason": req.get("_flush_reason"),
                "queue_wait_ms": req.get("_queue_wait_ms"),
            }
        yield response


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


def _annotate_batch_timing(batch: list[dict[str, Any]], *, flush_reason: str) -> None:
    processed_at = time.perf_counter()
    for request in batch:
        received_at = float(request.get("_received_at") or processed_at)
        request["_flush_reason"] = flush_reason
        request["_queue_wait_ms"] = round(max(0.0, (processed_at - received_at) * 1000.0), 2)


def _collect_synthesize_batch(
    messages: queue.Queue,
    first_request: dict[str, Any],
    *,
    max_batch: int,
    flush_sec: float,
    incomplete_batch_timeout_sec: float = DEFAULT_INCOMPLETE_BATCH_TIMEOUT_SEC,
) -> tuple[list[dict[str, Any]], str, dict[str, Any] | None]:
    """Collect synthesize requests. Returns (batch, flush_reason, deferred_request)."""
    first_request["_received_at"] = time.perf_counter()
    batch = [first_request]
    batch_id = first_request.get("batch_id")
    expected_size_raw = first_request.get("batch_size")

    if batch_id is not None and expected_size_raw is not None:
        expected_size = max(1, int(expected_size_raw))
        if expected_size == 1:
            return batch, "explicit_batch_complete", None
        deadline = time.perf_counter() + max(0.1, float(incomplete_batch_timeout_sec))
        while len(batch) < expected_size:
            timeout = max(0.0, deadline - time.perf_counter())
            try:
                next_request = messages.get(timeout=timeout)
            except queue.Empty:
                return batch, "batch_incomplete_timeout", None
            if next_request is None:
                return batch, "shutdown_or_eof", None
            next_request["_received_at"] = time.perf_counter()
            next_op = next_request.get("op") or "synthesize"
            if next_op != "synthesize":
                return batch, "explicit_batch_complete", next_request
            if str(next_request.get("batch_id")) != str(batch_id):
                return batch, "explicit_batch_complete", next_request
            batch.append(next_request)
        return batch, "explicit_batch_complete", None

    deadline = time.perf_counter() + flush_sec
    while len(batch) < max_batch:
        timeout = max(0.0, deadline - time.perf_counter())
        if flush_sec <= 0.0:
            break
        try:
            next_request = messages.get(timeout=timeout)
        except queue.Empty:
            return batch, "flush_timeout", None
        if next_request is None:
            return batch, "shutdown_or_eof", None
        next_request["_received_at"] = time.perf_counter()
        next_op = next_request.get("op") or "synthesize"
        if next_op == "synthesize":
            batch.append(next_request)
            continue
        flush_reason = "max_batch_reached" if len(batch) >= max_batch else "flush_timeout"
        return batch, flush_reason, next_request
    flush_reason = "max_batch_reached" if len(batch) >= max_batch else "flush_timeout"
    return batch, flush_reason, None


def _run_group(engine: OmniVoiceEngine, batch: list[dict[str, Any]], *, flush_reason: str) -> None:
    if not batch:
        return
    _annotate_batch_timing(batch, flush_reason=flush_reason)
    model, device = _batch_key(batch[0])
    _log(f"Handling synthesize batch size={len(batch)} model={model} device={device} flush={flush_reason}")
    engine.get(model=model, device=device)
    for response in _iter_batch(engine, batch):
        _emit(response)


def _run_synthesize_batch(
    engine: OmniVoiceEngine,
    batch: list[dict[str, Any]],
    *,
    flush_reason: str,
) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for request in batch:
        request = _sanitize_synthesize_request(request)
        groups.setdefault(_batch_key(request), []).append(request)
    for group in groups.values():
        _run_group(engine, group, flush_reason=flush_reason)


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

            deferred: dict[str, Any] | None = None
            while True:
                batch, flush_reason, deferred = _collect_synthesize_batch(
                    messages,
                    request,
                    max_batch=max_batch,
                    flush_sec=flush_sec,
                )
                if batch:
                    _run_synthesize_batch(engine, batch, flush_reason=flush_reason)
                if flush_reason == "shutdown_or_eof":
                    _log("Worker stdin closed")
                    return 0
                if deferred is None:
                    break
                if deferred.get("op") == "shutdown":
                    _log("Worker shutdown requested")
                    return 0
                if deferred.get("op") == "ping":
                    _emit({"id": deferred.get("id"), "ok": True, "pong": True})
                    request = messages.get()
                    if request is None:
                        _log("Worker stdin closed")
                        return 0
                    continue
                request = deferred
                if (request.get("op") or "synthesize") != "synthesize":
                    _emit_unknown(request, str(request.get("op") or "unknown"))
                    request = messages.get()
                    if request is None:
                        _log("Worker stdin closed")
                        return 0
                    continue
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
