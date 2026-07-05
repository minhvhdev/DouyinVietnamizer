"""Long-lived VoxCPM2 inference worker.

This script runs in the main backend Python environment and delegates synthesis
to the native ``voxcpm2-cli`` binary (llama.cpp-omni + GGUF weights). The
backend process spawns this worker as a child process and exchanges
newline-delimited JSON messages on stdin/stdout.

Request (backend -> worker)::

    {"id": "req-1", "op": "synthesize",
     "text": "(female, low pitch)Xin chào",
     "mode": "design" | "reference" | "ultimate",
     "reference_wav_path": "/abs/anchor.wav",
     "prompt_wav_path": "/abs/anchor.wav",
     "prompt_text": "transcript of anchor.wav",
     "anchor_text": "transcript of anchor.wav",
     "voice_design": "female, low pitch",
     "cfg_value": 2.0, "inference_timesteps": 10,
     "model": "gguf-q8", "device": "cuda:0",
     "output_path": "/abs/out.wav"}

Response (worker -> backend)::

    {"id": "req-1", "ok": true, "output_path": "/abs/out.wav",
     "duration_sec": 1.42, "sample_rate": 48000}
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import wave
from pathlib import Path
from typing import Any

from dv_backend.voxcpm_gguf import (
    GgufTtsServerSession,
    VOXCPM_DEFAULT_MODEL,
    VOXCPM_DEFAULT_SAMPLE_RATE,
    _cli_env,
    build_voxcpm_cli_command,
    normalize_voxcpm_model_id,
    resolve_llama_tts_server,
    resolve_voxcpm_cli,
    resolve_voxcpm_gguf_paths,
    is_gguf_runtime_ready,
)

DEFAULT_DEVICE = "cuda:0"
DEFAULT_INFERENCE_TIMESTEPS = 10
DEFAULT_CFG_VALUE = 2.0
DEFAULT_FLUSH_MS = 150
DEFAULT_MAX_BATCH = 4
DEFAULT_SYNTHESIS_TIMEOUT_SEC = 300.0
SUPPORTED_MODES = ("design", "reference", "ultimate")


def _log(message: str) -> None:
    sys.stderr.write(f"[voxcpm-worker] {message}\n")
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
        sample_rate = int(wav_file.getframerate() or VOXCPM_DEFAULT_SAMPLE_RATE)
        frames = int(wav_file.getnframes())
        if sample_rate <= 0 or frames <= 0:
            raise ValueError(f"Generated WAV is empty: {output_path}")
        return frames / float(sample_rate), sample_rate


class GgufVoxCPMEngine:
    """Keeps a hot llama-tts-server for GGUF inference across segments."""

    def __init__(self) -> None:
        self._model_id: str | None = None
        self._device: str | None = None
        self._baselm: Path | None = None
        self._acoustic: Path | None = None
        self._server: GgufTtsServerSession | None = None
        self._cli: Path | None = None

    def get(self, *, model: str, device: str) -> "GgufVoxCPMEngine":
        model_id = normalize_voxcpm_model_id(model)
        device_key = (device or DEFAULT_DEVICE).strip() or DEFAULT_DEVICE
        if (
            self._server is not None
            and self._model_id == model_id
            and self._device == device_key
            and self._baselm is not None
            and self._acoustic is not None
        ):
            return self

        self.release()
        self._baselm, self._acoustic = resolve_voxcpm_gguf_paths(model_id)
        self._model_id = model_id
        self._device = device_key
        self._server = GgufTtsServerSession()
        try:
            self._server.ensure_running(baselm=self._baselm, acoustic=self._acoustic, device=device_key)
            _log(
                f"TTS server ready baselm={self._baselm.name} "
                f"acoustic={self._acoustic.name} device={device_key}"
            )
        except Exception as exc:
            _log(f"TTS server unavailable, falling back to CLI: {exc!r}")
            self._server.shutdown()
            self._server = None
            self._cli = resolve_voxcpm_cli()
            _log(f"CLI fallback ready cli={self._cli.name}")
        return self

    def synthesize(
        self,
        text: str,
        output_path: str,
        *,
        prompt_wav_path: str | None,
        prompt_text: str | None,
        voice_design: str | None,
        inference_timesteps: int,
        cfg_value: float,
        reference_wav_path: str | None = None,
        anchor_text: str | None = None,
        mode: str = "design",
        timeout_sec: float = DEFAULT_SYNTHESIS_TIMEOUT_SEC,
    ) -> tuple[float, int]:
        if self._baselm is None or self._acoustic is None:
            raise RuntimeError("GGUF engine is not initialized.")

        resolved_mode = (mode or "design").strip().lower() or "design"
        if resolved_mode not in SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported VoxCPM clone mode: {mode!r}. Expected one of {SUPPORTED_MODES}."
            )
        anchor = reference_wav_path or prompt_wav_path
        use_server = self._server is not None and resolved_mode != "ultimate"
        if use_server:
            _log(
                f"Server synthesize mode={resolved_mode} timesteps={inference_timesteps} "
                f"cfg={cfg_value} reference_wav={bool(anchor)}"
            )
            return self._server.synthesize(
                text=text,
                output_path=output_path,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                reference_wav_path=anchor,
                mode=resolved_mode,
                timeout_sec=timeout_sec,
            )

        if resolved_mode == "ultimate" and self._cli is None:
            self._cli = resolve_voxcpm_cli()
        if self._cli is None:
            raise RuntimeError("Neither TTS server nor CLI fallback is available.")
        if resolved_mode == "ultimate":
            anchor_clean = (anchor_text or prompt_text or "").strip()
            if not anchor_clean:
                raise ValueError(
                    "Ultimate clone mode requires a non-empty anchor_text "
                    "matching the reference_wav_path transcript."
                )
            if len(anchor_clean) > 400:
                raise ValueError(
                    "Ultimate clone anchor transcript is too long for CLI fallback. "
                    "Use reference mode or shorten the .txt sidecar next to the voice WAV."
                )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = build_voxcpm_cli_command(
            self._cli,
            text=text,
            output_path=output_path,
            baselm=self._baselm,
            acoustic=self._acoustic,
            device=self._device or DEFAULT_DEVICE,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            reference_wav_path=reference_wav_path,
            prompt_wav_path=prompt_wav_path,
            prompt_text=anchor_text or prompt_text,
            mode=resolved_mode,
        )
        env = _cli_env(self._cli.parent)
        _log(f"Running voxcpm2-cli mode={resolved_mode}")
        started = time.perf_counter()
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(30.0, float(timeout_sec)),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            env=env,
        )
        elapsed = time.perf_counter() - started
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(
                f"voxcpm2-cli failed (exit {completed.returncode}) after {elapsed:.1f}s: {detail[-2000:]}"
            )
        if not out.is_file():
            raise RuntimeError(f"voxcpm2-cli did not write output: {output_path}")
        return _wav_duration(output_path)

    def release(self) -> None:
        if self._server is not None:
            self._server.shutdown()
        self._server = None
        self._cli = None
        self._baselm = None
        self._acoustic = None
        self._model_id = None
        self._device = None


def _iter_batch(engine: GgufVoxCPMEngine, batch: list[dict[str, Any]]):
    for req in batch:
        try:
            duration, sample_rate = engine.synthesize(
                _sanitize_text(req["text"]),
                req["output_path"],
                prompt_wav_path=req.get("prompt_wav_path"),
                prompt_text=req.get("prompt_text"),
                voice_design=req.get("voice_design"),
                inference_timesteps=int(req.get("inference_timesteps") or DEFAULT_INFERENCE_TIMESTEPS),
                cfg_value=float(req.get("cfg_value") or DEFAULT_CFG_VALUE),
                reference_wav_path=req.get("reference_wav_path"),
                anchor_text=req.get("anchor_text"),
                mode=req.get("mode") or "design",
            )
        except ValueError as exc:
            _log(f"Synthesize rejected: {exc!r}")
            yield {
                "id": req.get("id"),
                "ok": False,
                "code": "VOXCPM_BAD_REQUEST",
                "message": str(exc),
                "retryable": False,
            }
            continue
        except subprocess.TimeoutExpired as exc:
            _log(f"Synthesize timed out: {exc!r}")
            yield {
                "id": req.get("id"),
                "ok": False,
                "code": "VOXCPM_TIMEOUT",
                "message": "VoxCPM2 synthesis timed out.",
                "retryable": True,
            }
            continue
        except Exception as exc:  # noqa: BLE001
            _log(f"Synthesize failed: {exc!r}")
            traceback.print_exc(file=sys.stderr)
            yield {
                "id": req.get("id"),
                "ok": False,
                "code": "VOXCPM_INFERENCE_FAILED",
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

def _run_batch(engine: GgufVoxCPMEngine, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    for response in _iter_batch(engine, batch):
        responses.append(response)
    return responses


def _read_request(line: str) -> dict[str, Any] | None:
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        _emit({"id": None, "ok": False, "code": "BAD_REQUEST", "message": str(exc)})
        return None


def _batch_key(request: dict[str, Any]) -> tuple[str, str]:
    return (
        str(request.get("model") or VOXCPM_DEFAULT_MODEL),
        str(request.get("device") or DEFAULT_DEVICE),
    )


def _sanitize_synthesize_request(request: dict[str, Any]) -> dict[str, Any]:
    model = request.get("model") or VOXCPM_DEFAULT_MODEL
    device = request.get("device") or DEFAULT_DEVICE
    request_text = _sanitize_text(str(request.get("text") or ""))
    request["model"] = model
    request["device"] = device
    request["text"] = request_text
    return request


def _run_group(engine: GgufVoxCPMEngine, batch: list[dict[str, Any]]) -> None:
    if not batch:
        return
    model, device = _batch_key(batch[0])
    modes = sorted({str(req.get("mode") or "design") for req in batch})
    _log(f"Handling synthesize batch size={len(batch)} model={model} device={device} modes={','.join(modes)}")
    from dv_backend.gpu_lease import gpu_lease

    with gpu_lease(f"voxcpm-worker:{model}", device=device):
        engine.get(model=model, device=device)
        for response in _iter_batch(engine, batch):
            _emit(response)


def _run_synthesize_batch(engine: GgufVoxCPMEngine, batch: list[dict[str, Any]]) -> None:
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
    engine = GgufVoxCPMEngine()
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
        threading.Thread(target=_reader, name="voxcpm-worker-stdin-reader", daemon=True).start()
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
        help="Verify voxcpm2-cli and GGUF weights; used by the runtime smoke test.",
    )
    args = parser.parse_args()

    if args.health_check:
        if not is_gguf_runtime_ready():
            _emit(
                {
                    "ok": False,
                    "code": "VOXCPM_NOT_INSTALLED",
                    "message": "voxcpm2-cli or GGUF weights are missing.",
                }
            )
            return 1
        _emit({"ok": True, "code": "READY"})
        return 0

    return serve(max_batch=args.max_batch, flush_ms=args.flush_ms, idle_timeout_sec=args.idle_timeout_sec)


if __name__ == "__main__":
    raise SystemExit(main())
