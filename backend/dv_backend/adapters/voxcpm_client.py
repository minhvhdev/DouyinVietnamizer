"""Client for the long-lived VoxCPM2 worker.

The client is responsible for:
* Locating the Python executable for the JSONL worker process.
* Spawning the worker subprocess and managing its lifecycle.
* Forwarding synthesize requests and reading responses.
* Re-spawning the worker transparently if it dies (e.g. OOM, crash).
* Honouring cancellation via the ``JobRunner`` process registry.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from ..errors import AppError
from ..models import ErrorInfo
from ..voxcpm_env import resolve_voxcpm_python

WORKER_SCRIPT = "dv_backend.adapters.voxcpm_worker"
DEFAULT_MAX_BATCH = 4
DEFAULT_FLUSH_MS = 150
PROCESS_READY_TIMEOUT_SEC = 600.0
RESPONSE_QUEUE_GET_TIMEOUT_SEC = 0.1
STARTUP_PING_TIMEOUT_SEC = 30.0
IDLE_SHUTDOWN_SEC = 300.0
PING_INTERVAL_SEC = 60.0


def _json_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True) + "\n"


class VoxCPMWorkerClient:
    """Manages a single worker subprocess and a request/response correlation queue.

    A ``threading.Lock`` serializes writes to the worker stdin pipe (the
    protocol is line-based so interleaved writes would corrupt the stream).
    A dedicated reader thread populates a per-request response queue and a
    shared ``alive`` flag.
    """

    def __init__(
        self,
        *,
        data_dir: Path,
        model: str,
        device: str,
        num_steps: int,
        max_batch: int = DEFAULT_MAX_BATCH,
        flush_ms: int = DEFAULT_FLUSH_MS,
        idle_shutdown_sec: float = IDLE_SHUTDOWN_SEC,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.model = model
        self.device = device
        self.num_steps = max(4, min(64, int(num_steps)))
        self.max_batch = max(1, int(max_batch))
        self.flush_ms = max(20, int(flush_ms))
        self.idle_shutdown_sec = float(idle_shutdown_sec)
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending: dict[str, queue.Queue] = {}
        self._response_queues: dict[str, queue.Queue] = {}
        self._pending_lock = threading.Lock()
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._last_used = time.perf_counter()
        self._last_ping = 0.0
        self._start_error: str | None = None
        self._closed = False

    # ------------------------------------------------------------------ lifecycle

    def _spawn_locked(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        python = resolve_voxcpm_python()
        cmd = [
            str(python),
            "-m",
            WORKER_SCRIPT,
            "--max-batch",
            str(self.max_batch),
            "--flush-ms",
            str(self.flush_ms),
            "--idle-timeout-sec",
            str(self.idle_shutdown_sec),
        ]
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        # The worker runs in an isolated venv that does not have dv_backend
        # installed. Inject the *parent* of the dv_backend package directory
        # into PYTHONPATH so the worker can import dv_backend.adapters.voxcpm_worker
        # regardless of the parent cwd (e.g. portable runtime with nested
        # dv_backend layout where cwd=backend/dv_backend/ has no __init__.py).
        import dv_backend as _dv_backend_pkg
        worker_pythonpath = str(Path(_dv_backend_pkg.__file__).resolve().parent.parent)
        env["PYTHONPATH"] = (
            worker_pythonpath
            if not env.get("PYTHONPATH")
            else worker_pythonpath + os.pathsep + env["PYTHONPATH"]
        )
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                env=env,
            )
        except FileNotFoundError as exc:
            raise AppError(
                400,
                ErrorInfo(
                    code="VOXCPM_NOT_INSTALLED",
                    message="VoxCPM environment is not installed.",
                    action="Run 'python scripts/setup_voxcpm.py' and install voxcpm2-cli from llama.cpp-omni.",
                    detail=str(exc),
                ),
            ) from exc
        except OSError as exc:
            raise AppError(
                502,
                ErrorInfo(
                    code="VOXCPM_TTS_FAILED",
                    message="VoxCPM2 inference subprocess failed to start.",
                    action="Verify the isolated VoxCPM virtualenv is configured correctly.",
                    detail=str(exc),
                    retryable=True,
                ),
            ) from exc

        self._reader_thread = threading.Thread(
            target=self._reader_loop, name="voxcpm-worker-reader", daemon=True
        )
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(
            target=self._stderr_loop, name="voxcpm-worker-stderr", daemon=True
        )
        self._stderr_thread.start()
        self._wait_ready()
        self._last_used = time.perf_counter()

    def _terminate_proc(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass

    def _wait_ready(self) -> None:
        if self._proc is None:
            return
        deadline = time.perf_counter() + STARTUP_PING_TIMEOUT_SEC
        request_id = f"startup-{uuid.uuid4().hex}"
        q: queue.Queue = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[request_id] = q
        message = _json_line({"id": request_id, "op": "ping"})
        try:
            assert self._proc.stdin is not None
            with self._write_lock:
                self._proc.stdin.write(message)
                self._proc.stdin.flush()
        except Exception as exc:  # noqa: BLE001
            with self._pending_lock:
                self._pending.pop(request_id, None)
            self._terminate_proc()
            raise AppError(
                502,
                ErrorInfo(
                    code="VOXCPM_TTS_FAILED",
                    message="Worker did not respond to startup ping.",
                    detail=str(exc),
                    retryable=True,
                ),
            ) from exc
        try:
            response = q.get(timeout=max(1.0, deadline - time.perf_counter()))
        except queue.Empty as exc:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            self._terminate_proc()
            raise AppError(
                502,
                ErrorInfo(
                    code="VOXCPM_TTS_FAILED",
                    message="VoxCPM worker failed to start within timeout.",
                    retryable=True,
                ),
            ) from exc
        if not response.get("ok"):
            self._terminate_proc()
            self._raise_worker_error(response)

    def _drain_pending_with_error(self, request_id: str, *, code: str, message: str, detail: str | None = None, retryable: bool = True) -> None:
        with self._pending_lock:
            q = self._pending.pop(request_id, None)
        if q is None:
            return
        try:
            q.put_nowait(
                {
                    "id": request_id,
                    "ok": False,
                    "code": code,
                    "message": message,
                    "detail": detail,
                    "retryable": retryable,
                }
            )
        except queue.Full:
            pass

    def _reader_loop(self) -> None:
        if self._proc is None or self._proc.stdout is None:
            return
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            req_id = payload.get("id")
            if not req_id:
                continue
            with self._pending_lock:
                q = self._pending.pop(req_id, None)
            if q is None:
                continue
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass
        self._fail_pending("VoxCPM worker exited unexpectedly.", retryable=True)

    def _stderr_loop(self) -> None:
        if self._proc is None or self._proc.stderr is None:
            return
        for line in self._proc.stderr:
            if not line:
                continue
            sys.stderr.write(f"[voxcpm-worker] {line}")
            sys.stderr.flush()

    def _fail_pending(self, message: str, *, retryable: bool) -> None:
        with self._pending_lock:
            pending = list(self._pending.items())
            self._pending.clear()
        for _req_id, q in pending:
            try:
                q.put_nowait(
                    {
                        "ok": False,
                        "code": "VOXCPM_WORKER_DIED",
                        "message": message,
                        "retryable": retryable,
                    }
                )
            except queue.Full:
                pass

    def _ensure_alive(self) -> None:
        with self._lock:
            if self._closed:
                raise AppError(
                    502,
                    ErrorInfo(
                        code="VOXCPM_TTS_FAILED",
                        message="VoxCPM client is closed.",
                        retryable=True,
                    ),
                )
            if self._proc is None or self._proc.poll() is not None:
                self._spawn_locked()
            self._last_used = time.perf_counter()

    def _keep_alive(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        now = time.perf_counter()
        if now - self._last_ping < PING_INTERVAL_SEC:
            return
        if now - self._last_used < PING_INTERVAL_SEC:
            return
        self._last_ping = now
        try:
            assert self._proc.stdin is not None
            with self._write_lock:
                self._proc.stdin.write(_json_line({"id": f"ping-{uuid.uuid4().hex}", "op": "ping"}))
                self._proc.stdin.flush()
        except Exception:
            pass

    # ------------------------------------------------------------------ public API

    def register_with_runner(self, runner: Any) -> None:
        if runner is None or self._proc is None:
            return
        if hasattr(runner, "register_process"):
            try:
                runner.register_process("_voxcpm_worker", self._proc)
            except Exception:
                pass

    def _build_synthesize_request(
        self,
        *,
        text: str,
        output_path: Path,
        prompt_wav_path: str | None,
        prompt_text: str | None,
        voice_design: str | None,
        cfg_value: float,
        inference_timesteps: int,
        cache_key: str | None = None,
        reference_wav_path: str | None = None,
        anchor_text: str | None = None,
        mode: str = "design",
    ) -> tuple[str, dict[str, Any], queue.Queue]:
        output_path = Path(output_path)
        if not text or not text.strip():
            raise AppError(
                422,
                ErrorInfo(
                    code="EMPTY_TTS_TEXT",
                    message="Cannot synthesize empty narration text.",
                    action="Verify translation output for this segment.",
                ),
            )
        request_id = f"req-{uuid.uuid4().hex}"
        response_q: queue.Queue = queue.Queue(maxsize=1)
        request: dict[str, Any] = {
            "id": request_id,
            "op": "synthesize",
            "text": text,
            "output_path": str(output_path),
            "model": self.model,
            "device": self.device,
            "inference_timesteps": int(inference_timesteps),
            "cfg_value": float(cfg_value),
            "mode": mode,
            "reference_wav_path": reference_wav_path,
            "prompt_wav_path": prompt_wav_path,
            "prompt_text": prompt_text,
            "anchor_text": anchor_text,
            "voice_design": voice_design,
        }
        if cache_key:
            request["cache_key"] = cache_key
        return request_id, request, response_q

    def submit(
        self,
        *,
        text: str,
        output_path: Path,
        prompt_wav_path: str | None,
        prompt_text: str | None,
        voice_design: str | None,
        cfg_value: float,
        inference_timesteps: int,
        cache_key: str | None = None,
        reference_wav_path: str | None = None,
        anchor_text: str | None = None,
        mode: str = "design",
    ) -> str:
        return self.submit_batch(
            [
                {
                    "text": text,
                    "output_path": output_path,
                    "prompt_wav_path": prompt_wav_path,
                    "prompt_text": prompt_text,
                    "voice_design": voice_design,
                    "cfg_value": cfg_value,
                    "inference_timesteps": inference_timesteps,
                    "cache_key": cache_key,
                    "reference_wav_path": reference_wav_path,
                    "anchor_text": anchor_text,
                    "mode": mode,
                }
            ]
        )[0]

    def submit_batch(self, requests: list[dict[str, Any]]) -> list[str]:
        if not requests:
            return []
        self._ensure_alive()
        self._keep_alive()
        entries = [self._build_synthesize_request(**request) for request in requests]
        if len(entries) > 1:
            for _request_id, request, _response_q in entries:
                request["batch_size_hint"] = len(entries)
        request_ids = [entry[0] for entry in entries]
        with self._pending_lock:
            for request_id, _request, response_q in entries:
                self._pending[request_id] = response_q
                self._response_queues[request_id] = response_q
        try:
            assert self._proc is not None and self._proc.stdin is not None
            with self._write_lock:
                for _request_id, request, _response_q in entries:
                    self._proc.stdin.write(_json_line(request))
                self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            with self._pending_lock:
                for request_id in request_ids:
                    self._pending.pop(request_id, None)
                    self._response_queues.pop(request_id, None)
            raise AppError(
                502,
                ErrorInfo(
                    code="VOXCPM_TTS_FAILED",
                    message="VoxCPM worker is not accepting requests.",
                    action="Verify the isolated VoxCPM virtualenv and GPU availability.",
                    detail=str(exc),
                    retryable=True,
                ),
            ) from exc
        return request_ids

    def _raise_worker_error(self, response: dict[str, Any]) -> None:
        code = response.get("code") or "VOXCPM_TTS_FAILED"
        if code == "VOXCPM_BAD_REQUEST":
            raise AppError(
                422,
                ErrorInfo(
                    code=code,
                    message=response.get("message") or "VoxCPM worker rejected the request.",
                    action="Verify the request parameters and the cloning mode configuration.",
                    detail=response.get("detail"),
                    retryable=False,
                ),
            )
        raise AppError(
            502,
            ErrorInfo(
                code=code,
                message=response.get("message") or "VoxCPM2 could not generate narration.",
                action=(
                    "Check VoxCPM2 model, GPU availability, and reference audio settings. "
                    "Run 'python scripts/setup_voxcpm.py' if the isolated env is missing."
                ),
                detail=response.get("detail"),
                retryable=bool(response.get("retryable", True)),
            ),
        )

    def wait(self, request_id: str) -> dict[str, Any]:
        with self._pending_lock:
            response_q = self._response_queues.get(request_id)
        if response_q is None:
            raise AppError(
                504,
                ErrorInfo(
                    code="VOXCPM_TIMEOUT",
                    message="VoxCPM worker response queue is missing.",
                    action="Retry the TTS step.",
                    retryable=True,
                ),
            )
        try:
            response = response_q.get(timeout=PROCESS_READY_TIMEOUT_SEC)
        except queue.Empty:
            with self._pending_lock:
                self._pending.pop(request_id, None)
                self._response_queues.pop(request_id, None)
            raise AppError(
                504,
                ErrorInfo(
                    code="VOXCPM_TIMEOUT",
                    message="VoxCPM worker did not respond within the expected time.",
                    action="Check the worker log and GPU availability, then retry.",
                    retryable=True,
                ),
            )
        with self._pending_lock:
            self._response_queues.pop(request_id, None)
        self._last_used = time.perf_counter()
        if response.get("ok"):
            return response
        self._raise_worker_error(response)
        return response

    def wait_batch(self, request_ids: list[str]) -> list[dict[str, Any]]:
        return [self.wait(request_id) for request_id in request_ids]

    def synthesize(
        self,
        *,
        text: str,
        output_path: Path,
        prompt_wav_path: str | None,
        prompt_text: str | None,
        voice_design: str | None,
        cfg_value: float,
        inference_timesteps: int,
        cache_key: str | None = None,
        reference_wav_path: str | None = None,
        anchor_text: str | None = None,
        mode: str = "design",
    ) -> dict[str, Any]:
        request_id = self.submit(
            text=text,
            output_path=output_path,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
            voice_design=voice_design,
            reference_wav_path=reference_wav_path,
            anchor_text=anchor_text,
            mode=mode,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            cache_key=cache_key,
        )
        return self.wait(request_id)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin is not None and proc.poll() is None:
                with self._write_lock:
                    proc.stdin.write(_json_line({"op": "shutdown"}))
                    proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.wait(timeout=5.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


_client_lock = threading.Lock()
_clients: dict[str, "VoxCPMWorkerClient"] = {}


def acquire_client(
    *,
    data_dir: Path,
    model: str,
    device: str,
    num_steps: int,
    max_batch: int = DEFAULT_MAX_BATCH,
    flush_ms: int = DEFAULT_FLUSH_MS,
) -> "VoxCPMWorkerClient":
    """Return a shared worker client keyed by runtime and batch settings."""
    max_batch = max(1, int(max_batch or DEFAULT_MAX_BATCH))
    flush_ms = max(20, int(flush_ms or DEFAULT_FLUSH_MS))
    key = f"{model}|{device}|{int(num_steps)}|batch={max_batch}|flush={flush_ms}"
    with _client_lock:
        client = _clients.get(key)
        if client is None:
            client = VoxCPMWorkerClient(
                data_dir=data_dir,
                model=model,
                device=device,
                num_steps=num_steps,
                max_batch=max_batch,
                flush_ms=flush_ms,
            )
            _clients[key] = client
        return client


def release_all_clients() -> None:
    with _client_lock:
        clients = list(_clients.values())
        _clients.clear()
    for client in clients:
        client.close()


def client_debug_snapshot() -> dict[str, object]:
    with _client_lock:
        items = [
            {
                "key": key,
                "closed": client._closed,
            }
            for key, client in sorted(_clients.items())
        ]
    return {
        "count": len(items),
        "clients": items,
    }
