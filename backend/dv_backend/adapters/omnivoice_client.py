"""Client for the long-lived OmniVoice worker.

The client is responsible for:
* Locating the Python executable in the isolated ``.venv-omnivoice``.
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
from ..omnivoice_env import resolve_omnivoice_python

WORKER_SCRIPT = "dv_backend.adapters.omnivoice_worker"
DEFAULT_MAX_BATCH = 4
DEFAULT_FLUSH_MS = 150
PROCESS_READY_TIMEOUT_SEC = 120.0
RESPONSE_QUEUE_GET_TIMEOUT_SEC = 0.1
STARTUP_PING_TIMEOUT_SEC = 30.0
IDLE_SHUTDOWN_SEC = 300.0
PING_INTERVAL_SEC = 60.0


def _default_cache_dir(data_dir: Path) -> Path | None:
    if os.environ.get("DV_OMNIVOICE_CACHE_DISABLED") == "1":
        return None
    override = os.environ.get("DV_OMNIVOICE_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(data_dir) / "cache" / "omnivoice"


class OmniVoiceWorkerClient:
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
        self.num_steps = max(8, min(64, int(num_steps)))
        self.max_batch = max(1, int(max_batch))
        self.flush_ms = max(20, int(flush_ms))
        self.idle_shutdown_sec = float(idle_shutdown_sec)
        self.cache_dir = _default_cache_dir(self.data_dir)
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending: dict[str, queue.Queue] = {}
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
        python = resolve_omnivoice_python()
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
                    code="OMNIVOICE_NOT_INSTALLED",
                    message="OmniVoice environment is not installed.",
                    action="Run 'python scripts/setup_omnivoice.py' in the backend folder.",
                    detail=str(exc),
                ),
            ) from exc
        except OSError as exc:
            raise AppError(
                502,
                ErrorInfo(
                    code="OMNIVOICE_TTS_FAILED",
                    message="OmniVoice inference subprocess failed to start.",
                    action="Verify the isolated OmniVoice virtualenv is configured correctly.",
                    detail=str(exc),
                    retryable=True,
                ),
            ) from exc

        self._reader_thread = threading.Thread(
            target=self._reader_loop, name="omnivoice-worker-reader", daemon=True
        )
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(
            target=self._stderr_loop, name="omnivoice-worker-stderr", daemon=True
        )
        self._stderr_thread.start()
        # Wait for the worker to confirm it is ready by pinging.
        self._wait_ready()
        self._last_used = time.perf_counter()

    def _wait_ready(self) -> None:
        if self._proc is None:
            return
        deadline = time.perf_counter() + STARTUP_PING_TIMEOUT_SEC
        request_id = f"startup-{uuid.uuid4().hex}"
        q: queue.Queue = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[request_id] = q
        message = json.dumps({"id": request_id, "op": "ping"}, ensure_ascii=False) + "\n"
        try:
            assert self._proc.stdin is not None
            with self._write_lock:
                self._proc.stdin.write(message)
                self._proc.stdin.flush()
        except Exception as exc:  # noqa: BLE001
            self._drain_pending_with_error(
                request_id,
                code="OMNIVOICE_TTS_FAILED",
                message="Worker did not respond to startup ping.",
                detail=str(exc),
                retryable=True,
            )
            return
        try:
            q.get(timeout=max(1.0, deadline - time.perf_counter()))
        except queue.Empty:
            self._drain_pending_with_error(
                request_id,
                code="OMNIVOICE_TTS_FAILED",
                message="OmniVoice worker failed to start within timeout.",
                retryable=True,
            )
            return

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
        # stdout closed -> worker exited
        self._fail_pending("OmniVoice worker exited unexpectedly.", retryable=True)

    def _stderr_loop(self) -> None:
        if self._proc is None or self._proc.stderr is None:
            return
        for line in self._proc.stderr:
            if not line:
                continue
            # Surface worker logs to the backend stderr at debug verbosity.
            sys.stderr.write(f"[omnivoice-worker] {line}")
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
                        "code": "OMNIVOICE_WORKER_DIED",
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
                        code="OMNIVOICE_TTS_FAILED",
                        message="OmniVoice client is closed.",
                        retryable=True,
                    ),
                )
            if self._proc is None or self._proc.poll() is not None:
                self._spawn_locked()
            self._last_used = time.perf_counter()

    def _keep_alive(self) -> None:
        # Periodic ping so we detect a dead worker before sending real work.
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
                self._proc.stdin.write(json.dumps({"id": f"ping-{uuid.uuid4().hex}", "op": "ping"}) + "\n")
                self._proc.stdin.flush()
        except Exception:
            # Reader thread will detect broken pipe and fail pending requests.
            pass

    # ------------------------------------------------------------------ public API

    def register_with_runner(self, runner: Any) -> None:
        if runner is None or self._proc is None:
            return
        if hasattr(runner, "register_process"):
            try:
                runner.register_process("_omnivoice_worker", self._proc)
            except Exception:
                pass

    def synthesize(
        self,
        *,
        text: str,
        output_path: Path,
        ref_audio: str | None,
        ref_text: str | None,
        instruct: str | None,
        cache_key: str | None = None,
    ) -> dict[str, Any]:
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
        self._ensure_alive()
        self._keep_alive()

        request_id = f"req-{uuid.uuid4().hex}"
        response_q: queue.Queue = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[request_id] = response_q

        request: dict[str, Any] = {
            "id": request_id,
            "op": "synthesize",
            "text": text,
            "output_path": str(output_path),
            "model": self.model,
            "device": self.device,
            "num_step": self.num_steps,
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "instruct": instruct,
        }
        if cache_key:
            request["cache_key"] = cache_key
        try:
            assert self._proc is not None and self._proc.stdin is not None
            with self._write_lock:
                self._proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise AppError(
                502,
                ErrorInfo(
                    code="OMNIVOICE_TTS_FAILED",
                    message="OmniVoice worker is not accepting requests.",
                    action="Verify the isolated OmniVoice virtualenv and GPU availability.",
                    detail=str(exc),
                    retryable=True,
                ),
            ) from exc

        try:
            response = response_q.get(timeout=PROCESS_READY_TIMEOUT_SEC)
        except queue.Empty:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise AppError(
                504,
                ErrorInfo(
                    code="OMNIVOICE_TIMEOUT",
                    message="OmniVoice worker did not respond within the expected time.",
                    action="Check the worker log and GPU availability, then retry.",
                    retryable=True,
                ),
            )

        self._last_used = time.perf_counter()
        if response.get("ok"):
            return response
        raise AppError(
            502,
            ErrorInfo(
                code=response.get("code") or "OMNIVOICE_TTS_FAILED",
                message=response.get("message") or "OmniVoice could not generate narration.",
                action=(
                    "Check OmniVoice model, GPU availability, and reference audio settings. "
                    "Run 'python scripts/setup_omnivoice.py' if the isolated env is missing."
                ),
                detail=response.get("detail"),
                retryable=bool(response.get("retryable", True)),
            ),
        )

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
                    proc.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
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
_clients: dict[str, OmniVoiceWorkerClient] = {}


def acquire_client(
    *,
    data_dir: Path,
    model: str,
    device: str,
    num_steps: int,
) -> OmniVoiceWorkerClient:
    """Return a shared worker client keyed by (model, device, num_steps).

    A single worker is reused across segments that share the same
    (model, device, num_steps) tuple so the GPU model stays hot.
    """
    key = f"{model}|{device}|{int(num_steps)}"
    with _client_lock:
        client = _clients.get(key)
        if client is None:
            client = OmniVoiceWorkerClient(
                data_dir=data_dir,
                model=model,
                device=device,
                num_steps=num_steps,
            )
            _clients[key] = client
        return client


def release_all_clients() -> None:
    with _client_lock:
        clients = list(_clients.values())
        _clients.clear()
    for client in clients:
        client.close()
