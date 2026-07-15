"""Client for the long-lived OmniVoice worker subprocess."""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

from ..errors import AppError
from ..models import ErrorInfo
from ..omnivoice_env import OMNIVOICE_DEFAULT_MODEL, build_omnivoice_subprocess_env, resolve_omnivoice_python

WORKER_SCRIPT = "dv_backend.adapters.omnivoice_worker"
DEFAULT_MAX_BATCH = 4
DEFAULT_FLUSH_MS = 150
STARTUP_PING_TIMEOUT_SEC = 60.0
PING_INTERVAL_SEC = 60.0


def _json_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True) + "\n"


class OmniVoiceWorkerClient:
    def __init__(
        self,
        *,
        data_dir: Path,
        model: str,
        device: str,
        num_step: int,
        speed: float,
        language_id: str | None,
        audio_chunk_threshold: float,
        audio_chunk_duration: float,
        max_batch: int = DEFAULT_MAX_BATCH,
        flush_ms: int = DEFAULT_FLUSH_MS,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.model = (model or OMNIVOICE_DEFAULT_MODEL).strip() or OMNIVOICE_DEFAULT_MODEL
        self.device = (device or "cuda:0").strip() or "cuda:0"
        self.num_step = max(4, min(64, int(num_step)))
        self.speed = max(0.5, min(2.0, float(speed)))
        self.language_id = (language_id or "").strip() or None
        self.audio_chunk_threshold = max(4.0, min(60.0, float(audio_chunk_threshold)))
        self.audio_chunk_duration = max(4.0, min(30.0, float(audio_chunk_duration)))
        self.max_batch = max(1, int(max_batch))
        self.flush_ms = max(20, int(flush_ms))
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending: dict[str, queue.Queue] = {}
        self._pending_lock = threading.Lock()
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_tail: deque[str] = deque(maxlen=80)
        self._last_used = time.perf_counter()
        self._last_ping = 0.0
        self._closed = False

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
        ]
        env = build_omnivoice_subprocess_env(os.environ.copy())
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
                    code="OMNIVOICE_NOT_INSTALLED",
                    message="OmniVoice environment is not installed.",
                    action="Run 'python scripts/setup_omnivoice.py' to create the isolated virtualenv.",
                    detail=str(exc),
                ),
            ) from exc
        except OSError as exc:
            raise AppError(
                502,
                ErrorInfo(
                    code="OMNIVOICE_TTS_FAILED",
                    message="OmniVoice inference subprocess failed to start.",
                    action="Verify the isolated OmniVoice virtualenv and GPU availability.",
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
                    code="OMNIVOICE_TTS_FAILED",
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
                    code="OMNIVOICE_TTS_FAILED",
                    message="OmniVoice worker failed to start within timeout.",
                    retryable=True,
                ),
            ) from exc
        if not response.get("ok"):
            self._terminate_proc()
            self._raise_worker_error(response)

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
        self._fail_pending("OmniVoice worker exited unexpectedly.", retryable=True)

    def _stderr_loop(self) -> None:
        if self._proc is None or self._proc.stderr is None:
            return
        for line in self._proc.stderr:
            if not line:
                continue
            stripped = line.strip()
            if stripped:
                self._stderr_tail.append(stripped)
            sys.stderr.write(f"[omnivoice-worker] {line}")
            sys.stderr.flush()

    def _fail_pending(self, message: str, *, retryable: bool) -> None:
        detail = "\n".join(self._stderr_tail).strip() or None
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
                        "detail": detail,
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

    def _raise_worker_error(self, response: dict[str, Any]) -> None:
        code = response.get("code") or "OMNIVOICE_TTS_FAILED"
        if code == "OMNIVOICE_BAD_REQUEST":
            raise AppError(
                422,
                ErrorInfo(
                    code=code,
                    message=response.get("message") or "OmniVoice worker rejected the request.",
                    action="Verify reference audio, voice design, and text parameters.",
                    detail=response.get("detail"),
                    retryable=False,
                ),
            )
        raise AppError(
            502,
            ErrorInfo(
                code=code,
                message=response.get("message") or "OmniVoice could not generate narration.",
                action=(
                    "Check OmniVoice model, GPU availability, and reference audio settings. "
                    "Run 'python scripts/setup_omnivoice.py' if the isolated env is missing."
                ),
                detail=response.get("detail"),
                retryable=bool(response.get("retryable", True)),
            ),
        )

    def register_with_runner(self, runner: object | None) -> None:
        return None

    @property
    def pending_count(self) -> int:
        with self._pending_lock:
            return len(self._pending)

    def wait_result(self, request_id: str, *, timeout_sec: float = 600.0) -> dict[str, Any]:
        """Wait for a response without raising on worker-reported synthesis failure."""
        with self._pending_lock:
            response_q = self._pending.get(request_id)
        if response_q is None:
            raise AppError(
                502,
                ErrorInfo(
                    code="OMNIVOICE_TTS_FAILED",
                    message="Unknown OmniVoice request id.",
                    retryable=True,
                ),
            )
        try:
            response = response_q.get(timeout=timeout_sec)
        except queue.Empty as exc:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise AppError(
                502,
                ErrorInfo(
                    code="OMNIVOICE_TIMEOUT",
                    message="OmniVoice synthesis timed out.",
                    action="Retry TTS for this segment or restart the OmniVoice worker.",
                    retryable=True,
                ),
            ) from exc
        with self._pending_lock:
            self._pending.pop(request_id, None)
        self._last_used = time.perf_counter()
        return response

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
        request_id = f"req-{uuid.uuid4().hex}"
        response_q: queue.Queue = queue.Queue(maxsize=1)
        request: dict[str, Any] = {
            "id": request_id,
            "op": "synthesize",
            "text": text,
            "output_path": str(output_path),
            "model": self.model,
            "device": self.device,
            "num_step": self.num_step,
            "speed": self.speed,
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "anchor_text": anchor_text,
            "instruct": instruct,
            "language_id": self.language_id,
            "audio_chunk_threshold": self.audio_chunk_threshold,
            "audio_chunk_duration": self.audio_chunk_duration,
            "include_perf": bool(include_perf),
        }
        if batch_id is not None and batch_size is not None:
            request["batch_id"] = str(batch_id)
            request["batch_size"] = max(1, int(batch_size))
            if batch_index is not None:
                request["batch_index"] = max(0, int(batch_index))
        with self._pending_lock:
            self._pending[request_id] = response_q
        try:
            assert self._proc is not None and self._proc.stdin is not None
            with self._write_lock:
                self._proc.stdin.write(_json_line(request))
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
        return request_id

    def wait(self, request_id: str, *, timeout_sec: float = 600.0) -> dict[str, Any]:
        response = self.wait_result(request_id, timeout_sec=timeout_sec)
        if response.get("ok"):
            return response
        self._raise_worker_error(response)
        return response

    def wait_many(self, request_ids: list[str], *, timeout_sec: float = 600.0) -> list[dict[str, Any]]:
        if not request_ids:
            return []
        return [self.wait_result(request_id, timeout_sec=timeout_sec) for request_id in request_ids]

    def synthesize_many(
        self,
        requests: list[dict[str, Any]],
        *,
        timeout_sec: float = 600.0,
    ) -> list[dict[str, Any]]:
        """Submit every request before waiting for the first response."""
        if not requests:
            return []
        batch_id = uuid.uuid4().hex
        batch_size = len(requests)
        request_ids: list[str] = []
        for index, req in enumerate(requests):
            request_ids.append(
                self.submit(
                    text=str(req["text"]),
                    output_path=Path(req["output_path"]),
                    ref_audio=req.get("ref_audio"),
                    ref_text=req.get("ref_text"),
                    anchor_text=req.get("anchor_text"),
                    instruct=req.get("instruct"),
                    include_perf=bool(req.get("include_perf")),
                    batch_id=str(req.get("batch_id") or batch_id),
                    batch_index=int(req.get("batch_index") if req.get("batch_index") is not None else index),
                    batch_size=int(req.get("batch_size") if req.get("batch_size") is not None else batch_size),
                )
            )
        return self.wait_many(request_ids, timeout_sec=timeout_sec)

    def synthesize(
        self,
        *,
        text: str,
        output_path: Path,
        ref_audio: str | None,
        ref_text: str | None,
        anchor_text: str | None = None,
        instruct: str | None,
        include_perf: bool = False,
    ) -> dict[str, Any]:
        request_id = self.submit(
            text=text,
            output_path=output_path,
            ref_audio=ref_audio,
            ref_text=ref_text,
            anchor_text=anchor_text,
            instruct=instruct,
            include_perf=include_perf,
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
_clients: dict[str, OmniVoiceWorkerClient] = {}


def acquire_client(
    *,
    data_dir: Path,
    model: str,
    device: str,
    num_step: int,
    speed: float,
    language_id: str | None = None,
    audio_chunk_threshold: float = 30.0,
    audio_chunk_duration: float = 15.0,
    max_batch: int = DEFAULT_MAX_BATCH,
    flush_ms: int = DEFAULT_FLUSH_MS,
    scope: str = "shared",
) -> OmniVoiceWorkerClient:
    max_batch = max(1, int(max_batch or DEFAULT_MAX_BATCH))
    flush_ms = max(20, int(flush_ms or DEFAULT_FLUSH_MS))
    resolved_scope = (scope or "shared").strip().replace("|", "_") or "shared"
    key = (
        f"scope={resolved_scope}|{model}|{device}|{int(num_step)}|{float(speed)}|"
        f"{language_id or ''}|chunk={float(audio_chunk_threshold)}|{float(audio_chunk_duration)}|"
        f"batch={max_batch}|flush={flush_ms}"
    )
    with _client_lock:
        client = _clients.get(key)
        if client is None:
            client = OmniVoiceWorkerClient(
                data_dir=data_dir,
                model=model,
                device=device,
                num_step=num_step,
                speed=speed,
                language_id=language_id,
                audio_chunk_threshold=audio_chunk_threshold,
                audio_chunk_duration=audio_chunk_duration,
                max_batch=max_batch,
                flush_ms=flush_ms,
            )
            _clients[key] = client
        return client


def release_clients(scope: str | None = None) -> None:
    with _client_lock:
        if scope is None:
            clients = list(_clients.values())
            _clients.clear()
        else:
            prefix = f"scope={scope}|"
            keys = [key for key in _clients if key.startswith(prefix)]
            clients = [_clients.pop(key) for key in keys]
    for client in clients:
        client.close()


def release_all_clients() -> None:
    release_clients()


def client_debug_snapshot() -> dict[str, int]:
    with _client_lock:
        return {"count": len(_clients)}
