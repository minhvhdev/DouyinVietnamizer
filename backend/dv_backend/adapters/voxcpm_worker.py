"""Long-lived VoxCPM2 inference worker.

This script runs *inside* the isolated ``.venv-voxcpm`` virtualenv so the
heavy ``torch`` / ``voxcpm`` stack stays isolated from the main backend
environment. The backend process spawns this worker as a child process and
exchanges newline-delimited JSON messages on stdin/stdout.

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
     "model": "openbmb/VoxCPM2", "device": "cuda:0",
     "output_path": "/abs/out.wav"}

Mode semantics:

* ``design`` (default) — pure voice-design / zero-shot; no anchor audio
  parameters are forwarded to ``model.generate``.
* ``reference`` — voice cloning via VoxCPM2 reference mode. The anchor WAV
  is passed as ``reference_wav_path`` only. ``prompt_wav_path`` and
  ``prompt_text`` must NOT be sent (would invoke continuation-style speech
  context conditioning).
* ``ultimate`` — anchor is used as BOTH a reference identity signal AND a
  continuation-style speech prompt. ``reference_wav_path`` and
  ``prompt_wav_path`` are both set to the anchor; ``prompt_text`` is the
  exact transcript of the anchor audio. The worker rejects ultimate
  requests whose ``anchor_text`` is missing or blank.

Response (worker -> backend)::

    {"id": "req-1", "ok": true, "output_path": "/abs/out.wav",
     "duration_sec": 1.42, "sample_rate": 24000}
    {"id": "req-1", "ok": false, "code": "...", "message": "...",
     "retryable": true}

VoxCPM2's ``generate()`` is single-text; we call it per request after
optionally coalescing compatible requests (same model/device/voice
parameters) so the model stays hot in VRAM.
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


DEFAULT_MODEL = "openbmb/VoxCPM2"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_INFERENCE_TIMESTEPS = 10
DEFAULT_CFG_VALUE = 2.0
DEFAULT_FLUSH_MS = 150
DEFAULT_MAX_BATCH = 4
DEFAULT_SAMPLE_RATE = 24000
SUPPORTED_MODES = ("design", "reference", "ultimate")


def _log(message: str) -> None:
    sys.stderr.write(f"[voxcpm-worker] {message}\n")
    sys.stderr.flush()


def _emit(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=True) + "\n")
    sys.stdout.flush()


class VoxCPMEngine:
    """Lazy wrapper around the ``voxcpm`` package.

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
        from voxcpm import VoxCPM  # type: ignore

        _log(f"Loading VoxCPM model={model} device={device}")
        with contextlib.redirect_stdout(sys.stderr):
            engine = VoxCPM.from_pretrained(model, device=device or None)
        _log("VoxCPM model ready")
        self._engine = engine
        return self._engine

    def sample_rate(self, engine_obj: Any) -> int:
        tts = getattr(engine_obj, "tts_model", None)
        return int(getattr(tts, "sample_rate", DEFAULT_SAMPLE_RATE) or DEFAULT_SAMPLE_RATE)

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


def _generate(
    engine_obj: Any,
    text: str,
    *,
    prompt_wav_path: str | None,
    prompt_text: str | None,
    voice_design: str | None,
    inference_timesteps: int,
    cfg_value: float,
    reference_wav_path: str | None = None,
    anchor_text: str | None = None,
    mode: str = "design",
) -> Any:
    from voxcpm import VoxCPM  # type: ignore  # noqa: F401  (ensures package is importable)

    text = "".join(
        character
        for character in str(text or "")
        if not (0xD800 <= ord(character) <= 0xDFFF)
    )
    resolved_mode = (mode or "design").strip().lower() or "design"
    if resolved_mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported VoxCPM clone mode: {mode!r}. Expected one of {SUPPORTED_MODES}."
        )
    if resolved_mode == "ultimate":
        anchor_clean = (anchor_text or "").strip()
        if not anchor_clean:
            raise ValueError(
                "Ultimate clone mode requires a non-empty anchor_text "
                "matching the reference_wav_path transcript."
            )
    kwargs: dict[str, Any] = {
        "cfg_value": float(cfg_value),
        "inference_timesteps": int(inference_timesteps),
        "normalize": False,
        "denoise": False,
    }
    if resolved_mode == "reference":
        # VoxCPM2 reference mode: anchor WAV carries speaker identity via
        # isolated ref tokens. No prompt text, no prompt audio: those would
        # trigger continuation-style speech-context conditioning and the
        # output would sound like it continues from the anchor.
        if reference_wav_path:
            kwargs["reference_wav_path"] = reference_wav_path
    elif resolved_mode == "ultimate":
        # Ultimate mode: anchor is used both as reference identity signal
        # AND as a continuation-style speech prompt. prompt_text must equal
        # the transcript of prompt_wav_path so the model conditions on the
        # right speech context.
        if reference_wav_path:
            kwargs["reference_wav_path"] = reference_wav_path
        kwargs["prompt_wav_path"] = reference_wav_path
        kwargs["prompt_text"] = (anchor_text or "").strip()
    elif prompt_wav_path:
        # Legacy preview/job settings use prompt audio directly.
        kwargs["prompt_wav_path"] = prompt_wav_path
        if prompt_text:
            kwargs["prompt_text"] = prompt_text
    elif voice_design:
        # Voice design is carried in the text prefix; no audio parameters.
        pass
    _log(
        f"Generating 1 segment mode={resolved_mode} "
        f"inference_timesteps={inference_timesteps} "
        f"cfg_value={cfg_value} voice_design={bool(voice_design)} "
        f"reference_wav={bool(reference_wav_path)}"
    )
    with contextlib.redirect_stdout(sys.stderr):
        return engine_obj.generate(text=text, **kwargs)


def _coalesce(requests: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[tuple, list[dict[str, Any]]] = {}
    order: list[tuple] = []
    for req in requests:
        key = (
            req.get("model") or DEFAULT_MODEL,
            req.get("device") or DEFAULT_DEVICE,
            req.get("mode") or "design",
            req.get("reference_wav_path") or "",
            req.get("prompt_wav_path") or "",
            req.get("prompt_text") or "",
            req.get("voice_design") or "",
            int(req.get("inference_timesteps") or DEFAULT_INFERENCE_TIMESTEPS),
            float(req.get("cfg_value") or DEFAULT_CFG_VALUE),
        )
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(req)
    return [groups[key] for key in order]


def _run_batch(engine_obj: Any, batch: list[dict[str, Any]], sample_rate: int) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    for req in batch:
        try:
            audio = _generate(
                engine_obj,
                req["text"],
                prompt_wav_path=req.get("prompt_wav_path"),
                prompt_text=req.get("prompt_text"),
                voice_design=req.get("voice_design"),
                inference_timesteps=int(req.get("inference_timesteps") or DEFAULT_INFERENCE_TIMESTEPS),
                cfg_value=float(req.get("cfg_value") or DEFAULT_CFG_VALUE),
                reference_wav_path=req.get("reference_wav_path"),
                anchor_text=req.get("anchor_text"),
                mode=req.get("mode") or "design",
            )
            duration = _write_wav(req["output_path"], audio, sample_rate)
        except ValueError as exc:
            _log(f"Synthesize rejected: {exc!r}")
            responses.append(
                {
                    "id": req.get("id"),
                    "ok": False,
                    "code": "VOXCPM_BAD_REQUEST",
                    "message": str(exc),
                    "retryable": False,
                }
            )
            continue
        except Exception as exc:  # noqa: BLE001
            _log(f"Synthesize failed: {exc!r}")
            traceback.print_exc(file=sys.stderr)
            responses.append(
                {
                    "id": req.get("id"),
                    "ok": False,
                    "code": "VOXCPM_INFERENCE_FAILED",
                    "message": str(exc),
                    "retryable": True,
                }
            )
            continue
        responses.append(
            {
                "id": req.get("id"),
                "ok": True,
                "output_path": req["output_path"],
                "duration_sec": round(duration, 3),
                "sample_rate": sample_rate,
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
    _ = max_batch, flush_ms, idle_timeout_sec  # batching is by JSONL request ordering; flushed eagerly per request
    engine = VoxCPMEngine()

    def _handle_synthesize(request: dict[str, Any]) -> None:
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
            f"mode={request.get('mode') or 'design'} "
            f"text_len={len(request_text)} "
            f"reference_wav={bool(request.get('reference_wav_path'))} "
            f"prompt_wav={bool(request.get('prompt_wav_path'))} "
            f"prompt_text_len={len(str(request.get('prompt_text') or ''))} "
            f"voice_design={bool(request.get('voice_design'))}"
        )
        engine_obj = engine.get(model=model, device=device)
        sample_rate = engine.sample_rate(engine_obj)
        for response in _run_batch(engine_obj, [request], sample_rate):
            _emit(response)

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
    parser.add_argument("--health-check", action="store_true", help="Import voxcpm and exit; used by the runtime smoke test.")
    args = parser.parse_args()

    if args.health_check:
        try:
            import voxcpm  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            _emit({"ok": False, "code": "VOXCPM_NOT_INSTALLED", "message": str(exc)})
            return 1
        _emit({"ok": True, "code": "READY"})
        return 0

    return serve(max_batch=args.max_batch, flush_ms=args.flush_ms, idle_timeout_sec=args.idle_timeout_sec)


if __name__ == "__main__":
    raise SystemExit(main())
