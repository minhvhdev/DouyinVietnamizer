from __future__ import annotations

from contextlib import contextmanager
import json
import logging
from pathlib import Path
import time
from typing import Any, Iterator

logger = logging.getLogger(__name__)


DEFAULT_MAX_FILE_BYTES = 16 * 1024 * 1024
_SENSITIVE_KEYS = {"text", "translation", "ref_audio", "api_key", "prompt_text"}


def _scrub_sensitive(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return payload
    return {
        key: ("[redacted]" if key in _SENSITIVE_KEYS and value else value)
        for key, value in payload.items()
    }


def _gpu_peak_vram_mb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return round(float(torch.cuda.max_memory_allocated()) / (1024 * 1024), 2)
    except Exception:
        return None


def _maybe_rotate(path: Path, max_bytes: int) -> None:
    if max_bytes <= 0:
        return
    try:
        if path.exists() and path.stat().st_size > max_bytes:
            rotated = path.with_suffix(path.suffix + ".1")
            if rotated.exists():
                rotated.unlink()
            path.rename(rotated)
    except OSError as error:
        logger.debug("Telemetry rotation failed for %s: %s", path, error)


class TelemetrySink:
    def __init__(self, data_dir: Path, job_id: str, *, max_file_bytes: int = DEFAULT_MAX_FILE_BYTES) -> None:
        self.data_dir = Path(data_dir)
        self.job_id = job_id
        self.path = self.data_dir / "jobs" / job_id / "artifacts" / "telemetry.jsonl"
        self.max_file_bytes = int(max_file_bytes)

    def configure_max(self, max_file_bytes: int) -> None:
        self.max_file_bytes = max(0, int(max_file_bytes))

    def record(self, step: str, metrics: dict[str, Any]) -> None:
        try:
            payload = {
                "job_id": self.job_id,
                "step": step,
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **_scrub_sensitive(metrics),
            }
            audio_duration = float(payload.get("audio_duration_sec") or 0.0)
            wall_time_ms = float(payload.get("wall_time_ms") or 0.0)
            if audio_duration > 0 and "real_time_factor" not in payload:
                payload["real_time_factor"] = round((wall_time_ms / 1000.0) / audio_duration, 4)
            gpu_peak = _gpu_peak_vram_mb()
            if gpu_peak is not None and "gpu_peak_vram_mb" not in payload:
                payload["gpu_peak_vram_mb"] = gpu_peak
            json.dumps(payload, ensure_ascii=False)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            _maybe_rotate(self.path, self.max_file_bytes)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as error:
            logger.debug("Failed to write telemetry for %s/%s: %s", self.job_id, step, error)

    @contextmanager
    def measure(self, step: str, **metrics: Any) -> Iterator[dict[str, Any]]:
        started = time.perf_counter()
        data = dict(metrics)
        try:
            yield data
        except Exception:
            data.setdefault("status", "failed")
            data["wall_time_ms"] = round((time.perf_counter() - started) * 1000)
            self.record(step, data)
            raise
        else:
            data.setdefault("status", "ok")
            data["wall_time_ms"] = round((time.perf_counter() - started) * 1000)
            self.record(step, data)
