from __future__ import annotations

from dataclasses import dataclass
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
import threading
import time
from typing import Callable

from .gpu_lease import gpu_lease

try:
    import torch as _torch
except Exception:  # pragma: no cover - torch optional at import
    _torch = None


def _peak_vram_mb() -> float | None:
    if _torch is None:
        return None
    try:
        if not _torch.cuda.is_available():
            return None
        return round(float(_torch.cuda.memory_allocated()) / (1024 * 1024), 2)
    except Exception:
        return None


@dataclass
class GpuLease(AbstractContextManager["GpuLease"]):
    manager: "GpuModelManager"
    family: str
    device: str
    model_key: str
    lease_cm: AbstractContextManager[None]
    cold_start: bool
    queue_wait_ms: int
    load_ms: int
    vram_before_mb: float | None
    vram_after_mb: float | None

    def __enter__(self) -> "GpuLease":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.manager.release(self, exc_type is not None)
        finally:
            self.lease_cm.__exit__(None, None, None)


class GpuModelManager:
    def __init__(self, *, lock_dir: Path | None = None) -> None:
        self.lock_dir = lock_dir
        self._loaded: dict[tuple[str, str], str] = {}
        self._guard = threading.Lock()
        self.evictions: list[dict[str, str]] = []
        self.lease_history: list[dict[str, object]] = []
        self.idle_timeout_sec: float = 60.0
        self.keep_warm: bool = True
        self.max_resident_families: int = 1
        self._last_used: dict[tuple[str, str], float] = {}

    def configure(
        self,
        *,
        idle_timeout_sec: float | None = None,
        keep_warm: bool | None = None,
        max_resident_families: int | None = None,
    ) -> None:
        if idle_timeout_sec is not None:
            self.idle_timeout_sec = float(idle_timeout_sec)
        if keep_warm is not None:
            self.keep_warm = bool(keep_warm)
        if max_resident_families is not None:
            self.max_resident_families = int(max_resident_families)

    def acquire(
        self,
        family: str,
        device: str,
        model_key: str,
        *,
        loader: Callable[[], None] | None = None,
        serialize: bool = True,
    ) -> GpuLease:
        device_key = device or "cpu"
        started = time.perf_counter()
        lease_cm = gpu_lease(f"{family}:{model_key}", device=device_key, lock_dir=self.lock_dir) if serialize else nullcontext()
        lease_cm.__enter__()
        wait_ms = round((time.perf_counter() - started) * 1000)
        key = (family, device_key)
        with self._guard:
            current = self._loaded.get(key)
            cold = current != model_key
            vram_before = _peak_vram_mb()
            self._evict_excess_families(device_key, family)
            if cold:
                self._loaded[key] = model_key
            self._last_used[key] = time.perf_counter()
        load_started = time.perf_counter()
        try:
            if loader is not None and cold:
                loader()
        except Exception:
            with self._guard:
                self._loaded.pop(key, None)
            lease_cm.__exit__(None, None, None)
            raise
        load_ms = round((time.perf_counter() - load_started) * 1000)
        vram_after = _peak_vram_mb()
        lease = GpuLease(
            manager=self,
            family=family,
            device=device_key,
            model_key=model_key,
            lease_cm=lease_cm,
            cold_start=cold,
            queue_wait_ms=wait_ms,
            load_ms=load_ms,
            vram_before_mb=vram_before,
            vram_after_mb=vram_after,
        )
        self.lease_history.append({
            "family": family,
            "device": device_key,
            "model": model_key,
            "cold_start": cold,
            "queue_wait_ms": wait_ms,
            "load_ms": load_ms,
            "vram_before_mb": vram_before,
            "vram_after_mb": vram_after,
        })
        return lease

    def release(self, lease: GpuLease, had_error: bool) -> None:
        with self._guard:
            if not had_error and not self.keep_warm:
                self._loaded.pop((lease.family, lease.device), None)
            residents = [key for key in self._loaded if key[1] == lease.device]
            while len(residents) > self.max_resident_families and self.max_resident_families >= 0:
                oldest = min(residents, key=lambda key: self._last_used.get(key, 0.0))
                self._loaded.pop(oldest, None)
                self.evictions.append({"family": oldest[0], "device": oldest[1], "reason": "max_resident_exceeded"})
                residents = [key for key in self._loaded if key[1] == lease.device]

    def evict(self, family: str, device: str, *, reason: str) -> None:
        device_key = device or "cpu"
        with self._guard:
            self._loaded.pop((family, device_key), None)
            self.evictions.append({"family": family, "device": device_key, "reason": reason})

    def snapshot(self) -> dict[str, object]:
        with self._guard:
            residents = [
                {
                    "family": family,
                    "device": device,
                    "model": model_key,
                }
                for (family, device), model_key in sorted(self._loaded.items())
            ]
            return {
                "resident_models": residents,
                "lease_history_size": len(self.lease_history),
                "eviction_count": len(self.evictions),
            }

    def reset(self) -> dict[str, object]:
        with self._guard:
            previous = [
                {
                    "family": family,
                    "device": device,
                    "model": model_key,
                }
                for (family, device), model_key in sorted(self._loaded.items())
            ]
            self._loaded.clear()
            self._last_used.clear()
            self.lease_history.clear()
            self.evictions.clear()
        return {"resident_models": previous}

    def _evict_excess_families(self, device_key: str, keep_family: str) -> None:
        if self.max_resident_families < 0:
            return
        residents = sorted(
            (
                key
                for key in self._loaded
                if key[1] == device_key
            ),
            key=lambda key: self._last_used.get(key, 0.0),
        )
        excess = len(residents) - self.max_resident_families
        for key in residents:
            if excess <= 0:
                break
            if key[0] == keep_family:
                continue
            self._loaded.pop(key, None)
            self.evictions.append({"family": key[0], "device": device_key, "reason": "max_resident_exceeded"})
            excess -= 1


_GLOBAL_MANAGER = GpuModelManager()


def global_gpu_manager() -> GpuModelManager:
    return _GLOBAL_MANAGER
