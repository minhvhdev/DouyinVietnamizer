"""Persistent on-disk cache for VoxCPM2 TTS outputs.

Cache key = sha256(version, voice_id, normalized_text, model, num_step,
voice_design, cfg_value). The same input always returns the same cached
file, so re-running a job or dubbing the same video twice becomes instant
for any repeated segment.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import threading
from pathlib import Path

VOXCPM_CACHE_VERSION = "v1"


def _normalize_text(text: str) -> str:
    return " ".join((text or "").split()).strip().lower()


def cache_key(
    *,
    voice_id: str,
    text: str,
    model: str,
    num_step: int,
    voice_design: str | None = None,
    cfg_value: float = 2.0,
) -> str:
    payload = "|".join(
        [
            VOXCPM_CACHE_VERSION,
            voice_id or "",
            _normalize_text(text),
            model or "",
            str(int(num_step or 0)),
            (voice_design or "").strip(),
            f"{float(cfg_value):.4f}",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_path_for(cache_dir: Path, key: str) -> Path:
    return Path(cache_dir) / f"{key}.wav"


class VoxCPMCache:
    """File-backed cache with thread-safe access.

    Two segments with the same cache key share a single WAV on disk; the
    backend copies the cached file to the segment output path on hit.
    """

    def __init__(self, cache_dir: Path | None) -> None:
        self.enabled = cache_dir is not None
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._lock = threading.Lock()
        if self.enabled and self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Path | None:
        if not self.enabled or self.cache_dir is None:
            return None
        candidate = cache_path_for(self.cache_dir, key)
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
        return None

    def put(self, key: str, source_path: Path) -> Path | None:
        if not self.enabled or self.cache_dir is None:
            return None
        target = cache_path_for(self.cache_dir, key)
        with self._lock:
            if target.is_file() and target.stat().st_size > 0:
                return target
            try:
                shutil.copy2(source_path, target)
            except OSError:
                return None
        return target

    def materialize(self, key: str, destination: Path) -> bool:
        """Copy a cached file to ``destination`` if present. Returns True on hit."""
        cached = self.get(key)
        if cached is None:
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, destination)
        return True

    def clear(self) -> None:
        if not self.enabled or self.cache_dir is None:
            return
        with self._lock:
            for entry in self.cache_dir.glob("*.wav"):
                try:
                    entry.unlink()
                except OSError:
                    pass


def _default_cache_dir(data_dir: Path) -> Path | None:
    if os.environ.get("DV_VOXCPM_CACHE_DISABLED") == "1":
        return None
    override = os.environ.get("DV_VOXCPM_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(data_dir) / "cache" / "voxcpm"
