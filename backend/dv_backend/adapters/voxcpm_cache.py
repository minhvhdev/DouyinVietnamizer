"""Persistent on-disk cache for VoxCPM2 TTS outputs.

Cache key = sha256(version, mode, voice_id, normalized_text, model, num_step,
voice_design, cfg_value, reference_audio_content_hash, anchor_text). The same
input always returns the same cached file, so re-running a job or dubbing the
same video twice becomes instant for any repeated segment. Reference and
Ultimate mode outputs are kept strictly separate even when the target text and
anchor audio match.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import threading
import wave
from pathlib import Path

VOXCPM_CACHE_VERSION = "v6-gguf"
MODES = ("design", "reference", "ultimate")


def _normalize_text(text: str) -> str:
    return " ".join((text or "").split()).strip().lower()


def _normalize_mode(mode: str | None) -> str:
    candidate = (mode or "design").strip().lower() or "design"
    if candidate not in MODES:
        return "design"
    return candidate


_REFERENCE_AUDIO_HASH_CACHE: dict[tuple[str, int, int], str] = {}
_REFERENCE_AUDIO_HASH_LOCK = threading.Lock()


def _reference_audio_cache_key(path: str | os.PathLike | None) -> tuple[str, int, int] | None:
    if not path:
        return None
    p = Path(path)
    try:
        stat = p.stat()
    except OSError:
        return None
    if not p.is_file():
        return None
    try:
        resolved = str(p.resolve())
    except OSError:
        resolved = str(p.absolute())
    return (resolved, int(stat.st_size), int(stat.st_mtime_ns))

def _file_hash(path: str | os.PathLike | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    try:
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def reference_audio_content_hash(path: str | os.PathLike | None) -> str:
    """SHA-256 of the anchor audio file content, cached by path/size/mtime."""
    key = _reference_audio_cache_key(path)
    if key is None:
        return ""
    with _REFERENCE_AUDIO_HASH_LOCK:
        cached = _REFERENCE_AUDIO_HASH_CACHE.get(key)
    if cached is not None:
        return cached
    digest = _file_hash(path)
    if digest:
        with _REFERENCE_AUDIO_HASH_LOCK:
            _REFERENCE_AUDIO_HASH_CACHE[key] = digest
    return digest


def reference_text_hash(text_or_path: str | os.PathLike | None) -> str:
    if not text_or_path:
        return ""
    value = str(text_or_path)
    path = Path(value)
    if path.is_file():
        return _file_hash(path)
    return hashlib.sha256(_normalize_text(value).encode("utf-8")).hexdigest()


def is_valid_wav(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size <= 44:
            return False
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() > 0 and handle.getframerate() > 0
    except Exception:
        return False


def cache_key(
    *,
    voice_id: str,
    text: str,
    model: str,
    num_step: int,
    voice_design: str | None = None,
    cfg_value: float = 2.0,
    mode: str = "design",
    reference_wav_path: str | None = None,
    reference_text: str | None = None,
    anchor_text: str | None = None,
) -> str:
    mode_norm = _normalize_mode(mode)
    anchor_norm = _normalize_text(anchor_text or "") if mode_norm == "ultimate" else ""
    payload = "|".join(
        [
            VOXCPM_CACHE_VERSION,
            mode_norm,
            voice_id or "",
            _normalize_text(text),
            model or "",
            str(int(num_step or 0)),
            (voice_design or "").strip(),
            f"{float(cfg_value):.4f}",
            reference_audio_content_hash(reference_wav_path),
            reference_text_hash(reference_text),
            anchor_norm,
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
        if is_valid_wav(candidate):
            return candidate
        return None

    def put(self, key: str, source_path: Path) -> Path | None:
        if not self.enabled or self.cache_dir is None or not is_valid_wav(source_path):
            return None
        target = cache_path_for(self.cache_dir, key)
        with self._lock:
            if is_valid_wav(target):
                return target
            fd, temp_name = tempfile.mkstemp(dir=str(self.cache_dir), suffix=".tmp")
            os.close(fd)
            temp_path = Path(temp_name)
            try:
                shutil.copy2(source_path, temp_path)
                if not is_valid_wav(temp_path):
                    temp_path.unlink(missing_ok=True)
                    return None
                os.replace(temp_path, target)
            except OSError:
                temp_path.unlink(missing_ok=True)
                return None
        return target

    def materialize(self, key: str, destination: Path) -> bool:
        """Copy a cached file to ``destination`` if present. Returns True on hit."""
        cached = self.get(key)
        if cached is None:
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, destination)
        if not is_valid_wav(destination):
            destination.unlink(missing_ok=True)
            return False
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
