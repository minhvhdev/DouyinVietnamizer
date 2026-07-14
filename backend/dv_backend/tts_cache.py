"""TTS cache identity and sidecar metadata for Phase 2."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

CACHE_SCHEMA_VERSION = 3
CHUNK_MANIFEST_SCHEMA_VERSION = 3


def _sha256_text(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def _file_fingerprint(path: str | None) -> str | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(65536):
            digest.update(chunk)
    return digest.hexdigest()[:32]


def build_tts_cache_identity(settings: dict[str, Any], *, text: str, language: str = "vi") -> dict[str, Any]:
    backend = str(settings.get("tts_backend") or "omnivoice")
    ref_audio = str(settings.get("omnivoice_ref_audio") or "").strip()
    ref_text = str(settings.get("omnivoice_ref_text") or "").strip()
    model = str(settings.get("omnivoice_model") or settings.get("edge_tts_voice") or settings.get("google_tts_voice") or "")
    external_chunking = str(settings.get("omnivoice_external_chunking_enabled", False)).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "translation_text_hash": _sha256_text(text),
        "tts_backend": backend,
        "language": language,
        "model": model,
        "ref_audio_fingerprint": _file_fingerprint(ref_audio),
        "ref_text_hash": _sha256_text(ref_text) if ref_text else None,
        "voice_instruct_hash": _sha256_text(str(settings.get("omnivoice_instruct") or "")),
        "num_steps": int(settings.get("omnivoice_num_steps", 32) or 32),
        "omnivoice_external_chunking_enabled": external_chunking,
        "omnivoice_chunk_fidelity_fallback_full_segment": str(
            settings.get("omnivoice_chunk_fidelity_fallback_full_segment", True)
        ).strip().lower()
        not in {"0", "false", "no", "off"},
    }


def cache_key_from_identity(identity: dict[str, Any]) -> str:
    payload = json.dumps(identity, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def sidecar_path(wav_path: Path) -> Path:
    return wav_path.with_suffix(wav_path.suffix + ".meta.json")


def write_tts_sidecar(wav_path: Path, identity: dict[str, Any], *, extra: dict[str, Any] | None = None) -> None:
    payload = {
        **identity,
        "cache_key": cache_key_from_identity(identity),
        "wav_path": wav_path.name,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **(extra or {}),
    }
    sidecar_path(wav_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_tts_sidecar(wav_path: Path) -> dict[str, Any] | None:
    path = sidecar_path(wav_path)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def wav_cache_valid(wav_path: Path, expected_identity: dict[str, Any]) -> bool:
    if not wav_path.is_file() or wav_path.stat().st_size <= 44:
        return False
    sidecar = read_tts_sidecar(wav_path)
    if not sidecar:
        return False
    expected_key = cache_key_from_identity(expected_identity)
    if sidecar.get("cache_key") != expected_key:
        return False
    if sidecar.get("schema_version") != CACHE_SCHEMA_VERSION:
        return False
    return sidecar.get("translation_text_hash") == expected_identity.get("translation_text_hash")


def segment_wav_cache_valid(
    wav_path: Path,
    expected_identity: dict[str, Any],
    *,
    text: str,
    settings: dict[str, Any] | None,
    tts_dir: Path,
    segment_index: int,
) -> bool:
    """Segment-level cache validity; long text requires chunk manifest."""
    if not wav_cache_valid(wav_path, expected_identity):
        return False
    from .omnivoice_chunking import chunking_required, omnivoice_chunk_settings

    cfg = omnivoice_chunk_settings(settings)
    cleaned_len = len(str(text or "").strip())
    if cleaned_len > int(cfg["long_text_threshold"]) or chunking_required(text, settings):
        manifest = tts_dir / "chunks" / str(segment_index) / "manifest.json"
        if not manifest.is_file():
            return False
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if payload.get("schema_version") != CHUNK_MANIFEST_SCHEMA_VERSION:
            return False
        if payload.get("canonical_text_hash") != expected_identity.get("translation_text_hash"):
            return False
    return True
