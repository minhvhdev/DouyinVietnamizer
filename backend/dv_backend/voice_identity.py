"""Canonical TTS voice identity for duration profile keys."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

IDENTITY_SCHEMA_VERSION = 1

DEFAULT_GENERATION_CONFIG = {
    "speed": 1.0,
    "clone_mode": "reference",
    "trim_speech": False,
    "duration_repair": False,
    "automatic_tempo": False,
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes((text or "").encode("utf-8"))


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def generation_config_hash(generation_config: dict[str, Any] | None = None) -> str:
    payload = {**DEFAULT_GENERATION_CONFIG, **(generation_config or {})}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return _sha256_text(canonical)[:32]


def resolve_tts_voice_identity(
    *,
    tts_backend: str = "omnivoice",
    target_language: str = "vi",
    model: str = "",
    voice_id: str = "",
    reference_audio_path: str | Path | None = None,
    reference_text: str | None = None,
    generation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ref_audio_sha256 = _sha256_file(Path(reference_audio_path)) if reference_audio_path else None
    ref_text_sha256 = _sha256_text(reference_text or "") if reference_text is not None else None
    gen_hash = generation_config_hash(generation_config)
    stable_voice_id = voice_id.strip()
    if not stable_voice_id and ref_audio_sha256:
        stable_voice_id = f"ref:{ref_audio_sha256[:32]}"
    return {
        "identity_schema_version": IDENTITY_SCHEMA_VERSION,
        "tts_backend": tts_backend,
        "target_language": target_language,
        "model": model or "",
        "voice_id": stable_voice_id,
        "reference_audio_sha256": ref_audio_sha256,
        "reference_text_sha256": ref_text_sha256,
        "generation_config_hash": gen_hash,
    }


def identity_profile_key(identity: dict[str, Any]) -> str:
    canonical = {
        "identity_schema_version": identity.get("identity_schema_version", IDENTITY_SCHEMA_VERSION),
        "tts_backend": identity.get("tts_backend"),
        "target_language": identity.get("target_language") or identity.get("language"),
        "model": identity.get("model") or "",
        "voice_id": identity.get("voice_id") or "",
        "reference_audio_sha256": identity.get("reference_audio_sha256"),
        "reference_text_sha256": identity.get("reference_text_sha256"),
        "generation_config_hash": identity.get("generation_config_hash"),
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return f"id:{_sha256_text(payload)[:40]}"


def identity_from_settings(
    settings: dict[str, Any],
    *,
    language: str = "vi",
    cloned_voice_id: str | None = None,
    reference_audio_path: str | Path | None = None,
    reference_text: str | None = None,
) -> dict[str, Any]:
    backend = str(settings.get("tts_backend") or "omnivoice")
    model = str(settings.get("omnivoice_model") or settings.get("edge_tts_voice") or settings.get("google_tts_voice") or "")
    ref_path = reference_audio_path or settings.get("omnivoice_ref_audio")
    ref_text = reference_text if reference_text is not None else settings.get("omnivoice_ref_text")
    voice_id = cloned_voice_id or ""
    if not voice_id:
        ref = str(ref_path or "").strip()
        if ref:
            fp = _sha256_file(Path(ref))
            voice_id = f"ref:{fp[:32]}" if fp else ref
        else:
            instruct = str(settings.get("omnivoice_instruct") or "").strip()
            if instruct:
                voice_id = f"instruct:{hashlib.sha256(instruct.encode()).hexdigest()[:16]}"
            else:
                voice = str(settings.get("edge_tts_voice") or settings.get("google_tts_voice") or "default")
                voice_id = f"{backend}:{voice}"
    return resolve_tts_voice_identity(
        tts_backend=backend,
        target_language=language,
        model=model,
        voice_id=voice_id,
        reference_audio_path=ref_path,
        reference_text=str(ref_text or ""),
        generation_config={
            "speed": float(settings.get("tts_global_speed") or 1.0),
            "clone_mode": str(settings.get("omnivoice_clone_mode") or "reference"),
        },
    )


def identity_matches_stored(stored_identity: dict[str, Any] | None, current: dict[str, Any]) -> bool:
    if not stored_identity:
        return False
    for field in (
        "tts_backend",
        "target_language",
        "model",
        "voice_id",
        "reference_audio_sha256",
        "reference_text_sha256",
        "generation_config_hash",
    ):
        stored_val = stored_identity.get(field) or stored_identity.get("language" if field == "target_language" else "")
        current_val = current.get(field) or current.get("language" if field == "target_language" else "")
        if stored_val != current_val:
            return False
    return True


def settings_for_cloned_voice(
    base_settings: dict[str, Any],
    *,
    voice_id: str,
    wav_path: Path,
    transcript: str,
) -> dict[str, Any]:
    merged = dict(base_settings)
    merged["tts_backend"] = "omnivoice"
    merged["omnivoice_ref_audio"] = str(wav_path)
    merged["omnivoice_ref_text"] = transcript
    merged["tts_global_speed"] = 1.0
    merged["cloned_voice_id"] = voice_id
    return merged
