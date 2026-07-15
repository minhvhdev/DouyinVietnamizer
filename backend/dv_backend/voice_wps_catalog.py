"""Catalog and WPS management for all TTS voices across providers."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from .dubbing_languages import (
    SUPPORTED_DUB_LANGUAGES,
    default_speaking_rate_wps,
    dub_language_config,
    normalize_dub_language,
)
from .voice_calibration_dataset import load_calibration_dataset, select_calibration_samples
from .voice_calibration_samples import (
    aggregate_calibration_profile,
    compute_validation_metrics_for_samples,
    evaluate_calibration_sample,
)
from .voice_duration_profile import load_profiles, merge_bootstrap_profile, save_manual_wps_profile
from .voice_profile_policy import classify_profile_quality
from .voice_identity import identity_profile_key, resolve_tts_voice_identity

PROVIDER_LABELS = {
    "omnivoice": "OmniVoice",
    "edge_tts": "Edge TTS",
    "google_tts": "Google TTS",
}

MIN_WPS = 2.0
MAX_WPS = 5.0
SPS_PER_WPS = 1.15


def make_catalog_key(provider: str, kind: str, voice_id: str) -> str:
    return f"{provider}:{kind}:{voice_id}"


def parse_catalog_key(catalog_key: str) -> tuple[str, str, str]:
    parts = (catalog_key or "").strip().split(":", 2)
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise ValueError("Invalid catalog_key format.")
    return parts[0].strip().lower(), parts[1].strip().lower(), parts[2].strip()


def wps_from_profile(profile: dict[str, Any] | None, *, language: str) -> tuple[float, str | None]:
    if not isinstance(profile, dict):
        return default_speaking_rate_wps(language), None
    if profile.get("words_per_second") is not None:
        return round(float(profile["words_per_second"]), 2), str(profile.get("source") or "manual")
    sps = profile.get("syllables_per_second")
    if sps:
        return round(float(sps) / SPS_PER_WPS, 2), str(profile.get("source") or "calibration")
    return default_speaking_rate_wps(language), None


def _identity_for_entry(
    *,
    provider: str,
    kind: str,
    voice_id: str,
    language: str,
    wav_path: str | Path | None = None,
    transcript: str | None = None,
) -> dict[str, Any]:
    if provider == "omnivoice" and kind == "clone":
        return resolve_tts_voice_identity(
            tts_backend="omnivoice",
            target_language=language,
            voice_id=voice_id,
            reference_audio_path=wav_path,
            reference_text=transcript or "",
            generation_config={"clone_mode": "reference"},
        )
    if provider == "edge_tts":
        return resolve_tts_voice_identity(
            tts_backend="edge_tts",
            target_language=language,
            model=voice_id,
            voice_id=f"edge_tts:{voice_id}",
        )
    if provider == "google_tts":
        return resolve_tts_voice_identity(
            tts_backend="google_tts",
            target_language=language,
            model=voice_id,
            voice_id=f"google_tts:{voice_id}",
        )
    raise ValueError(f"Unsupported provider: {provider}")


def _profile_for_identity(data_dir: Path | None, identity: dict[str, Any]) -> dict[str, Any] | None:
    store = load_profiles(data_dir)
    key = identity_profile_key(identity)
    stored = store.get("profiles", {}).get(key)
    return stored if isinstance(stored, dict) else None


def _entry_from_voice(
    *,
    provider: str,
    kind: str,
    voice_id: str,
    voice_name: str,
    language: str,
    data_dir: Path | None,
    cloned_voice_id: str | None = None,
    wav_path: str | Path | None = None,
    transcript: str | None = None,
    duration_profile_status: str | None = None,
) -> dict[str, Any]:
    identity = _identity_for_entry(
        provider=provider,
        kind=kind,
        voice_id=voice_id,
        language=language,
        wav_path=wav_path,
        transcript=transcript,
    )
    profile = _profile_for_identity(data_dir, identity)
    wps, source = wps_from_profile(profile, language=language)
    default_wps = default_speaking_rate_wps(language)
    provider_label = PROVIDER_LABELS.get(provider, provider)
    if kind == "clone":
        provider_label = f"{provider_label} (Clone)"
    return {
        "catalog_key": make_catalog_key(provider, kind, voice_id),
        "provider": provider,
        "provider_label": provider_label,
        "kind": kind,
        "voice_id": voice_id,
        "voice_name": voice_name,
        "language": language,
        "words_per_second": wps if source else None,
        "effective_words_per_second": wps,
        "default_words_per_second": default_wps,
        "profile_source": source,
        "profile_key": identity_profile_key(identity),
        "cloned_voice_id": cloned_voice_id,
        "duration_profile_status": duration_profile_status,
        "measure_supported": True,
    }


def _list_edge_voices(language: str) -> list[dict[str, str]]:
    from .adapters.edge_tts import list_edge_tts_voices

    lang_config = dub_language_config(language)
    voices = list_edge_tts_voices(locale_prefix=str(lang_config["edge_locale"]))
    return [{"id": voice["id"], "name": voice.get("name") or voice["id"]} for voice in voices]


def _list_google_voices(language: str) -> list[dict[str, str]]:
    from .adapters.google_tts import list_google_tts_voices

    lang_config = dub_language_config(language)
    voices = list_google_tts_voices(locale=str(lang_config["google_locale"]))
    return [{"id": voice["id"], "name": voice.get("name") or voice["id"]} for voice in voices]


def _catalog_languages(language: str | None) -> list[str]:
    raw = str(language or "all").strip().lower()
    if raw in {"", "all", "*"}:
        return sorted(SUPPORTED_DUB_LANGUAGES)
    return [normalize_dub_language(raw)]


def build_wps_catalog(
    *,
    data_dir: Path | None,
    database: Any,
    language: str | None = None,
) -> list[dict[str, Any]]:
    languages = _catalog_languages(language)
    entries: list[dict[str, Any]] = []

    # Cloned voices are language-agnostic assets; attach the first catalog language for identity.
    clone_language = languages[0] if languages else "vi"
    rows = database.connection.execute(
        """
        SELECT id, name, wav_filename, transcript, duration_profile_status
        FROM cloned_voices
        WHERE backend = ?
        ORDER BY created_at DESC
        """,
        ("omnivoice",),
    ).fetchall()
    cloned_dir = data_dir / "cloned_voices_omnivoice" if data_dir else None
    for row in rows:
        wav_path = cloned_dir / row["wav_filename"] if cloned_dir else None
        if wav_path is None or not wav_path.is_file():
            continue
        transcript = (row["transcript"] or "").strip()
        entries.append(
            _entry_from_voice(
                provider="omnivoice",
                kind="clone",
                voice_id=row["id"],
                voice_name=row["name"],
                language=clone_language,
                data_dir=data_dir,
                cloned_voice_id=row["id"],
                wav_path=wav_path,
                transcript=transcript,
                duration_profile_status=row["duration_profile_status"],
            )
        )

    for lang in languages:
        for voice in _list_edge_voices(lang):
            entries.append(
                _entry_from_voice(
                    provider="edge_tts",
                    kind="preset",
                    voice_id=voice["id"],
                    voice_name=voice["name"],
                    language=lang,
                    data_dir=data_dir,
                )
            )

        for voice in _list_google_voices(lang):
            entries.append(
                _entry_from_voice(
                    provider="google_tts",
                    kind="preset",
                    voice_id=voice["id"],
                    voice_name=voice["name"],
                    language=lang,
                    data_dir=data_dir,
                )
            )

    entries.sort(key=lambda item: (item["language"], item["provider"], item["voice_name"].lower()))
    return entries


def set_voice_wps(
    *,
    data_dir: Path | None,
    catalog_key: str,
    words_per_second: float,
    language: str | None = None,
    database: Any | None = None,
    wav_path: str | Path | None = None,
    transcript: str | None = None,
) -> dict[str, Any]:
    provider, kind, voice_id = parse_catalog_key(catalog_key)
    lang = normalize_dub_language(language)
    rate = max(MIN_WPS, min(MAX_WPS, float(words_per_second)))

    if provider == "omnivoice" and kind == "clone" and database is not None:
        row = database.connection.execute(
            "SELECT wav_filename, transcript FROM cloned_voices WHERE id = ? AND backend = ?",
            (voice_id, "omnivoice"),
        ).fetchone()
        if not row:
            raise ValueError("Cloned voice not found.")
        if data_dir:
            wav_path = data_dir / "cloned_voices_omnivoice" / row["wav_filename"]
        transcript = (row["transcript"] or transcript or "").strip()

    identity = _identity_for_entry(
        provider=provider,
        kind=kind,
        voice_id=voice_id,
        language=lang,
        wav_path=wav_path,
        transcript=transcript,
    )
    profile = save_manual_wps_profile(
        data_dir,
        identity=identity,
        words_per_second=rate,
        source="manual",
    )
    return {
        "catalog_key": catalog_key,
        "words_per_second": rate,
        "syllables_per_second": profile.get("syllables_per_second"),
        "profile_key": profile.get("profile_key"),
        "profile_source": profile.get("source"),
    }


def _synthesize_measure_sample(
    *,
    settings: dict[str, Any],
    provider: str,
    kind: str,
    voice_id: str,
    phrase: str,
    wav_path: str | Path | None = None,
    transcript: str | None = None,
) -> Path:
    from .adapters.tts import TTS_VOICE_INSTRUCT_PREFIX, create_tts_adapter, resolve_tts_voice

    measure_settings = dict(settings)
    measure_settings["tts_backend"] = provider
    if provider == "edge_tts":
        measure_settings["edge_tts_voice"] = voice_id
    elif provider == "google_tts":
        measure_settings["google_tts_voice"] = voice_id
    elif provider == "omnivoice" and kind == "clone":
        measure_settings["omnivoice_ref_audio"] = str(wav_path or "")
        measure_settings["omnivoice_ref_text"] = transcript or ""

    tts = create_tts_adapter(measure_settings)
    output_wav = Path(tempfile.gettempdir()) / f"wps_measure_{uuid4().hex}.wav"
    preview_voice = resolve_tts_voice(measure_settings)
    synthesize_kwargs: dict[str, Any] = {
        "text": phrase,
        "output_path": output_wav,
        "voice": preview_voice,
    }
    if provider == "omnivoice" and kind == "clone":
        synthesize_kwargs.update(
            clone=True,
            clone_mode="reference",
            anchor_text=transcript or "",
            voice=str(wav_path or ""),
        )
    elif provider == "omnivoice":
        instruct = str(measure_settings.get("omnivoice_instruct") or "").strip()
        if instruct:
            synthesize_kwargs["voice"] = f"{TTS_VOICE_INSTRUCT_PREFIX}{instruct}"

    tts.synthesize(**synthesize_kwargs)
    return output_wav


def measure_voice_wps(
    *,
    data_dir: Path | None,
    settings: dict[str, Any],
    catalog_key: str,
    language: str | None = None,
    database: Any | None = None,
) -> dict[str, Any]:
    provider, kind, voice_id = parse_catalog_key(catalog_key)
    lang = normalize_dub_language(language)
    if lang not in SUPPORTED_DUB_LANGUAGES:
        raise ValueError(f"Unsupported calibration language: {lang}")

    wav_path: str | Path | None = None
    transcript: str | None = None
    if provider == "omnivoice" and kind == "clone":
        if database is None:
            raise ValueError("Database is required for cloned voice measurement.")
        row = database.connection.execute(
            "SELECT wav_filename, transcript FROM cloned_voices WHERE id = ? AND backend = ?",
            (voice_id, "omnivoice"),
        ).fetchone()
        if not row:
            raise ValueError("Cloned voice not found.")
        if not data_dir:
            raise ValueError("Data directory is required for cloned voice measurement.")
        wav_path = data_dir / "cloned_voices_omnivoice" / row["wav_filename"]
        if not wav_path.is_file():
            raise ValueError("Cloned voice audio file not found.")
        transcript = (row["transcript"] or "").strip()
        if not transcript:
            raise ValueError("Cloned voice requires ref_text before auto-measure.")

    identity = _identity_for_entry(
        provider=provider,
        kind=kind,
        voice_id=voice_id,
        language=lang,
        wav_path=wav_path,
        transcript=transcript,
    )
    dataset = load_calibration_dataset(language=lang)
    target_samples = select_calibration_samples(dataset, "full")
    evaluations = []
    synthesized_count = 0

    for sample in target_samples:
        output_wav: Path | None = None
        try:
            output_wav = _synthesize_measure_sample(
                settings=settings,
                provider=provider,
                kind=kind,
                voice_id=voice_id,
                phrase=sample.text,
                wav_path=wav_path,
                transcript=transcript,
            )
            synthesized_count += 1
            evaluation = evaluate_calibration_sample(sample, wav_path=output_wav, speed=1.0, language=lang)
        except Exception:
            evaluation = evaluate_calibration_sample(sample, wav_path=None, tts_failed=True, language=lang)
        finally:
            if output_wav is not None:
                try:
                    output_wav.unlink(missing_ok=True)
                except OSError:
                    pass
        evaluations.append(evaluation)

    aggregate = aggregate_calibration_profile(evaluations)
    accepted_count = int(aggregate.get("accepted_after_outlier_filter") or 0)
    if accepted_count < 5:
        raise ValueError(
            f"Calibration produced only {accepted_count} valid samples out of {len(target_samples)}."
        )

    aggregate_sps = float(aggregate.get("syllables_per_second") or 0)
    validation = compute_validation_metrics_for_samples(
        target_samples,
        evaluations,
        profile_sps=aggregate_sps,
    )
    quality = classify_profile_quality(
        accepted_count=accepted_count,
        validation_mae_ms=validation.get("prediction_mae_ms"),
        mode="full",
        status="ready" if accepted_count >= 10 else "partial",
    )
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest = {
        "job_id": f"wps-{uuid4().hex}",
        "mode": "full",
        "dataset_version": dataset.get("version"),
        "sample_total": len(target_samples),
        "sample_synthesized": synthesized_count,
        "sample_accepted": accepted_count,
        "sample_rejected": len(target_samples) - accepted_count,
        "created_at": now,
        "updated_at": now,
    }
    profile = merge_bootstrap_profile(
        data_dir,
        identity=identity,
        aggregate=aggregate,
        validation=validation,
        manifest=manifest,
        quality=quality if quality in {"good", "partial", "poor"} else "partial",
    )
    measured_wps = max(MIN_WPS, min(MAX_WPS, round(aggregate_sps / SPS_PER_WPS, 2)))
    return {
        "catalog_key": catalog_key,
        "words_per_second": measured_wps,
        "syllables_per_second": profile.get("syllables_per_second"),
        "sample_count_total": len(target_samples),
        "sample_count_accepted": accepted_count,
        "sample_count_rejected": len(target_samples) - accepted_count,
        "prediction_mae_ms": validation.get("prediction_mae_ms"),
        "profile_source": profile.get("source"),
        "profile_key": profile.get("profile_key"),
    }
