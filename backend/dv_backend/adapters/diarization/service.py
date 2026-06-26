"""Diarization backend adapters and orchestration."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Protocol

from ...diarization_models import DiarizationOptions, DiarizationResult, DiarizationTimeline, DiarizationTurn
from ...errors import AppError
from ...gpu_lease import gpu_lease
from ...models import ErrorInfo
from ...pyannote_vendor import (
    PYANNOTE_MODEL_DIRNAME,
    huggingface_token,
    pyannote_bootstrap_action,
    validate_pyannote_model_dir,
)
from ...torchaudio_compat import apply_torchaudio_compat, bypass_lightning_inspect_stack

logger = logging.getLogger(__name__)

PYANNOTE_MODEL_ID = "pyannote/speaker-diarization-community-1"


def resolve_pyannote_local_model(cache_dir: str | None) -> Path:
    if not cache_dir:
        raise AppError(
            503,
            ErrorInfo(
                code="PYANNOTE_MODEL_NOT_BOOTSTRAPPED",
                message="Pyannote Community-1 is not available in the local vendor cache.",
                action=pyannote_bootstrap_action(token_present=bool(huggingface_token())),
                retryable=True,
            ),
        )
    candidate = (Path(cache_dir) / PYANNOTE_MODEL_DIRNAME).resolve()
    issue = validate_pyannote_model_dir(candidate)
    if issue:
        raise AppError(
            503,
            ErrorInfo(
                code="PYANNOTE_MODEL_NOT_BOOTSTRAPPED",
                message="Pyannote Community-1 files are missing or incomplete in the local vendor cache.",
                action=pyannote_bootstrap_action(token_present=bool(huggingface_token())),
                detail=f"{issue}; path={candidate}",
                retryable=True,
            ),
        )
    return candidate


def _pyannote_backend_version() -> str:
    try:
        apply_torchaudio_compat()
        import pyannote.audio

        return getattr(pyannote.audio, "__version__", "unknown")
    except Exception:
        return "unknown"


class DiarizationBackend(Protocol):
    backend_id: str

    def diarize(self, audio_path: Path, options: DiarizationOptions) -> DiarizationResult:
        ...


def _resolve_device(requested: str) -> tuple[str, bool]:
    try:
        import torch

        if str(requested).lower().startswith("cuda") and torch.cuda.is_available():
            return requested if ":" in requested else "cuda:0", False
    except Exception:
        pass
    return "cpu", True


def _load_audio_waveform_dict(audio_path: Path) -> dict[str, Any]:
    import soundfile as sf
    import torch

    waveform, sample_rate = sf.read(str(audio_path), always_2d=True)
    tensor = torch.from_numpy(waveform.T).float()
    return {"waveform": tensor, "sample_rate": int(sample_rate)}


def _normalize_pyannote_annotation(annotation: Any, *, backend: str, model: str, device: str) -> DiarizationTimeline:
    if hasattr(annotation, "speaker_diarization"):
        annotation = annotation.speaker_diarization
    turns: list[DiarizationTurn] = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        turns.append(
            DiarizationTurn(
                speaker_id=f"SPK_{speaker}",
                start=round(float(segment.start), 3),
                end=round(float(segment.end), 3),
            )
        )
    turns.sort(key=lambda turn: (turn.start, turn.end))
    return DiarizationTimeline(backend=backend, model=model, device=device, turns=turns)


def derive_exclusive_timeline(regular: DiarizationTimeline) -> DiarizationTimeline:
    """Build a one-speaker-at-a-time timeline by resolving overlaps greedily."""
    if not regular.turns:
        return DiarizationTimeline(
            backend=regular.backend,
            model=regular.model,
            device=regular.device,
            turns=[],
            metadata={"derived": True},
        )

    events: list[tuple[float, int, str]] = []
    for turn in regular.turns:
        events.append((turn.start, 1, turn.speaker_id))
        events.append((turn.end, -1, turn.speaker_id))
    events.sort(key=lambda item: (item[0], -item[1]))

    active: dict[str, int] = {}
    exclusive_turns: list[DiarizationTurn] = []
    current_speaker: str | None = None
    current_start = events[0][0]

    def dominant(active_counts: dict[str, int]) -> str | None:
        if not active_counts:
            return None
        return max(active_counts.items(), key=lambda item: item[1])[0]

    for time_point, delta, speaker in events:
        if current_speaker is not None and time_point > current_start:
            exclusive_turns.append(
                DiarizationTurn(
                    speaker_id=current_speaker,
                    start=round(current_start, 3),
                    end=round(time_point, 3),
                )
            )
        active[speaker] = active.get(speaker, 0) + delta
        if active[speaker] <= 0:
            active.pop(speaker, None)
        current_speaker = dominant(active)
        current_start = time_point

    merged: list[DiarizationTurn] = []
    for turn in exclusive_turns:
        if merged and merged[-1].speaker_id == turn.speaker_id and abs(merged[-1].end - turn.start) < 0.01:
            merged[-1].end = turn.end
        elif turn.end > turn.start:
            merged.append(turn)

    return DiarizationTimeline(
        backend=regular.backend,
        model=regular.model,
        device=regular.device,
        turns=merged,
        metadata={"derived": True, "source": "regular"},
    )


class PyannoteCommunity1Backend:
    backend_id = "pyannote_community_1"

    def diarize(self, audio_path: Path, options: DiarizationOptions) -> DiarizationResult:
        device, used_cpu_fallback = _resolve_device(options.device)
        if used_cpu_fallback:
            logger.warning("Pyannote running on CPU; diarization will be slower.")

        apply_torchaudio_compat()
        try:
            from pyannote.audio import Pipeline
        except ImportError as error:
            raise AppError(
                503,
                ErrorInfo(
                    code="PYANNOTE_NOT_INSTALLED",
                    message="Pyannote Audio is not installed.",
                    action="Run setup/bootstrap to install pyannote.audio and download the Community-1 model.",
                    detail=str(error),
                ),
            ) from error

        token = options.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        local_model = resolve_pyannote_local_model(options.model_cache_dir)
        model_ref = str(local_model.resolve())
        config_path = local_model / "config.yaml"
        started = time.perf_counter()
        previous_offline = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            with bypass_lightning_inspect_stack():
                try:
                    pipeline = Pipeline.from_pretrained(model_ref, token=token)
                except TypeError:
                    pipeline = Pipeline.from_pretrained(str(config_path.resolve()), token=token)
        except Exception as error:
            message = str(error)
            if "401" in message or "403" in message or "gated" in message.lower():
                raise AppError(
                    503,
                    ErrorInfo(
                        code="PYANNOTE_ACCESS_DENIED",
                        message="Local Pyannote model files require valid Hugging Face access during bootstrap.",
                        action=(
                            "Accept the model license at huggingface.co/pyannote/speaker-diarization-community-1 "
                            "and bootstrap the model locally with HF_TOKEN."
                        ),
                        detail=message,
                    ),
                ) from error
            raise AppError(
                503,
                ErrorInfo(
                    code="PYANNOTE_LOAD_FAILED",
                    message="Failed to load Pyannote Community-1 from the local vendor cache.",
                    action="Verify vendor/pyannote/speaker-diarization-community-1/ is complete and rerun bootstrap.",
                    detail=message,
                    retryable=True,
                ),
            ) from error
        finally:
            if previous_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = previous_offline

        try:
            import torch

            if device.startswith("cuda"):
                pipeline.to(torch.device(device))
        except Exception:
            logger.warning("Could not move Pyannote pipeline to %s; using default device.", device)

        try:
            audio_input = _load_audio_waveform_dict(audio_path)
            output = pipeline(
                audio_input,
                min_speakers=options.min_speakers,
                max_speakers=options.max_speakers,
            )
        except Exception as error:
            raise AppError(
                502,
                ErrorInfo(
                    code="PYANNOTE_DIARIZATION_FAILED",
                    message="Pyannote diarization failed on the input audio.",
                    action="Verify the audio file and Pyannote model installation.",
                    detail=str(error),
                    retryable=True,
                ),
            ) from error

        runtime_ms = round((time.perf_counter() - started) * 1000)
        regular = _normalize_pyannote_annotation(
            output,
            backend=self.backend_id,
            model=model_ref,
            device=device,
        )
        regular.metadata["offline_local_load"] = True
        regular.metadata["resolved_model_path"] = model_ref
        regular.metadata["backend_version"] = _pyannote_backend_version()
        exclusive = derive_exclusive_timeline(regular)
        exclusive.metadata["offline_local_load"] = True
        exclusive.metadata["resolved_model_path"] = model_ref
        if used_cpu_fallback:
            regular.metadata["cpu_fallback"] = True
            exclusive.metadata["cpu_fallback"] = True
        return DiarizationResult(regular=regular, exclusive=exclusive, runtime_ms=runtime_ms)


class FunASRCampPlusBackend:
    backend_id = "funasr_campp"

    def diarize(self, audio_path: Path, options: DiarizationOptions) -> DiarizationResult:
        from ...adapters.asr import DEFAULT_FUNASR_ASR_MODEL, _funasr_generate, _load_funasr_model

        device, used_cpu_fallback = _resolve_device(options.device)
        started = time.perf_counter()
        model = _load_funasr_model(DEFAULT_FUNASR_ASR_MODEL, "", device)
        segments = _funasr_generate(model, audio_path, language="Chinese")
        turns: list[DiarizationTurn] = []
        for segment in segments:
            speaker = segment.get("speaker_id") or segment.get("spk") or "SPK_00"
            turns.append(
                DiarizationTurn(
                    speaker_id=f"SPK_{speaker}" if not str(speaker).startswith("SPK_") else str(speaker),
                    start=round(float(segment["start"]), 3),
                    end=round(float(segment["end"]), 3),
                )
            )
        regular = DiarizationTimeline(
            backend=self.backend_id,
            model=DEFAULT_FUNASR_ASR_MODEL,
            device=device,
            turns=turns,
            metadata={"cpu_fallback": used_cpu_fallback},
        )
        exclusive = derive_exclusive_timeline(regular)
        runtime_ms = round((time.perf_counter() - started) * 1000)
        return DiarizationResult(regular=regular, exclusive=exclusive, runtime_ms=runtime_ms)


def get_backend(backend_id: str) -> DiarizationBackend:
    if backend_id in {"pyannote_community_1", "auto"}:
        return PyannoteCommunity1Backend()
    if backend_id == "funasr_campp":
        return FunASRCampPlusBackend()
    raise AppError(
        422,
        ErrorInfo(
            code="UNSUPPORTED_DIARIZATION_BACKEND",
            message=f"Unsupported diarization backend: {backend_id}",
            action="Choose pyannote_community_1, funasr_campp, or auto.",
        ),
    )


def run_diarization_with_fallback(
    audio_path: Path,
    options: DiarizationOptions,
    *,
    primary_backend: str,
    fallback_backend: str | None,
    job_id: str,
) -> tuple[DiarizationResult, str, str | None, str | None]:
    """Return (result, backend_used, fallback_backend, fallback_reason)."""
    primary_id = "pyannote_community_1" if primary_backend == "auto" else primary_backend
    with gpu_lease(f"job-{job_id}:diarize"):
        try:
            result = get_backend(primary_id).diarize(audio_path, options)
            return result, primary_id, None, None
        except AppError as error:
            recoverable = {
                "PYANNOTE_NOT_INSTALLED",
                "PYANNOTE_MODEL_NOT_BOOTSTRAPPED",
                "PYANNOTE_ACCESS_DENIED",
                "PYANNOTE_LOAD_FAILED",
                "PYANNOTE_DIARIZATION_FAILED",
            }
            if not fallback_backend or (
                not error.info.retryable and error.info.code not in recoverable
            ):
                raise
            logger.warning(
                "Primary diarization backend %s failed (%s); falling back to %s",
                primary_id,
                error.info.code,
                fallback_backend,
            )
            result = get_backend(fallback_backend).diarize(audio_path, options)
            return result, fallback_backend, fallback_backend, error.info.code
