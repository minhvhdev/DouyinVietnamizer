import os
import sys
import threading
from pathlib import Path
from typing import Any

from ..errors import AppError
from ..models import ErrorInfo

DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
SENTENCE_PUNCTUATION = set("。！？；.!?;")
MAX_ASR_SEGMENT_SECONDS = 12.0
MAX_ASR_SEGMENT_CHARS = 72

_model_lock = threading.Lock()
_model_instance: Any = None
_model_cache_key: tuple[str, str, str] | None = None


def cuda_available() -> bool:
    from ..hardware import accelerator_available

    return accelerator_available()


def _require_asr_accelerator() -> None:
    from ..hardware import accelerator_available, default_inference_device

    if accelerator_available():
        return
    raise AppError(
        503,
        ErrorInfo(
            code="ACCELERATOR_UNAVAILABLE",
            message="Qwen3-ASR requires a GPU accelerator (NVIDIA CUDA or Apple MPS).",
            action=(
                "Use a CUDA-capable GPU on Windows/Linux, or run on Apple Silicon with "
                f"PyTorch MPS enabled (resolved device: {default_inference_device()})."
            ),
        ),
    )


def _resolve_asr_device(device: str) -> str:
    from ..hardware import resolve_inference_device

    return resolve_inference_device(device)


def configure_gpu_manager(settings: dict | None) -> None:
    """Apply pipeline settings to the global GpuModelManager."""
    from ..gpu_manager import global_gpu_manager

    settings = settings or {}
    manager = global_gpu_manager()
    try:
        idle_sec = float(settings.get("gpu_model_idle_timeout_sec", 60.0) or 60.0)
    except (TypeError, ValueError):
        idle_sec = 60.0
    try:
        max_models = int(settings.get("gpu_max_resident_models", 1) or 1)
    except (TypeError, ValueError):
        max_models = 1
    manager.configure(
        idle_timeout_sec=idle_sec,
        keep_warm=bool(settings.get("gpu_keep_warm_enabled", True)),
        max_resident_families=max(1, max_models),
    )


def resolve_model_reference(vendor_dir: Path, configured_path: str, default_hf_id: str) -> str:
    if configured_path:
        path = Path(configured_path)
        if path.is_dir():
            return str(path)
        if path.is_file():
            return str(path)
    local_name = default_hf_id.split("/")[-1]
    bundled = vendor_dir / "qwen3-asr" / local_name
    if bundled.is_dir():
        return str(bundled)
    return default_hf_id


def _split_at_last_sentence_punctuation(text: str) -> tuple[str, str] | None:
    for index in range(len(text) - 1, -1, -1):
        if text[index] in SENTENCE_PUNCTUATION:
            head = text[: index + 1].strip()
            tail = text[index + 1 :].strip()
            if head and tail:
                return head, tail
    return None


def _group_time_stamps(time_stamps: list[Any]) -> list[dict[str, float | str]]:
    segments: list[dict[str, float | str]] = []
    current_text = ""
    current_start: float | None = None
    current_end: float | None = None

    def flush_buffer() -> None:
        nonlocal current_text, current_start, current_end
        if not current_text.strip() or current_start is None or current_end is None:
            return
        segments.append(
            {
                "start": round(current_start, 2),
                "end": round(current_end, 2),
                "text": current_text.strip(),
            }
        )
        current_text = ""
        current_start = None
        current_end = None

    for stamp in time_stamps:
        text = str(getattr(stamp, "text", "") or "").strip()
        if not text:
            continue
        start = float(getattr(stamp, "start_time", 0.0))
        end = float(getattr(stamp, "end_time", start))
        if current_start is None:
            current_start = start
        current_end = end
        current_text += text
        duration = (current_end - current_start) if current_start is not None else 0.0
        punct_in_token = any(character in text for character in SENTENCE_PUNCTUATION)
        hit_limit = duration >= MAX_ASR_SEGMENT_SECONDS or len(current_text) >= MAX_ASR_SEGMENT_CHARS
        should_flush = punct_in_token or hit_limit
        if not should_flush:
            continue

        if hit_limit and not punct_in_token and current_start is not None and current_end is not None:
            split = _split_at_last_sentence_punctuation(current_text)
            if split is not None:
                head, tail = split
                total_len = max(1, len(current_text))
                split_time = current_start + (current_end - current_start) * (len(head) / total_len)
                segments.append(
                    {
                        "start": round(current_start, 2),
                        "end": round(split_time, 2),
                        "text": head,
                    }
                )
                current_text = tail
                current_start = split_time
                continue

        flush_buffer()

    if current_text.strip() and current_start is not None and current_end is not None:
        flush_buffer()
    return segments


def _result_to_segments(result: Any) -> list[dict[str, float | str]]:
    text = str(getattr(result, "text", "") or "").strip()
    time_stamps = getattr(result, "time_stamps", None) or []
    if time_stamps:
        segments = _group_time_stamps(list(time_stamps))
        if segments:
            return segments
    if not text:
        return []
    end = 0.0
    if time_stamps:
        end = float(getattr(time_stamps[-1], "end_time", 0.0))
    return [{"start": 0.0, "end": round(end, 2), "text": text}]


def _load_model(asr_model: str, aligner_model: str, device: str) -> Any:
    global _model_instance, _model_cache_key
    resolved_device = _resolve_asr_device(device)
    cache_key = (asr_model, aligner_model, resolved_device)
    with _model_lock:
        if _model_instance is not None and _model_cache_key == cache_key:
            return _model_instance

        from qwen_asr import Qwen3ASRModel

        from ..hardware import inference_dtype_for_device

        _require_asr_accelerator()
        dtype = inference_dtype_for_device(resolved_device)
        model = Qwen3ASRModel.from_pretrained(
            asr_model,
            dtype=dtype,
            device_map=resolved_device,
            forced_aligner=aligner_model,
            forced_aligner_kwargs={
                "dtype": dtype,
                "device_map": resolved_device,
            },
            max_inference_batch_size=1,
            max_new_tokens=4096,
        )
        _model_instance = model
        _model_cache_key = cache_key
        return model


def reset_model_cache() -> None:
    global _model_instance, _model_cache_key
    with _model_lock:
        if _model_instance is not None:
            del _model_instance
        _model_instance = None
        _model_cache_key = None
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif sys.platform == "darwin" and torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


def _extract_aligned_units(result: Any) -> list[dict[str, float | str]]:
    """Extract finest-grain aligned units from Qwen forced-aligner timestamps."""
    time_stamps = getattr(result, "time_stamps", None) or []
    units: list[dict[str, float | str]] = []
    for stamp in time_stamps:
        text = str(getattr(stamp, "text", "") or "")
        if not text:
            continue
        start = float(getattr(stamp, "start_time", 0.0))
        end = float(getattr(stamp, "end_time", start))
        if end <= start:
            end = start + 0.01
        units.append({"text": text, "start": round(start, 3), "end": round(end, 3)})
    return units


def _transcribe_with_qwen_asr(
    model: Any,
    audio_path: Path,
    *,
    language: str,
) -> list[dict[str, float | str]]:
    details = _transcribe_details_with_qwen(model, audio_path, language=language)
    return details["segments"]


def _transcribe_details_with_qwen(
    model: Any,
    audio_path: Path,
    *,
    language: str,
) -> dict[str, list[dict[str, float | str]]]:
    results = model.transcribe(
        audio=str(audio_path),
        language=language,
        return_time_stamps=True,
    )
    if not results:
        return {"segments": [], "aligned_units": []}
    result = results[0]
    segments = [segment for segment in _result_to_segments(result) if segment.get("text")]
    aligned_units = _extract_aligned_units(result)
    if not aligned_units and segments:
        aligned_units = [
            {
                "text": str(segment["text"]),
                "start": float(segment["start"]),
                "end": float(segment["end"]),
            }
            for segment in segments
        ]
    return {"segments": segments, "aligned_units": aligned_units}


def transcribe_audio(
    audio_path: Path,
    *,
    vendor_dir: Path,
    asr_model: str = "",
    aligner_model: str = "",
    device: str = "cuda:0",
    language: str = "Chinese",
    speaker_diarization: bool = False,
    funasr_model: str = "",
    diarization_audio_path: Path | None = None,
    include_alignment: bool = False,
) -> list[dict[str, float | str]] | dict[str, list[dict[str, float | str]]]:
    del funasr_model, diarization_audio_path
    resolved_asr = resolve_model_reference(
        vendor_dir,
        asr_model,
        DEFAULT_ASR_MODEL,
    )
    resolved_aligner = resolve_model_reference(
        vendor_dir,
        aligner_model,
        DEFAULT_ALIGNER_MODEL,
    )
    device = _resolve_asr_device(device)

    from ..gpu_manager import global_gpu_manager

    model_key = f"{resolved_asr}|{resolved_aligner}|{device}"
    manager = global_gpu_manager()

    def loader() -> None:
        _load_model(resolved_asr, resolved_aligner, device)

    with manager.acquire("asr", device, model_key, loader=loader):
        model = _load_model(resolved_asr, resolved_aligner, device)
        if include_alignment or speaker_diarization:
            return _transcribe_details_with_qwen(model, audio_path, language=language)
        return _transcribe_with_qwen_asr(model, audio_path, language=language)
