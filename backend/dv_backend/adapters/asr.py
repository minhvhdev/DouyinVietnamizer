import os
import threading
from pathlib import Path
from typing import Any

from ..errors import AppError
from ..models import ErrorInfo

DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
SENTENCE_PUNCTUATION = set("。！？；.!?;")

_model_lock = threading.Lock()
_model_instance: Any = None
_model_cache_key: tuple[str, str, str] | None = None
_funasr_model_instance: Any = None
_funasr_model_cache_key: tuple[str, str, str] | None = None


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


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


def _group_time_stamps(time_stamps: list[Any]) -> list[dict[str, float | str]]:
    segments: list[dict[str, float | str]] = []
    current_text = ""
    current_start: float | None = None
    current_end: float | None = None

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
        if any(character in text for character in SENTENCE_PUNCTUATION):
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

    if current_text.strip() and current_start is not None and current_end is not None:
        segments.append(
            {
                "start": round(current_start, 2),
                "end": round(current_end, 2),
                "text": current_text.strip(),
            }
        )
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


def _normalize_funasr_time(value: Any) -> float:
    raw = float(value or 0.0)
    if raw >= 100 and raw == int(raw):
        return round(raw / 1000.0, 2)
    return round(raw, 2)


def _parse_funasr_segments(result: dict[str, Any]) -> list[dict[str, float | str]]:
    segments: list[dict[str, float | str]] = []
    sentence_info = result.get("sentence_info") or []
    for item in sentence_info:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "") or "").strip()
        if not text:
            continue
        start = _normalize_funasr_time(item.get("start", item.get("start_time", 0.0)))
        end = _normalize_funasr_time(item.get("end", item.get("end_time", start)))
        if end < start:
            end = start
        segment: dict[str, float | str] = {
            "start": start,
            "end": end,
            "text": text,
        }
        spk = item.get("spk")
        if spk is not None and str(spk).strip() != "":
            segment["speaker_id"] = str(spk)
        segments.append(segment)
    return segments


def _load_model(asr_model: str, aligner_model: str, device: str) -> Any:
    global _model_instance, _model_cache_key
    cache_key = (asr_model, aligner_model, device)
    with _model_lock:
        if _model_instance is not None and _model_cache_key == cache_key:
            return _model_instance

        import torch
        from qwen_asr import Qwen3ASRModel

        if not torch.cuda.is_available():
            raise AppError(
                503,
                ErrorInfo(
                    code="CUDA_UNAVAILABLE",
                    message="Qwen3-ASR requires an NVIDIA GPU with CUDA.",
                    action="Install NVIDIA drivers and a CUDA-enabled PyTorch build, then retry.",
                ),
            )

        resolved_device = device if device.startswith("cuda") else "cuda:0"
        model = Qwen3ASRModel.from_pretrained(
            asr_model,
            dtype=torch.bfloat16,
            device_map=resolved_device,
            forced_aligner=aligner_model,
            forced_aligner_kwargs={
                "dtype": torch.bfloat16,
                "device_map": resolved_device,
            },
            max_inference_batch_size=1,
            max_new_tokens=4096,
        )
        _model_instance = model
        _model_cache_key = cache_key
        return model


def _load_funasr_model(asr_model: str, aligner_model: str, device: str) -> Any:
    global _funasr_model_instance, _funasr_model_cache_key
    cache_key = (asr_model, aligner_model, device)
    with _model_lock:
        if _funasr_model_instance is not None and _funasr_model_cache_key == cache_key:
            return _funasr_model_instance

        import torch
        from funasr import AutoModel

        if not torch.cuda.is_available():
            raise AppError(
                503,
                ErrorInfo(
                    code="CUDA_UNAVAILABLE",
                    message="FunASR speaker diarization requires an NVIDIA GPU with CUDA.",
                    action="Install NVIDIA drivers and a CUDA-enabled PyTorch build, then retry.",
                ),
            )

        resolved_device = device if device.startswith("cuda") else "cuda:0"
        os.environ.setdefault("MODELSCOPE_CACHE", str(Path.home() / ".cache" / "modelscope"))
        model = AutoModel(
            model=asr_model,
            hub="hf",
            trust_remote_code=True,
            vad_model="funasr/fsmn-vad",
            spk_model="funasr/campplus",
            spk_mode="vad_segment",
            forced_aligner=aligner_model,
            forced_aligner_kwargs={
                "dtype": torch.bfloat16,
                "device_map": resolved_device,
            },
            device=resolved_device,
            disable_update=True,
            disable_pbar=True,
        )
        _funasr_model_instance = model
        _funasr_model_cache_key = cache_key
        return model


def reset_model_cache() -> None:
    global _model_instance, _model_cache_key, _funasr_model_instance, _funasr_model_cache_key
    with _model_lock:
        if _model_instance is not None:
            del _model_instance
        if _funasr_model_instance is not None:
            del _funasr_model_instance
        _model_instance = None
        _model_cache_key = None
        _funasr_model_instance = None
        _funasr_model_cache_key = None
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _transcribe_with_qwen_asr(
    model: Any,
    audio_path: Path,
    *,
    language: str,
) -> list[dict[str, float | str]]:
    results = model.transcribe(
        audio=str(audio_path),
        language=language,
        return_time_stamps=True,
    )
    if not results:
        return []
    segments = _result_to_segments(results[0])
    return [segment for segment in segments if segment["text"]]


def _transcribe_with_funasr(
    model: Any,
    audio_path: Path,
    *,
    language: str,
) -> list[dict[str, float | str]]:
    try:
        results = model.generate(
            input=str(audio_path),
            batch_size=1,
            language=language,
            return_spk_res=True,
            sentence_timestamp=True,
        )
    except Exception as error:
        raise AppError(
            502,
            ErrorInfo(
                code="FUNASR_DIARIZATION_FAILED",
                message="FunASR could not transcribe audio with speaker diarization.",
                action="Disable speaker diarization in settings or verify funasr is installed.",
                detail=str(error),
                retryable=True,
            ),
        ) from error

    if not results:
        return []

    result = results[0] if isinstance(results, list) else results
    if not isinstance(result, dict):
        return []

    segments = _parse_funasr_segments(result)
    if segments:
        return segments

    text = str(result.get("text", "") or "").strip()
    if not text:
        return []
    return [{"start": 0.0, "end": 0.0, "text": text}]


def transcribe_audio(
    audio_path: Path,
    *,
    vendor_dir: Path,
    asr_model: str = "",
    aligner_model: str = "",
    device: str = "cuda:0",
    language: str = "Chinese",
    speaker_diarization: bool = False,
) -> list[dict[str, float | str]]:
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

    if speaker_diarization:
        model = _load_funasr_model(resolved_asr, resolved_aligner, device)
        return _transcribe_with_funasr(model, audio_path, language=language)

    model = _load_model(resolved_asr, resolved_aligner, device)
    return _transcribe_with_qwen_asr(model, audio_path, language=language)
