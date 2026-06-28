import os
import re
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import AppError
from ..models import ErrorInfo

DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
DEFAULT_FUNASR_ASR_MODEL = "paraformer-zh"
DEFAULT_FUNASR_PUNC_MODEL = "funasr/ct-punc"
DEFAULT_FUNASR_VAD_MODEL = "funasr/fsmn-vad"
DEFAULT_FUNASR_SPK_MODEL = "funasr/campplus"
CAMPPLUS_MODEL = "iic/speech_campplus_sv_zh_en_16k-common_advanced"
FUNASR_LONG_AUDIO_BATCH_SECONDS = 300
FUNASR_MAX_SINGLE_SEGMENT_MS = 20000
MAX_SPEAKER_VOICE_SLOTS = 10
PRIMARY_SPEAKER_SLOTS = 9
MINOR_SPEAKER_SLOT = str(MAX_SPEAKER_VOICE_SLOTS - 1)
SPEAKER_CLUSTER_SIMILARITY = 0.62
SPEAKER_CONFIDENCE_LOW = 0.12
MIN_EMBEDDING_SECONDS = 0.35
SAMPLE_RATE_16K = 16000
SENTENCE_PUNCTUATION = set("。！？；.!?;")
MAX_ASR_SEGMENT_SECONDS = 6.0
MAX_ASR_SEGMENT_CHARS = 45

_model_lock = threading.Lock()
_model_instance: Any = None
_model_cache_key: tuple[str, str, str] | None = None
_funasr_model_instance: Any = None
_funasr_model_cache_key: tuple[str, str, str] | None = None
_campplus_model_instance: Any = None
_campplus_model_cache_key: tuple[str, str] | None = None


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
        duration = (current_end - current_start) if current_start is not None else 0.0
        should_flush = (
            any(character in text for character in SENTENCE_PUNCTUATION)
            or duration >= MAX_ASR_SEGMENT_SECONDS
            or len(current_text) >= MAX_ASR_SEGMENT_CHARS
        )
        if should_flush:
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
    sentence_info = result.get("sentence_info") or result.get("sentences") or []
    for item in sentence_info:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("sentence") or "").strip()
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
        spk = item.get("spk", item.get("spk_id", item.get("speaker_id", item.get("speaker"))))
        if spk is not None and str(spk).strip() != "":
            segment["speaker_id"] = str(spk)
        segments.append(segment)
    return segments


def _require_funasr_speaker_labels(segments: list[dict[str, float | str]]) -> None:
    if not segments:
        raise AppError(
            422,
            ErrorInfo(
                code="FUNASR_DIARIZATION_INCOMPLETE",
                message="FunASR did not return any timed subtitle segments.",
                action="Disable speaker diarization or verify FunASR models are installed.",
            ),
        )
    if not any(segment.get("speaker_id") for segment in segments):
        raise AppError(
            422,
            ErrorInfo(
                code="FUNASR_DIARIZATION_INCOMPLETE",
                message="FunASR did not assign speaker labels to any segment.",
                action=(
                    "Speaker diarization needs the FunASR paraformer model. "
                    "Keep diarization enabled and rerun ASR, or disable diarization to use Qwen3-ASR."
                ),
                detail=f"Received {len(segments)} segment(s) without speaker_id.",
            ),
        )


def _assign_speaker_ids_by_overlap(
    segments: list[dict[str, float | str]],
    labeled_segments: list[dict[str, float | str]],
    *,
    audio_path: Path | None = None,
) -> list[dict[str, float | str]]:
    labels = [
        segment
        for segment in labeled_segments
        if segment.get("speaker_id") is not None
    ]
    if not labels:
        return segments

    label_weights: list[float] = []
    if audio_path is not None and audio_path.is_file():
        samples = _load_mono_audio_16k(audio_path)
        for label in labels:
            label_start = float(label["start"])
            label_end = float(label.get("end") or label_start)
            label_weights.append(
                _segment_mean_rms(samples, SAMPLE_RATE_16K, label_start, label_end)
            )
    else:
        label_weights = [1.0] * len(labels)

    merged: list[dict[str, float | str]] = []
    for segment in segments:
        updated = dict(segment)
        seg_start = float(segment["start"])
        seg_end = float(segment.get("end") or seg_start)
        if seg_end <= seg_start:
            seg_end = seg_start + 0.05

        best_speaker: str | None = None
        best_score = 0.0
        for label, weight in zip(labels, label_weights, strict=True):
            label_start = float(label["start"])
            label_end = float(label.get("end") or label_start)
            if label_end <= label_start:
                label_end = label_start + 0.05
            overlap = max(0.0, min(seg_end, label_end) - max(seg_start, label_start))
            score = overlap * max(weight, 1e-6)
            if score > best_score:
                best_score = score
                best_speaker = str(label["speaker_id"])

        if best_speaker is not None:
            updated["speaker_id"] = best_speaker
        merged.append(updated)
    return merged


def _load_mono_audio_16k(audio_path: Path) -> np.ndarray:
    try:
        import librosa
    except ImportError as error:
        raise AppError(
            503,
            ErrorInfo(
                code="LIBROSA_UNAVAILABLE",
                message="Speaker diarization requires librosa for audio analysis.",
                action="Install librosa in the backend environment.",
            ),
        ) from error

    samples, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE_16K, mono=True)
    return np.asarray(samples, dtype=np.float32)


def _slice_audio(samples: np.ndarray, start: float, end: float) -> np.ndarray:
    sample_start = max(0, int(start * SAMPLE_RATE_16K))
    sample_end = min(len(samples), int(end * SAMPLE_RATE_16K))
    if sample_end <= sample_start:
        sample_end = min(len(samples), sample_start + int(MIN_EMBEDDING_SECONDS * SAMPLE_RATE_16K))
    return samples[sample_start:sample_end]


def _segment_mean_rms(
    samples: np.ndarray,
    sample_rate: int,
    start: float,
    end: float,
) -> float:
    chunk = _slice_audio(samples, start, end)
    if chunk.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))


def _write_mono_wav(path: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE_16K) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _normalize_embedding(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return vector
    return vector / norm


def _clear_campplus_model() -> None:
    global _campplus_model_instance, _campplus_model_cache_key
    if _campplus_model_instance is not None:
        del _campplus_model_instance
    _campplus_model_instance = None
    _campplus_model_cache_key = None


def _load_campplus_model(device: str) -> Any:
    global _campplus_model_instance, _campplus_model_cache_key
    resolved_device = device if device.startswith("cuda") else "cuda:0"
    cache_key = (CAMPPLUS_MODEL, resolved_device)
    with _model_lock:
        if _campplus_model_instance is not None and _campplus_model_cache_key == cache_key:
            return _campplus_model_instance

        import torch
        from funasr import AutoModel

        if not torch.cuda.is_available():
            raise AppError(
                503,
                ErrorInfo(
                    code="CUDA_UNAVAILABLE",
                    message="CampPlus speaker embedding requires an NVIDIA GPU with CUDA.",
                    action="Install NVIDIA drivers and a CUDA-enabled PyTorch build, then retry.",
                ),
            )

        os.environ.setdefault("MODELSCOPE_CACHE", str(Path.home() / ".cache" / "modelscope"))
        model = AutoModel(
            model=CAMPPLUS_MODEL,
            hub="ms",
            trust_remote_code=True,
            device=resolved_device,
            disable_update=True,
            disable_pbar=True,
        )
        _campplus_model_instance = model
        _campplus_model_cache_key = cache_key
        return model


def _extract_campplus_embedding(model: Any, samples: np.ndarray) -> np.ndarray:
    if samples.size < int(MIN_EMBEDDING_SECONDS * SAMPLE_RATE_16K):
        raise ValueError("Segment audio is too short for CampPlus embedding.")

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_path = Path(handle.name)
        _write_mono_wav(temp_path, samples)
        results = model.generate(input=str(temp_path))
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    if not results:
        raise ValueError("CampPlus returned no embedding.")

    payload = results[0] if isinstance(results, list) else results
    if not isinstance(payload, dict):
        raise ValueError("CampPlus returned an unexpected payload.")

    embedding = payload.get("spk_embedding")
    if embedding is None:
        raise ValueError("CampPlus payload did not include spk_embedding.")

    if hasattr(embedding, "detach"):
        embedding = embedding.detach().cpu().numpy()
    elif hasattr(embedding, "numpy"):
        embedding = embedding.numpy()
    return np.asarray(embedding, dtype=np.float32).reshape(-1)


def _cluster_speaker_embeddings(
    embeddings: list[np.ndarray],
    *,
    similarity_threshold: float = SPEAKER_CLUSTER_SIMILARITY,
) -> tuple[list[int], list[float]]:
    centroids: list[np.ndarray] = []
    cluster_ids: list[int] = []
    confidences: list[float] = []

    for embedding in embeddings:
        normalized = _normalize_embedding(embedding)
        if not centroids:
            centroids.append(normalized)
            cluster_ids.append(0)
            confidences.append(1.0)
            continue

        similarities = [float(np.dot(normalized, centroid)) for centroid in centroids]
        best_index = int(np.argmax(similarities))
        best_similarity = similarities[best_index]
        second_similarity = max(
            (score for index, score in enumerate(similarities) if index != best_index),
            default=0.0,
        )
        margin = best_similarity - second_similarity

        if best_similarity >= similarity_threshold:
            cluster_ids.append(best_index)
            confidences.append(max(0.0, margin))
            centroids[best_index] = _normalize_embedding(centroids[best_index] + normalized)
        else:
            centroids.append(normalized)
            cluster_ids.append(len(centroids) - 1)
            confidences.append(1.0)

    return cluster_ids, confidences


def _fill_missing_speaker_ids(
    segments: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    filled: list[dict[str, float | str]] = [dict(segment) for segment in segments]
    last_speaker: str | None = None
    last_confidence: float | None = None

    for segment in filled:
        speaker_id = segment.get("speaker_id")
        if speaker_id is not None:
            last_speaker = str(speaker_id)
            last_confidence = (
                float(segment["speaker_confidence"])
                if segment.get("speaker_confidence") is not None
                else None
            )
            continue
        if last_speaker is not None:
            segment["speaker_id"] = last_speaker
            segment["speaker_confidence"] = round(
                min(last_confidence or SPEAKER_CONFIDENCE_LOW, SPEAKER_CONFIDENCE_LOW),
                3,
            )

    next_speaker: str | None = None
    next_confidence: float | None = None
    for segment in reversed(filled):
        speaker_id = segment.get("speaker_id")
        if speaker_id is not None:
            next_speaker = str(speaker_id)
            next_confidence = (
                float(segment["speaker_confidence"])
                if segment.get("speaker_confidence") is not None
                else None
            )
            continue
        if next_speaker is not None:
            segment["speaker_id"] = next_speaker
            segment["speaker_confidence"] = round(
                min(next_confidence or SPEAKER_CONFIDENCE_LOW, SPEAKER_CONFIDENCE_LOW),
                3,
            )

    return filled


def _assign_speakers_with_campplus(
    audio_path: Path,
    segments: list[dict[str, float | str]],
    *,
    device: str,
) -> list[dict[str, float | str]]:
    if not segments:
        return []

    samples = _load_mono_audio_16k(audio_path)
    model = _load_campplus_model(device)

    embeddings: list[np.ndarray | None] = []
    for segment in segments:
        start = float(segment["start"])
        end = float(segment.get("end") or start)
        duration = max(0.0, end - start)
        if duration < MIN_EMBEDDING_SECONDS:
            embeddings.append(None)
            continue
        try:
            slice_samples = _slice_audio(samples, start, end)
            embeddings.append(_extract_campplus_embedding(model, slice_samples))
        except ValueError:
            embeddings.append(None)

    valid_embeddings = [embedding for embedding in embeddings if embedding is not None]
    if not valid_embeddings:
        raise AppError(
            422,
            ErrorInfo(
                code="SPEAKER_EMBEDDING_FAILED",
                message="CampPlus could not extract speaker embeddings for this audio.",
                action="Disable speaker diarization or verify CampPlus is installed.",
            ),
        )

    cluster_ids, confidences = _cluster_speaker_embeddings(valid_embeddings)

    labeled: list[dict[str, float | str]] = []
    cluster_pointer = 0
    for segment, embedding in zip(segments, embeddings, strict=True):
        updated = dict(segment)
        if embedding is None:
            labeled.append(updated)
            continue
        updated["speaker_id"] = str(cluster_ids[cluster_pointer])
        updated["speaker_confidence"] = round(confidences[cluster_pointer], 3)
        cluster_pointer += 1
        labeled.append(updated)

    labeled = _fill_missing_speaker_ids(labeled)
    if not any(segment.get("speaker_id") for segment in labeled):
        raise AppError(
            422,
            ErrorInfo(
                code="SPEAKER_EMBEDDING_FAILED",
                message="Speaker diarization did not assign any speaker labels.",
                action="Disable speaker diarization or retry ASR.",
            ),
        )
    return labeled


def _split_qwen_text_by_diarization_timing(
    qwen_text: str,
    diar_segments: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    cleaned = re.sub(r"\s+", " ", (qwen_text or "").strip())
    if not cleaned or not diar_segments:
        return []

    total_chars = len(cleaned)
    total_time = sum(
        max(0.05, float(segment.get("end") or segment["start"]) - float(segment["start"]))
        for segment in diar_segments
    )
    if total_time <= 0:
        return []

    split_segments: list[dict[str, float | str]] = []
    offset = 0
    for index, segment in enumerate(diar_segments):
        start = float(segment["start"])
        end = float(segment.get("end") or start)
        duration = max(0.05, end - start)
        if index == len(diar_segments) - 1:
            piece = cleaned[offset:]
        else:
            char_count = max(1, round(total_chars * (duration / total_time)))
            piece = cleaned[offset : offset + char_count]
            offset += char_count
        piece = piece.strip()
        if not piece:
            continue
        entry: dict[str, float | str] = {
            "start": round(start, 2),
            "end": round(end, 2),
            "text": piece,
        }
        speaker_id = segment.get("speaker_id")
        if speaker_id is not None:
            entry["speaker_id"] = str(speaker_id)
        split_segments.append(entry)
    return split_segments


def _merge_qwen_and_diarization_segments(
    qwen_segments: list[dict[str, float | str]],
    diar_segments: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    if len(qwen_segments) <= 1 and len(diar_segments) >= 2:
        qwen_text = str(qwen_segments[0].get("text") or "") if qwen_segments else ""
        split_segments = _split_qwen_text_by_diarization_timing(qwen_text, diar_segments)
        if split_segments:
            return split_segments
    return _assign_speaker_ids_by_overlap(qwen_segments, diar_segments)


def _remap_speaker_ids_to_slots(
    segments: list[dict[str, float | str]],
    *,
    max_speakers: int = MAX_SPEAKER_VOICE_SLOTS,
    primary_speakers: int = PRIMARY_SPEAKER_SLOTS,
) -> list[dict[str, float | str]]:
    if max_speakers <= 0:
        return segments

    minor_slot = str(max_speakers - 1)
    primary_count = min(primary_speakers, max_speakers - 1)

    durations: dict[str, float] = {}
    for segment in segments:
        speaker_id = segment.get("speaker_id")
        if speaker_id is None:
            continue
        key = str(speaker_id)
        start = float(segment["start"])
        end = float(segment.get("end") or start)
        durations[key] = durations.get(key, 0.0) + max(0.05, end - start)

    ranked = sorted(durations.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) <= primary_count:
        mapping = {speaker: str(index) for index, (speaker, _) in enumerate(ranked)}
    else:
        mapping = {
            speaker: str(index)
            for index, (speaker, _) in enumerate(ranked[:primary_count])
        }
        for speaker, _duration in ranked[primary_count:]:
            mapping[speaker] = minor_slot

    remapped: list[dict[str, float | str]] = []
    for segment in segments:
        updated = dict(segment)
        speaker_id = segment.get("speaker_id")
        if speaker_id is not None:
            updated["speaker_id"] = mapping.get(str(speaker_id), minor_slot)
        remapped.append(updated)
    return remapped


def _clear_funasr_model() -> None:
    global _funasr_model_instance, _funasr_model_cache_key
    if _funasr_model_instance is not None:
        del _funasr_model_instance
    _funasr_model_instance = None
    _funasr_model_cache_key = None


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


def _load_funasr_model(funasr_model: str, aligner_model: str, device: str) -> Any:
    global _funasr_model_instance, _funasr_model_cache_key
    cache_key = (funasr_model, aligner_model, device)
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
            model=funasr_model or DEFAULT_FUNASR_ASR_MODEL,
            hub="hf",
            trust_remote_code=True,
            vad_model=DEFAULT_FUNASR_VAD_MODEL,
            punc_model=DEFAULT_FUNASR_PUNC_MODEL,
            spk_model=DEFAULT_FUNASR_SPK_MODEL,
            spk_mode="punc_segment",
            vad_kwargs={"max_single_segment_time": FUNASR_MAX_SINGLE_SEGMENT_MS},
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
        _clear_funasr_model()
        _clear_campplus_model()
        _model_instance = None
        _model_cache_key = None
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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


def _funasr_generate(
    model: Any,
    audio_path: Path,
    *,
    language: str,
) -> list[dict[str, float | str]]:
    try:
        results = model.generate(
            input=str(audio_path),
            batch_size_s=FUNASR_LONG_AUDIO_BATCH_SECONDS,
            batch_size_threshold_s=60,
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
    return _parse_funasr_segments(result)


def _transcribe_with_diarization_hybrid(
    *,
    vendor_dir: Path,
    audio_path: Path,
    diarization_audio_path: Path | None = None,
    asr_model: str,
    aligner_model: str,
    funasr_model: str,
    device: str,
    language: str,
) -> list[dict[str, float | str]]:
    del vendor_dir, funasr_model

    qwen = _load_model(asr_model, aligner_model, device)
    qwen_segments = _transcribe_with_qwen_asr(qwen, audio_path, language=language)
    if not qwen_segments:
        raise AppError(
            422,
            ErrorInfo(
                code="EMPTY_ASR_TRANSCRIPTION",
                message="ASR completed without detecting any spoken text.",
                action="Verify the source audio and ASR models, then retry.",
            ),
        )

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
    except Exception:
        pass

    diar_audio = diarization_audio_path or audio_path
    labeled = _assign_speakers_with_campplus(
        diar_audio,
        qwen_segments,
        device=device,
    )
    _clear_campplus_model()
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return _remap_speaker_ids_to_slots(labeled)


def _transcribe_with_funasr(
    model: Any,
    audio_path: Path,
    *,
    language: str,
) -> list[dict[str, float | str]]:
    segments = _funasr_generate(model, audio_path, language=language)
    if segments:
        _require_funasr_speaker_labels(segments)
        return segments

    raise AppError(
        422,
        ErrorInfo(
            code="FUNASR_DIARIZATION_INCOMPLETE",
            message="FunASR returned plain text without speaker/timestamp segments.",
            action=(
                "Speaker diarization requires FunASR paraformer output. "
                "Rerun ASR after models download, or disable speaker diarization."
            ),
            detail="Empty FunASR sentence_info.",
        ),
    )


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

    from ..gpu_lease import gpu_lease

    with gpu_lease(f"asr:{audio_path.name}"):
        model = _load_model(resolved_asr, resolved_aligner, device)
        if include_alignment or speaker_diarization:
            return _transcribe_details_with_qwen(model, audio_path, language=language)
        return _transcribe_with_qwen_asr(model, audio_path, language=language)
