import os
from pathlib import Path
import re
import shutil
import wave
from typing import Any

from ..errors import AppError
from ..models import ErrorInfo
from ..omnivoice_mps import plan_omnivoice_device

SUPPORTED_TTS_BACKENDS = ("omnivoice", "edge_tts", "google_tts", "gemini_tts")
CLOUD_TTS_BACKENDS = frozenset({"edge_tts", "google_tts", "gemini_tts"})
GEMINI_TTS_VOICES = (
    {"id": "Zephyr", "name": "Zephyr (Bright)"},
    {"id": "Puck", "name": "Puck (Upbeat)"},
    {"id": "Charon", "name": "Charon (Informative)"},
    {"id": "Kore", "name": "Kore (Firm)"},
    {"id": "Fenrir", "name": "Fenrir (Excitable)"},
    {"id": "Aoede", "name": "Aoede (Breezy)"},
)
OMNIVOICE_DEFAULT_MODEL = "k2-fsa/OmniVoice"
TTS_VOICE_INSTRUCT_PREFIX = "instruct:"
MAX_TTS_CHARS = 450
OMNIVOICE_MAX_TTS_CHARS = 240
OMNIVOICE_EXTERNAL_SPLIT_CHARS = 280
_TTS_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…。，！？；;])\s+")
_OMNI_TTS_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…。，！？；;,:\-—])\s+")


def sanitize_tts_text(text: str) -> str:
    """Remove characters that tokenizer/audio backends cannot encode.

    Some upstream ASR/translation paths can preserve lone UTF-16 surrogate
    code points from malformed source text. HuggingFace tokenizers reject
    those with ``TextEncodeInput`` errors, so strip them before chunking.
    """
    return "".join(
        character
        for character in (text or "")
        if not (0xD800 <= ord(character) <= 0xDFFF)
    )


def prepare_spoken_text_for_tts(text: str, *, speech_duration: float | None = None) -> str:
    """Normalize text immediately before TTS synthesis."""
    cleaned = sanitize_tts_text(text).strip()
    compact = re.sub(r"\s+", "", cleaned)
    is_short = (speech_duration is not None and speech_duration < 1.2) or len(compact) < 10
    if is_short:
        cleaned = re.sub(r"^\.{2,}\s*", "", cleaned).strip()
    return cleaned


def split_tts_text(text: str, *, max_chars: int = MAX_TTS_CHARS) -> list[str]:
    cleaned = re.sub(r"\s+", " ", sanitize_tts_text(text).strip())
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    sentences = [
        part.strip()
        for part in _TTS_SENTENCE_SPLIT_RE.split(cleaned)
        if part.strip()
    ] or [cleaned]

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(sentence) <= max_chars:
            current = sentence
            continue
        for offset in range(0, len(sentence), max_chars):
            chunks.append(sentence[offset : offset + max_chars].strip())
        current = ""
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def estimate_omnivoice_duration_sec(text: str, *, speed: float = 1.0, buffer: float = 1.45) -> float:
    """Estimate narration duration for OmniVoice clone TTS.

    OmniVoice prepends ``ref_text`` to the spoken prompt when it is set. For dubbing
    we pass an empty ``ref_text`` and must supply an explicit duration that matches
    only the target narration text.
    """
    cleaned = re.sub(r"\s+", " ", sanitize_tts_text(text).strip())
    if not cleaned:
        return 2.0
    chars_per_sec = 10.5 * max(0.5, min(2.0, float(speed)))
    return max(2.0, min(45.0, (len(cleaned) / chars_per_sec) * max(1.0, float(buffer))))


_OMNI_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…。！？])\s+")


def split_omnivoice_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", sanitize_tts_text(text).strip())
    if not cleaned:
        return []
    parts = [part.strip() for part in _OMNI_SENTENCE_SPLIT_RE.split(cleaned) if part.strip()]
    return parts or [cleaned]


def _append_word_safe_chunks(chunks: list[str], sentence: str, max_chars: int) -> None:
    """Split an overlong sentence on spaces; never cut mid-token when a space exists."""
    start = 0
    length = len(sentence)
    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            space = sentence.rfind(" ", start, end)
            if space > start:
                end = space
            # else: no space in window — hard-cut unbroken overlong token as last resort
        piece = sentence[start:end].strip()
        if piece:
            chunks.append(piece)
        if end <= start:
            end = min(start + max_chars, length)
        if end < length and sentence[end : end + 1].isspace():
            end += 1
        start = end if end > start else start + max_chars


def split_omnivoice_tts_text(text: str, *, max_chars: int = OMNIVOICE_MAX_TTS_CHARS) -> list[str]:
    """Split narration for OmniVoice at punctuation and word boundaries.

    Smaller chunks than the generic TTS splitter reduce dropped syllables when
    the model synthesizes long Vietnamese segments.
    """
    cleaned = re.sub(r"\s+", " ", sanitize_tts_text(text).strip())
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    sentences = [
        part.strip()
        for part in _OMNI_TTS_SENTENCE_SPLIT_RE.split(cleaned)
        if part.strip()
    ] or [cleaned]

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(sentence) <= max_chars:
            current = sentence
            continue
        _append_word_safe_chunks(chunks, sentence, max_chars)
        current = ""
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def parse_tts_voice_string(voice: str | None) -> tuple[str | None, str | None, str | None]:
    """Return ``(prompt_wav_path, prompt_text, voice_design)`` for a voice string."""
    value = str(voice or "auto").strip()
    if not value or value.lower() == "auto":
        return None, None, None
    if value.startswith(TTS_VOICE_INSTRUCT_PREFIX):
        voice_design = value[len(TTS_VOICE_INSTRUCT_PREFIX):].strip()
        return None, None, voice_design or None
    path = Path(value)
    if path.is_file():
        return str(path), None, None
    return None, None, None


def tts_backend_from_settings(settings: dict) -> str:
    backend = str(settings.get("tts_backend") or "omnivoice").strip().lower()
    if backend in SUPPORTED_TTS_BACKENDS:
        return backend
    return "omnivoice"


def is_cloud_tts_backend(backend: str | None = None, *, settings: dict | None = None) -> bool:
    resolved = (backend or (tts_backend_from_settings(settings or {}) if settings else "") or "").strip().lower()
    return resolved in CLOUD_TTS_BACKENDS


def resolve_tts_voice(settings: dict) -> str:
    from ..dubbing_languages import dub_language_from_settings, dub_language_config

    lang_config = dub_language_config(dub_language_from_settings(settings))
    backend = tts_backend_from_settings(settings)
    if backend == "edge_tts":
        default_voice = str(lang_config["default_edge_voice"])
        return str(settings.get("edge_tts_voice") or default_voice).strip() or default_voice
    if backend == "google_tts":
        default_voice = str(lang_config["default_google_voice"])
        return str(settings.get("google_tts_voice") or default_voice).strip() or default_voice
    if backend == "gemini_tts":
        return str(settings.get("gemini_tts_voice") or "Zephyr").strip() or "Zephyr"
    if backend == "omnivoice":
        instruct = str(settings.get("omnivoice_instruct") or "").strip()
        if instruct:
            return f"{TTS_VOICE_INSTRUCT_PREFIX}{instruct}"
        ref_audio = str(settings.get("omnivoice_ref_audio") or "").strip()
        if ref_audio:
            return ref_audio
        return "auto"
    return "auto"


def resolve_omnivoice_device(device: str | None) -> str:
    allow_cpu_fallback = os.environ.get("DV_OMNIVOICE_ALLOW_CPU_FALLBACK", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return plan_omnivoice_device(
        device,
        allow_cpu_fallback=allow_cpu_fallback,
    ).resolved_device


def _wav_format_key(params: wave._wave_params) -> tuple:
    return (
        params.nchannels,
        params.sampwidth,
        params.framerate,
        params.comptype,
        params.compname,
    )


def _read_wav_mono_float(path: Path) -> tuple["object", int, tuple]:
    import numpy as np

    with wave.open(str(path), "rb") as wav_file:
        params = wav_file.getparams()
        sample_rate = int(wav_file.getframerate())
        raw = wav_file.readframes(wav_file.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if params.nchannels > 1:
        samples = samples.reshape(-1, params.nchannels).mean(axis=1)
    return samples, sample_rate, params


def _concat_wav_files(paths: list[Path], output_path: Path) -> None:
    if not paths:
        raise ValueError("No WAV files to concatenate.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(paths) == 1:
        shutil.copy2(paths[0], output_path)
        return

    with wave.open(str(paths[0]), "rb") as first:
        format_key = _wav_format_key(first.getparams())
        params = first.getparams()
        frames = [first.readframes(first.getnframes())]
    for path in paths[1:]:
        with wave.open(str(path), "rb") as wav_file:
            if _wav_format_key(wav_file.getparams()) != format_key:
                raise ValueError(f"Incompatible WAV format: {path}")
            frames.append(wav_file.readframes(wav_file.getnframes()))
    with wave.open(str(output_path), "wb") as output:
        output.setparams(params)
        for frame in frames:
            output.writeframes(frame)


def _concat_wav_files_crossfade(paths: list[Path], output_path: Path, *, gap_ms: int = 100) -> None:
    """Join chunk WAVs with a short silence gap to avoid clipped words at seams."""
    import numpy as np

    if not paths:
        raise ValueError("No WAV files to concatenate.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(paths) == 1:
        shutil.copy2(paths[0], output_path)
        return

    arrays: list[np.ndarray] = []
    sample_rate: int | None = None
    params = None
    for path in paths:
        samples, rate, file_params = _read_wav_mono_float(path)
        if sample_rate is None:
            sample_rate = rate
            params = file_params
        elif rate != sample_rate:
            raise ValueError(f"Incompatible sample rate: {path}")
        arrays.append(samples)

    assert sample_rate is not None and params is not None
    gap = np.zeros(max(1, int(sample_rate * gap_ms / 1000)), dtype=np.float32)
    merged = arrays[0]
    for samples in arrays[1:]:
        merged = np.concatenate([merged, gap, samples])
    pcm16 = np.clip(merged, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    with wave.open(str(output_path), "wb") as output:
        mono_params = params._replace(nchannels=1)
        output.setparams(mono_params)
        output.writeframes(pcm16.tobytes())



class TtsSession:
    def __init__(
        self,
        settings: dict,
        *,
        data_dir: Path | None,
        runner: object | None,
        adapter_factory=create_tts_adapter if "create_tts_adapter" in globals() else None,
        gpu_manager=None,
    ) -> None:
        self.settings = dict(settings)
        self.data_dir = data_dir
        self.runner = runner
        self._adapter_factory = adapter_factory or create_tts_adapter
        self._adapter = None
        self.backend = tts_backend_from_settings(self.settings)
        self.voice = resolve_tts_voice(self.settings)
        self.clone = self.backend == "omnivoice" and bool(
            str(self.settings.get("omnivoice_ref_audio") or "").strip()
        )
        self.clone_mode = "reference"
        self.anchor_text = self._anchor_transcript() if self.clone else None
        if gpu_manager is None:
            from ..gpu_manager import global_gpu_manager
            self.gpu_manager = global_gpu_manager()
        else:
            self.gpu_manager = gpu_manager
        self._lease = None
        self._model_key = self._build_model_key()
        self._last_batch_mode: str | None = None
        self._last_batch_diagnostics: dict[str, Any] = {}

    def _build_model_key(self) -> str:
        if self.backend == "omnivoice":
            model = str(self.settings.get("omnivoice_model", "") or OMNIVOICE_DEFAULT_MODEL)
            device = resolve_omnivoice_device(str(self.settings.get("omnivoice_device", "cuda:0") or "cuda:0"))
            steps = self.settings.get("omnivoice_num_steps", 32)
            return f"{model}|{device}|{int(steps or 0)}|1.0"
        if is_cloud_tts_backend(self.backend):
            return f"{self.backend}|{self.voice}"
        raise AppError(
            422,
            ErrorInfo(
                code="UNSUPPORTED_TTS_BACKEND",
                message=f"Unsupported TTS backend for session: {self.backend}",
                action="Use omnivoice, edge_tts, google_tts, or gemini_tts.",
            ),
        )

    def __enter__(self) -> "TtsSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _adapter_instance(self):
        if self._adapter is not None:
            return self._adapter
        device = resolve_omnivoice_device(str(self.settings.get("omnivoice_device", "cuda:0") or "cuda:0")) if self.backend == "omnivoice" else "cpu"
        if (
            self.gpu_manager is not None
            and not self._session_disabled()
            and not is_cloud_tts_backend(self.backend)
        ):
            self._lease = self.gpu_manager.acquire("tts", device, self._model_key, serialize=False)
            self._lease.__enter__()
        try:
            self._adapter = self._adapter_factory(self.settings, data_dir=self.data_dir, runner=self.runner)
        except Exception:
            if self._lease is not None:
                self._lease.__exit__(None, None, None)
                self._lease = None
            raise
        return self._adapter

    def _session_disabled(self) -> bool:
        return not bool(self.settings.get("tts_session_reuse_enabled", True))

    def _anchor_transcript(self) -> str | None:
        anchor, source = self._anchor_transcript_meta()
        if anchor is not None:
            try:
                from ..omnivoice_diagnostics import diagnostics_enabled, file_content_hash, log_event, short_hash

                if diagnostics_enabled():
                    ref_audio = str(self.settings.get("omnivoice_ref_audio") or "").strip()
                    log_event(
                        "tts_session_anchor_resolve",
                        {
                            "ref_audio_hash": file_content_hash(ref_audio),
                            "anchor_text_hash": short_hash(anchor),
                            "anchor_text_length": len(anchor),
                            "anchor_source": source,
                        },
                    )
            except Exception:
                pass
        return anchor

    def _anchor_transcript_meta(self) -> tuple[str | None, str]:
        ref_audio = str(self.settings.get("omnivoice_ref_audio") or "").strip()
        if not ref_audio:
            return None, "none"
        if self.backend == "omnivoice":
            manual = str(self.settings.get("omnivoice_ref_text") or "").strip()
            if manual:
                if len(manual) > 400:
                    return manual[:400].rstrip(), "explicit_omnivoice_ref_text_truncated"
                return manual, "explicit_omnivoice_ref_text"
        sidecar = Path(ref_audio).with_suffix(".txt")
        if not sidecar.is_file():
            return None, "missing_sidecar"
        try:
            transcript = sidecar.read_text(encoding="utf-8").strip()
        except OSError:
            return None, "sidecar_read_error"
        if not transcript:
            return None, "empty_sidecar"
        # Long anchor transcripts crash or stall GGUF ultimate mode; keep a
        # short prefix that still conditions timbre without blowing the graph.
        if len(transcript) > 400:
            return transcript[:400].rstrip(), "sidecar_truncated"
        return transcript, "sidecar"

    def synthesize(self, text: str, output_path: Path, *, segment: dict | None = None) -> None:
        segment = segment or {}
        if not bool(self.settings.get("tts_session_reuse_enabled", True)):
            factory = self._adapter_factory
            for attempt in range(2):
                try:
                    factory(self.settings, data_dir=self.data_dir, runner=self.runner).synthesize(
                        text,
                        output_path,
                        voice=self.voice,
                        ref_text=str(segment.get("text") or "").strip() or None,
                        clone=self.clone,
                        clone_mode=self.clone_mode,
                        anchor_text=self.anchor_text,
                        segment=segment,
                    )
                    return
                except AppError as error:
                    if not error.info.retryable or attempt:
                        raise
                    self._reset_caches_after_failure()
            return
        last_error: AppError | None = None
        for attempt in range(2):
            try:
                self._adapter_instance().synthesize(
                    text,
                    output_path,
                    voice=self.voice,
                    ref_text=str(segment.get("text") or "").strip() or None,
                    clone=self.clone,
                    clone_mode=self.clone_mode,
                    anchor_text=self.anchor_text,
                    segment=segment,
                )
                return
            except AppError as error:
                last_error = error
                if not error.info.retryable or attempt:
                    raise
                self._reset_caches_after_failure()
        if last_error is not None:
            raise last_error

    @property
    def last_batch_mode(self) -> str | None:
        return self._last_batch_mode

    @property
    def last_batch_diagnostics(self) -> dict[str, Any]:
        return dict(self._last_batch_diagnostics)

    def _apply_adapter_batch_diagnostics(self, adapter: object) -> None:
        diagnostics = getattr(adapter, "last_batch_diagnostics", None)
        if isinstance(diagnostics, dict):
            self._last_batch_diagnostics = dict(diagnostics)
            if diagnostics.get("mode"):
                self._last_batch_mode = str(diagnostics["mode"])

    def synthesize_batch(self, items: list[dict]) -> None:
        if not items:
            return
        adapter_items = [
            {
                "text": item["text"],
                "output_path": item["output_path"],
                "voice": self.voice,
                "ref_text": str((item.get("segment") or {}).get("text") or "").strip() or None,
                "clone": self.clone,
                "clone_mode": self.clone_mode,
                "anchor_text": self.anchor_text,
                "segment": item.get("segment") or {},
            }
            for item in items
        ]
        if not bool(self.settings.get("tts_session_reuse_enabled", True)):
            factory = self._adapter_factory
            for attempt in range(2):
                try:
                    adapter = factory(self.settings, data_dir=self.data_dir, runner=self.runner)
                    synthesize_batch = getattr(adapter, "synthesize_batch", None)
                    if callable(synthesize_batch):
                        self._last_batch_mode = "adapter_synthesize_batch"
                        synthesize_batch(adapter_items)
                        self._apply_adapter_batch_diagnostics(adapter)
                    else:
                        self._last_batch_mode = (
                            "omnivoice_sequential_fallback"
                            if self.backend == "omnivoice"
                            else "sequential_fallback"
                        )
                        for item in adapter_items:
                            adapter.synthesize(
                                item["text"],
                                item["output_path"],
                                voice=item["voice"],
                                ref_text=item["ref_text"],
                                clone=item["clone"],
                                clone_mode=item["clone_mode"],
                                anchor_text=item["anchor_text"],
                                segment=item.get("segment") or {},
                            )
                    return
                except AppError as error:
                    if not error.info.retryable or attempt:
                        raise
                    self._reset_caches_after_failure()
            return
        last_error: AppError | None = None
        for attempt in range(2):
            try:
                adapter = self._adapter_instance()
                synthesize_batch = getattr(adapter, "synthesize_batch", None)
                if callable(synthesize_batch):
                    self._last_batch_mode = "adapter_synthesize_batch"
                    synthesize_batch(adapter_items)
                    self._apply_adapter_batch_diagnostics(adapter)
                else:
                    self._last_batch_mode = (
                        "omnivoice_sequential_fallback"
                        if self.backend == "omnivoice"
                        else "sequential_fallback"
                    )
                    for item in adapter_items:
                        adapter.synthesize(
                            item["text"],
                            item["output_path"],
                            voice=item["voice"],
                            ref_text=item["ref_text"],
                            clone=item["clone"],
                            clone_mode=item["clone_mode"],
                            anchor_text=item["anchor_text"],
                            segment=item.get("segment") or {},
                        )
                return
            except AppError as error:
                last_error = error
                if not error.info.retryable or attempt:
                    raise
                self._reset_caches_after_failure()
        if last_error is not None:
            raise last_error

    @staticmethod
    def _reset_caches_after_failure() -> None:
        try:
            from .asr import reset_model_cache
            from .. import pipeline as _pipeline  # noqa: F401  pragma: no cover - imported for side effects

            reset_model_cache()
        except Exception:
            pass
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def close(self) -> None:
        adapter = self._adapter
        self._adapter = None
        close = getattr(adapter, "close", None)
        if callable(close):
            close()
        lease = self._lease
        self._lease = None
        if lease is not None:
            lease.__exit__(None, None, None)


def create_tts_adapter(settings: dict, *, data_dir: Path | None = None, runner: object | None = None):
    backend = tts_backend_from_settings(settings)
    if backend == "edge_tts":
        from .edge_tts import DEFAULT_EDGE_TTS_VOICE, EdgeTtsAdapter

        voice = str(settings.get("edge_tts_voice") or DEFAULT_EDGE_TTS_VOICE).strip() or DEFAULT_EDGE_TTS_VOICE
        return EdgeTtsAdapter(voice=voice)
    if backend == "google_tts":
        from .google_tts import DEFAULT_GOOGLE_TTS_SPEAKING_RATE, DEFAULT_GOOGLE_TTS_VOICE, GoogleTtsAdapter

        voice = str(settings.get("google_tts_voice") or DEFAULT_GOOGLE_TTS_VOICE).strip() or DEFAULT_GOOGLE_TTS_VOICE
        try:
            speaking_rate = float(settings.get("google_tts_speaking_rate", DEFAULT_GOOGLE_TTS_SPEAKING_RATE) or 1.0)
        except (TypeError, ValueError):
            speaking_rate = DEFAULT_GOOGLE_TTS_SPEAKING_RATE
        return GoogleTtsAdapter(
            api_key=str(settings.get("google_tts_api_key") or "").strip(),
            voice=voice,
            speaking_rate=max(0.5, min(1.5, speaking_rate)),
        )
    if backend == "gemini_tts":
        from .gemini import GeminiKeyPool, GeminiTtsAdapter

        keys = [
            item for item in settings.get("gemini_api_keys", [])
            if isinstance(item, dict) and item.get("key")
        ]
        return GeminiTtsAdapter(
            GeminiKeyPool(keys, cursor=int(settings.get("gemini_key_cursor", 0) or 0)),
            model=str(settings.get("gemini_tts_model") or "gemini-2.5-flash-preview-tts"),
        )
    if backend == "omnivoice":
        from ..omnivoice_env import OMNIVOICE_DEFAULT_MODEL
        from .omnivoice_tts import (
            OMNIVOICE_DEFAULT_CHUNK_DURATION_SEC,
            OMNIVOICE_DEFAULT_CHUNK_THRESHOLD_SEC,
            OmniVoiceTtsAdapter,
        )

        return OmniVoiceTtsAdapter(
            model=str(settings.get("omnivoice_model", OMNIVOICE_DEFAULT_MODEL) or OMNIVOICE_DEFAULT_MODEL),
            device=resolve_omnivoice_device(str(settings.get("omnivoice_device", "cuda:0") or "cuda:0")),
            num_step=int(settings.get("omnivoice_num_steps", 32) or 32),
            speed=1.0,
            language_id=str(settings.get("omnivoice_language_id") or "").strip() or None,
            audio_chunk_threshold=OMNIVOICE_DEFAULT_CHUNK_THRESHOLD_SEC,
            audio_chunk_duration=OMNIVOICE_DEFAULT_CHUNK_DURATION_SEC,
            data_dir=data_dir,
            runner=runner,
            settings=settings,
        )
    raise AppError(
        422,
        ErrorInfo(
            code="UNSUPPORTED_TTS_BACKEND",
            message=f"Unsupported TTS backend: {backend}",
            action="Choose omnivoice, edge_tts, google_tts, or gemini_tts.",
        ),
    )
