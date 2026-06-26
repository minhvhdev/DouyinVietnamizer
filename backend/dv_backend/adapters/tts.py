from pathlib import Path
import re
import shutil
import subprocess
import wave

from ..errors import AppError
from ..models import ErrorInfo
from ..omnivoice_env import resolve_omnivoice_python

SUPPORTED_TTS_BACKENDS = ("omnivoice",)
OMNIVOICE_DEFAULT_MODEL = "k2-fsa/OmniVoice"
OMNIVOICE_INSTRUCT_PREFIX = "instruct:"


MAX_TTS_CHARS = 450
_TTS_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?â€¦ã€‚ï¼ï¼Ÿï¼›;])\s+")


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



def parse_omnivoice_voice(voice: str | None) -> tuple[str | None, str | None, str | None]:
    value = str(voice or "auto").strip()
    if not value or value.lower() == "auto":
        return None, None, None
    if value.startswith(OMNIVOICE_INSTRUCT_PREFIX):
        instruct = value[len(OMNIVOICE_INSTRUCT_PREFIX):].strip()
        return None, instruct or None, None
    path = Path(value)
    if path.is_file():
        return str(path), None, None
    return None, None, value


def _is_omnivoice_voice_clone(voice: str | None) -> bool:
    ref_audio, _, _ = parse_omnivoice_voice(voice)
    return ref_audio is not None

def _wav_format_key(params: wave._wave_params) -> tuple:
    return (
        params.nchannels,
        params.sampwidth,
        params.framerate,
        params.comptype,
        params.compname,
    )


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



class OmniVoiceTtsAdapter:
    """Adapter backed by a long-lived OmniVoice worker.

    The adapter no longer spawns a fresh ``python -m omnivoice.cli.infer``
    process per segment. Instead it acquires a shared :class:`OmniVoiceWorkerClient`
    which keeps the OmniVoice model resident in VRAM and coalesces requests
    that share the same voice signature. Combined with the on-disk cache
    (key = sha256(voice, text, model, num_step, instruct)) repeated or
    re-run jobs become near-instant.
    """

    def __init__(
        self,
        *,
        model: str = OMNIVOICE_DEFAULT_MODEL,
        device: str = "cuda:0",
        num_steps: int = 32,
        data_dir: Path | None = None,
        runner: object | None = None,
        max_batch: int | None = None,
        flush_ms: int | None = None,
        enable_cache: bool = True,
        # Legacy keyword accepted for backwards compatibility with tests /
        # older callers. The new adapter no longer spawns subprocesses
        # itself; instead it routes through OmniVoiceWorkerClient. Passing
        # ``omnivoice_python`` raises so callers know to switch to the new
        # flow.
        omnivoice_python: Path | None = None,
        # Test seams: inject a fake client / cache. Production code never
        # sets these.
        _client: object | None = None,
        _cache: object | None = None,
    ) -> None:
        if omnivoice_python is not None:
            raise TypeError(
                "OmniVoiceTtsAdapter no longer accepts 'omnivoice_python'; "
                "the new flow uses OmniVoiceWorkerClient. Inject a fake "
                "client via '_client' in tests, or pass 'data_dir' to use "
                "the real worker."
            )
        self.model = (model or OMNIVOICE_DEFAULT_MODEL).strip() or OMNIVOICE_DEFAULT_MODEL
        self.device = (device or "cuda:0").strip() or "cuda:0"
        self.num_steps = max(8, min(64, int(num_steps)))
        self._data_dir = Path(data_dir) if data_dir is not None else None
        self._runner = runner
        self._max_batch = max_batch
        self._flush_ms = flush_ms
        self._enable_cache = enable_cache
        self._client = None
        self._cache = None
        self._injected_client = _client
        self._injected_cache = _cache

    def _resolve_data_dir(self) -> Path:
        if self._data_dir is not None:
            return self._data_dir
        from ..config import AppConfig

        try:
            return AppConfig.from_env().data_dir
        except Exception:
            return Path.cwd() / "data"

    def _ensure_runtime(self) -> None:
        if self._client is not None:
            return
        if self._injected_client is not None:
            self._client = self._injected_client
            self._cache = self._injected_cache
            return
        from .omnivoice_cache import OmniVoiceCache
        from .omnivoice_client import acquire_client

        data_dir = self._resolve_data_dir()
        if self._enable_cache:
            self._cache = OmniVoiceCache(data_dir / "cache" / "omnivoice")
        else:
            self._cache = None
        self._client = acquire_client(
            data_dir=data_dir,
            model=self.model,
            device=self.device,
            num_steps=self.num_steps,
        )
        if self._max_batch is not None or self._flush_ms is not None:
            self._client.max_batch = self._max_batch or self._client.max_batch
            self._client.flush_ms = self._flush_ms or self._client.flush_ms
        self._client.register_with_runner(self._runner)

    def _run_infer(
        self,
        *,
        text: str,
        output_path: Path,
        ref_audio: str | None,
        ref_text: str | None,
        instruct: str | None,
        voice_id: str,
    ) -> None:
        self._ensure_runtime()
        cache_key = None
        if self._cache is not None and not _is_omnivoice_voice_clone(voice_id):
            from .omnivoice_cache import cache_key as make_cache_key

            cache_key = make_cache_key(
                voice_id=voice_id,
                text=text,
                model=self.model,
                num_step=self.num_steps,
                instruct=instruct,
            )
            if self._cache.materialize(cache_key, output_path):
                return
        assert self._client is not None
        response = self._client.synthesize(
            text=text,
            output_path=output_path,
            ref_audio=ref_audio,
            ref_text=ref_text,
            instruct=instruct,
            cache_key=cache_key,
        )
        if not response.get("ok", False):
            raise AppError(
                502,
                ErrorInfo(
                    code=response.get("code") or "OMNIVOICE_TTS_FAILED",
                    message=response.get("message") or "OmniVoice could not generate narration.",
                    action=(
                        "Check OmniVoice model, GPU availability, and reference audio settings. "
                        "Run 'python scripts/setup_omnivoice.py' if the isolated env is missing."
                    ),
                    detail=response.get("detail"),
                    retryable=bool(response.get("retryable", True)),
                ),
            )
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise AppError(
                502,
                ErrorInfo(
                    code="OMNIVOICE_TTS_FAILED",
                    message="OmniVoice produced an empty audio file.",
                    action="Try another reference clip or switch to auto voice mode.",
                    retryable=True,
                ),
            )
        if self._cache is not None and cache_key is not None:
            self._cache.put(cache_key, output_path)

    def _synthesize_single(
        self,
        text: str,
        output_path: Path,
        *,
        voice: str,
        ref_text: str | None,
    ) -> None:
        ref_audio, instruct, fallback = parse_omnivoice_voice(voice)
        if ref_audio is None and instruct is None and fallback:
            maybe_path = Path(fallback)
            if maybe_path.is_file():
                ref_audio = str(maybe_path)
        self._run_infer(
            text=text,
            output_path=output_path,
            ref_audio=ref_audio,
            ref_text=ref_text,
            instruct=instruct,
            voice_id=voice or "",
        )

    def synthesize(
        self,
        text: str,
        output_path: Path,
        *,
        voice: str,
        ref_text: str | None = None,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        chunks = split_tts_text(text)
        if not chunks:
            raise AppError(
                422,
                ErrorInfo(
                    code="EMPTY_TTS_TEXT",
                    message="Cannot synthesize empty narration text.",
                    action="Verify translation output for this segment.",
                ),
            )
        try:
            if len(chunks) == 1:
                self._synthesize_single(
                    chunks[0],
                    output_path,
                    voice=voice,
                    ref_text=ref_text,
                )
                return

            parts: list[Path] = []
            for index, chunk in enumerate(chunks):
                part_path = output_path.with_name(f"{output_path.stem}.part{index:03d}.wav")
                self._synthesize_single(
                    chunk,
                    part_path,
                    voice=voice,
                    ref_text=ref_text,
                )
                parts.append(part_path)
            _concat_wav_files(parts, output_path)
        except AppError:
            raise
        except Exception as cause:
            raise AppError(
                502,
                ErrorInfo(
                    code="OMNIVOICE_TTS_FAILED",
                    message="OmniVoice could not generate narration.",
                    action="Ensure the OmniVoice virtualenv is installed and the GPU is available.",
                    detail=str(cause),
                    retryable=True,
                ),
            ) from cause
        finally:
            for part_path in output_path.parent.glob(f"{output_path.stem}.part*.wav"):
                part_path.unlink(missing_ok=True)


def create_tts_adapter(settings: dict, *, data_dir: Path | None = None, runner: object | None = None):
    try:
        batch_size = int(settings.get("omnivoice_batch_size", 4) or 4)
    except (TypeError, ValueError):
        batch_size = 4
    try:
        flush_ms = int(settings.get("omnivoice_batch_flush_ms", 150) or 150)
    except (TypeError, ValueError):
        flush_ms = 150
    return OmniVoiceTtsAdapter(
        model=str(settings.get("omnivoice_model", OMNIVOICE_DEFAULT_MODEL) or OMNIVOICE_DEFAULT_MODEL),
        device=str(settings.get("omnivoice_device", "cuda:0") or "cuda:0"),
        num_steps=int(settings.get("omnivoice_num_steps", 32) or 32),
        data_dir=data_dir,
        runner=runner,
        max_batch=max(1, batch_size),
        flush_ms=max(0, flush_ms),
        enable_cache=str(settings.get("omnivoice_cache_enabled", True)).lower()
        not in {"0", "false", "no"},
    )
