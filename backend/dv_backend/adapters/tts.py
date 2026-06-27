from pathlib import Path
import re
import shutil
import wave

from ..errors import AppError
from ..models import ErrorInfo

SUPPORTED_TTS_BACKENDS = ("voxcpm",)
VOXCPM_DEFAULT_MODEL = "openbmb/VoxCPM2"
VOXCPM_INSTRUCT_PREFIX = "instruct:"

MAX_TTS_CHARS = 450
_TTS_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…。，！？；;])\s+")


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


def parse_voxcpm_voice(voice: str | None) -> tuple[str | None, str | None, str | None]:
    """Return ``(prompt_wav_path, prompt_text, voice_design)`` for a voice string."""
    value = str(voice or "auto").strip()
    if not value or value.lower() == "auto":
        return None, None, None
    if value.startswith(VOXCPM_INSTRUCT_PREFIX):
        voice_design = value[len(VOXCPM_INSTRUCT_PREFIX):].strip()
        return None, None, voice_design or None
    path = Path(value)
    if path.is_file():
        return str(path), None, None
    return None, None, None


def _is_voxcpm_voice_clone(voice: str | None) -> bool:
    prompt_wav_path, _, _ = parse_voxcpm_voice(voice)
    return prompt_wav_path is not None


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


class VoxCPMTtsAdapter:
    """Adapter backed by a long-lived VoxCPM worker.

    The adapter routes every segment through a shared
    :class:`VoxCPMWorkerClient` which keeps the VoxCPM2 model resident in VRAM
    and coalesces compatible requests. Combined with the on-disk cache
    (key = sha256(voice_id, text, model, num_step, voice_design, cfg_value))
    repeated or re-run jobs become near-instant.
    """

    def __init__(
        self,
        *,
        model: str = VOXCPM_DEFAULT_MODEL,
        device: str = "cuda:0",
        num_steps: int = 10,
        data_dir: Path | None = None,
        runner: object | None = None,
        max_batch: int | None = None,
        flush_ms: int | None = None,
        enable_cache: bool = True,
        # Test seams: inject a fake client / cache. Production code never
        # sets these.
        _client: object | None = None,
        _cache: object | None = None,
    ) -> None:
        self.model = (model or VOXCPM_DEFAULT_MODEL).strip() or VOXCPM_DEFAULT_MODEL
        self.device = (device or "cuda:0").strip() or "cuda:0"
        self.num_steps = max(4, min(64, int(num_steps)))
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
        from .voxcpm_cache import VoxCPMCache
        from .voxcpm_client import acquire_client

        data_dir = self._resolve_data_dir()
        if self._enable_cache:
            self._cache = VoxCPMCache(data_dir / "cache" / "voxcpm")
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
        prompt_wav_path: str | None,
        prompt_text: str | None,
        voice_design: str | None,
        voice_id: str,
    ) -> None:
        self._ensure_runtime()
        cache_key = None
        if self._cache is not None and not _is_voxcpm_voice_clone(voice_id):
            from .voxcpm_cache import cache_key as make_cache_key

            cache_key = make_cache_key(
                voice_id=voice_id,
                text=text,
                model=self.model,
                num_step=self.num_steps,
                voice_design=voice_design,
                cfg_value=2.0,
            )
            if self._cache.materialize(cache_key, output_path):
                return
        assert self._client is not None
        response = self._client.synthesize(
            text=text,
            output_path=output_path,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
            voice_design=voice_design,
            cfg_value=2.0,
            inference_timesteps=self.num_steps,
            cache_key=cache_key,
        )
        if not response.get("ok", False):
            raise AppError(
                502,
                ErrorInfo(
                    code=response.get("code") or "VOXCPM_TTS_FAILED",
                    message=response.get("message") or "VoxCPM2 could not generate narration.",
                    action=(
                        "Check VoxCPM2 model, GPU availability, and reference audio settings. "
                        "Run 'python scripts/setup_voxcpm.py' if the isolated env is missing."
                    ),
                    detail=response.get("detail"),
                    retryable=bool(response.get("retryable", True)),
                ),
            )
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise AppError(
                502,
                ErrorInfo(
                    code="VOXCPM_TTS_FAILED",
                    message="VoxCPM2 produced an empty audio file.",
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
        prompt_wav_path, _, voice_design = parse_voxcpm_voice(voice)
        if voice_design:
            text = f"({voice_design}){text}"
        self._run_infer(
            text=text,
            output_path=output_path,
            prompt_wav_path=prompt_wav_path,
            prompt_text=ref_text,
            voice_design=voice_design,
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
                    code="VOXCPM_TTS_FAILED",
                    message="VoxCPM2 could not generate narration.",
                    action="Ensure the VoxCPM virtualenv is installed and the GPU is available.",
                    detail=str(cause),
                    retryable=True,
                ),
            ) from cause
        finally:
            for part_path in output_path.parent.glob(f"{output_path.stem}.part*.wav"):
                part_path.unlink(missing_ok=True)


def create_tts_adapter(settings: dict, *, data_dir: Path | None = None, runner: object | None = None):
    try:
        batch_size = int(settings.get("voxcpm_batch_size", 4) or 4)
    except (TypeError, ValueError):
        batch_size = 4
    try:
        flush_ms = int(settings.get("voxcpm_batch_flush_ms", 150) or 150)
    except (TypeError, ValueError):
        flush_ms = 150
    return VoxCPMTtsAdapter(
        model=str(settings.get("voxcpm_model", VOXCPM_DEFAULT_MODEL) or VOXCPM_DEFAULT_MODEL),
        device=str(settings.get("voxcpm_device", "cuda:0") or "cuda:0"),
        num_steps=int(settings.get("voxcpm_num_steps", 10) or 10),
        data_dir=data_dir,
        runner=runner,
        max_batch=max(1, batch_size),
        flush_ms=max(0, flush_ms),
        enable_cache=str(settings.get("voxcpm_cache_enabled", True)).lower()
        not in {"0", "false", "no"},
    )
