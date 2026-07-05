import os
from pathlib import Path
import re
import shutil
import wave

from ..errors import AppError
from ..models import ErrorInfo

SUPPORTED_TTS_BACKENDS = ("voxcpm",)
VOXCPM_DEFAULT_MODEL = "gguf-q8"
VOXCPM_INSTRUCT_PREFIX = "instruct:"
VOXCPM_MODE_DESIGN = "design"
VOXCPM_MODE_REFERENCE = "reference"
VOXCPM_MODE_ULTIMATE = "ultimate"
VOXCPM_CLONE_MODES = frozenset(
    {VOXCPM_MODE_REFERENCE, VOXCPM_MODE_ULTIMATE}
)

MAX_TTS_CHARS = 450
_TTS_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…。，！？；;])\s+")


def _normalize_clone_mode(value: object) -> str:
    """Coerce caller-supplied clone mode into a known enum value.

    Unknown / missing values fall back to ``"reference"`` so existing callers
    that only know about the legacy boolean clone flag keep working.
    """
    if not value:
        return VOXCPM_MODE_REFERENCE
    candidate = str(value).strip().lower()
    if candidate in VOXCPM_CLONE_MODES:
        return candidate
    return VOXCPM_MODE_REFERENCE


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


def resolve_voxcpm_model(model: str) -> str:
    from ..voxcpm_gguf import normalize_voxcpm_model_id

    configured = (model or VOXCPM_DEFAULT_MODEL).strip() or VOXCPM_DEFAULT_MODEL
    path = Path(configured)
    if path.exists():
        return str(path.resolve())
    return normalize_voxcpm_model_id(configured)


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
        self.model = resolve_voxcpm_model(model)
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
            max_batch=self._max_batch or 4,
            flush_ms=self._flush_ms or 150,
        )
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
        reference_wav_path: str | None = None,
        anchor_text: str | None = None,
        mode: str = VOXCPM_MODE_DESIGN,
    ) -> None:
        self._ensure_runtime()
        cache_key = None
        if self._cache is not None:
            from .voxcpm_cache import cache_key as make_cache_key

            cache_key = make_cache_key(
                voice_id=voice_id,
                text=text,
                model=self.model,
                num_step=self.num_steps,
                voice_design=voice_design,
                cfg_value=2.0,
                mode=mode,
                reference_wav_path=reference_wav_path,
                reference_text=prompt_text,
                anchor_text=anchor_text,
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
            reference_wav_path=reference_wav_path,
            anchor_text=anchor_text,
            mode=mode,
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

    def _single_infer_kwargs(
        self,
        text: str,
        output_path: Path,
        *,
        voice: str,
        ref_text: str | None,
        anchor_text: str | None = None,
        mode: str = VOXCPM_MODE_DESIGN,
    ) -> dict[str, object]:
        prompt_wav_path, _, voice_design = parse_voxcpm_voice(voice)
        if voice_design:
            text = f"({voice_design}){text}"
        if mode == VOXCPM_MODE_ULTIMATE:
            if not prompt_wav_path:
                raise AppError(
                    422,
                    ErrorInfo(
                        code="VOXCPM_ULTIMATE_REQUIRES_REFERENCE_AUDIO",
                        message=(
                            "Ultimate clone mode requires a reference audio path "
                            "(the voice argument must point to a WAV file)."
                        ),
                        action="Provide a reference audio file in the voice argument.",
                    ),
                )
            anchor_clean = (anchor_text or "").strip()
            if not anchor_clean:
                raise AppError(
                    422,
                    ErrorInfo(
                        code="VOXCPM_ULTIMATE_REQUIRES_ANCHOR_TRANSCRIPT",
                        message=(
                            "Ultimate clone mode requires the exact transcript of "
                            "the reference audio (anchor_text), not source or "
                            "target text."
                        ),
                        action=(
                            "Provide the verbatim transcript of the reference audio "
                            "file, or switch to reference mode for the ordinary "
                            "clone path that does not require a transcript."
                        ),
                    ),
                )
            return {
                "text": text,
                "output_path": output_path,
                "prompt_wav_path": prompt_wav_path,
                "prompt_text": anchor_clean,
                "reference_wav_path": prompt_wav_path,
                "anchor_text": anchor_clean,
                "voice_design": voice_design,
                "voice_id": voice or "",
                "mode": VOXCPM_MODE_ULTIMATE,
            }
        if mode == VOXCPM_MODE_REFERENCE and prompt_wav_path is not None:
            return {
                "text": text,
                "output_path": output_path,
                "prompt_wav_path": None,
                "prompt_text": None,
                "reference_wav_path": prompt_wav_path,
                "anchor_text": None,
                "voice_design": voice_design,
                "voice_id": voice or "",
                "mode": VOXCPM_MODE_REFERENCE,
            }
        return {
            "text": text,
            "output_path": output_path,
            "prompt_wav_path": prompt_wav_path,
            "prompt_text": ref_text,
            "reference_wav_path": prompt_wav_path,
            "anchor_text": None,
            "voice_design": voice_design,
            "voice_id": voice or "",
            "mode": VOXCPM_MODE_DESIGN,
        }

    def _synthesize_single(
        self,
        text: str,
        output_path: Path,
        *,
        voice: str,
        ref_text: str | None,
        anchor_text: str | None = None,
        mode: str = VOXCPM_MODE_DESIGN,
    ) -> None:
        self._run_infer(
            **self._single_infer_kwargs(
                text,
                output_path,
                voice=voice,
                ref_text=ref_text,
                anchor_text=anchor_text,
                mode=mode,
            )
        )

    def close(self) -> None:
        return None

    def synthesize(
        self,
        text: str,
        output_path: Path,
        *,
        voice: str,
        ref_text: str | None = None,
        anchor_text: str | None = None,
        clone: bool = False,
        clone_mode: str | None = None,
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
        # Resolve explicit mode only when the caller signals cloning. Legacy
        # callers that pass clone=True without a mode get the default
        # reference (ordinary clone) path. clone=False always means design.
        if clone:
            mode = _normalize_clone_mode(clone_mode) if clone_mode else VOXCPM_MODE_REFERENCE
        else:
            mode = VOXCPM_MODE_DESIGN
        try:
            if len(chunks) == 1:
                self._synthesize_single(
                    chunks[0],
                    output_path,
                    voice=voice,
                    ref_text=ref_text,
                    anchor_text=anchor_text,
                    mode=mode,
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
                    anchor_text=anchor_text,
                    mode=mode,
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

    def synthesize_batch(self, items: list[dict]) -> None:
        if not items:
            return
        self._ensure_runtime()
        assert self._client is not None
        from .voxcpm_cache import cache_key as make_cache_key

        submitted: list[tuple[str, Path, str | None]] = []
        requests: list[dict] = []
        for item in items:
            text = str(item.get("text") or "")
            output_path = Path(item["output_path"])
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
            if len(chunks) != 1:
                self.synthesize(
                    text,
                    output_path,
                    voice=str(item.get("voice") or "auto"),
                    ref_text=item.get("ref_text"),
                    anchor_text=item.get("anchor_text"),
                    clone=bool(item.get("clone", False)),
                    clone_mode=item.get("clone_mode"),
                )
                continue
            mode = (
                _normalize_clone_mode(item.get("clone_mode"))
                if bool(item.get("clone", False)) and item.get("clone_mode")
                else (VOXCPM_MODE_REFERENCE if bool(item.get("clone", False)) else VOXCPM_MODE_DESIGN)
            )
            infer = self._single_infer_kwargs(
                chunks[0],
                output_path,
                voice=str(item.get("voice") or "auto"),
                ref_text=item.get("ref_text"),
                anchor_text=item.get("anchor_text"),
                mode=mode,
            )
            cache_key = None
            if self._cache is not None:
                cache_key = make_cache_key(
                    voice_id=str(infer["voice_id"]),
                    text=str(infer["text"]),
                    model=self.model,
                    num_step=self.num_steps,
                    voice_design=infer.get("voice_design"),
                    cfg_value=2.0,
                    mode=str(infer["mode"]),
                    reference_wav_path=infer.get("reference_wav_path"),
                    reference_text=infer.get("prompt_text"),
                    anchor_text=infer.get("anchor_text"),
                )
                if self._cache.materialize(cache_key, output_path):
                    continue
            requests.append(
                {
                    "text": str(infer["text"]),
                    "output_path": output_path,
                    "prompt_wav_path": infer.get("prompt_wav_path"),
                    "prompt_text": infer.get("prompt_text"),
                    "voice_design": infer.get("voice_design"),
                    "reference_wav_path": infer.get("reference_wav_path"),
                    "anchor_text": infer.get("anchor_text"),
                    "mode": str(infer["mode"]),
                    "cfg_value": 2.0,
                    "inference_timesteps": self.num_steps,
                    "cache_key": cache_key,
                }
            )
            submitted.append(("", output_path, cache_key))
        request_ids = self._client.submit_batch(requests)
        for index, request_id in enumerate(request_ids):
            submitted[index] = (request_id, submitted[index][1], submitted[index][2])
        responses = self._client.wait_batch(request_ids)
        for (_request_id, output_path, cache_key), response in zip(submitted, responses):
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
        self.voice = self._default_voice()
        self.clone = bool(str(self.settings.get("voxcpm_ref_audio") or "").strip())
        self.clone_mode = self._clone_mode() if self.clone else "reference"
        self.anchor_text = self._anchor_transcript() if self.clone else None
        if gpu_manager is None:
            from ..gpu_manager import global_gpu_manager
            self.gpu_manager = global_gpu_manager()
        else:
            self.gpu_manager = gpu_manager
        self._lease = None
        self._model_key = self._build_model_key()

    def _build_model_key(self) -> str:
        model = str(self.settings.get("voxcpm_model", "") or "")
        device = str(self.settings.get("voxcpm_device", "cuda:0") or "cuda:0")
        steps = self.settings.get("voxcpm_num_steps", 10)
        return f"{model}|{device}|{int(steps or 0)}|{self.clone_mode}"

    def __enter__(self) -> "TtsSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _adapter_instance(self):
        if self._adapter is not None:
            return self._adapter
        device = str(self.settings.get("voxcpm_device", "cuda:0") or "cuda:0")
        if self.gpu_manager is not None and not self._session_disabled():
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

    def _default_voice(self) -> str:
        instruct = str(self.settings.get("voxcpm_instruct") or "").strip()
        if instruct:
            return f"{VOXCPM_INSTRUCT_PREFIX}{instruct}"
        ref_audio = str(self.settings.get("voxcpm_ref_audio") or "").strip()
        if ref_audio:
            return ref_audio
        return "auto"

    def _anchor_transcript(self) -> str | None:
        ref_audio = str(self.settings.get("voxcpm_ref_audio") or "").strip()
        if not ref_audio:
            return None
        sidecar = Path(ref_audio).with_suffix(".txt")
        if not sidecar.is_file():
            return None
        try:
            transcript = sidecar.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not transcript:
            return None
        # Long anchor transcripts crash or stall GGUF ultimate mode; keep a
        # short prefix that still conditions timbre without blowing the graph.
        if len(transcript) > 400:
            return transcript[:400].rstrip()
        return transcript

    def _clone_mode(self) -> str:
        return _normalize_clone_mode(self.settings.get("voxcpm_clone_mode"))

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
                )
                return
            except AppError as error:
                last_error = error
                if not error.info.retryable or attempt:
                    raise
                self._reset_caches_after_failure()
        if last_error is not None:
            raise last_error

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
                        synthesize_batch(adapter_items)
                    else:
                        for item in adapter_items:
                            adapter.synthesize(
                                item["text"],
                                item["output_path"],
                                voice=item["voice"],
                                ref_text=item["ref_text"],
                                clone=item["clone"],
                                clone_mode=item["clone_mode"],
                                anchor_text=item["anchor_text"],
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
                    synthesize_batch(adapter_items)
                else:
                    for item in adapter_items:
                        adapter.synthesize(
                            item["text"],
                            item["output_path"],
                            voice=item["voice"],
                            ref_text=item["ref_text"],
                            clone=item["clone"],
                            clone_mode=item["clone_mode"],
                            anchor_text=item["anchor_text"],
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
