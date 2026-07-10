from __future__ import annotations

from pathlib import Path

from ..errors import AppError
from ..models import ErrorInfo
from ..omnivoice_env import OMNIVOICE_DEFAULT_MODEL
from .omnivoice_infer import _strip_surrogates
from .tts import parse_tts_voice_string

OMNIVOICE_INSTRUCT_PREFIX = "instruct:"
OMNIVOICE_DEFAULT_CHUNK_THRESHOLD_SEC = 30.0
OMNIVOICE_DEFAULT_CHUNK_DURATION_SEC = 15.0


class OmniVoiceTtsAdapter:
    """Adapter backed by a long-lived OmniVoice worker subprocess."""

    def __init__(
        self,
        *,
        model: str = OMNIVOICE_DEFAULT_MODEL,
        device: str = "cuda:0",
        num_step: int = 32,
        speed: float = 1.0,
        language_id: str | None = None,
        audio_chunk_threshold: float = OMNIVOICE_DEFAULT_CHUNK_THRESHOLD_SEC,
        audio_chunk_duration: float = OMNIVOICE_DEFAULT_CHUNK_DURATION_SEC,
        data_dir: Path | None = None,
        runner: object | None = None,
        _client: object | None = None,
    ) -> None:
        self.model = (model or OMNIVOICE_DEFAULT_MODEL).strip() or OMNIVOICE_DEFAULT_MODEL
        self.device = (device or "cuda:0").strip() or "cuda:0"
        self.num_step = max(4, min(64, int(num_step)))
        self.speed = max(0.5, min(1.5, float(speed)))
        self.language_id = (language_id or "").strip() or None
        self.audio_chunk_threshold = max(4.0, min(60.0, float(audio_chunk_threshold)))
        self.audio_chunk_duration = max(4.0, min(30.0, float(audio_chunk_duration)))
        self._data_dir = Path(data_dir) if data_dir is not None else None
        self._runner = runner
        self._client = None
        self._injected_client = _client

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
            return
        from .omnivoice_client import acquire_client

        self._client = acquire_client(
            data_dir=self._resolve_data_dir(),
            model=self.model,
            device=self.device,
            num_step=self.num_step,
            speed=self.speed,
            language_id=self.language_id,
        )
        self._client.register_with_runner(self._runner)

    def _voice_kwargs(self, voice: str, ref_text: str | None) -> dict[str, str | None]:
        ref_audio, _, voice_design = parse_tts_voice_string(voice)
        return {
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "instruct": voice_design,
        }

    def _synthesize_single(
        self,
        text: str,
        output_path: Path,
        *,
        voice: str,
        ref_text: str | None,
        anchor_text: str | None = None,
    ) -> None:
        self._ensure_runtime()
        kwargs = self._voice_kwargs(voice, ref_text)
        client = self._client
        assert client is not None
        response = client.synthesize(
            text=text,
            output_path=output_path,
            ref_audio=kwargs["ref_audio"],
            ref_text=kwargs["ref_text"],
            anchor_text=anchor_text,
            instruct=kwargs["instruct"],
        )
        if not response.get("ok", False):
            raise AppError(
                502,
                ErrorInfo(
                    code=response.get("code") or "OMNIVOICE_TTS_FAILED",
                    message=response.get("message") or "OmniVoice could not generate narration.",
                    action="Check OmniVoice settings and reference audio.",
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
        _ = clone, clone_mode, ref_text
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned = _strip_surrogates(text).strip()
        if not cleaned:
            raise AppError(
                422,
                ErrorInfo(
                    code="EMPTY_TTS_TEXT",
                    message="Cannot synthesize empty narration text.",
                    action="Verify translation output for this segment.",
                ),
            )
        try:
            self._synthesize_single(
                cleaned,
                output_path,
                voice=voice,
                ref_text=None,
                anchor_text=anchor_text,
            )
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
