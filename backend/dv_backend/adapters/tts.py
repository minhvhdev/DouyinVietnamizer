import asyncio
from collections.abc import Callable
from pathlib import Path

from ..errors import AppError
from ..models import ErrorInfo


class EdgeTtsAdapter:
    def __init__(self, communicate_factory: Callable | None = None) -> None:
        self.communicate_factory = communicate_factory or self._default_communicate_factory

    @staticmethod
    def _default_communicate_factory(text: str, voice: str, rate: str, pitch: str, volume: str):
        import edge_tts

        return edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate,
            pitch=pitch,
            volume=volume,
        )

    def synthesize(
        self,
        text: str,
        output_path: Path,
        *,
        voice: str,
        rate: str,
        pitch: str,
        volume: str,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        communicate = self.communicate_factory(text, voice, rate, pitch, volume)
        try:
            asyncio.run(communicate.save(str(output_path)))
        except Exception as cause:
            raise AppError(
                502,
                ErrorInfo(
                    code="EDGE_TTS_FAILED",
                    message="Microsoft Edge TTS could not generate narration.",
                    action="Wait briefly and resume the job.",
                    detail=str(cause),
                    retryable=True,
                ),
            ) from cause

        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise AppError(
                502,
                ErrorInfo(
                    code="EDGE_TTS_EMPTY_OUTPUT",
                    message="Microsoft Edge TTS returned empty audio.",
                    action="Wait briefly and resume the job.",
                    retryable=True,
                ),
            )
