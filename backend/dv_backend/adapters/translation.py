import time
from collections.abc import Callable

from ..errors import AppError
from ..models import ErrorInfo


class GoogleFreeTranslator:
    def __init__(
        self,
        client_factory: Callable | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_attempts: int = 3,
    ) -> None:
        self.client_factory = client_factory or self._default_client_factory
        self.sleep = sleep
        self.max_attempts = max_attempts

    @staticmethod
    def _default_client_factory(source: str, target: str):
        from deep_translator import GoogleTranslator

        return GoogleTranslator(source=source, target=target)

    def translate(self, texts: list[str], source: str, target: str) -> list[str]:
        client = self.client_factory(source, target)
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                translated = client.translate_batch(texts)
                if len(translated) != len(texts) or any(not item.strip() for item in translated):
                    raise AppError(
                        502,
                        ErrorInfo(
                            code="TRANSLATION_INVALID_OUTPUT",
                            message="Google Translate returned incomplete output.",
                            action="Retry the job after checking the network connection.",
                            retryable=True,
                        ),
                    )
                return translated
            except AppError:
                raise
            except Exception as cause:
                last_error = cause
                if attempt + 1 < self.max_attempts:
                    self.sleep(float(2**attempt))

        raise AppError(
            502,
            ErrorInfo(
                code="GOOGLE_TRANSLATE_FAILED",
                message="Google Translate could not translate the segments.",
                action="Wait briefly and resume the job.",
                detail=str(last_error),
                retryable=True,
            ),
        )
