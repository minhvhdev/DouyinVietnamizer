from fastapi import Request
from fastapi.responses import JSONResponse

from .models import ErrorInfo


class AppError(Exception):
    def __init__(self, status_code: int, info: ErrorInfo) -> None:
        super().__init__(info.message)
        self.status_code = status_code
        self.info = info


async def app_error_handler(_request: Request, error: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content={"error": error.info.model_dump()},
    )

