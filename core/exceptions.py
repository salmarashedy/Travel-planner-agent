from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class ApplicationError(Exception):
    def __init__(self, message: str, status_code: int = 500) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class LLMClientError(ApplicationError):
    def __init__(self, message: str = "The LLM service is unavailable.") -> None:
        super().__init__(message, status_code=503)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApplicationError)
    async def application_error_handler(
        request: Request, exc: ApplicationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message},
        )
