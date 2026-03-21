"""RFC 7807 Problem Details error handling for FastAPI."""

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from corail.core.errors import (
    AdapterNotFoundError,
    ConfigValidationError,
    CoreError,
    ExecutionError,
    LLMError,
)

ERROR_TYPE_BASE = "https://corail.dev/errors"

_ERROR_STATUS_MAP: dict[type[CoreError], tuple[int, str]] = {
    AdapterNotFoundError: (400, "adapter-not-found"),
    ConfigValidationError: (422, "config-validation"),
    LLMError: (502, "llm-error"),
    ExecutionError: (500, "execution-error"),
}


def register_error_status(error_type: type[CoreError], status: int, slug: str) -> None:
    """Register a custom CoreError subclass for RFC 7807 mapping."""
    _ERROR_STATUS_MAP[error_type] = (status, slug)


class ProblemDetail(BaseModel):
    """RFC 7807 Problem Details for HTTP APIs."""

    type: str
    title: str
    status: int
    detail: str
    instance: str = ""
    request_id: str = ""


def core_error_to_problem(error: CoreError, request_path: str = "", request_id: str = "") -> ProblemDetail:
    """Convert a CoreError to an RFC 7807 ProblemDetail."""
    status, slug = _ERROR_STATUS_MAP.get(type(error), (500, "internal"))
    return ProblemDetail(
        type=f"{ERROR_TYPE_BASE}/{slug}",
        title=error.code,
        status=status,
        detail=error.message,
        instance=request_path,
        request_id=request_id,
    )


async def core_error_handler(request: Request, exc: CoreError) -> JSONResponse:
    """FastAPI exception handler for CoreError subtypes."""
    request_id = getattr(request.state, "request_id", "")
    problem = core_error_to_problem(exc, request_path=str(request.url.path), request_id=request_id)
    return JSONResponse(
        status_code=problem.status,
        content=problem.model_dump(),
        media_type="application/problem+json",
    )


async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unhandled exceptions."""
    request_id = getattr(request.state, "request_id", "")
    problem = ProblemDetail(
        type=f"{ERROR_TYPE_BASE}/internal",
        title="Internal Server Error",
        status=500,
        detail=str(exc),
        instance=str(request.url.path),
        request_id=request_id,
    )
    return JSONResponse(
        status_code=500,
        content=problem.model_dump(),
        media_type="application/problem+json",
    )
