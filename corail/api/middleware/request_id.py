"""Request ID middleware — generates or propagates X-Request-ID."""

from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from ulid import ULID

REQUEST_ID_HEADER = "X-Request-ID"
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def _generate_request_id() -> str:
    return f"req_{ULID()}"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Extracts or generates X-Request-ID and attaches to request state and response."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request with request ID."""
        rid = request.headers.get(REQUEST_ID_HEADER) or _generate_request_id()
        request.state.request_id = rid
        request_id_var.set(rid)

        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = rid
        return response
