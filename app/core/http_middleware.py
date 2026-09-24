"""HTTP access logging and early upload-size hint rejection."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.request_id import request_id_from_request
from app.schemas.errors import ErrorData, ErrorResponse

logger = logging.getLogger(__name__)


class HttpObservabilityMiddleware(BaseHTTPMiddleware):
    """Log request outcomes and reject certainly oversized multipart bodies early."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        upload_path: str,
        max_upload_size_bytes: int,
        multipart_overhead_allowance_bytes: int,
    ) -> None:
        super().__init__(app)
        self._upload_path = upload_path
        self._max_body_hint = max_upload_size_bytes + multipart_overhead_allowance_bytes

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        started = time.perf_counter()
        response: Response
        content_length = request.headers.get("content-length")
        if request.method == "POST" and request.url.path == self._upload_path and content_length:
            try:
                announced_size = int(content_length)
            except ValueError:
                announced_size = -1
            if announced_size > self._max_body_hint:
                request_id = request_id_from_request(request)
                body = ErrorResponse(
                    error=ErrorData(
                        code="UPLOAD_TOO_LARGE",
                        message="Uploaded file exceeds the size limit",
                        request_id=request_id,
                    )
                )
                response = JSONResponse(status_code=413, content=body.model_dump(mode="json"))
                self._log(request, response, started)
                return response
        response = await call_next(request)
        self._log(request, response, started)
        return response

    def _log(self, request: Request, response: Response, started: float) -> None:
        logger.info(
            "http_request",
            extra={
                "event": "http_request",
                "request_id": request_id_from_request(request),
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            },
        )
