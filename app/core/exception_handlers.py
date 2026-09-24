"""Global conversion of domain, validation, and unexpected errors."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.errors import DomainError
from app.core.request_id import request_id_from_request
from app.schemas.errors import ErrorData, ErrorResponse

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Install the stable JSON error envelope for every failure class."""

    @app.exception_handler(DomainError)
    async def handle_domain_error(request: Request, exc: DomainError) -> JSONResponse:
        request_id = request_id_from_request(request)
        logger.warning(
            exc.code,
            extra={
                "event": "domain_error",
                "request_id": request_id,
                "error_code": exc.code,
            },
        )
        return _error_response(
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            details=dict(exc.details),
            request_id=request_id,
            headers=dict(exc.headers),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code, message = {
            404: ("ROUTE_NOT_FOUND", "Route was not found"),
            405: ("METHOD_NOT_ALLOWED", "Method is not allowed"),
        }.get(exc.status_code, ("HTTP_ERROR", "HTTP request failed"))
        return _error_response(
            status_code=exc.status_code,
            code=code,
            message=message,
            details={},
            request_id=request_id_from_request(request),
            headers=dict(exc.headers or {}),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        details: dict[str, object] = {
            "errors": [
                {
                    "location": [str(part) for part in item.get("loc", ())],
                    "message": str(item.get("msg", "Invalid value")),
                    "type": str(item.get("type", "validation_error")),
                }
                for item in exc.errors()
            ]
        }
        return _error_response(
            status_code=422,
            code="REQUEST_VALIDATION_ERROR",
            message="Request validation failed",
            details=details,
            request_id=request_id_from_request(request),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        del exc
        request_id = request_id_from_request(request)
        logger.error(
            "unexpected_error",
            extra={
                "event": "unexpected_error",
                "request_id": request_id,
                "error_code": "INTERNAL_ERROR",
            },
        )
        return _error_response(
            status_code=500,
            code="INTERNAL_ERROR",
            message="An internal error occurred",
            details={},
            request_id=request_id,
        )


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, object],
    request_id: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    response_headers = dict(headers or {})
    response_headers["X-Request-ID"] = request_id
    body = ErrorResponse(
        error=ErrorData(
            code=code,
            message=message,
            details=details,
            request_id=request_id,
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
        headers=response_headers,
    )
