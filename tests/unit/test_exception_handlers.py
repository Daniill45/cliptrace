from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from app.core.exception_handlers import register_exception_handlers
from app.core.request_id import RequestIdMiddleware


@pytest.mark.asyncio
async def test_unexpected_error_uses_safe_stable_envelope() -> None:
    application = FastAPI()

    @application.get("/explode")
    async def explode() -> None:
        raise RuntimeError("private failure detail")

    register_exception_handlers(application)
    application.add_middleware(RequestIdMiddleware)
    transport = httpx.ASGITransport(app=application, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/explode")

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "INTERNAL_ERROR",
            "message": "An internal error occurred",
            "details": {},
            "request_id": response.headers["x-request-id"],
        }
    }
    assert "private failure detail" not in response.text
