"""Request correlation middleware."""

from __future__ import annotations

from uuid import uuid4

from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestIdMiddleware:
    """Assign a server-generated UUID to every HTTP request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        request_id = str(uuid4())
        state = scope.setdefault("state", {})
        if isinstance(state, dict):
            state["request_id"] = request_id

        async def send_with_request_id(message: Message) -> None:
            if message.get("type") == "http.response.start":
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        await self.app(scope, receive, send_with_request_id)


def request_id_from_request(request: Request) -> str:
    """Read the request ID assigned by middleware."""
    return str(getattr(request.state, "request_id", "unknown"))
