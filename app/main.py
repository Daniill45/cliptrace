"""ClipTrace FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.api.v1.endpoints.health import router as health_router
from app.api.v1.router import router as api_v1_router
from app.core.config import Settings, get_settings
from app.core.exception_handlers import register_exception_handlers
from app.core.http_middleware import HttpObservabilityMiddleware
from app.core.logging import configure_logging
from app.core.request_id import RequestIdMiddleware
from app.db.models.video import Video
from app.db.session import create_engine, create_session_factory
from app.repositories.videos import VideoRepository
from app.services.media.probe import MediaProbe
from app.services.media.storage import LocalStorage
from app.services.search.mock import MockFixedTimestampSearchEngine

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build an isolated application instance with explicit infrastructure state."""
    resolved = settings or get_settings()
    configure_logging(resolved.log_level)
    engine = create_engine(resolved.database_url)
    session_factory = create_session_factory(engine)
    storage = LocalStorage(
        resolved.upload_dir,
        max_size_bytes=resolved.max_upload_size_bytes,
        chunk_size_bytes=resolved.upload_chunk_size_bytes,
        temp_max_age_seconds=resolved.temp_file_max_age_seconds,
    )
    media_probe = MediaProbe(
        binary=resolved.ffprobe_binary,
        timeout_seconds=resolved.ffprobe_timeout_seconds,
        max_duration_seconds=resolved.max_video_duration_seconds,
        min_duration_seconds=(
            resolved.mock_result_start_seconds + resolved.mock_result_duration_seconds
        ),
    )
    search_engine = MockFixedTimestampSearchEngine(
        start_seconds=resolved.mock_result_start_seconds,
        duration_seconds=resolved.mock_result_duration_seconds,
        min_fragment_duration_seconds=resolved.min_fragment_duration_seconds,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await storage.initialize()
        try:
            async with session_factory() as session:
                await session.execute(select(Video.id).limit(1))
                referenced = await VideoRepository(session).storage_keys()
                await storage.reconcile(referenced)
        except SQLAlchemyError:
            logger.warning(
                "startup_database_not_ready",
                extra={"event": "startup_database_not_ready"},
            )
        yield
        await engine.dispose()

    docs_url = "/docs" if resolved.app_env != "production" else None
    openapi_url = "/openapi.json" if resolved.app_env != "production" else None
    app = FastAPI(
        title=resolved.app_name,
        version="1.0.0",
        docs_url=docs_url,
        openapi_url=openapi_url,
        lifespan=lifespan,
    )
    app.state.settings = resolved
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.storage = storage
    app.state.media_probe = media_probe
    app.state.search_engine = search_engine

    app.include_router(health_router)
    app.include_router(api_v1_router, prefix=resolved.api_v1_prefix)

    @app.get("/", include_in_schema=False, response_class=FileResponse)
    async def frontend() -> FileResponse:
        """Serve the local ClipTrace single-page interface."""
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="frontend-assets")
    register_exception_handlers(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "HEAD", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Range"],
        expose_headers=["Content-Range", "Content-Length", "Location", "X-Request-ID"],
    )
    app.add_middleware(
        HttpObservabilityMiddleware,
        upload_path=f"{resolved.api_v1_prefix}/videos",
        max_upload_size_bytes=resolved.max_upload_size_bytes,
        multipart_overhead_allowance_bytes=resolved.upload_chunk_size_bytes,
    )
    app.add_middleware(RequestIdMiddleware)
    return app


app = create_app()
