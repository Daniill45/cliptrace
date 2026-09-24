"""Infrastructure liveness and dependency readiness endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_media_probe, get_session, get_storage
from app.core.errors import ServiceNotReadyError
from app.db.models.video import Video
from app.schemas.common import HealthResponse
from app.services.media.probe import MediaProbe
from app.services.media.storage import LocalStorage

router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    """Report process liveness without consulting dependencies."""
    return HealthResponse(status="ok")


@router.get("/health/ready", response_model=HealthResponse)
async def readiness(
    session: Annotated[AsyncSession, Depends(get_session)],
    storage: Annotated[LocalStorage, Depends(get_storage)],
    media_probe: Annotated[MediaProbe, Depends(get_media_probe)],
) -> HealthResponse:
    """Verify migrated DB table, writable storage, and executable ffprobe."""
    try:
        await session.execute(select(Video.id).limit(1))
        await storage.readiness_probe()
        await media_probe.check_available()
    except Exception as exc:
        raise ServiceNotReadyError from exc
    return HealthResponse(status="ready")
