"""Persistence-only access to video metadata."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.video import Video, VideoStatus


class VideoRepository:
    """SQLAlchemy repository with no transaction ownership."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, video: Video) -> Video:
        self._session.add(video)
        await self._session.flush()
        return video

    async def get(self, video_id: UUID) -> Video | None:
        return await self._session.get(Video, video_id)

    async def get_for_update(self, video_id: UUID) -> Video | None:
        result = await self._session.execute(
            select(Video).where(Video.id == video_id).with_for_update()
        )
        return result.scalar_one_or_none()

    async def list(self, *, limit: int, offset: int) -> Sequence[Video]:
        result = await self._session.execute(
            select(Video).order_by(Video.created_at.desc()).limit(limit).offset(offset)
        )
        return result.scalars().all()

    async def count(self) -> int:
        return int(await self._session.scalar(select(func.count()).select_from(Video)) or 0)

    async def delete(self, video: Video) -> None:
        await self._session.delete(video)
        await self._session.flush()

    async def update_status(
        self,
        video: Video,
        *,
        status: VideoStatus,
        error_code: str | None,
        error_message: str | None,
    ) -> Video:
        video.status = status
        video.error_code = error_code
        video.error_message = error_message
        await self._session.flush()
        return video

    async def storage_keys(self) -> set[str]:
        result = await self._session.scalars(select(Video.storage_key))
        return set(result.all())
