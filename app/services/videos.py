"""Transactional application service for the video resource."""

from __future__ import annotations

import asyncio
import logging
import posixpath
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from starlette.datastructures import UploadFile

from app.core.errors import (
    InternalApplicationError,
    RequestValidationDomainError,
    UnsupportedMediaTypeError,
    VideoNotFoundError,
    VideoNotReadyError,
)
from app.db.models.video import Video, VideoStatus
from app.repositories.search import SearchIndexRepository
from app.repositories.videos import VideoRepository
from app.services.media.probe import MediaProbe
from app.services.media.storage import LocalStorage

logger = logging.getLogger(__name__)
ALLOWED_UPLOAD_CONTENT_TYPES = {"video/mp4", "application/octet-stream"}


@dataclass(frozen=True)
class StreamAsset:
    """Verified metadata needed to build a streaming response."""

    video: Video
    size_bytes: int


class VideoService:
    """Coordinate persistence, media validation, and local storage atomically."""

    def __init__(
        self,
        *,
        session: AsyncSession,
        repository: VideoRepository,
        storage: LocalStorage,
        media_probe: MediaProbe,
        search_indexes: SearchIndexRepository | None = None,
    ) -> None:
        self._session = session
        self._repository = repository
        self._storage = storage
        self._media_probe = media_probe
        self._search_indexes = search_indexes

    async def create(self, *, upload: UploadFile, title: str | None) -> Video:
        """Validate, store, and persist one upload or leave no final artifact."""
        staged = None
        storage_key: str | None = None
        try:
            original_filename = normalize_original_filename(upload.filename)
            normalized_title = normalize_title(title)
            if upload.content_type not in ALLOWED_UPLOAD_CONTENT_TYPES:
                raise UnsupportedMediaTypeError("Declared upload content type is not supported")
            if not original_filename.lower().endswith(".mp4"):
                raise UnsupportedMediaTypeError("Uploaded filename must end with .mp4")

            staged = await self._storage.stage_upload(upload)
            metadata = await self._media_probe.probe(staged.path)
            video_id = uuid4()
            storage_key = await self._storage.finalize(staged, video_id)
            now = datetime.now(UTC)
            video = Video(
                id=video_id,
                title=normalized_title,
                original_filename=original_filename,
                storage_key=storage_key,
                content_type="video/mp4",
                size_bytes=staged.size_bytes,
                sha256=staged.sha256,
                duration_seconds=metadata.duration_seconds,
                width=metadata.width,
                height=metadata.height,
                fps=metadata.fps,
                video_codec=metadata.video_codec,
                video_profile=metadata.video_profile,
                pixel_format=metadata.pixel_format,
                audio_codec=metadata.audio_codec,
                status=VideoStatus.READY,
                error_code=None,
                error_message=None,
                created_at=now,
                updated_at=now,
                ready_at=now,
            )
            try:
                async with self._session.begin():
                    await self._repository.create(video)
                    if self._search_indexes is not None:
                        await self._search_indexes.create_pending(video.id)
            except BaseException:
                await self._storage.remove_final(storage_key)
                raise
            return video
        finally:
            if staged is not None:
                await self._storage.discard_staged(staged)
            await upload.close()

    async def get(self, video_id: UUID) -> Video:
        """Return one resource or raise the public not-found domain error."""
        video = await self._repository.get(video_id)
        if video is None:
            raise VideoNotFoundError
        return video

    async def list(self, *, limit: int, offset: int) -> tuple[Sequence[Video], int]:
        """Return a stable newest-first page and the total row count."""
        return (
            await self._repository.list(limit=limit, offset=offset),
            await self._repository.count(),
        )

    async def get_stream_asset(self, video_id: UUID) -> StreamAsset:
        """Resolve a ready video and verify current physical metadata."""
        video = await self.get(video_id)
        if video.status is not VideoStatus.READY:
            raise VideoNotReadyError
        size = await self._storage.verify(video.storage_key, video.size_bytes)
        return StreamAsset(video=video, size_bytes=size)

    async def delete(self, video_id: UUID) -> None:
        """Delete via rename/commit/unlink so crashes cannot expose an orphan DB row."""
        tombstone_key: str | None = None
        storage_key: str | None = None
        try:
            async with self._session.begin():
                video = await self._repository.get_for_update(video_id)
                if video is None:
                    raise VideoNotFoundError
                storage_key = video.storage_key
                tombstone_key = await self._storage.move_to_tombstone(storage_key)
                await self._repository.delete(video)
        except (Exception, asyncio.CancelledError) as exc:
            if storage_key is not None and tombstone_key is not None:
                try:
                    await self._storage.restore_tombstone(storage_key, tombstone_key)
                except OSError:
                    logger.error(
                        "delete_tombstone_restore_failed",
                        extra={
                            "event": "delete_tombstone_restore_failed",
                            "video_id": str(video_id),
                        },
                    )
            if isinstance(exc, VideoNotFoundError | asyncio.CancelledError):
                raise
            raise InternalApplicationError from exc
        if tombstone_key is not None:
            try:
                await self._storage.remove_tombstone(tombstone_key)
            except OSError:
                logger.error(
                    "delete_tombstone_unlink_failed",
                    extra={"event": "delete_tombstone_unlink_failed", "video_id": str(video_id)},
                )


def normalize_original_filename(filename: str | None) -> str:
    """Extract display-only basename without ever using it as a path."""
    if not filename:
        raise RequestValidationDomainError("A non-empty filename is required")
    basename = posixpath.basename(filename.replace("\\", "/"))
    if not basename or "\x00" in basename or len(basename) > 255:
        raise RequestValidationDomainError("Filename must contain 1 to 255 safe characters")
    return basename


def normalize_title(title: str | None) -> str | None:
    """Trim an optional title and enforce its public length contract."""
    if title is None:
        return None
    normalized = title.strip()
    if not 1 <= len(normalized) <= 200:
        raise RequestValidationDomainError("Title must contain 1 to 200 characters after trim")
    return normalized
