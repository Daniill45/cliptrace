"""Request-scoped dependency wiring."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated, cast

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.datastructures import FormData, UploadFile

from app.core.config import Settings
from app.core.errors import RequestValidationDomainError
from app.repositories.videos import VideoRepository
from app.services.media.probe import MediaProbe
from app.services.media.storage import LocalStorage
from app.services.search.base import FragmentSearchEngine
from app.services.search.service import SearchService
from app.services.videos import VideoService


@dataclass(frozen=True)
class UploadFormData:
    file: UploadFile
    title: str | None


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield one session and always rollback any uncommitted read transaction."""
    factory = cast(async_sessionmaker[AsyncSession], request.app.state.session_factory)
    async with factory() as session:
        yield session


def get_settings_dependency(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_storage(request: Request) -> LocalStorage:
    return cast(LocalStorage, request.app.state.storage)


def get_media_probe(request: Request) -> MediaProbe:
    return cast(MediaProbe, request.app.state.media_probe)


def get_search_engine(request: Request) -> FragmentSearchEngine:
    return cast(FragmentSearchEngine, request.app.state.search_engine)


def get_video_repository(session: Annotated[AsyncSession, Depends(get_session)]) -> VideoRepository:
    return VideoRepository(session)


def get_video_service(
    session: Annotated[AsyncSession, Depends(get_session)],
    repository: Annotated[VideoRepository, Depends(get_video_repository)],
    storage: Annotated[LocalStorage, Depends(get_storage)],
    media_probe: Annotated[MediaProbe, Depends(get_media_probe)],
) -> VideoService:
    return VideoService(
        session=session,
        repository=repository,
        storage=storage,
        media_probe=media_probe,
    )


def get_search_service(
    repository: Annotated[VideoRepository, Depends(get_video_repository)],
    storage: Annotated[LocalStorage, Depends(get_storage)],
    engine: Annotated[FragmentSearchEngine, Depends(get_search_engine)],
    settings: Annotated[Settings, Depends(get_settings_dependency)],
) -> SearchService:
    return SearchService(
        repository=repository,
        storage=storage,
        engine=engine,
        api_prefix=settings.api_v1_prefix,
    )


async def parse_upload_form(request: Request) -> AsyncIterator[UploadFormData]:
    """Validate form fields and retain ownership until all request work completes."""
    try:
        form: FormData = await request.form()
    except Exception as exc:
        raise RequestValidationDomainError("Malformed multipart form") from exc
    try:
        file_values = form.getlist("file")
        title_values = form.getlist("title")
        uploaded_files = [item for _, item in form.multi_items() if isinstance(item, UploadFile)]
        if (
            len(file_values) != 1
            or not isinstance(file_values[0], UploadFile)
            or len(uploaded_files) != 1
        ):
            raise RequestValidationDomainError("Exactly one file field is required")
        if len(title_values) > 1 or any(not isinstance(item, str) for item in title_values):
            raise RequestValidationDomainError("At most one title field is supported")
        title: str | None = None
        if title_values:
            title_value = title_values[0]
            if not isinstance(title_value, str):
                raise RequestValidationDomainError("Title must be a string")
            title = title_value
        yield UploadFormData(file=file_values[0], title=title)
    finally:
        await form.close()
