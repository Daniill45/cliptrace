"""Thin REST endpoints for video resources and fragment search."""

from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query, Response, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import (
    UploadFormData,
    get_search_index_service,
    get_search_service,
    get_settings_dependency,
    get_storage,
    get_video_service,
    parse_upload_form,
)
from app.core.config import Settings
from app.core.errors import RangeNotSatisfiableError
from app.schemas.errors import ErrorResponse
from app.schemas.search import SearchRequest, SearchResponse
from app.schemas.search_indexes import SearchIndexStatusResponse
from app.schemas.videos import VideoListResponse, VideoResponse, video_to_response
from app.services.media.range_requests import InvalidByteRange, parse_byte_range
from app.services.media.storage import LocalStorage
from app.services.search.indexes import SearchIndexService, SearchIndexSnapshot
from app.services.search.service import SearchService
from app.services.videos import VideoService

router = APIRouter(prefix="/videos", tags=["videos"])

# FastAPI types decorator-level OpenAPI metadata as nested dictionaries containing Any.
_VIDEO_STREAM_HEADERS: dict[str, Any] = {
    "Accept-Ranges": {"schema": {"type": "string"}},
    "Content-Length": {"schema": {"type": "integer", "format": "int64"}},
    "Content-Disposition": {"schema": {"type": "string"}},
}
_VIDEO_PARTIAL_HEADERS: dict[str, Any] = {
    **_VIDEO_STREAM_HEADERS,
    "Content-Range": {"schema": {"type": "string"}},
}
_VIDEO_CONTENT_ERRORS: dict[int | str, dict[str, Any]] = {
    404: {"description": "Video was not found", "model": ErrorResponse},
    409: {"description": "Video is not ready", "model": ErrorResponse},
    410: {"description": "Video file is missing or inconsistent", "model": ErrorResponse},
    416: {
        "description": "Requested byte range is not satisfiable",
        "model": ErrorResponse,
        "headers": {"Content-Range": {"schema": {"type": "string"}}},
    },
    422: {"description": "Request validation failed", "model": ErrorResponse},
    500: {"description": "Internal application error", "model": ErrorResponse},
}
VIDEO_CONTENT_GET_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "Complete MP4 content",
        "content": {"video/mp4": {"schema": {"type": "string", "format": "binary"}}},
        "headers": _VIDEO_STREAM_HEADERS,
    },
    206: {
        "description": "Partial MP4 content for one byte range",
        "content": {"video/mp4": {"schema": {"type": "string", "format": "binary"}}},
        "headers": _VIDEO_PARTIAL_HEADERS,
    },
    **_VIDEO_CONTENT_ERRORS,
}
VIDEO_CONTENT_HEAD_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "Headers for complete MP4 content; response body is empty",
        "headers": _VIDEO_STREAM_HEADERS,
    },
    206: {
        "description": "Headers for one MP4 byte range; response body is empty",
        "headers": _VIDEO_PARTIAL_HEADERS,
    },
    **_VIDEO_CONTENT_ERRORS,
}


@router.post(
    "",
    response_model=VideoResponse,
    status_code=status.HTTP_201_CREATED,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["file"],
                        "properties": {
                            "file": {"type": "string", "format": "binary"},
                            "title": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 200,
                            },
                        },
                    }
                }
            },
        }
    },
)
async def upload_video(
    payload: Annotated[UploadFormData, Depends(parse_upload_form)],
    service: Annotated[VideoService, Depends(get_video_service)],
    settings: Annotated[Settings, Depends(get_settings_dependency)],
    response: Response,
) -> VideoResponse:
    video = await service.create(upload=payload.file, title=payload.title)
    response.headers["Location"] = f"{settings.api_v1_prefix}/videos/{video.id}"
    return video_to_response(video, api_prefix=settings.api_v1_prefix)


@router.get("", response_model=VideoListResponse)
async def list_videos(
    service: Annotated[VideoService, Depends(get_video_service)],
    settings: Annotated[Settings, Depends(get_settings_dependency)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> VideoListResponse:
    videos, total = await service.list(limit=limit, offset=offset)
    return VideoListResponse(
        items=[video_to_response(item, api_prefix=settings.api_v1_prefix) for item in videos],
        limit=limit,
        offset=offset,
        total=total,
    )


@router.get("/{video_id}", response_model=VideoResponse)
async def get_video(
    video_id: UUID,
    service: Annotated[VideoService, Depends(get_video_service)],
    settings: Annotated[Settings, Depends(get_settings_dependency)],
) -> VideoResponse:
    return video_to_response(await service.get(video_id), api_prefix=settings.api_v1_prefix)


@router.get(
    "/{video_id}/content",
    operation_id="get_video_content",
    response_class=Response,
    responses=VIDEO_CONTENT_GET_RESPONSES,
)
async def get_video_content(
    video_id: UUID,
    service: Annotated[VideoService, Depends(get_video_service)],
    storage: Annotated[LocalStorage, Depends(get_storage)],
    range_header: Annotated[
        str | None,
        Header(
            alias="Range",
            description="Optional single byte range, for example: bytes=0-1023",
        ),
    ] = None,
) -> Response:
    return await _build_stream_response(
        video_id=video_id,
        service=service,
        storage=storage,
        requested_range=range_header,
        head_only=False,
    )


@router.head(
    "/{video_id}/content",
    operation_id="head_video_content",
    response_class=Response,
    responses=VIDEO_CONTENT_HEAD_RESPONSES,
)
async def head_video_content(
    video_id: UUID,
    service: Annotated[VideoService, Depends(get_video_service)],
    storage: Annotated[LocalStorage, Depends(get_storage)],
    range_header: Annotated[
        str | None,
        Header(
            alias="Range",
            description="Optional single byte range, for example: bytes=0-1023",
        ),
    ] = None,
) -> Response:
    return await _build_stream_response(
        video_id=video_id,
        service=service,
        storage=storage,
        requested_range=range_header,
        head_only=True,
    )


async def _build_stream_response(
    *,
    video_id: UUID,
    service: VideoService,
    storage: LocalStorage,
    requested_range: str | None,
    head_only: bool,
) -> Response:
    """Build GET/HEAD responses from the same verified single-range contract."""
    asset = await service.get_stream_asset(video_id)
    if requested_range is None:
        response_status = 200
        start = 0
        end = asset.size_bytes - 1
    else:
        try:
            selected = parse_byte_range(requested_range, asset.size_bytes)
        except InvalidByteRange as exc:
            raise RangeNotSatisfiableError(asset.size_bytes) from exc
        response_status = 206
        start = selected.start
        end = selected.end
    length = end - start + 1
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Type": "video/mp4",
        "Content-Length": str(length),
        "Content-Disposition": "inline",
    }
    if response_status == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{asset.size_bytes}"
    if head_only:
        return Response(status_code=response_status, headers=headers)
    file_handle = await storage.open_verified(asset.video.storage_key, asset.size_bytes)
    return StreamingResponse(
        storage.stream(file_handle, start=start, length=length),
        status_code=response_status,
        headers=headers,
    )


@router.post("/{video_id}/search", response_model=SearchResponse)
async def search_video(
    video_id: UUID,
    payload: SearchRequest,
    service: Annotated[SearchService, Depends(get_search_service)],
) -> SearchResponse:
    return await service.search(video_id=video_id, query=payload.query, limit=payload.limit)


@router.get("/{video_id}/search-index", response_model=SearchIndexStatusResponse)
async def get_video_search_index(
    video_id: UUID,
    service: Annotated[SearchIndexService, Depends(get_search_index_service)],
) -> SearchIndexStatusResponse:
    return _search_index_to_response(await service.get(video_id))


@router.post(
    "/{video_id}/search-index/reindex",
    response_model=SearchIndexStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def reindex_video_search(
    video_id: UUID,
    service: Annotated[SearchIndexService, Depends(get_search_index_service)],
) -> SearchIndexStatusResponse:
    return _search_index_to_response(await service.reindex(video_id))


@router.delete("/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_video(
    video_id: UUID,
    service: Annotated[VideoService, Depends(get_video_service)],
) -> Response:
    await service.delete(video_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _search_index_to_response(index: SearchIndexSnapshot) -> SearchIndexStatusResponse:
    return SearchIndexStatusResponse(
        video_id=index.video_id,
        status=index.status.value,
        attempts=index.attempts,
        fragment_count=index.fragment_count,
        pipeline_version=index.pipeline_version,
        asr_model_version=index.asr_model_version,
        embedding_model_version=index.embedding_model_version,
        error_code=index.error_code,
        error_message=index.error_message,
        created_at=index.created_at,
        updated_at=index.updated_at,
        started_at=index.started_at,
        indexed_at=index.indexed_at,
    )
