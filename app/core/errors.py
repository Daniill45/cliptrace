"""Typed domain errors independent from the HTTP framework."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(eq=False)
class DomainError(Exception):
    """Base class for safe client-visible application errors."""

    code: str
    message: str
    status_code: int
    details: Mapping[str, object] = field(default_factory=dict)
    headers: Mapping[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.code


class VideoNotFoundError(DomainError):
    def __init__(self) -> None:
        super().__init__("VIDEO_NOT_FOUND", "Video was not found", 404)


class VideoNotReadyError(DomainError):
    def __init__(self) -> None:
        super().__init__("VIDEO_NOT_READY", "Video is not ready", 409)


class VideoFileMissingError(DomainError):
    def __init__(self) -> None:
        super().__init__(
            "VIDEO_FILE_MISSING",
            "Video file is missing or inconsistent with its metadata",
            410,
        )


class UploadTooLargeError(DomainError):
    def __init__(self) -> None:
        super().__init__("UPLOAD_TOO_LARGE", "Uploaded file exceeds the size limit", 413)


class UnsupportedMediaTypeError(DomainError):
    def __init__(self, message: str = "Only compatible MP4 video is supported") -> None:
        super().__init__("UNSUPPORTED_MEDIA_TYPE", message, 415)


class InvalidVideoError(DomainError):
    def __init__(self, message: str = "The uploaded file is not a valid video") -> None:
        super().__init__("INVALID_VIDEO", message, 422)


class VideoStreamRequiredError(DomainError):
    def __init__(self) -> None:
        super().__init__("VIDEO_STREAM_REQUIRED", "A video stream is required", 422)


class VideoAudioTrackRequiredError(DomainError):
    def __init__(self) -> None:
        super().__init__("VIDEO_AUDIO_TRACK_REQUIRED", "An AAC audio track is required", 422)


class VideoTooShortError(DomainError):
    def __init__(self) -> None:
        super().__init__("VIDEO_TOO_SHORT", "Video is too short for mock search", 422)


class VideoTooLongError(DomainError):
    def __init__(self) -> None:
        super().__init__("VIDEO_TOO_LONG", "Video exceeds the duration limit", 422)


class RangeNotSatisfiableError(DomainError):
    def __init__(self, file_size: int) -> None:
        super().__init__(
            "RANGE_NOT_SATISFIABLE",
            "Requested byte range is not satisfiable",
            416,
            headers={"Content-Range": f"bytes */{file_size}"},
        )


class ServiceNotReadyError(DomainError):
    def __init__(self) -> None:
        super().__init__("SERVICE_NOT_READY", "Service dependencies are not ready", 503)


class InternalApplicationError(DomainError):
    def __init__(self) -> None:
        super().__init__("INTERNAL_ERROR", "An internal error occurred", 500)


class RequestValidationDomainError(DomainError):
    def __init__(self, message: str = "Request validation failed") -> None:
        super().__init__("REQUEST_VALIDATION_ERROR", message, 422)
