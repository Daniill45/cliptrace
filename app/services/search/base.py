"""Framework-neutral online fragment search contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID


@dataclass(frozen=True)
class VideoDescriptor:
    id: UUID
    duration_seconds: float
    storage_key: str


@dataclass(frozen=True)
class SearchHit:
    start_seconds: float
    end_seconds: float
    score: float | None
    preview_text: str | None


class FragmentSearchEngine(Protocol):
    """Stable online-search seam for mock and future ML implementations."""

    @property
    def mode(self) -> str: ...

    @property
    def version(self) -> str: ...

    async def search(
        self,
        *,
        video: VideoDescriptor,
        query: str,
        limit: int,
    ) -> Sequence[SearchHit]: ...
