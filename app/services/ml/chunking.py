"""Deterministic temporal chunking for timestamped transcripts."""

from __future__ import annotations

import math
from collections.abc import Sequence

from app.services.ml.contracts import TranscriptChunk, TranscriptSegment


class TranscriptChunker:
    """Build overlapping transcript windows with video-safe boundaries."""

    def __init__(
        self,
        *,
        target_duration_seconds: float = 12.0,
        min_duration_seconds: float = 8.0,
        max_duration_seconds: float = 20.0,
        overlap_seconds: float = 2.0,
    ) -> None:
        values = (
            target_duration_seconds,
            min_duration_seconds,
            max_duration_seconds,
            overlap_seconds,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Chunk durations must be finite")
        if not 0 < min_duration_seconds <= target_duration_seconds <= max_duration_seconds:
            raise ValueError("Expected 0 < minimum <= target <= maximum chunk duration")
        if not 0 <= overlap_seconds < min_duration_seconds:
            raise ValueError("Chunk overlap must be non-negative and shorter than the minimum")
        if overlap_seconds >= target_duration_seconds:
            raise ValueError("Chunk overlap must be shorter than the target duration")
        self._target_duration = target_duration_seconds
        self._min_duration = min_duration_seconds
        self._max_duration = max_duration_seconds
        self._overlap = overlap_seconds

    def chunk(
        self,
        segments: Sequence[TranscriptSegment],
        *,
        video_duration_seconds: float,
        language: str | None = None,
    ) -> tuple[TranscriptChunk, ...]:
        """Return chronological windows, dropping only blank or out-of-video speech."""
        if not math.isfinite(video_duration_seconds) or video_duration_seconds <= 0:
            raise ValueError("Video duration must be positive and finite")

        normalized = self._normalize_segments(segments, video_duration_seconds)
        if not normalized:
            return ()

        speech_start = normalized[0].start_seconds
        speech_end = max(segment.end_seconds for segment in normalized)
        window_duration = min(self._target_duration, video_duration_seconds)
        step = window_duration - min(self._overlap, max(0.0, window_duration - 1e-9))
        start = speech_start
        chunks: list[TranscriptChunk] = []

        while start < speech_end:
            end = min(start + window_duration, video_duration_seconds)
            if end - start < min(self._min_duration, video_duration_seconds):
                start = max(0.0, end - min(self._min_duration, video_duration_seconds))

            selected = [
                segment
                for segment in normalized
                if segment.end_seconds > start and segment.start_seconds < end
            ]
            if selected:
                text = " ".join(segment.text for segment in selected)
                candidate = TranscriptChunk(
                    start_seconds=start,
                    end_seconds=end,
                    text=text,
                    language=language,
                )
                if not chunks or (
                    candidate.start_seconds,
                    candidate.end_seconds,
                    candidate.text,
                ) != (
                    chunks[-1].start_seconds,
                    chunks[-1].end_seconds,
                    chunks[-1].text,
                ):
                    chunks.append(candidate)

            if end >= speech_end or end >= video_duration_seconds:
                break
            next_start = start + step
            if next_start <= start:
                raise ValueError("Chunk configuration does not make forward progress")
            start = next_start

        return tuple(chunks)

    def _normalize_segments(
        self,
        segments: Sequence[TranscriptSegment],
        video_duration_seconds: float,
    ) -> list[TranscriptSegment]:
        normalized: list[TranscriptSegment] = []
        for segment in segments:
            if not math.isfinite(segment.start_seconds) or not math.isfinite(segment.end_seconds):
                raise ValueError("Transcript timestamps must be finite")
            text = " ".join(segment.text.split())
            start = max(0.0, min(segment.start_seconds, video_duration_seconds))
            end = max(0.0, min(segment.end_seconds, video_duration_seconds))
            if not text or start >= end:
                continue
            normalized.append(TranscriptSegment(start_seconds=start, end_seconds=end, text=text))
        normalized.sort(key=lambda segment: (segment.start_seconds, segment.end_seconds))
        return normalized
