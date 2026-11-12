"""Deterministic temporal chunking for timestamped transcripts."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from app.services.ml.contracts import TranscriptChunk, TranscriptSegment


class TranscriptChunker:
    """Build overlapping transcript windows with video-safe boundaries."""

    version = "fixed-window-v1"

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


_TERMINAL_PUNCTUATION = frozenset(".!?\u2026\u3002\uff01\uff1f")
_TRAILING_CLOSERS = frozenset("\"'\u00bb\u201d\u2019)]}")
_NO_SPACE_BEFORE = frozenset(
    ",.!?;:%\u2026)]}\u00bb\u201d\u2019\u3002\uff01\uff1f\uff0c\u3001\uff1a\uff1b"
)
_NO_SPACE_AFTER = frozenset("([{\u00ab\u201c\u2018")


@dataclass(frozen=True, slots=True)
class _TimedToken:
    start_seconds: float
    end_seconds: float
    text: str
    word_count: int
    source_position: int
    part_position: int


@dataclass(frozen=True, slots=True)
class _SentenceUnit:
    start_seconds: float
    end_seconds: float
    text: str
    word_count: int
    hard_break_after: bool


class SemanticTranscriptChunker:
    """Build deterministic sentence-like chunks from timestamped transcript words."""

    version = "semantic-pause-punct-v1"

    def __init__(
        self,
        *,
        target_duration_seconds: float = 12.0,
        min_duration_seconds: float = 8.0,
        max_duration_seconds: float = 20.0,
        target_words: int = 48,
        max_words: int = 80,
        max_sentence_duration_seconds: float = 15.0,
        max_sentence_words: int = 40,
        pause_boundary_seconds: float = 0.8,
        hard_pause_seconds: float = 2.5,
        overlap_sentences: int = 1,
    ) -> None:
        duration_values = (
            target_duration_seconds,
            min_duration_seconds,
            max_duration_seconds,
            max_sentence_duration_seconds,
            pause_boundary_seconds,
            hard_pause_seconds,
        )
        if not all(math.isfinite(value) for value in duration_values):
            raise ValueError("Semantic chunk durations and pauses must be finite")
        if not 0 < min_duration_seconds <= target_duration_seconds <= max_duration_seconds:
            raise ValueError("Expected 0 < minimum <= target <= maximum chunk duration")
        if not 0 < max_sentence_duration_seconds <= max_duration_seconds:
            raise ValueError("Sentence duration must be positive and at most the chunk maximum")
        if not 0 < pause_boundary_seconds <= hard_pause_seconds:
            raise ValueError("Expected 0 < sentence pause <= hard pause")
        integer_values = (target_words, max_words, max_sentence_words, overlap_sentences)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_values):
            raise ValueError("Semantic chunk word limits and overlap must be integers")
        if not 0 < target_words <= max_words:
            raise ValueError("Expected 0 < target words <= maximum words")
        if not 0 < max_sentence_words <= max_words:
            raise ValueError("Sentence word maximum must be positive and at most the chunk maximum")
        if overlap_sentences < 0:
            raise ValueError("Sentence overlap must be non-negative")

        self._target_duration = target_duration_seconds
        self._min_duration = min_duration_seconds
        self._max_duration = max_duration_seconds
        self._target_words = target_words
        self._max_words = max_words
        self._max_sentence_duration = max_sentence_duration_seconds
        self._max_sentence_words = max_sentence_words
        self._pause_boundary = pause_boundary_seconds
        self._hard_pause = hard_pause_seconds
        self._overlap_sentences = overlap_sentences

    def chunk(
        self,
        segments: Sequence[TranscriptSegment],
        *,
        video_duration_seconds: float,
        language: str | None = None,
    ) -> tuple[TranscriptChunk, ...]:
        """Return chunks aligned to punctuation, pauses, and whole-sentence overlap."""
        if not math.isfinite(video_duration_seconds) or video_duration_seconds <= 0:
            raise ValueError("Video duration must be positive and finite")

        tokens = self._normalize_tokens(segments, video_duration_seconds)
        if not tokens:
            return ()
        sentences = self._build_sentences(tokens)

        chunks: list[TranscriptChunk] = []
        region_start = 0
        for position, sentence in enumerate(sentences):
            if not sentence.hard_break_after:
                continue
            chunks.extend(self._chunk_region(sentences[region_start : position + 1], language))
            region_start = position + 1
        if region_start < len(sentences):
            chunks.extend(self._chunk_region(sentences[region_start:], language))
        return tuple(chunks)

    def _normalize_tokens(
        self,
        segments: Sequence[TranscriptSegment],
        video_duration_seconds: float,
    ) -> list[_TimedToken]:
        tokens: list[_TimedToken] = []
        for source_position, segment in enumerate(segments):
            if not math.isfinite(segment.start_seconds) or not math.isfinite(segment.end_seconds):
                raise ValueError("Transcript timestamps must be finite")
            text = " ".join(segment.text.split())
            start = max(0.0, min(segment.start_seconds, video_duration_seconds))
            end = max(0.0, min(segment.end_seconds, video_duration_seconds))
            if not text or start >= end:
                continue

            parts = text.split()
            duration = end - start
            for part_position, part in enumerate(parts):
                part_start = start + duration * part_position / len(parts)
                part_end = (
                    end
                    if part_position == len(parts) - 1
                    else start + duration * (part_position + 1) / len(parts)
                )
                tokens.append(
                    _TimedToken(
                        start_seconds=part_start,
                        end_seconds=part_end,
                        text=part,
                        word_count=int(any(character.isalnum() for character in part)),
                        source_position=source_position,
                        part_position=part_position,
                    )
                )

        tokens.sort(
            key=lambda token: (
                token.start_seconds,
                token.end_seconds,
                token.source_position,
                token.part_position,
            )
        )
        return tokens

    def _build_sentences(self, tokens: Sequence[_TimedToken]) -> list[_SentenceUnit]:
        sentences: list[_SentenceUnit] = []
        current: list[_TimedToken] = []
        current_start = 0.0
        current_end = 0.0
        current_words = 0

        for position, token in enumerate(tokens):
            if not current:
                current_start = token.start_seconds
                current_end = token.end_seconds
            current.append(token)
            current_end = max(current_end, token.end_seconds)
            current_words += token.word_count

            next_token = tokens[position + 1] if position + 1 < len(tokens) else None
            if next_token is None:
                should_close = True
                hard_break_after = False
            else:
                gap = max(0.0, next_token.start_seconds - current_end)
                hard_break_after = gap >= self._hard_pause
                exceeds_sentence_words = (
                    current_words + next_token.word_count > self._max_sentence_words
                )
                exceeds_sentence_duration = (
                    max(current_end, next_token.end_seconds) - current_start
                    > self._max_sentence_duration
                )
                should_close = (
                    _has_terminal_punctuation(token.text)
                    or gap >= self._pause_boundary
                    or exceeds_sentence_words
                    or exceeds_sentence_duration
                )

            if not should_close:
                continue
            sentences.append(
                _SentenceUnit(
                    start_seconds=current_start,
                    end_seconds=current_end,
                    text=_join_text_parts(token_value.text for token_value in current),
                    word_count=current_words,
                    hard_break_after=hard_break_after,
                )
            )
            current = []
            current_words = 0

        return sentences

    def _chunk_region(
        self,
        sentences: Sequence[_SentenceUnit],
        language: str | None,
    ) -> list[TranscriptChunk]:
        chunks: list[TranscriptChunk] = []
        start = 0

        while start < len(sentences):
            end = start
            chunk_start = sentences[start].start_seconds
            chunk_end = chunk_start
            chunk_words = 0

            while end < len(sentences):
                sentence = sentences[end]
                candidate_end = max(chunk_end, sentence.end_seconds)
                candidate_duration = candidate_end - chunk_start
                candidate_words = chunk_words + sentence.word_count
                exceeds_maximum = (
                    candidate_duration > self._max_duration or candidate_words > self._max_words
                )
                if exceeds_maximum:
                    if end == start:
                        # A raw ASR fallback can contain one unsplittable token with a
                        # very wide timestamp. Keep it searchable instead of failing
                        # the complete video index.
                        chunk_end = candidate_end
                        chunk_words = candidate_words
                        end += 1
                    break

                chunk_end = candidate_end
                chunk_words = candidate_words
                end += 1

                target_reached = (
                    chunk_end - chunk_start >= self._target_duration
                    or chunk_words >= self._target_words
                )
                if not target_reached or end >= len(sentences):
                    continue
                if self._can_absorb_short_tail(
                    sentences,
                    tail_start=end,
                    chunk_start=chunk_start,
                    chunk_words=chunk_words,
                ):
                    continue
                break

            selected = sentences[start:end]
            chunks.append(
                TranscriptChunk(
                    start_seconds=selected[0].start_seconds,
                    end_seconds=max(sentence.end_seconds for sentence in selected),
                    text=_join_text_parts(sentence.text for sentence in selected),
                    language=language,
                )
            )
            if end >= len(sentences):
                break
            next_start = max(start + 1, end - self._overlap_sentences)
            if next_start <= start:
                raise ValueError("Semantic chunk configuration does not make forward progress")
            start = next_start

        return chunks

    def _can_absorb_short_tail(
        self,
        sentences: Sequence[_SentenceUnit],
        *,
        tail_start: int,
        chunk_start: float,
        chunk_words: int,
    ) -> bool:
        if tail_start >= len(sentences):
            return False
        tail_duration = sentences[-1].end_seconds - sentences[tail_start].start_seconds
        if tail_duration >= self._min_duration:
            return False
        combined_duration = sentences[-1].end_seconds - chunk_start
        combined_words = chunk_words + sum(
            sentence.word_count for sentence in sentences[tail_start:]
        )
        return combined_duration <= self._max_duration and combined_words <= self._max_words


def _has_terminal_punctuation(text: str) -> bool:
    candidate = text.rstrip()
    while candidate and candidate[-1] in _TRAILING_CLOSERS:
        candidate = candidate[:-1].rstrip()
    return bool(candidate) and candidate[-1] in _TERMINAL_PUNCTUATION


def _join_text_parts(parts: Iterable[str]) -> str:
    result = ""
    for part_value in parts:
        part = part_value.strip()
        if not part:
            continue
        if not result:
            result = part
        elif part[0] in _NO_SPACE_BEFORE or part.startswith(("'", "\u2019")):
            result += part
        elif result[-1] in _NO_SPACE_AFTER:
            result += part
        else:
            result += f" {part}"
    return result
