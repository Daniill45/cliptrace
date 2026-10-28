from __future__ import annotations

import pytest

from app.services.ml.chunking import TranscriptChunker
from app.services.ml.contracts import TranscriptSegment


def test_chunker_builds_overlapping_bounded_windows_and_rebalances_tail() -> None:
    chunker = TranscriptChunker(
        target_duration_seconds=12.0,
        min_duration_seconds=8.0,
        max_duration_seconds=20.0,
        overlap_seconds=2.0,
    )
    segments = (
        TranscriptSegment(0.0, 1.0, "one"),
        TranscriptSegment(9.0, 10.0, "two"),
        TranscriptSegment(11.0, 12.0, "three"),
        TranscriptSegment(19.0, 20.0, "four"),
        TranscriptSegment(24.0, 25.0, "five"),
    )

    chunks = chunker.chunk(segments, video_duration_seconds=25.0, language="en")

    assert [(chunk.start_seconds, chunk.end_seconds) for chunk in chunks] == [
        (0.0, 12.0),
        (10.0, 22.0),
        (17.0, 25.0),
    ]
    assert [chunk.text for chunk in chunks] == [
        "one two three",
        "three four",
        "four five",
    ]
    assert all(8.0 <= chunk.end_seconds - chunk.start_seconds <= 20.0 for chunk in chunks)
    assert all(chunk.language == "en" for chunk in chunks)


def test_chunker_sorts_clamps_and_drops_blank_or_out_of_range_segments() -> None:
    chunker = TranscriptChunker()
    segments = (
        TranscriptSegment(29.0, 40.0, "  ending  "),
        TranscriptSegment(2.0, 3.0, "  "),
        TranscriptSegment(-2.0, 3.0, " opening   words "),
        TranscriptSegment(40.0, 50.0, "outside"),
    )

    chunks = chunker.chunk(segments, video_duration_seconds=30.0)

    assert [(chunk.start_seconds, chunk.end_seconds, chunk.text) for chunk in chunks] == [
        (0.0, 12.0, "opening words"),
        (20.0, 30.0, "ending"),
    ]


def test_chunker_uses_entire_video_when_it_is_shorter_than_minimum() -> None:
    chunks = TranscriptChunker().chunk(
        (TranscriptSegment(1.0, 2.0, "short"),),
        video_duration_seconds=5.0,
    )

    assert len(chunks) == 1
    assert (chunks[0].start_seconds, chunks[0].end_seconds) == (0.0, 5.0)


def test_chunker_returns_empty_for_no_usable_transcript() -> None:
    chunks = TranscriptChunker().chunk(
        (TranscriptSegment(1.0, 2.0, "   "),),
        video_duration_seconds=10.0,
    )

    assert chunks == ()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_duration_seconds": 0.0}, "minimum"),
        ({"min_duration_seconds": 10.0, "target_duration_seconds": 8.0}, "minimum"),
        ({"overlap_seconds": 8.0}, "overlap"),
        ({"target_duration_seconds": float("nan")}, "finite"),
    ],
)
def test_chunker_rejects_invalid_configuration(
    kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TranscriptChunker(**kwargs)


def test_chunker_rejects_invalid_video_or_transcript_timestamps() -> None:
    chunker = TranscriptChunker()

    with pytest.raises(ValueError, match="Video duration"):
        chunker.chunk((), video_duration_seconds=float("inf"))
    with pytest.raises(ValueError, match="timestamps"):
        chunker.chunk(
            (TranscriptSegment(float("nan"), 2.0, "bad"),),
            video_duration_seconds=10.0,
        )
