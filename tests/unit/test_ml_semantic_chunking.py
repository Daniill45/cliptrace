from __future__ import annotations

from typing import Any

import pytest

from app.services.ml.chunking import SemanticTranscriptChunker
from app.services.ml.contracts import TranscriptSegment


def _chunker(**overrides: Any) -> SemanticTranscriptChunker:
    settings: dict[str, Any] = {
        "target_duration_seconds": 4.0,
        "min_duration_seconds": 0.5,
        "max_duration_seconds": 8.0,
        "target_words": 100,
        "max_words": 100,
        "max_sentence_duration_seconds": 8.0,
        "max_sentence_words": 100,
        "pause_boundary_seconds": 0.8,
        "hard_pause_seconds": 2.5,
        "overlap_sentences": 1,
    }
    settings.update(overrides)
    return SemanticTranscriptChunker(**settings)


def test_semantic_chunker_overlaps_complete_punctuation_bounded_sentences() -> None:
    segments = (
        TranscriptSegment(0.0, 1.0, "One"),
        TranscriptSegment(1.0, 2.0, "sentence."),
        TranscriptSegment(2.1, 3.0, "Second"),
        TranscriptSegment(3.0, 4.0, "sentence."),
        TranscriptSegment(4.1, 5.0, "Third"),
        TranscriptSegment(5.0, 6.0, "sentence."),
    )

    chunks = _chunker(pause_boundary_seconds=5.0, hard_pause_seconds=6.0).chunk(
        segments,
        video_duration_seconds=10.0,
        language="en",
    )

    assert SemanticTranscriptChunker.version == "semantic-pause-punct-v1"
    assert [
        (chunk.start_seconds, chunk.end_seconds, chunk.text, chunk.language) for chunk in chunks
    ] == [
        (0.0, 4.0, "One sentence. Second sentence.", "en"),
        (2.1, 6.0, "Second sentence. Third sentence.", "en"),
    ]


def test_semantic_chunker_uses_pause_at_threshold_but_not_below_it() -> None:
    exact_threshold = (
        TranscriptSegment(0.0, 0.4, "first"),
        TranscriptSegment(0.4, 0.8, "thought"),
        TranscriptSegment(1.6, 2.0, "second"),
        TranscriptSegment(2.0, 2.4, "thought"),
    )
    below_threshold = (
        exact_threshold[0],
        exact_threshold[1],
        TranscriptSegment(1.599, 2.0, "second"),
        exact_threshold[3],
    )
    chunker = _chunker(
        target_duration_seconds=0.5,
        max_duration_seconds=5.0,
        max_sentence_duration_seconds=5.0,
        overlap_sentences=0,
    )

    exact_chunks = chunker.chunk(exact_threshold, video_duration_seconds=5.0)
    below_chunks = chunker.chunk(below_threshold, video_duration_seconds=5.0)

    assert [chunk.text for chunk in exact_chunks] == ["first thought", "second thought"]
    assert [chunk.text for chunk in below_chunks] == ["first thought second thought"]


def test_semantic_chunker_never_crosses_a_hard_pause_or_overlaps_across_it() -> None:
    chunks = _chunker(
        target_duration_seconds=8.0,
        max_duration_seconds=10.0,
        max_sentence_duration_seconds=10.0,
    ).chunk(
        (
            TranscriptSegment(0.0, 0.8, "Before."),
            TranscriptSegment(3.3, 4.0, "After."),
        ),
        video_duration_seconds=10.0,
    )

    assert [(chunk.start_seconds, chunk.end_seconds, chunk.text) for chunk in chunks] == [
        (0.0, 0.8, "Before."),
        (3.3, 4.0, "After."),
    ]


def test_semantic_chunker_enforces_word_caps_without_losing_tokens() -> None:
    segments = tuple(
        TranscriptSegment(position / 10, (position + 1) / 10, f"w{position + 1}")
        for position in range(10)
    )
    chunks = _chunker(
        target_duration_seconds=8.0,
        max_duration_seconds=10.0,
        target_words=4,
        max_words=5,
        max_sentence_duration_seconds=10.0,
        max_sentence_words=3,
        overlap_sentences=0,
    ).chunk(segments, video_duration_seconds=10.0)

    words_by_chunk = [chunk.text.split() for chunk in chunks]

    assert [len(words) for words in words_by_chunk] == [3, 3, 4]
    assert [word for words in words_by_chunk for word in words] == [
        f"w{position}" for position in range(1, 11)
    ]
    assert all(len(words) <= 5 for words in words_by_chunk)


def test_semantic_chunker_enforces_duration_cap_on_sentence_boundaries() -> None:
    chunks = _chunker(
        target_duration_seconds=5.0,
        max_duration_seconds=5.0,
        max_sentence_duration_seconds=3.0,
        pause_boundary_seconds=4.0,
        hard_pause_seconds=5.0,
        overlap_sentences=0,
    ).chunk(
        (
            TranscriptSegment(0.0, 2.0, "First."),
            TranscriptSegment(2.1, 4.1, "Second."),
            TranscriptSegment(4.2, 6.2, "Third."),
        ),
        video_duration_seconds=10.0,
    )

    assert [(chunk.start_seconds, chunk.end_seconds, chunk.text) for chunk in chunks] == [
        (0.0, 4.1, "First. Second."),
        (4.2, 6.2, "Third."),
    ]
    assert all(chunk.end_seconds - chunk.start_seconds <= 5.0 for chunk in chunks)


def test_semantic_chunker_absorbs_a_short_tail_only_when_the_union_fits() -> None:
    segments = (
        TranscriptSegment(0.0, 2.0, "A."),
        TranscriptSegment(2.1, 4.0, "B."),
        TranscriptSegment(4.1, 5.0, "C."),
    )
    chunker = _chunker(
        min_duration_seconds=2.0,
        max_duration_seconds=6.0,
        max_sentence_duration_seconds=6.0,
        pause_boundary_seconds=5.0,
        hard_pause_seconds=6.0,
    )

    chunks = chunker.chunk(segments, video_duration_seconds=10.0)

    assert [(chunk.start_seconds, chunk.end_seconds, chunk.text) for chunk in chunks] == [
        (0.0, 5.0, "A. B. C."),
    ]


def test_semantic_chunker_keeps_short_tail_when_union_exceeds_maximum() -> None:
    segments = (
        TranscriptSegment(0.0, 2.0, "A."),
        TranscriptSegment(2.1, 4.0, "B."),
        TranscriptSegment(6.5, 8.0, "C."),
    )
    chunks = _chunker(
        min_duration_seconds=2.0,
        max_duration_seconds=6.0,
        max_sentence_duration_seconds=6.0,
        pause_boundary_seconds=9.0,
        hard_pause_seconds=10.0,
    ).chunk(segments, video_duration_seconds=10.0)

    assert [(chunk.start_seconds, chunk.end_seconds, chunk.text) for chunk in chunks] == [
        (0.0, 4.0, "A. B."),
        (2.1, 8.0, "B. C."),
    ]


def test_semantic_chunker_interpolates_multiword_fallback_segments() -> None:
    chunks = _chunker(
        target_duration_seconds=20.0,
        min_duration_seconds=1.0,
        max_duration_seconds=20.0,
        target_words=2,
        max_words=2,
        max_sentence_duration_seconds=20.0,
        max_sentence_words=2,
        pause_boundary_seconds=15.0,
        hard_pause_seconds=16.0,
        overlap_sentences=0,
    ).chunk(
        (TranscriptSegment(0.0, 10.0, "one two three four five"),),
        video_duration_seconds=10.0,
    )

    assert [(chunk.start_seconds, chunk.end_seconds, chunk.text) for chunk in chunks] == [
        (0.0, 4.0, "one two"),
        (4.0, 8.0, "three four"),
        (8.0, 10.0, "five"),
    ]


def test_semantic_chunker_joins_standalone_punctuation() -> None:
    chunks = _chunker(target_duration_seconds=8.0).chunk(
        (
            TranscriptSegment(0.0, 0.3, "He"),
            TranscriptSegment(0.3, 0.6, "said"),
            TranscriptSegment(0.6, 0.7, ":"),
            TranscriptSegment(0.7, 0.8, "«"),
            TranscriptSegment(0.8, 1.1, "yes"),
            TranscriptSegment(1.1, 1.2, "!"),
            TranscriptSegment(1.2, 1.3, "»"),
        ),
        video_duration_seconds=5.0,
    )

    assert [chunk.text for chunk in chunks] == ["He said: «yes!»"]


def test_semantic_chunker_sorts_clamps_and_drops_unusable_segments() -> None:
    chunks = SemanticTranscriptChunker().chunk(
        (
            TranscriptSegment(29.0, 40.0, " ending "),
            TranscriptSegment(2.0, 3.0, "  "),
            TranscriptSegment(-2.0, 3.0, " opening words "),
            TranscriptSegment(40.0, 50.0, "outside"),
        ),
        video_duration_seconds=30.0,
        language="en",
    )

    assert [(chunk.start_seconds, chunk.end_seconds, chunk.text) for chunk in chunks] == [
        (0.0, 3.0, "opening words"),
        (29.0, 30.0, "ending"),
    ]
    assert all(chunk.language == "en" for chunk in chunks)


def test_semantic_chunker_keeps_short_speech_and_is_repeatable() -> None:
    segments = (
        TranscriptSegment(1.0, 2.0, "short."),
        TranscriptSegment(1.0, 2.0, "stable."),
    )
    chunker = SemanticTranscriptChunker()

    first = chunker.chunk(segments, video_duration_seconds=5.0)
    second = chunker.chunk(segments, video_duration_seconds=5.0)

    assert first == second
    assert [(chunk.start_seconds, chunk.end_seconds, chunk.text) for chunk in first] == [
        (1.0, 2.0, "short. stable."),
    ]


def test_semantic_chunker_keeps_one_oversized_raw_fallback_token() -> None:
    chunks = SemanticTranscriptChunker().chunk(
        (TranscriptSegment(0.0, 30.0, "[Music]"),),
        video_duration_seconds=30.0,
    )

    assert [(chunk.start_seconds, chunk.end_seconds, chunk.text) for chunk in chunks] == [
        (0.0, 30.0, "[Music]"),
    ]


def test_semantic_chunker_returns_empty_for_no_usable_transcript() -> None:
    assert (
        SemanticTranscriptChunker().chunk(
            (TranscriptSegment(1.0, 2.0, "   "),),
            video_duration_seconds=10.0,
        )
        == ()
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_duration_seconds": 0.0}, "minimum"),
        ({"target_duration_seconds": float("nan")}, "finite"),
        ({"target_words": 0}, "target words"),
        ({"max_sentence_words": 81}, "Sentence word maximum"),
        ({"pause_boundary_seconds": 3.0}, "sentence pause"),
        ({"overlap_sentences": -1}, "overlap"),
        ({"overlap_sentences": 1.5}, "integers"),
    ],
)
def test_semantic_chunker_rejects_invalid_configuration(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SemanticTranscriptChunker(**kwargs)


def test_semantic_chunker_rejects_nonfinite_video_or_transcript_timestamps() -> None:
    chunker = SemanticTranscriptChunker()

    with pytest.raises(ValueError, match="Video duration"):
        chunker.chunk((), video_duration_seconds=float("inf"))
    with pytest.raises(ValueError, match="timestamps"):
        chunker.chunk(
            (TranscriptSegment(float("nan"), 2.0, "bad"),),
            video_duration_seconds=10.0,
        )
