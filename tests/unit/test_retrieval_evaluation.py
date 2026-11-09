from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from app.services.ml.contracts import Embedding
from app.services.ml.evaluation import (
    EvaluationCorpus,
    EvaluationDataset,
    EvaluationFragment,
    EvaluationQuery,
    evaluate_retrieval,
    load_evaluation_dataset,
    parse_evaluation_dataset,
)


class FakeTextEmbedder:
    model_id = "fake-e5-v1"

    def __init__(
        self,
        embeddings: Mapping[str, Embedding],
        *,
        dimension: int = 2,
    ) -> None:
        self.dimension = dimension
        self._embeddings = embeddings
        self.document_batches: list[tuple[str, ...]] = []
        self.queries: list[str] = []

    async def embed_documents(self, texts: Sequence[str]) -> Sequence[Embedding]:
        batch = tuple(texts)
        self.document_batches.append(batch)
        return tuple(self._embeddings[text] for text in batch)

    async def embed_query(self, text: str) -> Embedding:
        self.queries.append(text)
        return self._embeddings[text]


def _dataset(
    *,
    fragments: tuple[EvaluationFragment, ...] | None = None,
    queries: tuple[EvaluationQuery, ...] | None = None,
) -> EvaluationDataset:
    return EvaluationDataset(
        version=1,
        name="unit-eval",
        corpora=(
            EvaluationCorpus(
                id="video-1",
                fragments=fragments
                or (
                    EvaluationFragment(id="a", text="fragment a"),
                    EvaluationFragment(id="b", text="fragment b"),
                    EvaluationFragment(id="c", text="fragment c"),
                ),
                queries=queries
                or (
                    EvaluationQuery(
                        id="q1",
                        text="query one",
                        relevance={"b": 3, "c": 1},
                    ),
                ),
            ),
        ),
    )


@pytest.mark.asyncio
async def test_evaluate_retrieval_calculates_recall_mrr_and_graded_ndcg() -> None:
    dataset = _dataset()
    embedder = FakeTextEmbedder(
        {
            "fragment a": (1.0, 0.0),
            "fragment b": (0.8, 0.6),
            "fragment c": (0.0, 1.0),
            "query one": (1.0, 0.0),
        }
    )

    report = await evaluate_retrieval(embedder, dataset, cutoffs=(1, 2, 3))

    ideal_at_two = 7.0 + 1.0 / math.log2(3.0)
    actual_at_two = 7.0 / math.log2(3.0)
    ideal_at_three = ideal_at_two
    actual_at_three = actual_at_two + 1.0 / math.log2(4.0)
    assert report.metrics == pytest.approx(
        {
            "recall@1": 0.0,
            "mrr@1": 0.0,
            "ndcg@1": 0.0,
            "recall@2": 0.5,
            "mrr@2": 0.5,
            "ndcg@2": actual_at_two / ideal_at_two,
            "recall@3": 1.0,
            "mrr@3": 0.5,
            "ndcg@3": actual_at_three / ideal_at_three,
        }
    )
    assert report.model_id == "fake-e5-v1"
    assert report.corpus_count == 1
    assert report.fragment_count == 3
    assert report.query_count == 1
    assert report.query_results[0].ranked_fragment_ids == ("a", "b", "c")
    assert report.query_results[0].first_relevant_rank == 2
    assert embedder.document_batches == [("fragment a", "fragment b", "fragment c")]
    assert embedder.queries == ["query one"]


@pytest.mark.asyncio
async def test_evaluate_retrieval_macro_averages_queries() -> None:
    dataset = _dataset(
        queries=(
            EvaluationQuery(id="q1", text="query one", relevance={"a": 1}),
            EvaluationQuery(id="q2", text="query two", relevance={"c": 1}),
        )
    )
    embedder = FakeTextEmbedder(
        {
            "fragment a": (1.0, 0.0),
            "fragment b": (0.8, 0.6),
            "fragment c": (0.0, 1.0),
            "query one": (1.0, 0.0),
            "query two": (1.0, 0.0),
        }
    )

    report = await evaluate_retrieval(embedder, dataset, cutoffs=(1, 3))

    assert report.metrics["recall@1"] == pytest.approx(0.5)
    assert report.metrics["mrr@1"] == pytest.approx(0.5)
    assert report.metrics["ndcg@1"] == pytest.approx(0.5)
    assert report.metrics["recall@3"] == pytest.approx(1.0)
    assert report.metrics["mrr@3"] == pytest.approx((1.0 + 1.0 / 3.0) / 2.0)


@pytest.mark.asyncio
async def test_evaluate_retrieval_preserves_fragment_order_when_scores_tie() -> None:
    dataset = _dataset(
        fragments=(
            EvaluationFragment(id="first", text="first text"),
            EvaluationFragment(id="second", text="second text"),
        ),
        queries=(EvaluationQuery(id="q1", text="query", relevance={"second": 1}),),
    )
    embedder = FakeTextEmbedder(
        {
            "first text": (1.0, 0.0),
            "second text": (1.0, 0.0),
            "query": (1.0, 0.0),
        }
    )

    report = await evaluate_retrieval(embedder, dataset, cutoffs=(2,))

    assert report.query_results[0].ranked_fragment_ids == ("first", "second")
    assert report.query_results[0].first_relevant_rank == 2


def test_evaluation_dataset_rejects_duplicate_fragment_ids() -> None:
    with pytest.raises(ValueError, match="Duplicate fragment id"):
        _dataset(
            fragments=(
                EvaluationFragment(id="duplicate", text="one"),
                EvaluationFragment(id="duplicate", text="two"),
            )
        )


def test_evaluation_dataset_rejects_duplicate_query_ids() -> None:
    with pytest.raises(ValueError, match="Duplicate query id"):
        _dataset(
            queries=(
                EvaluationQuery(id="duplicate", text="one", relevance={"a": 1}),
                EvaluationQuery(id="duplicate", text="two", relevance={"a": 1}),
            )
        )


def test_evaluation_dataset_rejects_unknown_relevance_fragment() -> None:
    with pytest.raises(ValueError, match="unknown fragments"):
        _dataset(queries=(EvaluationQuery(id="q1", text="query", relevance={"missing": 1}),))


def test_evaluation_fragment_rejects_blank_text() -> None:
    with pytest.raises(ValueError, match="Fragment text"):
        EvaluationFragment(id="blank", text="   ")


@pytest.mark.parametrize("grade", [0, 4, 1024])
def test_evaluation_query_rejects_grades_outside_documented_scale(grade: int) -> None:
    with pytest.raises(ValueError, match="1 to 3"):
        EvaluationQuery(id="query", text="text", relevance={"fragment": grade})


def test_repository_smoke_dataset_matches_the_versioned_contract() -> None:
    repository_root = Path(__file__).parents[2]

    dataset = load_evaluation_dataset(repository_root / "eval" / "retrieval_smoke.json")

    assert dataset.version == 1
    assert dataset.name == "cliptrace-retrieval-smoke-v1"
    assert len(dataset.corpora) == 2
    assert sum(len(corpus.fragments) for corpus in dataset.corpora) == 6
    assert sum(len(corpus.queries) for corpus in dataset.corpora) == 4


def test_dataset_parser_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="Unknown keys"):
        parse_evaluation_dataset(
            {
                "version": 1,
                "name": "bad-contract",
                "corpora": [],
                "unexpected": True,
            }
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("cutoffs", [(), (0,), (-1,), (1, 1)])
async def test_evaluate_retrieval_rejects_invalid_cutoffs(cutoffs: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="cutoff"):
        await evaluate_retrieval(FakeTextEmbedder({}), _dataset(), cutoffs=cutoffs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("document_embedding", "query_embedding", "message"),
    [
        ((1.0,), (1.0, 0.0), "dimension"),
        ((math.nan, 0.0), (1.0, 0.0), "finite"),
        ((0.0, 0.0), (1.0, 0.0), "(?i)zero norm"),
        ((1.0, 0.0), (math.inf, 0.0), "finite"),
        ((1.0, 0.0), (0.0, 0.0), "(?i)zero norm"),
    ],
)
async def test_evaluate_retrieval_rejects_invalid_embeddings(
    document_embedding: Embedding,
    query_embedding: Embedding,
    message: str,
) -> None:
    dataset = _dataset(
        fragments=(EvaluationFragment(id="fragment", text="document"),),
        queries=(EvaluationQuery(id="query", text="query", relevance={"fragment": 1}),),
    )
    embedder = FakeTextEmbedder(
        {"document": document_embedding, "query": query_embedding},
        dimension=2,
    )

    with pytest.raises(ValueError, match=message):
        await evaluate_retrieval(embedder, dataset, cutoffs=(1,))
