"""Offline evaluation utilities for dense transcript-fragment retrieval."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from app.services.ml.contracts import Embedding, TextEmbedder


class _Identified(Protocol):
    @property
    def id(self) -> str: ...


@dataclass(frozen=True, slots=True)
class EvaluationFragment:
    """One candidate transcript fragment in an evaluation corpus."""

    id: str
    text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required_text(self.id, "Fragment id"))
        object.__setattr__(self, "text", _required_text(self.text, "Fragment text"))


@dataclass(frozen=True, slots=True)
class EvaluationQuery:
    """A search query and its graded fragment relevance judgments."""

    id: str
    text: str
    relevance: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required_text(self.id, "Query id"))
        object.__setattr__(self, "text", _required_text(self.text, "Query text"))
        normalized_relevance: dict[str, int] = {}
        for fragment_id, grade in self.relevance.items():
            normalized_id = _required_text(fragment_id, "Relevance fragment id")
            if isinstance(grade, bool) or not isinstance(grade, int) or grade < 1 or grade > 3:
                raise ValueError("Relevance grades must be integers from 1 to 3")
            if normalized_id in normalized_relevance:
                raise ValueError(f"Duplicate relevance fragment id: {normalized_id}")
            normalized_relevance[normalized_id] = grade
        if not normalized_relevance:
            raise ValueError("Each evaluation query must have a relevance judgment")
        object.__setattr__(self, "relevance", normalized_relevance)


@dataclass(frozen=True, slots=True)
class EvaluationCorpus:
    """Fragments and queries belonging to one production search scope."""

    id: str
    fragments: tuple[EvaluationFragment, ...]
    queries: tuple[EvaluationQuery, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required_text(self.id, "Corpus id"))
        if not self.fragments:
            raise ValueError("Each evaluation corpus must contain fragments")
        if not self.queries:
            raise ValueError("Each evaluation corpus must contain queries")
        fragment_ids = _require_unique_ids(self.fragments, "fragment")
        _require_unique_ids(self.queries, "query")
        for query in self.queries:
            unknown = set(query.relevance) - fragment_ids
            if unknown:
                names = ", ".join(sorted(unknown))
                raise ValueError(
                    f"Query {query.id!r} references unknown fragments in corpus {self.id!r}: "
                    f"{names}"
                )


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    """Versioned collection of independent retrieval corpora."""

    version: int
    name: str
    corpora: tuple[EvaluationCorpus, ...]

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError("Only evaluation dataset version 1 is supported")
        object.__setattr__(self, "name", _required_text(self.name, "Dataset name"))
        if not self.corpora:
            raise ValueError("Evaluation dataset must contain corpora")
        _require_unique_ids(self.corpora, "corpus")


@dataclass(frozen=True, slots=True)
class RankedFragment:
    """One ranked candidate with its evaluation grade."""

    rank: int
    fragment_id: str
    score: float
    relevance: int

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "fragment_id": self.fragment_id,
            "score": self.score,
            "relevance": self.relevance,
        }


@dataclass(frozen=True, slots=True)
class QueryEvaluation:
    """Per-query diagnostics behind the aggregate retrieval metrics."""

    corpus_id: str
    query_id: str
    first_relevant_rank: int | None
    metrics: Mapping[str, float]
    ranking: tuple[RankedFragment, ...]

    @property
    def ranked_fragment_ids(self) -> tuple[str, ...]:
        """Expose the ranking compactly for diagnostics and regression tests."""
        return tuple(item.fragment_id for item in self.ranking)

    def to_dict(self) -> dict[str, object]:
        return {
            "corpus_id": self.corpus_id,
            "query_id": self.query_id,
            "first_relevant_rank": self.first_relevant_rank,
            "metrics": dict(self.metrics),
            "ranking": [item.to_dict() for item in self.ranking],
        }


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationReport:
    """Macro-averaged retrieval metrics plus inspectable query results."""

    dataset: str
    dataset_version: int
    model: str
    cutoffs: tuple[int, ...]
    corpora: int
    fragments: int
    queries: int
    metrics: Mapping[str, float]
    per_query: tuple[QueryEvaluation, ...]

    @property
    def model_id(self) -> str:
        return self.model

    @property
    def corpus_count(self) -> int:
        return self.corpora

    @property
    def fragment_count(self) -> int:
        return self.fragments

    @property
    def query_count(self) -> int:
        return self.queries

    @property
    def query_results(self) -> tuple[QueryEvaluation, ...]:
        return self.per_query

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "model": self.model,
            "cutoffs": list(self.cutoffs),
            "corpora": self.corpora,
            "fragments": self.fragments,
            "queries": self.queries,
            "metrics": dict(self.metrics),
            "per_query": [result.to_dict() for result in self.per_query],
        }


async def evaluate_retrieval(
    embedder: TextEmbedder,
    dataset: EvaluationDataset,
    cutoffs: Sequence[int] = (1, 3, 5),
) -> RetrievalEvaluationReport:
    """Evaluate dense retrieval independently within each video-like corpus."""
    normalized_cutoffs = _validate_cutoffs(cutoffs)
    if embedder.dimension <= 0:
        raise ValueError("Embedder dimension must be positive")

    query_results: list[QueryEvaluation] = []
    metric_totals = {
        f"{metric}@{cutoff}": 0.0
        for cutoff in normalized_cutoffs
        for metric in ("recall", "mrr", "ndcg")
    }
    for corpus in dataset.corpora:
        document_embeddings = await embedder.embed_documents(
            tuple(fragment.text for fragment in corpus.fragments)
        )
        if len(document_embeddings) != len(corpus.fragments):
            raise ValueError("Embedder returned an unexpected document count")
        validated_documents = tuple(
            _validate_embedding(
                embedding,
                expected_dimension=embedder.dimension,
                description=f"document embedding {fragment.id!r}",
            )
            for fragment, embedding in zip(corpus.fragments, document_embeddings, strict=True)
        )

        for query in corpus.queries:
            query_embedding = _validate_embedding(
                await embedder.embed_query(query.text),
                expected_dimension=embedder.dimension,
                description=f"query embedding {query.id!r}",
            )
            scored = [
                (
                    fragment,
                    _cosine_similarity(query_embedding, document_embedding),
                    position,
                )
                for position, (fragment, document_embedding) in enumerate(
                    zip(corpus.fragments, validated_documents, strict=True)
                )
            ]
            scored.sort(key=lambda item: (-item[1], item[2]))
            ranked = tuple(
                RankedFragment(
                    rank=rank,
                    fragment_id=fragment.id,
                    score=score,
                    relevance=query.relevance.get(fragment.id, 0),
                )
                for rank, (fragment, score, _) in enumerate(scored, start=1)
            )
            first_relevant_rank = next(
                (item.rank for item in ranked if item.relevance > 0),
                None,
            )
            query_metrics: dict[str, float] = {}
            for cutoff in normalized_cutoffs:
                query_metrics.update(_metrics_at_cutoff(ranked, query.relevance, cutoff))
            for name, value in query_metrics.items():
                metric_totals[name] += value
            query_results.append(
                QueryEvaluation(
                    corpus_id=corpus.id,
                    query_id=query.id,
                    first_relevant_rank=first_relevant_rank,
                    metrics=query_metrics,
                    ranking=ranked[: max(normalized_cutoffs)],
                )
            )

    query_count = len(query_results)
    if query_count == 0:
        raise ValueError("Evaluation dataset must contain queries")
    aggregate_metrics = {name: total / query_count for name, total in metric_totals.items()}
    return RetrievalEvaluationReport(
        dataset=dataset.name,
        dataset_version=dataset.version,
        model=embedder.model_id,
        cutoffs=normalized_cutoffs,
        corpora=len(dataset.corpora),
        fragments=sum(len(corpus.fragments) for corpus in dataset.corpora),
        queries=query_count,
        metrics=aggregate_metrics,
        per_query=tuple(query_results),
    )


def load_evaluation_dataset(path: Path) -> EvaluationDataset:
    """Load and validate a versioned JSON retrieval dataset."""
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid evaluation JSON: {exc.msg}") from exc
    return parse_evaluation_dataset(raw)


def parse_evaluation_dataset(raw: object) -> EvaluationDataset:
    """Parse the public JSON dataset contract without accepting silent typos."""
    root = _mapping(raw, "dataset")
    _require_keys(root, {"version", "name", "corpora"}, "dataset")
    version = root["version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("Dataset version must be an integer")
    name = _string(root["name"], "Dataset name")
    corpora = tuple(
        _parse_corpus(value, position)
        for position, value in enumerate(_list(root["corpora"], "Dataset corpora"))
    )
    return EvaluationDataset(version=version, name=name, corpora=corpora)


def _parse_corpus(raw: object, position: int) -> EvaluationCorpus:
    context = f"corpus[{position}]"
    value = _mapping(raw, context)
    _require_keys(value, {"id", "fragments", "queries"}, context)
    fragments = tuple(
        _parse_fragment(item, item_position, context)
        for item_position, item in enumerate(_list(value["fragments"], f"{context}.fragments"))
    )
    queries = tuple(
        _parse_query(item, item_position, context)
        for item_position, item in enumerate(_list(value["queries"], f"{context}.queries"))
    )
    return EvaluationCorpus(
        id=_string(value["id"], f"{context}.id"),
        fragments=fragments,
        queries=queries,
    )


def _parse_fragment(raw: object, position: int, corpus_context: str) -> EvaluationFragment:
    context = f"{corpus_context}.fragments[{position}]"
    value = _mapping(raw, context)
    _require_keys(value, {"id", "text"}, context)
    return EvaluationFragment(
        id=_string(value["id"], f"{context}.id"),
        text=_string(value["text"], f"{context}.text"),
    )


def _parse_query(raw: object, position: int, corpus_context: str) -> EvaluationQuery:
    context = f"{corpus_context}.queries[{position}]"
    value = _mapping(raw, context)
    _require_keys(value, {"id", "text", "relevance"}, context)
    raw_relevance = _mapping(value["relevance"], f"{context}.relevance")
    relevance: dict[str, int] = {}
    for fragment_id, grade in raw_relevance.items():
        if isinstance(grade, bool) or not isinstance(grade, int):
            raise ValueError(f"{context}.relevance grades must be integers")
        relevance[fragment_id] = grade
    return EvaluationQuery(
        id=_string(value["id"], f"{context}.id"),
        text=_string(value["text"], f"{context}.text"),
        relevance=relevance,
    )


def _metrics_at_cutoff(
    ranked: Sequence[RankedFragment],
    relevance: Mapping[str, int],
    cutoff: int,
) -> dict[str, float]:
    top = ranked[:cutoff]
    recalled = sum(item.relevance > 0 for item in top) / len(relevance)
    first_rank = next((item.rank for item in top if item.relevance > 0), None)
    reciprocal_rank = 0.0 if first_rank is None else 1.0 / first_rank
    dcg = sum(
        (2.0**item.relevance - 1.0) / math.log2(item.rank + 1.0)
        for item in top
        if item.relevance > 0
    )
    ideal_grades = sorted(relevance.values(), reverse=True)[:cutoff]
    ideal_dcg = sum(
        (2.0**grade - 1.0) / math.log2(rank + 1.0)
        for rank, grade in enumerate(ideal_grades, start=1)
    )
    ndcg = 0.0 if ideal_dcg == 0 else dcg / ideal_dcg
    return {
        f"recall@{cutoff}": recalled,
        f"mrr@{cutoff}": reciprocal_rank,
        f"ndcg@{cutoff}": ndcg,
    }


def _validate_cutoffs(cutoffs: Sequence[int]) -> tuple[int, ...]:
    if not cutoffs:
        raise ValueError("At least one evaluation cutoff is required")
    normalized: list[int] = []
    for cutoff in cutoffs:
        if isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff <= 0:
            raise ValueError("Evaluation cutoffs must be positive integers")
        if cutoff in normalized:
            raise ValueError("Evaluation cutoffs must be unique")
        normalized.append(cutoff)
    return tuple(sorted(normalized))


def _validate_embedding(
    embedding: Sequence[float],
    *,
    expected_dimension: int,
    description: str,
) -> Embedding:
    if len(embedding) != expected_dimension:
        raise ValueError(f"Unexpected dimension for {description}")
    normalized = tuple(float(value) for value in embedding)
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError(f"Non-finite value in {description}")
    if math.fsum(value * value for value in normalized) <= 0:
        raise ValueError(f"Zero norm for {description}")
    return normalized


def _cosine_similarity(left: Embedding, right: Embedding) -> float:
    dot_product = math.fsum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    score = dot_product / (left_norm * right_norm)
    return max(-1.0, min(1.0, score))


def _required_text(value: str, description: str) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{description} must not be blank")
    return normalized


def _require_unique_ids(values: Sequence[_Identified], description: str) -> set[str]:
    seen: set[str] = set()
    for value in values:
        item_id = value.id
        if item_id in seen:
            raise ValueError(f"Duplicate {description} id: {item_id}")
        seen.add(item_id)
    return seen


def _mapping(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{description} must be a JSON object with string keys")
    return cast(dict[str, object], value)


def _list(value: object, description: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{description} must be a JSON array")
    return cast(list[object], value)


def _string(value: object, description: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{description} must be a string")
    return value


def _require_keys(value: Mapping[str, object], expected: set[str], description: str) -> None:
    actual = set(value)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise ValueError(f"Missing keys in {description}: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"Unknown keys in {description}: {', '.join(sorted(unknown))}")
