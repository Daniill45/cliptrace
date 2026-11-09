"""Run the real E5 adapter against a versioned offline retrieval dataset."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from app.core.config import get_settings
from app.services.ml.adapters import OnnxE5Embedder
from app.services.ml.evaluation import evaluate_retrieval, load_evaluation_dataset


@dataclass(frozen=True, slots=True)
class CliArguments:
    dataset: Path
    cutoffs: tuple[int, ...]
    model_cache_dir: Path | None


def _parse_arguments(argv: Sequence[str] | None = None) -> CliArguments:
    parser = argparse.ArgumentParser(
        description="Evaluate ClipTrace dense retrieval on a versioned JSON dataset.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the retrieval evaluation JSON file.",
    )
    parser.add_argument(
        "--k",
        dest="cutoffs",
        action="append",
        type=int,
        help="Evaluation cutoff; repeat for multiple values (default: 1, 3, 5).",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        help="Override ML_MODEL_CACHE_DIR for this evaluation run.",
    )
    namespace = parser.parse_args(argv)
    supplied_cutoffs = cast(list[int] | None, namespace.cutoffs)
    return CliArguments(
        dataset=cast(Path, namespace.dataset),
        cutoffs=tuple(supplied_cutoffs) if supplied_cutoffs is not None else (1, 3, 5),
        model_cache_dir=cast(Path | None, namespace.model_cache_dir),
    )


async def _evaluate(arguments: CliArguments) -> dict[str, object]:
    settings = get_settings()
    dataset = load_evaluation_dataset(arguments.dataset)
    embedder = OnnxE5Embedder(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        revision=settings.embedding_model_revision,
        cache_folder=arguments.model_cache_dir or settings.ml_model_cache_dir,
        batch_size=settings.embedding_batch_size,
    )
    report = await evaluate_retrieval(embedder, dataset, arguments.cutoffs)
    return report.to_dict()


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate retrieval and print a stable, machine-readable JSON report."""
    arguments = _parse_arguments(argv)
    report = asyncio.run(_evaluate(arguments))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
