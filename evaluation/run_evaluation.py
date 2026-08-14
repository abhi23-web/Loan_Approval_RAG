#!/usr/bin/env python3
"""Run the golden dataset against the current pipeline.

    python evaluation/run_evaluation.py
    python evaluation/run_evaluation.py --strategy recursive_1000_150 --top-k 8
    python evaluation/run_evaluation.py --judge          # model-scored faithfulness
    python evaluation/run_evaluation.py --repeat 5       # stronger consistency check

Writes a JSON result file and a Markdown summary to
``evaluation/experiment_results/`` and prints the headline numbers.

This is the regression gate. Run it before and after any change to chunking, the
embedding model, the retriever, top-k, the prompt or the LLM, and compare.
"""

from __future__ import annotations

import argparse
import sys

# Running this file directly puts its own directory on sys.path rather than the
# repository root, which would hide the "app" package. `pip install -e .` makes
# this redundant but never harmful; keeping it means a fresh clone runs with no
# install step at all.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from app.core.config import get_settings
from app.core.exceptions import EvaluationError, HomeLoanRagError
from app.core.logging_config import configure_logging, get_logger
from app.core.tracing import configure_tracing
from app.rag.llm import build_llm_provider
from app.services.container import get_container
from evaluation.dataset import GoldenDataset
from evaluation.report import render_run_summary, write_markdown
from evaluation.runner import EvaluationRunner, write_run

_logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the golden dataset evaluation")
    parser.add_argument("--strategy", default=None, help="chunking strategy to evaluate")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=None, help="executions per case")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="score faithfulness and relevancy with the configured LLM as judge",
    )
    parser.add_argument(
        "--skip-precondition",
        action="store_true",
        help="run even if the corpus is not in the state the dataset grades against",
    )
    arguments = parser.parse_args()

    load_dotenv()
    settings = get_settings()
    configure_logging(settings.app.log_level)
    configure_tracing(settings.observability.langsmith_project)

    container = get_container()
    dataset = GoldenDataset.load()

    # The judge is a separate provider instance so its calls are visibly distinct
    # from answer generation in LangSmith rather than blended into the same span.
    judge_provider = build_llm_provider(settings.llm) if arguments.judge else None

    runner = EvaluationRunner(container, dataset, judge_provider=judge_provider)
    try:
        evaluation_run = runner.run(
            strategy_name=arguments.strategy,
            top_k=arguments.top_k,
            repeat_runs=arguments.repeat,
            enforce_precondition=not arguments.skip_precondition,
        )
    except EvaluationError as evaluation_error:
        print(f"\nevaluation could not run:\n{evaluation_error}\n", file=sys.stderr)
        return 2
    except HomeLoanRagError as pipeline_error:
        print(f"\npipeline error: {pipeline_error}\n", file=sys.stderr)
        return 3

    results_directory = settings.paths.experiment_results_dir
    result_path = write_run(evaluation_run, results_directory)
    summary_path = write_markdown(
        render_run_summary(evaluation_run), results_directory / f"{evaluation_run.run_id}.md"
    )

    aggregate = evaluation_run.aggregate
    print("\nGolden dataset evaluation")
    print("-" * 62)
    print(f"strategy            : {evaluation_run.configuration['chunking_strategy']}")
    print(f"top_k               : {evaluation_run.configuration['top_k']}")
    print(f"cases x runs        : {aggregate.case_count} x "
          f"{evaluation_run.configuration['repeat_runs']}")
    print(f"correct answer rate : {aggregate.correct_answer_rate * 100:.0f}%")
    print(f"context precision   : {aggregate.context_precision:.3f}")
    print(f"context recall      : {aggregate.context_recall:.3f}")
    print(f"MRR                 : {aggregate.mrr:.3f}")
    print(f"version correctness : {aggregate.version_correct_rate * 100:.0f}%")
    print(f"hallucinated cites  : {aggregate.hallucination_rate * 100:.0f}%")
    print(f"mean total latency  : {aggregate.mean_total_latency_ms:.0f} ms")
    if aggregate.consistency:
        print(f"answer consistency  : {aggregate.consistency.answer_consistency * 100:.0f}%")
        print(f"retrieval consistency: {aggregate.consistency.retrieval_consistency * 100:.0f}%")
    print("-" * 62)
    for note in aggregate.notes:
        print(f"note: {note}")
    print(f"\nresults : {result_path}")
    print(f"summary : {summary_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
