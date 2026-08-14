#!/usr/bin/env python3
"""Controlled experiments: chunking strategy and top-k sweeps.

The discipline these scripts enforce is one variable at a time. Everything else
— the corpus, the version state, the embedding model, the prompt, the seed — is
held constant across the runs in a sweep, and the configuration fingerprint is
written into each result file so that can be verified afterwards rather than
taken on trust.

    python evaluation/experiments.py chunking
    python evaluation/experiments.py top-k --values 3 5 8 10
    python evaluation/experiments.py chunking --strategies recursive_800_100 semantic

Chunking sweeps require every strategy to have been indexed first:

    python scripts/ingest.py --all-strategies
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

# Running this file directly puts its own directory on sys.path rather than the
# repository root, which would hide the "app" package. `pip install -e .` makes
# this redundant but never harmful; keeping it means a fresh clone runs with no
# install step at all.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from app.core.config import get_chunking_config, get_settings
from app.core.exceptions import EvaluationError, HomeLoanRagError
from app.core.logging_config import configure_logging, get_logger
from app.core.tracing import configure_tracing
from app.services.container import get_container
from evaluation.dataset import GoldenDataset
from evaluation.report import render_comparison_table, write_markdown
from evaluation.runner import EvaluationRun, EvaluationRunner, write_run

_logger = get_logger(__name__)


def _run_sweep(
    runner: EvaluationRunner,
    *,
    strategies: list[str] | None,
    top_k_values: list[int] | None,
    repeat_runs: int | None,
) -> list[EvaluationRun]:
    """Execute one run per value of the swept variable."""
    settings = get_settings()
    results_directory = settings.paths.experiment_results_dir
    runs: list[EvaluationRun] = []

    variable_values: list[tuple[str | None, int | None]]
    if strategies is not None:
        variable_values = [(strategy, None) for strategy in strategies]
    else:
        variable_values = [(None, top_k) for top_k in (top_k_values or [])]

    for strategy_name, top_k in variable_values:
        label = strategy_name or f"top_k={top_k}"
        print(f"\n=== running: {label} ===")
        try:
            evaluation_run = runner.run(
                strategy_name=strategy_name, top_k=top_k, repeat_runs=repeat_runs
            )
        except HomeLoanRagError as run_error:
            # A strategy that was never indexed should not abort the whole sweep;
            # the other strategies still produce a valid comparison.
            print(f"  skipped ({run_error})", file=sys.stderr)
            continue
        write_run(evaluation_run, results_directory)
        runs.append(evaluation_run)
        aggregate = evaluation_run.aggregate
        print(
            f"  precision={aggregate.context_precision:.3f} "
            f"recall={aggregate.context_recall:.3f} "
            f"mrr={aggregate.mrr:.3f} "
            f"correct={aggregate.correct_answer_rate * 100:.0f}% "
            f"latency={aggregate.mean_total_latency_ms:.0f}ms"
        )
    return runs


def _write_comparison(
    runs: list[EvaluationRun], variable_name: str, variable_key: str, output_path: Path
) -> None:
    if not runs:
        print("no runs completed; nothing to compare", file=sys.stderr)
        return

    generated_at = datetime.now(UTC).isoformat()
    reference_configuration = runs[0].configuration
    held_constant = {
        key: value
        for key, value in sorted(reference_configuration.items())
        if key != variable_key
    }

    content = "\n".join(
        [
            f"# {variable_name} comparison",
            "",
            f"Generated {generated_at} from {len(runs)} evaluation run(s) in "
            "`evaluation/experiment_results/`.",
            "",
            "## Held constant",
            "",
            "| Setting | Value |",
            "| --- | --- |",
            *(f"| {key} | `{value}` |" for key, value in held_constant.items()),
            "",
            "## Results",
            "",
            render_comparison_table(runs, variable_name, variable_key),
            "",
            "Faithfulness is the LLM-judge score when the sweep was run with "
            "`--judge`, and the lexical proxy otherwise.",
            "",
        ]
    )
    write_markdown(content, output_path)
    print(f"\ncomparison written to {output_path}\n")
    print(render_comparison_table(runs, variable_name, variable_key))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run controlled RAG experiments")
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    chunking_parser = subparsers.add_parser("chunking", help="compare chunking strategies")
    chunking_parser.add_argument("--strategies", nargs="*", default=None)
    chunking_parser.add_argument("--repeat", type=int, default=1)

    top_k_parser = subparsers.add_parser("top-k", help="compare retrieval depth")
    top_k_parser.add_argument("--values", nargs="*", type=int, default=[3, 5, 8, 10])
    top_k_parser.add_argument("--repeat", type=int, default=1)

    arguments = parser.parse_args()

    load_dotenv()
    settings = get_settings()
    configure_logging(settings.app.log_level)
    configure_tracing(settings.observability.langsmith_project)

    runner = EvaluationRunner(get_container(), GoldenDataset.load())
    results_directory = settings.paths.experiment_results_dir

    try:
        if arguments.experiment == "chunking":
            strategies = arguments.strategies or sorted(get_chunking_config().strategies)
            runs = _run_sweep(
                runner, strategies=strategies, top_k_values=None, repeat_runs=arguments.repeat
            )
            _write_comparison(
                runs,
                "Chunking strategy",
                "chunking_strategy",
                results_directory / "comparison_chunking.md",
            )
        else:
            runs = _run_sweep(
                runner, strategies=None, top_k_values=arguments.values, repeat_runs=arguments.repeat
            )
            _write_comparison(
                runs, "top_k", "top_k", results_directory / "comparison_top_k.md"
            )
    except EvaluationError as evaluation_error:
        print(f"\nexperiment could not run:\n{evaluation_error}\n", file=sys.stderr)
        return 2

    return 0 if runs else 1


if __name__ == "__main__":
    sys.exit(main())
