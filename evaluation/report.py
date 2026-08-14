"""Rendering evaluation results as Markdown.

Reports are generated from result files, never typed by hand. That is the
mechanism behind the "no fabricated results" rule: if a number appears in a
table, a JSON file in ``evaluation/experiment_results/`` contains the execution
that produced it.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evaluation.runner import EvaluationRun

_NOT_MEASURED = "not measured"


def _format_metric(value: float | None, *, as_percentage: bool = False, digits: int = 3) -> str:
    if value is None:
        return _NOT_MEASURED
    if as_percentage:
        return f"{value * 100:.0f}%"
    return f"{value:.{digits}f}"


def render_run_summary(evaluation_run: EvaluationRun) -> str:
    """A single run, in full."""
    aggregate = evaluation_run.aggregate
    configuration = evaluation_run.configuration

    lines = [
        f"# Evaluation run `{evaluation_run.run_id}`",
        "",
        f"- Dataset: `{evaluation_run.dataset_id}` v{evaluation_run.dataset_version}",
        f"- Started: {evaluation_run.started_at}",
        f"- Cases: {aggregate.case_count}, executions: {aggregate.execution_count}",
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "| --- | --- |",
    ]
    lines.extend(f"| {key} | `{value}` |" for key, value in sorted(configuration.items()))

    lines += [
        "",
        "## Retrieval metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Context precision | {_format_metric(aggregate.context_precision)} |",
        f"| Context recall | {_format_metric(aggregate.context_recall)} |",
        f"| MRR | {_format_metric(aggregate.mrr)} |",
        f"| NDCG | {_format_metric(aggregate.ndcg)} |",
        f"| Mean retrieved chunks | {aggregate.mean_retrieved_chunks:.1f} |",
        f"| Mean context characters | {aggregate.mean_context_characters:.0f} |",
        "",
        "## Generation and business metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Correct answer rate | {_format_metric(aggregate.correct_answer_rate, as_percentage=True)} |",
        f"| Citation correctness | {_format_metric(aggregate.citation_correct_rate, as_percentage=True)} |",
        f"| Version correctness | {_format_metric(aggregate.version_correct_rate, as_percentage=True)} |",
        f"| Grounded rate | {_format_metric(aggregate.grounded_rate, as_percentage=True)} |",
        f"| Hallucinated-citation rate | {_format_metric(aggregate.hallucination_rate, as_percentage=True)} |",
        f"| Insufficient-information rate | {_format_metric(aggregate.insufficient_information_rate, as_percentage=True)} |",
        f"| Faithfulness (lexical proxy) | {_format_metric(aggregate.faithfulness_lexical)} |",
        f"| Faithfulness (LLM judge) | {_format_metric(aggregate.faithfulness_judge)} |",
        f"| Answer relevancy (lexical proxy) | {_format_metric(aggregate.answer_relevancy_lexical)} |",
        f"| Answer relevancy (LLM judge) | {_format_metric(aggregate.answer_relevancy_judge)} |",
        "",
        "## System metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Mean retrieval latency | {aggregate.mean_retrieval_latency_ms:.0f} ms |",
        f"| Mean end-to-end latency | {aggregate.mean_total_latency_ms:.0f} ms |",
        f"| p95 end-to-end latency | {aggregate.p95_total_latency_ms:.0f} ms |",
        f"| Mean prompt tokens | {_format_metric(aggregate.mean_prompt_tokens, digits=0)} |",
        f"| Mean completion tokens | {_format_metric(aggregate.mean_completion_tokens, digits=0)} |",
    ]

    if aggregate.consistency:
        consistency = aggregate.consistency
        lines += [
            "",
            "## Reproducibility across repeated runs",
            "",
            "| Measure | Share of cases identical across runs |",
            "| --- | --- |",
            f"| Answer text | {_format_metric(consistency.answer_consistency, as_percentage=True)} |",
            f"| Retrieved chunks and order | {_format_metric(consistency.retrieval_consistency, as_percentage=True)} |",
            f"| Citations | {_format_metric(consistency.citation_consistency, as_percentage=True)} |",
            f"| Cited document version | {_format_metric(consistency.version_consistency, as_percentage=True)} |",
        ]

    lines += ["", "## Per-case results", "", "| Case | Correct | Precision | Recall | MRR | Version | Latency |", "| --- | --- | --- | --- | --- | --- | --- |"]
    seen_cases: set[str] = set()
    for execution in evaluation_run.executions:
        if execution.case_id in seen_cases:
            continue  # first run of each case; repeats are summarised above
        seen_cases.add(execution.case_id)
        case_metrics = execution.metrics
        lines.append(
            f"| {execution.case_id} "
            f"| {'yes' if case_metrics.answer_correct else 'NO'} "
            f"| {case_metrics.context_precision:.2f} "
            f"| {case_metrics.context_recall:.2f} "
            f"| {case_metrics.reciprocal_rank:.2f} "
            f"| {'yes' if case_metrics.version_correct else 'NO'} "
            f"| {case_metrics.total_latency_ms:.0f} ms |"
        )

    if aggregate.notes:
        lines += ["", "## Caveats", ""]
        lines.extend(f"- {note}" for note in aggregate.notes)

    if evaluation_run.warnings:
        lines += ["", "## Precondition warnings", ""]
        lines.extend(f"- {warning}" for warning in evaluation_run.warnings)

    return "\n".join(lines) + "\n"


def render_comparison_table(
    runs: list[EvaluationRun], variable_name: str, variable_key: str
) -> str:
    """Side-by-side comparison of runs that differ in exactly one variable."""
    header = (
        f"| {variable_name} | Precision | Recall | MRR | NDCG | Correct | Version ok "
        "| Faithfulness | Retrieval ms | Total ms | Ctx chars | Prompt tokens |"
    )
    lines = [
        header,
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for evaluation_run in runs:
        aggregate = evaluation_run.aggregate
        faithfulness = (
            aggregate.faithfulness_judge
            if aggregate.faithfulness_judge is not None
            else aggregate.faithfulness_lexical
        )
        lines.append(
            f"| {evaluation_run.configuration.get(variable_key)} "
            f"| {_format_metric(aggregate.context_precision)} "
            f"| {_format_metric(aggregate.context_recall)} "
            f"| {_format_metric(aggregate.mrr)} "
            f"| {_format_metric(aggregate.ndcg)} "
            f"| {_format_metric(aggregate.correct_answer_rate, as_percentage=True)} "
            f"| {_format_metric(aggregate.version_correct_rate, as_percentage=True)} "
            f"| {_format_metric(faithfulness)} "
            f"| {aggregate.mean_retrieval_latency_ms:.0f} "
            f"| {aggregate.mean_total_latency_ms:.0f} "
            f"| {aggregate.mean_context_characters:.0f} "
            f"| {_format_metric(aggregate.mean_prompt_tokens, digits=0)} |"
        )
    return "\n".join(lines) + "\n"


def load_run(result_path: Path) -> dict[str, Any]:
    with result_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_markdown(content: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def aggregate_as_dict(evaluation_run: EvaluationRun) -> dict[str, Any]:
    return asdict(evaluation_run.aggregate)
