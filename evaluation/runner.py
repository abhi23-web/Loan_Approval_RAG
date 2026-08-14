"""The evaluation harness.

Runs the golden dataset through the *same* pipeline the API uses, scores each
execution, measures reproducibility across repeats, and writes a machine-readable
result file that experiments and the README table are generated from.

Two rules this module exists to enforce:

* **No fabricated numbers.** Every figure in a result file came from an execution
  recorded in that same file. Nothing is estimated, and a metric that could not
  be computed is written as ``null``, never as a plausible default.
* **Precondition checking.** The dataset states which policy version must be
  active for its expected answers to be correct. If the corpus is in a different
  state the run stops with an explanation instead of producing a page of red
  that only means "you forgot to ingest".
"""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.exceptions import EvaluationError
from app.core.logging_config import get_logger
from app.core.tracing import add_trace_metadata, traced, tracing_enabled
from app.models.documents import RetrievedChunk
from app.rag.llm import LLMProvider
from app.rag.pipeline import QuestionAnswer
from app.rag.prompts import PROMPT_VERSION
from app.services.container import ApplicationContainer
from app.utils.json_store import write_json_atomic
from evaluation import metrics as metric_functions
from evaluation.dataset import GoldenCase, GoldenDataset
from evaluation.metrics import AggregateMetrics, CaseMetrics, ConsistencyMetrics

_logger = get_logger(__name__)


@dataclass
class CaseExecution:
    """One execution of one case: what was asked, what came back, how it scored."""

    case_id: str
    run_index: int
    question: str
    answer: str
    retrieved_chunk_ids: list[str]
    cited_markers: list[str]
    cited_versions: list[int]
    metrics: CaseMetrics


@dataclass
class EvaluationRun:
    run_id: str
    started_at: str
    finished_at: str
    dataset_id: str
    dataset_version: str
    configuration: dict[str, Any]
    environment: dict[str, str]
    aggregate: AggregateMetrics
    executions: list[CaseExecution] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "configuration": self.configuration,
            "environment": self.environment,
            "aggregate": asdict(self.aggregate),
            "warnings": self.warnings,
            "executions": [asdict(execution) for execution in self.executions],
        }


class EvaluationRunner:
    """Executes the golden dataset and scores it."""

    def __init__(
        self,
        container: ApplicationContainer,
        dataset: GoldenDataset,
        *,
        judge_provider: LLMProvider | None = None,
    ) -> None:
        self._container = container
        self._dataset = dataset
        # The judge is opt-in: it costs one model call per scored dimension per
        # execution, which triples the cost of a 30-execution run.
        self._judge = judge_provider

    # ------------------------------------------------------------ preconditions

    def check_precondition(self) -> list[str]:
        """Verify the corpus is in the state the expected answers assume."""
        version_store = self._container.fresh_version_store()
        problems: list[str] = []

        for source_name, required_version in (
            self._dataset.grading_precondition.required_active_version.items()
        ):
            active_version = version_store.active_version(source_name)
            if active_version is None:
                problems.append(
                    f"'{source_name}' has no ingested version; expected version "
                    f"{required_version} to be active"
                )
            elif active_version.version_number != required_version:
                problems.append(
                    f"'{source_name}' has version {active_version.version_number} active, "
                    f"but the dataset is graded against version {required_version}"
                )

        # Historical cases need the superseded version to still exist on disk.
        for case in self._dataset.cases:
            for source_name, version_number in (case.version_override or {}).items():
                if version_store.find_version(source_name, version_number) is None:
                    problems.append(
                        f"case {case.case_id} needs version {version_number} of "
                        f"'{source_name}', which has never been ingested"
                    )
        return problems

    # -------------------------------------------------------------------- run

    def run(
        self,
        *,
        strategy_name: str | None = None,
        top_k: int | None = None,
        repeat_runs: int | None = None,
        enforce_precondition: bool = True,
    ) -> EvaluationRun:
        problems = self.check_precondition()
        if problems and enforce_precondition:
            instructions = "\n  ".join(self._dataset.grading_precondition.how_to_reach_it)
            raise EvaluationError(
                "the corpus is not in the state this dataset grades against:\n  - "
                + "\n  - ".join(problems)
                + f"\n\nBring it to the expected state with:\n  {instructions}"
            )

        settings = self._container.settings
        effective_strategy = strategy_name or settings.retrieval.active_strategy
        effective_top_k = top_k or settings.retrieval.top_k
        effective_repeats = repeat_runs or self._dataset.reproducibility.repeat_runs

        started_at = datetime.now(UTC)
        run_id = (
            f"{started_at.strftime('%Y%m%dT%H%M%SZ')}"
            f"__{effective_strategy}__k{effective_top_k}"
        )
        _logger.info(
            "evaluation %s starting: %d case(s) x %d run(s)",
            run_id,
            len(self._dataset.cases),
            effective_repeats,
        )

        executions: list[CaseExecution] = []
        for case in self._dataset.cases:
            executions.extend(
                self._execute_case(case, effective_strategy, effective_top_k, effective_repeats)
            )

        consistency = self._measure_consistency(executions)
        aggregate = metric_functions.aggregate(
            [execution.metrics for execution in executions], consistency
        )
        aggregate.notes = self._build_notes(settings, effective_repeats)

        finished_at = datetime.now(UTC)
        evaluation_run = EvaluationRun(
            run_id=run_id,
            started_at=started_at.isoformat(),
            finished_at=finished_at.isoformat(),
            dataset_id=self._dataset.dataset_id,
            dataset_version=self._dataset.dataset_version,
            configuration={
                **settings.experiment_fingerprint(),
                "chunking_strategy": effective_strategy,
                "top_k": effective_top_k,
                "repeat_runs": effective_repeats,
                "prompt_version": PROMPT_VERSION,
                "judge_enabled": self._judge is not None,
            },
            environment={
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "langsmith_tracing": str(tracing_enabled()),
            },
            aggregate=aggregate,
            executions=executions,
            warnings=problems,
        )
        _logger.info(
            "evaluation %s finished in %.1fs: correct=%.0f%% recall=%.2f mrr=%.2f",
            run_id,
            (finished_at - started_at).total_seconds(),
            aggregate.correct_answer_rate * 100,
            aggregate.context_recall,
            aggregate.mrr,
        )
        return evaluation_run

    @traced("evaluation.case", run_type="chain")
    def _execute_case(
        self, case: GoldenCase, strategy_name: str, top_k: int, repeat_runs: int
    ) -> list[CaseExecution]:
        executions: list[CaseExecution] = []
        for run_index in range(1, repeat_runs + 1):
            answer = self._container.rag_pipeline.answer_question(
                case.question,
                strategy_name=strategy_name,
                top_k=top_k,
                version_numbers_by_source=case.version_override,
                # A historical lookup must be allowed to leave the active version.
                restrict_to_active_versions=None if case.version_override is None else False,
            )
            executions.append(self._score(case, answer, run_index))

        add_trace_metadata(
            case_id=case.case_id,
            expected_version=case.expected_document_version,
            correct=all(execution.metrics.answer_correct for execution in executions),
        )
        return executions

    def _score(self, case: GoldenCase, answer: QuestionAnswer, run_index: int) -> CaseExecution:
        chunks: list[RetrievedChunk] = answer.retrieved_chunks
        relevance_flags = [
            metric_functions.is_chunk_relevant(
                chunk,
                case.expected_source,
                case.expected_document_version,
                case.expected_context_keywords,
            )
            for chunk in chunks
        ]
        answer_text = answer.explanation.explanation

        faithfulness_judge_score = (
            metric_functions.judge_faithfulness(answer_text, chunks, self._judge)
            if self._judge
            else None
        )
        relevancy_judge_score = (
            metric_functions.judge_answer_relevancy(answer_text, case.question, self._judge)
            if self._judge
            else None
        )

        case_metrics = CaseMetrics(
            case_id=case.case_id,
            answer_correct=metric_functions.answer_matches_expectation(
                answer_text,
                case.expected_answer_patterns,
                case.forbidden_answer_patterns,
                case.acceptable_answer_variations,
            ),
            context_precision=metric_functions.context_precision(relevance_flags),
            context_recall=metric_functions.context_recall(
                chunks, case.expected_context_keywords
            ),
            reciprocal_rank=metric_functions.reciprocal_rank(relevance_flags),
            ndcg=metric_functions.normalised_discounted_cumulative_gain(relevance_flags),
            faithfulness_lexical=metric_functions.lexical_faithfulness(answer_text, chunks),
            faithfulness_judge=faithfulness_judge_score,
            answer_relevancy_lexical=metric_functions.lexical_answer_relevancy(
                answer_text, case.question
            ),
            answer_relevancy_judge=relevancy_judge_score,
            citation_correct=metric_functions.citation_correct(
                answer.explanation.citations, case.expected_source
            ),
            version_correct=metric_functions.version_correct(
                answer.explanation.citations,
                case.expected_source,
                case.expected_document_version,
            ),
            is_grounded=answer.explanation.is_grounded,
            insufficient_information=answer.explanation.insufficient_information,
            retrieval_latency_ms=answer.retrieval.retrieval_latency_ms,
            total_latency_ms=answer.total_latency_ms,
            retrieved_chunk_count=len(chunks),
            context_characters=answer.retrieval.context_characters,
            prompt_tokens=answer.explanation.prompt_tokens,
            completion_tokens=answer.explanation.completion_tokens,
        )

        return CaseExecution(
            case_id=case.case_id,
            run_index=run_index,
            question=case.question,
            answer=answer_text,
            retrieved_chunk_ids=[chunk.metadata.chunk_id for chunk in chunks],
            cited_markers=[citation.marker for citation in answer.explanation.citations],
            cited_versions=sorted(
                {citation.version_number for citation in answer.explanation.citations}
            ),
            metrics=case_metrics,
        )

    # ------------------------------------------------------------ consistency

    @staticmethod
    def _measure_consistency(executions: list[CaseExecution]) -> ConsistencyMetrics | None:
        """Share of cases whose repeated executions agreed with each other.

        Retrieval consistency is compared on the *ordered* chunk id list, not the
        set: two runs that retrieve the same chunks in a different order can
        still produce different answers, so order is part of the guarantee.
        """
        executions_by_case: dict[str, list[CaseExecution]] = {}
        for execution in executions:
            executions_by_case.setdefault(execution.case_id, []).append(execution)

        repeated_cases = [
            case_executions
            for case_executions in executions_by_case.values()
            if len(case_executions) > 1
        ]
        if not repeated_cases:
            return None

        def _share_identical(extractor) -> float:  # noqa: ANN001 - local helper
            identical = sum(
                1
                for case_executions in repeated_cases
                if len({extractor(execution) for execution in case_executions}) == 1
            )
            return round(identical / len(repeated_cases), 4)

        return ConsistencyMetrics(
            answer_consistency=_share_identical(lambda execution: execution.answer.strip()),
            retrieval_consistency=_share_identical(
                lambda execution: tuple(execution.retrieved_chunk_ids)
            ),
            citation_consistency=_share_identical(
                lambda execution: tuple(execution.cited_markers)
            ),
            version_consistency=_share_identical(
                lambda execution: tuple(execution.cited_versions)
            ),
        )

    def _build_notes(self, settings: Settings, repeat_runs: int) -> list[str]:
        """Caveats that must travel with the numbers, not be discovered later."""
        notes: list[str] = []
        if settings.llm.provider == "deterministic":
            notes.append(
                "LLM provider was the offline stub: generation metrics "
                "(faithfulness, relevancy, correctness) are not meaningful."
            )
        if settings.embeddings.provider == "deterministic":
            notes.append(
                "Embedding provider was the offline hashing test double: retrieval "
                "metrics reflect lexical overlap, not semantic similarity."
            )
        if self._judge is None:
            notes.append(
                "Faithfulness and relevancy are lexical proxies; run with --judge "
                "for model-scored versions."
            )
        if repeat_runs < 2:
            notes.append("repeat_runs < 2: consistency was not measured.")
        return notes


def write_run(evaluation_run: EvaluationRun, results_directory: Path) -> Path:
    """Persist a run as JSON, named so runs sort chronologically."""
    results_directory.mkdir(parents=True, exist_ok=True)
    output_path = results_directory / f"{evaluation_run.run_id}.json"
    write_json_atomic(output_path, evaluation_run.to_dict())
    _logger.info("wrote evaluation result to %s", output_path)
    return output_path
