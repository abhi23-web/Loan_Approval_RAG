"""The RAG pipeline.

query -> retrieve -> assemble context -> generate -> validated, cited answer

Two entry points share the machinery:

``assess_application``
    The product path. The rule engine decides; retrieval and generation supply
    the policy grounding and the explanation.

``answer_question``
    The evaluation path. Used by the golden dataset and the ad-hoc question
    endpoint, so evaluation exercises the same retrieval and generation code the
    product uses rather than a parallel implementation that could drift.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from app.core.config import Settings
from app.core.logging_config import get_logger
from app.core.tracing import add_trace_metadata, traced
from app.models.applicant import LoanApplication
from app.models.assessment import (
    GroundedExplanation,
    LoanAssessmentResponse,
    RetrievalDiagnostics,
    RuleAssessment,
)
from app.models.documents import RetrievedChunk
from app.rag.context_builder import assemble_context
from app.rag.generator import GroundedAnswerGenerator
from app.rag.prompts import build_assessment_prompt, build_question_prompt
from app.rag.query_builder import build_policy_query, build_question_query
from app.rag.retriever import PolicyRetriever, RetrievalRequest
from app.rules.eligibility import EligibilityRuleEngine
from app.utils.timing import measure_latency

_logger = get_logger(__name__)


@dataclass
class QuestionAnswer:
    """Result of the question path, used by evaluation and the ask endpoint."""

    question: str
    explanation: GroundedExplanation
    retrieval: RetrievalDiagnostics
    retrieved_chunks: list[RetrievedChunk]
    total_latency_ms: float


class HomeLoanRagPipeline:
    """Wires the rule engine, retriever and generator into the two entry points."""

    def __init__(
        self,
        settings: Settings,
        retriever: PolicyRetriever,
        generator: GroundedAnswerGenerator,
        rule_engine: EligibilityRuleEngine,
    ) -> None:
        self._settings = settings
        self._retriever = retriever
        self._generator = generator
        self._rule_engine = rule_engine

    @traced("pipeline.assess_application", run_type="chain")
    def assess_application(
        self,
        application: LoanApplication,
        *,
        strategy_name: str | None = None,
        top_k: int | None = None,
    ) -> tuple[LoanAssessmentResponse, list[RetrievedChunk]]:
        """Assess one application end to end.

        Returns the response plus the retrieved chunks, so the UI can show the
        evidence behind the answer without a second retrieval.
        """
        request_id = str(uuid.uuid4())
        with measure_latency() as stopwatch:
            rule_assessment: RuleAssessment = self._rule_engine.assess(application)

            retrieval_outcome = self._retriever.retrieve(
                RetrievalRequest(
                    query=build_policy_query(application, rule_assessment),
                    top_k=top_k,
                    strategy_name=strategy_name,
                )
            )
            context = assemble_context(retrieval_outcome.chunks, self._settings.retrieval)
            explanation = self._generator.generate(
                build_assessment_prompt(application, rule_assessment, context.sources_block),
                context,
            )

        add_trace_metadata(
            request_id=request_id,
            decision=rule_assessment.decision,
            grounded=explanation.is_grounded,
            # Financial context is useful for debugging a decision; the applicant's
            # identity is not, so it never reaches the trace.
            application=application.redacted()
            if self._settings.observability.redact_applicant_pii
            else application.model_dump(),
        )

        response = LoanAssessmentResponse(
            request_id=request_id,
            generated_at=datetime.now(UTC),
            decision=rule_assessment.decision,
            rule_assessment=rule_assessment,
            explanation=explanation,
            retrieval=retrieval_outcome.diagnostics,
            total_latency_ms=round(stopwatch.elapsed_ms, 2),
            langsmith_project=self._settings.observability.langsmith_project,
        )
        _logger.info(
            "assessment %s complete: decision=%s grounded=%s citations=%d in %.0fms",
            request_id[:8],
            response.decision,
            explanation.is_grounded,
            len(explanation.citations),
            response.total_latency_ms,
        )
        return response, context.included_chunks

    @traced("pipeline.answer_question", run_type="chain")
    def answer_question(
        self,
        question: str,
        *,
        strategy_name: str | None = None,
        top_k: int | None = None,
        version_numbers_by_source: dict[str, int] | None = None,
        restrict_to_active_versions: bool | None = None,
    ) -> QuestionAnswer:
        """Answer a direct policy question, optionally against a specific version."""
        with measure_latency() as stopwatch:
            retrieval_outcome = self._retriever.retrieve(
                RetrievalRequest(
                    query=build_question_query(question),
                    top_k=top_k,
                    strategy_name=strategy_name,
                    version_numbers_by_source=version_numbers_by_source,
                    restrict_to_active_versions=restrict_to_active_versions,
                )
            )
            context = assemble_context(retrieval_outcome.chunks, self._settings.retrieval)
            explanation = self._generator.generate(
                build_question_prompt(question, context.sources_block), context
            )

        return QuestionAnswer(
            question=question,
            explanation=explanation,
            retrieval=retrieval_outcome.diagnostics,
            retrieved_chunks=context.included_chunks,
            total_latency_ms=round(stopwatch.elapsed_ms, 2),
        )
