"""Assessment service.

The API route's only job is HTTP: parse, call this, serialise. Everything a loan
assessment actually involves lives here, so the same behaviour is reachable from
the evaluation harness and from a future queue worker without going through
FastAPI.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.exceptions import KnowledgeBaseEmptyError
from app.core.logging_config import get_logger
from app.models.applicant import LoanApplication
from app.models.assessment import LoanAssessmentResponse
from app.models.documents import RetrievedChunk
from app.rag.pipeline import HomeLoanRagPipeline, QuestionAnswer
from app.services.container import ApplicationContainer

_logger = get_logger(__name__)


@dataclass
class AssessmentResult:
    """Response plus the evidence behind it, for callers that show their work."""

    response: LoanAssessmentResponse
    retrieved_chunks: list[RetrievedChunk]


class AssessmentService:
    """Application-level operations for loan assessment."""

    def __init__(self, container: ApplicationContainer) -> None:
        self._container = container

    @property
    def _pipeline(self) -> HomeLoanRagPipeline:
        return self._container.rag_pipeline

    def _require_populated_index(self) -> None:
        """Fail loudly when nothing has been ingested.

        An empty index would otherwise produce a polite "insufficient
        information" for every applicant, which reads like a policy answer and
        hides an operational problem.
        """
        strategy_name = self._container.settings.retrieval.active_strategy
        if self._container.vector_store.count(strategy_name) == 0:
            raise KnowledgeBaseEmptyError(
                f"no chunks indexed for strategy '{strategy_name}'. "
                "Run: python scripts/ingest.py"
            )

    def assess(
        self,
        application: LoanApplication,
        *,
        strategy_name: str | None = None,
        top_k: int | None = None,
    ) -> AssessmentResult:
        self._container.fresh_version_store()
        self._require_populated_index()
        response, retrieved_chunks = self._pipeline.assess_application(
            application, strategy_name=strategy_name, top_k=top_k
        )
        return AssessmentResult(response=response, retrieved_chunks=retrieved_chunks)

    def ask(
        self,
        question: str,
        *,
        strategy_name: str | None = None,
        top_k: int | None = None,
        version_numbers_by_source: dict[str, int] | None = None,
        restrict_to_active_versions: bool | None = None,
    ) -> QuestionAnswer:
        """Answer a policy question, optionally pinned to specific versions."""
        self._container.fresh_version_store()
        self._require_populated_index()
        return self._pipeline.answer_question(
            question,
            strategy_name=strategy_name,
            top_k=top_k,
            version_numbers_by_source=version_numbers_by_source,
            restrict_to_active_versions=restrict_to_active_versions,
        )
