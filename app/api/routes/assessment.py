"""Loan assessment endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_assessment_service
from app.api.schemas import (
    LoanAssessmentApiResponse,
    LoanAssessmentRequest,
    PolicyQuestionRequest,
    PolicyQuestionResponse,
)
from app.core.logging_config import get_logger
from app.services.assessment_service import AssessmentService

_logger = get_logger(__name__)

router = APIRouter(tags=["assessment"])


@router.post("/loan-assessment", response_model=LoanAssessmentApiResponse)
def assess_loan_application(
    request: LoanAssessmentRequest,
    assessment_service: AssessmentService = Depends(get_assessment_service),
) -> LoanAssessmentApiResponse:
    """Assess an application: deterministic decision plus a grounded explanation."""
    result = assessment_service.assess(
        request.application,
        strategy_name=request.chunking_strategy,
        top_k=request.top_k,
    )
    return LoanAssessmentApiResponse(
        assessment=result.response,
        retrieved_chunks=result.retrieved_chunks if request.include_retrieved_chunks else None,
    )


@router.post("/policy-question", response_model=PolicyQuestionResponse)
def answer_policy_question(
    request: PolicyQuestionRequest,
    assessment_service: AssessmentService = Depends(get_assessment_service),
) -> PolicyQuestionResponse:
    """Answer a policy question against the corpus.

    This is the endpoint the golden dataset runs through, so evaluation measures
    the deployed path rather than a private one that can drift from it.
    """
    answer = assessment_service.ask(
        request.question,
        strategy_name=request.chunking_strategy,
        top_k=request.top_k,
        version_numbers_by_source=request.version_numbers_by_source,
        restrict_to_active_versions=request.restrict_to_active_versions,
    )
    return PolicyQuestionResponse(
        question=answer.question,
        answer=answer.explanation.explanation,
        is_grounded=answer.explanation.is_grounded,
        insufficient_information=answer.explanation.insufficient_information,
        citations=[citation.model_dump() for citation in answer.explanation.citations],
        retrieval=answer.retrieval.model_dump(),
        total_latency_ms=answer.total_latency_ms,
        retrieved_chunks=answer.retrieved_chunks if request.include_retrieved_chunks else None,
    )
