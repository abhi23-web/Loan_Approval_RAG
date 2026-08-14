"""API request and response schemas.

Wrappers around the domain models rather than copies of them. The domain model
stays the single definition of what a loan application is; these types add only
what is specific to the HTTP boundary — per-request overrides and debug toggles.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.applicant import LoanApplication
from app.models.assessment import LoanAssessmentResponse
from app.models.documents import RetrievedChunk


class LoanAssessmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    application: LoanApplication
    # Overrides exist so the experiment harness can sweep configurations through
    # the real API instead of a bypass path. They are optional and default to the
    # configured values.
    chunking_strategy: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    include_retrieved_chunks: bool = False


class LoanAssessmentApiResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessment: LoanAssessmentResponse
    retrieved_chunks: list[RetrievedChunk] | None = None


class PolicyQuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=500)
    chunking_strategy: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    # e.g. {"meridian_home_loan_policy": 2} to ask what the rule was under v2.
    version_numbers_by_source: dict[str, int] | None = None
    restrict_to_active_versions: bool | None = None
    include_retrieved_chunks: bool = False


class PolicyQuestionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str
    is_grounded: bool
    insufficient_information: bool
    citations: list[dict]
    retrieval: dict
    total_latency_ms: float
    retrieved_chunks: list[RetrievedChunk] | None = None


class DocumentRefreshRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_names: list[str] | None = None
    strategy_names: list[str] | None = None
    # Only for when the chunker, cleaner or embedding model changed — the
    # documents are unchanged but the vectors are stale.
    force_reindex: bool = False


class DocumentRefreshResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    duration_seconds: float
    results: list[dict]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    checked_at: datetime
    application: str
    environment: str
    active_chunking_strategy: str
    embedding_provider: str
    embedding_model: str
    llm_provider: str
    llm_model: str
    indexed_chunk_count: int
    active_source_count: int
    langsmith_tracing: bool
    warnings: list[str]


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: str
    detail: str
