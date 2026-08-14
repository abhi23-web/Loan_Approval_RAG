"""Assessment contracts: rule output, grounded answer, and the API response.

The type split is the design point of this system. ``RuleAssessment`` is what the
deterministic engine decided; ``GroundedExplanation`` is what the language model
said about it. They are never merged into one free-text blob, so a reader can
always tell "policy says X" apart from "the model wrote X".
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.documents import RetrievedChunk

Decision = Literal["ELIGIBLE", "CONDITIONALLY_ELIGIBLE", "NOT_ELIGIBLE"]
CheckOutcome = Literal["PASS", "CONDITIONAL", "FAIL"]


class RuleCheck(BaseModel):
    """One deterministic eligibility check and the numbers behind it."""

    model_config = ConfigDict(extra="forbid")

    check_name: str
    outcome: CheckOutcome
    observed_value: float | int | str | None
    threshold_value: float | int | str | None
    explanation: str


class RuleAssessment(BaseModel):
    """The decision. Computed in Python, never delegated to the model."""

    model_config = ConfigDict(extra="forbid")

    decision: Decision
    checks: list[RuleCheck]
    estimated_monthly_emi_inr: float
    computed_foir_percent: float
    requested_ltv_percent: float
    permitted_ltv_percent: float
    maximum_eligible_loan_amount_inr: float

    @property
    def failed_checks(self) -> list[RuleCheck]:
        return [check for check in self.checks if check.outcome == "FAIL"]

    @property
    def conditional_checks(self) -> list[RuleCheck]:
        return [check for check in self.checks if check.outcome == "CONDITIONAL"]


class Citation(BaseModel):
    """A source the explanation is allowed to point at.

    Built from the chunks that were actually retrieved. The model selects among
    these; it never authors one, which is what makes fabricated citations
    structurally impossible rather than merely discouraged.
    """

    model_config = ConfigDict(extra="forbid")

    marker: str  # e.g. "S1", matching the numbering used in the prompt
    source_name: str
    institution: str
    document_title: str
    version_number: int
    version_label: str
    url: str
    page_number: int | None = None
    effective_date: str | None = None
    excerpt: str
    similarity: float


class GroundedExplanation(BaseModel):
    """The model's explanation plus the citations it actually used."""

    model_config = ConfigDict(extra="forbid")

    explanation: str
    citations: list[Citation]
    # True when retrieval produced nothing usable, or the model's answer could
    # not be tied to any retrieved source. The explanation is then the fixed
    # insufficient-information sentence rather than a guess.
    is_grounded: bool
    insufficient_information: bool = False
    model_name: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class RetrievalDiagnostics(BaseModel):
    """What retrieval did, surfaced so the UI and LangSmith agree with each other."""

    model_config = ConfigDict(extra="forbid")

    query: str
    chunking_strategy: str
    top_k: int
    min_similarity: float
    restricted_to_active_versions: bool
    active_version_ids: list[str]
    retrieved_count: int
    dropped_below_threshold_count: int
    retrieval_latency_ms: float
    context_characters: int


class LoanAssessmentResponse(BaseModel):
    """What the API returns and the frontend renders."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    generated_at: datetime
    decision: Decision
    rule_assessment: RuleAssessment
    explanation: GroundedExplanation
    retrieval: RetrievalDiagnostics
    total_latency_ms: float
    # Present only when tracing is on; lets a reviewer jump from a decision to
    # the exact trace that produced it.
    langsmith_project: str | None = None


class RetrievedContextView(BaseModel):
    """Debug view of retrieved chunks, exposed to the UI behind an expander."""

    model_config = ConfigDict(extra="forbid")

    chunks: list[RetrievedChunk] = Field(default_factory=list)
