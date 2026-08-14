"""Turning a loan application into a retrieval query.

A raw dump of form fields makes a poor query: numbers dominate the embedding and
pull back whatever chunk happens to contain similar figures. What actually needs
retrieving is the *policy language* behind each check the rule engine ran, so the
query is assembled from the vocabulary of those checks.

The assembly is deterministic — the same application always produces the same
query string, in the same order — which is a precondition for the same
application producing the same retrieved context and therefore the same answer.
"""

from __future__ import annotations

from app.models.applicant import LoanApplication
from app.models.assessment import RuleAssessment

# The policy vocabulary each rule check should retrieve. Keyed by check name so
# adding a rule and adding its retrieval terms happen in an obvious pair.
_QUERY_TERMS_BY_CHECK: dict[str, str] = {
    "minimum_applicant_age": "minimum age of applicant at application",
    "age_at_loan_maturity": "maximum age of borrower at loan maturity",
    "maximum_tenure": "maximum loan tenure in years",
    "minimum_monthly_income": "minimum net monthly income requirement",
    "employment_stability": "employment continuity and business vintage requirement",
    "credit_score": "minimum CIBIL credit score for sanction",
    "repayment_capacity_foir": "fixed obligation to income ratio FOIR limit repayment capacity",
    "loan_to_value": "maximum loan to value ratio LTV slab by loan amount",
}

_BASE_QUERY = "home loan eligibility criteria"

# A failing or conditional check is the reason the applicant is reading this
# answer, so its policy language is repeated to weight it in the embedding.
_EMPHASIS_REPEATS = 2


def build_policy_query(
    application: LoanApplication, rule_assessment: RuleAssessment
) -> str:
    """Compose the retrieval query for one assessment."""
    decisive_terms: list[str] = []
    supporting_terms: list[str] = []

    for check in rule_assessment.checks:
        terms = _QUERY_TERMS_BY_CHECK.get(check.check_name)
        if not terms:
            continue
        if check.outcome in {"FAIL", "CONDITIONAL"}:
            decisive_terms.extend([terms] * _EMPHASIS_REPEATS)
        else:
            supporting_terms.append(terms)

    employment_phrase = application.employment_type.replace("_", " ")
    query_parts = [
        _BASE_QUERY,
        f"for a {employment_phrase} applicant",
        *decisive_terms,
        *supporting_terms,
    ]
    return "; ".join(query_parts)


def build_question_query(question: str) -> str:
    """Retrieval query for a direct policy question.

    The question is used as-is. Rewriting or expanding it would make the golden
    dataset measure the rewriter as much as the retriever, and there is no
    evidence yet that a rewrite helps here — that is an experiment to run, not an
    assumption to bake in.
    """
    return question.strip()
