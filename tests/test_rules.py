"""Deterministic eligibility rules.

These are the tests that matter most: this module, not the model, decides who is
approved. Each case pins one rule so a threshold change shows up as a named
failure rather than as a shifted aggregate.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import get_settings
from app.models.applicant import LoanApplication
from app.rules.eligibility import (
    EligibilityRuleEngine,
    calculate_equated_monthly_instalment,
    calculate_principal_for_instalment,
    resolve_permitted_ltv_percent,
)

RULES = get_settings().rules
ENGINE = EligibilityRuleEngine(RULES)


def _application(**overrides) -> LoanApplication:
    """A comfortably approvable applicant, adjusted per test."""
    base = {
        "applicant_name": "Test Applicant",
        "age_years": 34,
        "employment_type": "salaried",
        "employment_experience_months": 72,
        "monthly_income_inr": 200_000.0,
        "credit_score": 780,
        "existing_monthly_emi_inr": 0.0,
        "number_of_existing_loans": 0,
        "loan_amount_required_inr": 4_000_000.0,
        "property_value_inr": 8_000_000.0,
        "loan_tenure_years": 20,
    }
    return LoanApplication(**{**base, **overrides})


# ------------------------------------------------------------------- arithmetic


def test_emi_matches_the_standard_amortisation_formula() -> None:
    # INR 10,00,000 at 8.75% p.a. over 240 instalments.
    # r = 0.0875/12 = 0.00729167, (1+r)^240 = 5.71857
    # EMI = 1e6 * 0.00729167 * 5.71857 / 4.71857 = 8837.11
    instalment = calculate_equated_monthly_instalment(1_000_000, 8.75, 20)
    assert instalment == pytest.approx(8837.11, abs=0.5)


def test_emi_and_principal_are_inverses() -> None:
    principal = 5_000_000.0
    instalment = calculate_equated_monthly_instalment(principal, 8.75, 20)
    assert calculate_principal_for_instalment(instalment, 8.75, 20) == pytest.approx(
        principal, rel=1e-4
    )


def test_zero_interest_does_not_divide_by_zero() -> None:
    assert calculate_equated_monthly_instalment(1_200_000, 0.0, 10) == pytest.approx(10_000.0)


@pytest.mark.parametrize(
    ("loan_amount", "expected_ltv"),
    [(2_500_000, 90.0), (3_000_000, 90.0), (5_000_000, 80.0), (7_500_000, 80.0), (9_000_000, 75.0)],
)
def test_ltv_ladder_boundaries(loan_amount: float, expected_ltv: float) -> None:
    """Slab boundaries are inclusive of their upper bound."""
    assert resolve_permitted_ltv_percent(loan_amount, RULES) == expected_ltv


# ----------------------------------------------------------------------- checks


def test_a_strong_applicant_is_eligible() -> None:
    assessment = ENGINE.assess(_application())
    assert assessment.decision == "ELIGIBLE"
    assert not assessment.failed_checks


def test_a_borderline_credit_score_is_conditional_not_rejected() -> None:
    assessment = ENGINE.assess(_application(credit_score=670))
    assert assessment.decision == "CONDITIONALLY_ELIGIBLE"
    assert [check.check_name for check in assessment.conditional_checks] == ["credit_score"]


def test_a_low_credit_score_is_rejected() -> None:
    assessment = ENGINE.assess(_application(credit_score=610))
    assert assessment.decision == "NOT_ELIGIBLE"
    assert any(check.check_name == "credit_score" for check in assessment.failed_checks)


def test_excess_loan_to_value_is_rejected() -> None:
    assessment = ENGINE.assess(
        _application(loan_amount_required_inr=7_800_000.0, property_value_inr=8_000_000.0)
    )
    assert assessment.decision == "NOT_ELIGIBLE"
    assert any(check.check_name == "loan_to_value" for check in assessment.failed_checks)


def test_repayment_capacity_dominates_a_high_income_but_over_committed_applicant() -> None:
    assessment = ENGINE.assess(
        _application(
            monthly_income_inr=60_000.0,
            existing_monthly_emi_inr=25_000.0,
            number_of_existing_loans=2,
        )
    )
    assert assessment.decision == "NOT_ELIGIBLE"
    assert any(check.check_name == "repayment_capacity_foir" for check in assessment.failed_checks)
    assert assessment.computed_foir_percent > RULES.conditional_foir_percent


def test_tenure_beyond_retirement_age_is_rejected() -> None:
    assessment = ENGINE.assess(_application(age_years=58, loan_tenure_years=25))
    assert assessment.decision == "NOT_ELIGIBLE"
    assert any(check.check_name == "age_at_loan_maturity" for check in assessment.failed_checks)


def test_self_employed_applicants_need_a_longer_track_record() -> None:
    salaried = ENGINE.assess(_application(employment_type="salaried", employment_experience_months=18))
    self_employed = ENGINE.assess(
        _application(employment_type="self_employed", employment_experience_months=18)
    )
    assert salaried.decision == "ELIGIBLE"
    assert self_employed.decision == "NOT_ELIGIBLE"


def test_co_applicant_income_counts_only_when_they_co_own() -> None:
    without_ownership = _application(
        monthly_income_inr=60_000.0, co_applicant_monthly_income_inr=90_000.0
    )
    with_ownership = _application(
        monthly_income_inr=60_000.0,
        co_applicant_monthly_income_inr=90_000.0,
        co_applicant_is_joint_owner=True,
    )
    assert without_ownership.total_monthly_income_inr == 60_000.0
    assert with_ownership.total_monthly_income_inr == 150_000.0
    assert (
        ENGINE.assess(with_ownership).computed_foir_percent
        < ENGINE.assess(without_ownership).computed_foir_percent
    )


def test_maximum_eligible_loan_is_the_binding_constraint() -> None:
    """Reported headroom must respect both collateral and repayment capacity."""
    assessment = ENGINE.assess(
        _application(monthly_income_inr=50_000.0, property_value_inr=50_000_000.0,
                     loan_amount_required_inr=4_000_000.0)
    )
    collateral_cap = 50_000_000.0 * assessment.permitted_ltv_percent / 100.0
    assert assessment.maximum_eligible_loan_amount_inr < collateral_cap


def test_the_same_application_always_produces_the_same_assessment() -> None:
    """Determinism of the decision layer, independent of any model."""
    application = _application(credit_score=705, monthly_income_inr=90_000.0)
    first = ENGINE.assess(application)
    second = ENGINE.assess(application)
    assert first.model_dump() == second.model_dump()


# ------------------------------------------------------------------ validation


def test_a_loan_larger_than_the_property_is_rejected_at_the_contract() -> None:
    with pytest.raises(ValidationError, match="cannot exceed property_value"):
        _application(loan_amount_required_inr=9_000_000.0, property_value_inr=8_000_000.0)


def test_an_emi_without_a_loan_is_rejected_at_the_contract() -> None:
    with pytest.raises(ValidationError, match="number_of_existing_loans"):
        _application(existing_monthly_emi_inr=15_000.0, number_of_existing_loans=0)


def test_applicant_name_is_redacted_for_logging() -> None:
    redacted = _application().redacted()
    assert redacted["applicant_name"] == "[redacted]"
    assert redacted["monthly_income_inr"] == 200_000.0
