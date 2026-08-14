"""Deterministic eligibility rules.

This module — not the language model — decides. A model that can be talked into
a different answer by phrasing has no business approving credit, and a decision
that cannot be recomputed from the inputs cannot be defended to a regulator.

The division of labour across the system is:

* these rules produce the decision and the arithmetic behind it;
* retrieval produces the policy text that the thresholds correspond to;
* the model explains the first in the language of the second, and may not
  contradict either.

Every threshold is configuration (``rules`` in settings.yaml), so tuning credit
policy never requires touching this file.
"""

from __future__ import annotations

from app.core.config import RulesSection
from app.core.logging_config import get_logger
from app.models.applicant import LoanApplication
from app.models.assessment import Decision, RuleAssessment, RuleCheck

_logger = get_logger(__name__)

_MONTHS_PER_YEAR = 12


def calculate_equated_monthly_instalment(
    principal_inr: float, annual_interest_percent: float, tenure_years: int
) -> float:
    """Standard amortising EMI.

    EMI = P·r·(1+r)^n / ((1+r)^n − 1), with r the monthly rate and n the number
    of instalments. The zero-rate branch is not a real product but keeps the
    function total, so a misconfigured rate produces a sane number instead of a
    ZeroDivisionError deep inside a request.
    """
    monthly_rate = annual_interest_percent / 100.0 / _MONTHS_PER_YEAR
    instalment_count = tenure_years * _MONTHS_PER_YEAR
    if monthly_rate <= 0:
        return round(principal_inr / instalment_count, 2)
    growth_factor = (1.0 + monthly_rate) ** instalment_count
    return round(principal_inr * monthly_rate * growth_factor / (growth_factor - 1.0), 2)


def calculate_principal_for_instalment(
    instalment_inr: float, annual_interest_percent: float, tenure_years: int
) -> float:
    """Inverse of the EMI formula: the largest loan a given EMI can service."""
    monthly_rate = annual_interest_percent / 100.0 / _MONTHS_PER_YEAR
    instalment_count = tenure_years * _MONTHS_PER_YEAR
    if instalment_inr <= 0:
        return 0.0
    if monthly_rate <= 0:
        return round(instalment_inr * instalment_count, 2)
    growth_factor = (1.0 + monthly_rate) ** instalment_count
    return round(instalment_inr * (growth_factor - 1.0) / (monthly_rate * growth_factor), 2)


def resolve_permitted_ltv_percent(loan_amount_inr: float, rules: RulesSection) -> float:
    """Walk the LTV ladder and return the ceiling for this loan size."""
    for slab in rules.ltv_slabs:
        if slab.max_loan_amount_inr is None or loan_amount_inr <= slab.max_loan_amount_inr:
            return slab.max_ltv_percent
    return rules.ltv_slabs[-1].max_ltv_percent


class EligibilityRuleEngine:
    """Runs every check and combines them into a single decision."""

    def __init__(self, rules: RulesSection) -> None:
        self._rules = rules

    def assess(self, application: LoanApplication) -> RuleAssessment:
        proposed_instalment = calculate_equated_monthly_instalment(
            application.loan_amount_required_inr,
            self._rules.indicative_annual_interest_percent,
            application.loan_tenure_years,
        )
        total_monthly_obligation = (
            application.existing_monthly_emi_inr
            + application.other_monthly_liabilities_inr
            + proposed_instalment
        )
        foir_percent = round(
            100.0 * total_monthly_obligation / application.total_monthly_income_inr, 2
        )
        permitted_ltv_percent = resolve_permitted_ltv_percent(
            application.loan_amount_required_inr, self._rules
        )

        checks = [
            self._check_minimum_age(application),
            self._check_age_at_maturity(application),
            self._check_tenure(application),
            self._check_minimum_income(application),
            self._check_employment_stability(application),
            self._check_credit_score(application),
            self._check_repayment_capacity(application, foir_percent),
            self._check_loan_to_value(application, permitted_ltv_percent),
        ]

        assessment = RuleAssessment(
            decision=self._combine(checks),
            checks=checks,
            estimated_monthly_emi_inr=proposed_instalment,
            computed_foir_percent=foir_percent,
            requested_ltv_percent=application.requested_ltv_percent,
            permitted_ltv_percent=permitted_ltv_percent,
            maximum_eligible_loan_amount_inr=self._maximum_eligible_loan(
                application, permitted_ltv_percent
            ),
        )
        _logger.info(
            "rule decision=%s foir=%.1f%% ltv=%.1f%%/%.1f%% emi=%.0f",
            assessment.decision,
            foir_percent,
            application.requested_ltv_percent,
            permitted_ltv_percent,
            proposed_instalment,
        )
        return assessment

    # ---------------------------------------------------------- individual checks

    def _check_minimum_age(self, application: LoanApplication) -> RuleCheck:
        passed = application.age_years >= self._rules.min_applicant_age
        return RuleCheck(
            check_name="minimum_applicant_age",
            outcome="PASS" if passed else "FAIL",
            observed_value=application.age_years,
            threshold_value=self._rules.min_applicant_age,
            explanation=(
                f"Applicant is {application.age_years}; the minimum age is "
                f"{self._rules.min_applicant_age}."
            ),
        )

    def _check_age_at_maturity(self, application: LoanApplication) -> RuleCheck:
        age_at_maturity = application.age_at_loan_maturity_years
        passed = age_at_maturity <= self._rules.max_age_at_loan_maturity
        return RuleCheck(
            check_name="age_at_loan_maturity",
            outcome="PASS" if passed else "FAIL",
            observed_value=age_at_maturity,
            threshold_value=self._rules.max_age_at_loan_maturity,
            explanation=(
                f"Applicant would be {age_at_maturity} at the end of a "
                f"{application.loan_tenure_years}-year tenure; the limit is "
                f"{self._rules.max_age_at_loan_maturity}."
            ),
        )

    def _check_tenure(self, application: LoanApplication) -> RuleCheck:
        passed = application.loan_tenure_years <= self._rules.max_tenure_years
        return RuleCheck(
            check_name="maximum_tenure",
            outcome="PASS" if passed else "FAIL",
            observed_value=application.loan_tenure_years,
            threshold_value=self._rules.max_tenure_years,
            explanation=(
                f"Requested tenure is {application.loan_tenure_years} years against a "
                f"maximum of {self._rules.max_tenure_years}."
            ),
        )

    def _check_minimum_income(self, application: LoanApplication) -> RuleCheck:
        observed_income = application.total_monthly_income_inr
        passed = observed_income >= self._rules.min_monthly_income_inr
        return RuleCheck(
            check_name="minimum_monthly_income",
            outcome="PASS" if passed else "FAIL",
            observed_value=observed_income,
            threshold_value=self._rules.min_monthly_income_inr,
            explanation=(
                f"Considered monthly income is INR {observed_income:,.0f} against a "
                f"minimum of INR {self._rules.min_monthly_income_inr:,.0f}."
            ),
        )

    def _check_employment_stability(self, application: LoanApplication) -> RuleCheck:
        # Self-employed applicants carry income volatility a payslip does not, so
        # the required track record is longer. Retirees are held to the salaried
        # bar since their income is already established.
        required_months = (
            self._rules.min_employment_months_self_employed
            if application.employment_type in {"self_employed", "professional"}
            else self._rules.min_employment_months_salaried
        )
        passed = application.employment_experience_months >= required_months
        return RuleCheck(
            check_name="employment_stability",
            outcome="PASS" if passed else "FAIL",
            observed_value=application.employment_experience_months,
            threshold_value=required_months,
            explanation=(
                f"{application.employment_experience_months} month(s) of "
                f"{application.employment_type.replace('_', ' ')} history against a "
                f"requirement of {required_months}."
            ),
        )

    def _check_credit_score(self, application: LoanApplication) -> RuleCheck:
        if application.credit_score >= self._rules.min_credit_score:
            outcome = "PASS"
        elif application.credit_score >= self._rules.conditional_credit_score:
            outcome = "CONDITIONAL"
        else:
            outcome = "FAIL"
        return RuleCheck(
            check_name="credit_score",
            outcome=outcome,
            observed_value=application.credit_score,
            threshold_value=self._rules.min_credit_score,
            explanation=(
                f"Credit score is {application.credit_score}. Standard sanction needs "
                f"{self._rules.min_credit_score}; "
                f"{self._rules.conditional_credit_score}-{self._rules.min_credit_score - 1} "
                "is eligible only on a conditional basis."
            ),
        )

    def _check_repayment_capacity(
        self, application: LoanApplication, foir_percent: float
    ) -> RuleCheck:
        if foir_percent <= self._rules.max_foir_percent:
            outcome = "PASS"
        elif foir_percent <= self._rules.conditional_foir_percent:
            outcome = "CONDITIONAL"
        else:
            outcome = "FAIL"
        return RuleCheck(
            check_name="repayment_capacity_foir",
            outcome=outcome,
            observed_value=foir_percent,
            threshold_value=self._rules.max_foir_percent,
            explanation=(
                f"Total obligations would be {foir_percent:.1f}% of considered income "
                f"against a standard ceiling of {self._rules.max_foir_percent:.0f}% "
                f"and a conditional ceiling of {self._rules.conditional_foir_percent:.0f}%."
            ),
        )

    def _check_loan_to_value(
        self, application: LoanApplication, permitted_ltv_percent: float
    ) -> RuleCheck:
        passed = application.requested_ltv_percent <= permitted_ltv_percent
        return RuleCheck(
            check_name="loan_to_value",
            outcome="PASS" if passed else "FAIL",
            observed_value=application.requested_ltv_percent,
            threshold_value=permitted_ltv_percent,
            explanation=(
                f"Requested loan-to-value is {application.requested_ltv_percent:.1f}% "
                f"against a permitted {permitted_ltv_percent:.0f}% for a loan of "
                f"INR {application.loan_amount_required_inr:,.0f}."
            ),
        )

    # ---------------------------------------------------------------- combining

    @staticmethod
    def _combine(checks: list[RuleCheck]) -> Decision:
        """Worst outcome wins. No check can be offset by another passing check."""
        if any(check.outcome == "FAIL" for check in checks):
            return "NOT_ELIGIBLE"
        if any(check.outcome == "CONDITIONAL" for check in checks):
            return "CONDITIONALLY_ELIGIBLE"
        return "ELIGIBLE"

    def _maximum_eligible_loan(
        self, application: LoanApplication, permitted_ltv_percent: float
    ) -> float:
        """The binding constraint of collateral value and repayment capacity.

        Reported even on a rejection, because "you could borrow up to X" is the
        single most useful thing an applicant can be told.
        """
        collateral_cap = application.property_value_inr * permitted_ltv_percent / 100.0

        affordable_instalment = (
            application.total_monthly_income_inr * self._rules.max_foir_percent / 100.0
            - application.existing_monthly_emi_inr
            - application.other_monthly_liabilities_inr
        )
        capacity_cap = calculate_principal_for_instalment(
            max(0.0, affordable_instalment),
            self._rules.indicative_annual_interest_percent,
            min(application.loan_tenure_years, self._rules.max_tenure_years),
        )
        return round(max(0.0, min(collateral_cap, capacity_cap)), 2)
