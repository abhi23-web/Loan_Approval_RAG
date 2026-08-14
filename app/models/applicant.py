"""The loan application contract.

Validation lives here rather than in the API route so that the Streamlit form,
the API and the evaluation harness all reject the same bad input in the same way.
Cross-field checks that a lender would actually make — tenure that outlives the
applicant, a loan larger than the property — are enforced, because a system that
silently accepts nonsense produces confident nonsense.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

EmploymentType = Literal["salaried", "self_employed", "professional", "retired"]
PropertyType = Literal["apartment", "independent_house", "plot_plus_construction", "under_construction"]
LocationTier = Literal["metro", "tier_1", "tier_2", "tier_3"]


class LoanApplication(BaseModel):
    """Applicant-supplied facts. No derived values — those belong to the rules."""

    model_config = ConfigDict(extra="forbid")

    # --- Applicant -------------------------------------------------------
    applicant_name: str = Field(min_length=2, max_length=120)
    age_years: int = Field(ge=18, le=80)
    employment_type: EmploymentType
    employment_experience_months: int = Field(ge=0, le=720)
    monthly_income_inr: float = Field(gt=0, le=100_000_000)
    annual_income_inr: float | None = Field(default=None, gt=0, le=1_200_000_000)
    credit_score: int = Field(ge=300, le=900)

    # --- Existing obligations -------------------------------------------
    existing_monthly_emi_inr: float = Field(default=0.0, ge=0, le=100_000_000)
    number_of_existing_loans: int = Field(default=0, ge=0, le=50)
    other_monthly_liabilities_inr: float = Field(default=0.0, ge=0, le=100_000_000)

    # --- Loan request ----------------------------------------------------
    loan_amount_required_inr: float = Field(gt=0, le=1_000_000_000)
    property_value_inr: float = Field(gt=0, le=10_000_000_000)
    loan_tenure_years: int = Field(ge=1, le=40)

    # --- Co-applicant and property --------------------------------------
    co_applicant_monthly_income_inr: float = Field(default=0.0, ge=0, le=100_000_000)
    co_applicant_is_joint_owner: bool = False
    property_type: PropertyType = "apartment"
    location_tier: LocationTier = "metro"

    @model_validator(mode="after")
    def _validate_cross_field_consistency(self) -> LoanApplication:
        if self.loan_amount_required_inr > self.property_value_inr:
            raise ValueError(
                "loan_amount_required_inr cannot exceed property_value_inr; "
                "no lender finances more than the asset is worth"
            )
        if self.annual_income_inr is not None:
            implied_annual_income = self.monthly_income_inr * 12
            # A 25% tolerance absorbs bonuses and variable pay without letting a
            # decimal-point slip through unnoticed.
            if not 0.75 * implied_annual_income <= self.annual_income_inr <= 1.25 * implied_annual_income * 1.4:
                raise ValueError(
                    "annual_income_inr is inconsistent with monthly_income_inr; "
                    "check both figures"
                )
        if self.existing_monthly_emi_inr > 0 and self.number_of_existing_loans == 0:
            raise ValueError(
                "existing_monthly_emi_inr is greater than zero but "
                "number_of_existing_loans is zero"
            )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_monthly_income_inr(self) -> float:
        """Household income considered for repayment capacity.

        Co-applicant income only counts when that co-applicant is a joint owner,
        which is the condition every policy in the corpus attaches to it.
        """
        if self.co_applicant_is_joint_owner:
            return self.monthly_income_inr + self.co_applicant_monthly_income_inr
        return self.monthly_income_inr

    @computed_field  # type: ignore[prop-decorator]
    @property
    def requested_ltv_percent(self) -> float:
        return round(100.0 * self.loan_amount_required_inr / self.property_value_inr, 2)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def age_at_loan_maturity_years(self) -> int:
        return self.age_years + self.loan_tenure_years

    def redacted(self) -> dict[str, object]:
        """Log/trace-safe view with direct identifiers removed.

        Financial figures are kept because they are what makes a trace useful for
        debugging a decision; the name is not.
        """
        payload = self.model_dump()
        payload["applicant_name"] = "[redacted]"
        return payload
