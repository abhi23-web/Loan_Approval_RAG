"""Prompt templates.

Prompts live in one module, versioned with a constant, because a prompt change is
an experiment variable exactly like a chunk size. ``PROMPT_VERSION`` is recorded
on every trace and in every evaluation result, so a metric shift can be
attributed to the prompt that caused it.

The design constraints encoded below:

* the model is told the decision has already been made by a rule engine, so it
  cannot believe it is the decision-maker;
* it may cite only the numbered sources it was given, which makes an invented
  citation detectable rather than merely discouraged;
* it has an explicit, exact escape hatch when the context does not support an
  answer, so "I don't know" is an available action and not a failure.

Retrieved policy text is never passed through ``str.format``. Policy documents
contain braces often enough that doing so would eventually raise mid-request, so
the sources block is concatenated, not interpolated.
"""

from __future__ import annotations

from app.models.applicant import LoanApplication
from app.models.assessment import RuleAssessment

PROMPT_VERSION = "2026-08-14.v1"

INSUFFICIENT_INFORMATION_SENTENCE = (
    "Insufficient information in the available policy documents."
)

SYSTEM_PROMPT = f"""\
You are a home loan policy analyst working inside an automated assessment system.

Your role is narrow and you must stay inside it:

1. The eligibility DECISION has already been made by a deterministic rule engine.
   You do not make, change, soften or dispute that decision. You explain it.
2. You may use ONLY the numbered policy extracts provided under "POLICY SOURCES".
   You have no other knowledge of any lender's policy. Do not use general
   knowledge about home loans to fill a gap.
3. Every factual claim about policy must carry the marker of the source it came
   from, written in square brackets, for example [S1] or [S2].
4. Never invent a source marker, a page number, a document title or a version
   number. If a marker is not in the list you were given, you may not write it.
5. If the provided extracts do not support an explanation, reply with exactly
   this sentence and nothing else:
   {INSUFFICIENT_INFORMATION_SENTENCE}

Style: plain business English for an applicant. Four to eight sentences. No
headings, no bullet lists, no restating of the applicant's personal details, and
no description of your own reasoning process.
"""

_ASSESSMENT_HEADER_TEMPLATE = """\
DECISION FROM THE RULE ENGINE
{decision}

RULE CHECK RESULTS
{rule_checks}

APPLICATION FACTS RELEVANT TO POLICY
- Requested loan amount: INR {loan_amount:,.0f}
- Property value: INR {property_value:,.0f}
- Requested loan-to-value: {requested_ltv:.1f}%
- Permitted loan-to-value for this loan size: {permitted_ltv:.0f}%
- Estimated monthly instalment: INR {estimated_emi:,.0f}
- Fixed obligation to income ratio: {foir:.1f}%
- Credit score: {credit_score}
- Employment type: {employment_type}
- Requested tenure: {tenure_years} years
"""

_ASSESSMENT_TASK = """\

TASK
Explain to the applicant why the decision above was reached, grounding each
policy statement in the numbered sources with a [S#] marker. Where a check
failed or was conditional, say what the policy requires and what the applicant
would need to change. Do not state any threshold that does not appear in the
sources above.
"""

_QUESTION_TASK = """\

TASK
Answer the question using only the numbered sources above. Cite every factual
statement with its [S#] marker. Answer in one to three sentences, and state the
figure explicitly where the question asks for one.
"""


def _sources_section(policy_sources_block: str) -> str:
    return f"\nPOLICY SOURCES\n{policy_sources_block}\n"


def build_assessment_prompt(
    application: LoanApplication,
    rule_assessment: RuleAssessment,
    policy_sources_block: str,
) -> str:
    """Render the user prompt for a loan assessment.

    Applicant name and other identifiers are deliberately absent: they cannot
    affect a policy explanation, and leaving them out keeps personal data out of
    prompts, traces and any model-provider logs.
    """
    rule_check_lines = "\n".join(
        f"- {check.check_name}: {check.outcome} — {check.explanation}"
        for check in rule_assessment.checks
    )
    header = _ASSESSMENT_HEADER_TEMPLATE.format(
        decision=rule_assessment.decision.replace("_", " ").title(),
        rule_checks=rule_check_lines,
        loan_amount=application.loan_amount_required_inr,
        property_value=application.property_value_inr,
        requested_ltv=rule_assessment.requested_ltv_percent,
        permitted_ltv=rule_assessment.permitted_ltv_percent,
        estimated_emi=rule_assessment.estimated_monthly_emi_inr,
        foir=rule_assessment.computed_foir_percent,
        credit_score=application.credit_score,
        employment_type=application.employment_type.replace("_", " "),
        tenure_years=application.loan_tenure_years,
    )
    return header + _sources_section(policy_sources_block) + _ASSESSMENT_TASK


def build_question_prompt(question: str, policy_sources_block: str) -> str:
    """Render the user prompt for a direct policy question.

    Used by the golden dataset, which asks about policy content rather than about
    a specific applicant, and by the ad-hoc question endpoint.
    """
    return f"QUESTION\n{question}\n" + _sources_section(policy_sources_block) + _QUESTION_TASK
