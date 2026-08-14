"""Streamlit frontend for the Home Loan Approval Assistant.

The frontend is a client and nothing more. It validates enough to give immediate
feedback, then sends the application to the API; it never computes eligibility
itself. Two implementations of the same rules would drift, and the one the
applicant sees would eventually disagree with the one on record.

What it deliberately shows: the decision, the reasoning behind each rule check,
the policy sources with their document version, and the retrieval diagnostics.
What it deliberately hides: the model's raw output before validation, prompts,
and anything resembling chain of thought.

    streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

API_BASE_URL = os.environ.get("HOME_LOAN_API_URL", "http://localhost:8000").rstrip("/")
API_PREFIX = "/api/v1"
REQUEST_TIMEOUT_SECONDS = 300.0

DECISION_PRESENTATION: dict[str, tuple[str, str]] = {
    "ELIGIBLE": ("Eligible", "success"),
    "CONDITIONALLY_ELIGIBLE": ("Conditionally eligible", "warning"),
    "NOT_ELIGIBLE": ("Not eligible", "error"),
}

OUTCOME_ICON = {"PASS": "✅", "CONDITIONAL": "⚠️", "FAIL": "❌"}


# ------------------------------------------------------------------ API client


def _call_api(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call the backend, converting transport and domain errors into UI errors."""
    url = f"{API_BASE_URL}{API_PREFIX}{path}"
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = (
                client.post(url, json=payload) if payload is not None else client.get(url)
            )
    except httpx.HTTPError as transport_error:
        raise RuntimeError(
            f"Could not reach the backend at {API_BASE_URL}. "
            f"Start it with 'python run_backend.py'. ({transport_error})"
        ) from transport_error

    if response.status_code >= 400:
        try:
            error_body = response.json()
            detail = error_body.get("detail") or error_body
        except ValueError:
            detail = response.text[:400]
        raise RuntimeError(f"Backend returned HTTP {response.status_code}: {detail}")
    return response.json()


# --------------------------------------------------------------------- widgets


def _render_decision_banner(decision: str) -> None:
    label, style = DECISION_PRESENTATION.get(decision, (decision, "info"))
    getattr(st, style)(f"### Decision: {label}")


def _render_rule_checks(rule_assessment: dict[str, Any]) -> None:
    st.subheader("How this decision was reached")
    st.caption(
        "These checks are computed in code from your figures and the lender's "
        "configured thresholds. They are not produced by a language model."
    )
    for check in rule_assessment["checks"]:
        icon = OUTCOME_ICON.get(check["outcome"], "•")
        readable_name = check["check_name"].replace("_", " ").title()
        st.markdown(f"{icon} **{readable_name}** — {check['explanation']}")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Estimated EMI", f"₹{rule_assessment['estimated_monthly_emi_inr']:,.0f}")
    metric_columns[1].metric("FOIR", f"{rule_assessment['computed_foir_percent']:.1f}%")
    metric_columns[2].metric(
        "Loan to value",
        f"{rule_assessment['requested_ltv_percent']:.1f}%",
        delta=f"limit {rule_assessment['permitted_ltv_percent']:.0f}%",
        delta_color="off",
    )
    metric_columns[3].metric(
        "Max eligible loan",
        f"₹{rule_assessment['maximum_eligible_loan_amount_inr']:,.0f}",
    )


def _render_explanation(explanation: dict[str, Any]) -> None:
    st.subheader("Policy explanation")
    if explanation["insufficient_information"]:
        st.warning(
            explanation["explanation"]
            + "  \nThe decision above still stands — it comes from the rule engine — "
            "but no policy extract was retrieved that could support an explanation."
        )
    else:
        st.write(explanation["explanation"])
        if not explanation["is_grounded"]:
            st.warning(
                "Part of this explanation could not be tied to a retrieved source "
                "and has been marked as ungrounded."
            )


def _render_citations(explanation: dict[str, Any]) -> None:
    citations = explanation.get("citations") or []
    if not citations:
        return

    st.subheader("Sources")
    for citation in citations:
        page_fragment = f" · page {citation['page_number']}" if citation.get("page_number") else ""
        effective_fragment = (
            f" · effective {citation['effective_date']}" if citation.get("effective_date") else ""
        )
        with st.container(border=True):
            st.markdown(
                f"**[{citation['marker']}] {citation['document_title']}** — "
                f"{citation['institution']}"
            )
            st.caption(
                f"Version {citation['version_number']}{page_fragment}{effective_fragment} "
                f"· similarity {citation['similarity']:.3f}"
            )
            st.markdown(f"> {citation['excerpt']}")
            st.markdown(f"[Open source document]({citation['url']})")


def _render_retrieval_diagnostics(
    retrieval: dict[str, Any], retrieved_chunks: list[dict[str, Any]] | None
) -> None:
    with st.expander("Retrieval diagnostics"):
        diagnostic_columns = st.columns(4)
        diagnostic_columns[0].metric("Chunks used", retrieval["retrieved_count"])
        diagnostic_columns[1].metric("Retrieval", f"{retrieval['retrieval_latency_ms']:.0f} ms")
        diagnostic_columns[2].metric("Context", f"{retrieval['context_characters']} chars")
        diagnostic_columns[3].metric("Below threshold", retrieval["dropped_below_threshold_count"])
        st.caption(
            f"Strategy `{retrieval['chunking_strategy']}` · top_k {retrieval['top_k']} · "
            f"minimum similarity {retrieval['min_similarity']} · "
            f"active versions only: {retrieval['restricted_to_active_versions']}"
        )
        if retrieval["active_version_ids"]:
            st.caption("Version filter: " + ", ".join(retrieval["active_version_ids"]))
        for chunk in retrieved_chunks or []:
            metadata = chunk["metadata"]
            st.markdown(
                f"**#{chunk['rank']}** `{metadata['source_name']}` "
                f"v{metadata['version_number']} · similarity {chunk['similarity']:.3f}"
            )
            st.code(chunk["text"][:600], language=None)


# ----------------------------------------------------------------------- pages


def _application_form() -> dict[str, Any] | None:
    """Render the form and return a validated payload, or None if not submitted."""
    with st.form("loan_application"):
        st.subheader("Applicant information")
        first_column, second_column = st.columns(2)
        with first_column:
            applicant_name = st.text_input("Applicant name", value="Asha Menon")
            age_years = st.number_input("Age", min_value=18, max_value=80, value=34, step=1)
            employment_type = st.selectbox(
                "Employment type",
                ["salaried", "self_employed", "professional", "retired"],
                format_func=lambda value: value.replace("_", " ").title(),
            )
            employment_experience_months = st.number_input(
                "Employment / business experience (months)",
                min_value=0, max_value=720, value=72, step=1,
            )
        with second_column:
            monthly_income_inr = st.number_input(
                "Net monthly income (₹)", min_value=1.0, value=150_000.0, step=5_000.0, format="%.0f"
            )
            credit_score = st.number_input(
                "Credit score (CIBIL)", min_value=300, max_value=900, value=760, step=1
            )
            number_of_existing_loans = st.number_input(
                "Number of existing loans", min_value=0, max_value=50, value=1, step=1
            )
            existing_monthly_emi_inr = st.number_input(
                "Existing monthly EMI (₹)", min_value=0.0, value=18_000.0, step=1_000.0, format="%.0f"
            )

        st.subheader("Loan request")
        third_column, fourth_column, fifth_column = st.columns(3)
        with third_column:
            loan_amount_required_inr = st.number_input(
                "Loan amount required (₹)", min_value=1.0, value=6_000_000.0,
                step=100_000.0, format="%.0f",
            )
        with fourth_column:
            property_value_inr = st.number_input(
                "Property value (₹)", min_value=1.0, value=8_000_000.0,
                step=100_000.0, format="%.0f",
            )
        with fifth_column:
            loan_tenure_years = st.number_input(
                "Loan tenure (years)", min_value=1, max_value=40, value=20, step=1
            )

        st.subheader("Co-applicant, property and liabilities")
        sixth_column, seventh_column = st.columns(2)
        with sixth_column:
            co_applicant_monthly_income_inr = st.number_input(
                "Co-applicant monthly income (₹)", min_value=0.0, value=0.0,
                step=5_000.0, format="%.0f",
            )
            co_applicant_is_joint_owner = st.checkbox(
                "Co-applicant is a joint owner of the property",
                help="Most policies only aggregate a co-applicant's income when they co-own the property.",
            )
            other_monthly_liabilities_inr = st.number_input(
                "Other monthly liabilities (₹)", min_value=0.0, value=0.0,
                step=1_000.0, format="%.0f",
            )
        with seventh_column:
            property_type = st.selectbox(
                "Property type",
                ["apartment", "independent_house", "plot_plus_construction", "under_construction"],
                format_func=lambda value: value.replace("_", " ").title(),
            )
            location_tier = st.selectbox(
                "Location", ["metro", "tier_1", "tier_2", "tier_3"],
                format_func=lambda value: value.replace("_", " ").title(),
            )

        with st.expander("Retrieval settings (for evaluation and demos)"):
            override_top_k = st.slider("top_k", min_value=1, max_value=15, value=5)
            show_chunks = st.checkbox("Show retrieved chunks", value=False)

        submitted = st.form_submit_button("Assess application", type="primary")

    if not submitted:
        return None

    # Client-side checks that give instant feedback. The API validates the same
    # rules again — this is convenience, not the trust boundary.
    validation_errors: list[str] = []
    if loan_amount_required_inr > property_value_inr:
        validation_errors.append("Loan amount cannot exceed the property value.")
    if existing_monthly_emi_inr > 0 and number_of_existing_loans == 0:
        validation_errors.append(
            "You entered an existing EMI but zero existing loans."
        )
    if co_applicant_is_joint_owner and co_applicant_monthly_income_inr <= 0:
        validation_errors.append(
            "A co-applicant marked as joint owner should have an income entered."
        )
    if validation_errors:
        for message in validation_errors:
            st.error(message)
        return None

    return {
        "application": {
            "applicant_name": applicant_name,
            "age_years": int(age_years),
            "employment_type": employment_type,
            "employment_experience_months": int(employment_experience_months),
            "monthly_income_inr": float(monthly_income_inr),
            "credit_score": int(credit_score),
            "existing_monthly_emi_inr": float(existing_monthly_emi_inr),
            "number_of_existing_loans": int(number_of_existing_loans),
            "other_monthly_liabilities_inr": float(other_monthly_liabilities_inr),
            "loan_amount_required_inr": float(loan_amount_required_inr),
            "property_value_inr": float(property_value_inr),
            "loan_tenure_years": int(loan_tenure_years),
            "co_applicant_monthly_income_inr": float(co_applicant_monthly_income_inr),
            "co_applicant_is_joint_owner": bool(co_applicant_is_joint_owner),
            "property_type": property_type,
            "location_tier": location_tier,
        },
        "top_k": int(override_top_k),
        "include_retrieved_chunks": bool(show_chunks),
    }


def _render_assessment_page() -> None:
    st.title("Home Loan Approval Assistant")
    st.caption(
        "Eligibility is decided by deterministic rules. The explanation is generated "
        "from retrieved policy documents and is always shown with its source and "
        "document version."
    )

    request_payload = _application_form()
    if request_payload is None:
        return

    with st.status("Assessing application…", expanded=False) as status:
        try:
            status.write("Applying eligibility rules and retrieving policy…")
            api_response = _call_api("/loan-assessment", request_payload)
            status.update(label="Assessment complete", state="complete")
        except RuntimeError as api_error:
            status.update(label="Assessment failed", state="error")
            st.error(str(api_error))
            return

    assessment = api_response["assessment"]
    st.divider()
    _render_decision_banner(assessment["decision"])
    _render_explanation(assessment["explanation"])
    _render_citations(assessment["explanation"])
    _render_rule_checks(assessment["rule_assessment"])
    _render_retrieval_diagnostics(
        assessment["retrieval"], api_response.get("retrieved_chunks")
    )
    st.caption(
        f"Request {assessment['request_id'][:8]} · "
        f"{assessment['total_latency_ms']:.0f} ms end to end"
    )


def _render_corpus_page() -> None:
    st.title("Policy corpus")
    st.caption("Which documents are indexed, and which version of each is currently active.")

    try:
        corpus_status = _call_api("/documents/status")
    except RuntimeError as api_error:
        st.error(str(api_error))
        return

    overview_columns = st.columns(3)
    overview_columns[0].metric("Active strategy", corpus_status["active_chunking_strategy"])
    overview_columns[1].metric("Indexed chunks", corpus_status["chunk_count_active_strategy"])
    overview_columns[2].metric("Embedding model", corpus_status["embedding_model"])

    for source in corpus_status["sources"]:
        header = f"{source['document_title']} — {source['institution']}"
        with st.expander(header, expanded=source["total_versions"] > 0):
            st.caption(
                f"`{source['source_name']}` · {source['document_type']} · "
                f"{source['authority']} source · "
                f"{'enabled' if source['enabled'] else 'disabled'}"
            )
            if source["active_version_number"] is None:
                st.info("Not yet ingested.")
            else:
                st.markdown(
                    f"**Active: version {source['active_version_number']}**"
                    + (
                        f" · effective {source['active_effective_date']}"
                        if source["active_effective_date"]
                        else ""
                    )
                )
                st.dataframe(source["version_history"], hide_index=True, width="stretch")
            st.markdown(f"[Source document]({source['url']})")

    if st.button("Check sources for updates now"):
        with st.spinner("Running ingestion…"):
            try:
                refresh_result = _call_api("/documents/refresh", {})
                st.success(refresh_result["summary"])
                st.rerun()
            except RuntimeError as api_error:
                st.error(str(api_error))


def _render_question_page() -> None:
    st.title("Ask the policy corpus")
    st.caption(
        "The same retrieval and generation path the assessment uses. Useful for "
        "checking what the documents actually say, including superseded versions."
    )

    question = st.text_input(
        "Question", value="What is the minimum CIBIL score required for a standard sanction?"
    )
    first_column, second_column = st.columns(2)
    top_k = first_column.slider("top_k", min_value=1, max_value=15, value=5)
    historical_version = second_column.number_input(
        "Pin Meridian policy to version (0 = current)", min_value=0, max_value=9, value=0
    )

    if not st.button("Ask", type="primary"):
        return

    payload: dict[str, Any] = {
        "question": question,
        "top_k": int(top_k),
        "include_retrieved_chunks": True,
    }
    if historical_version:
        payload["version_numbers_by_source"] = {
            "meridian_home_loan_policy": int(historical_version)
        }
        payload["restrict_to_active_versions"] = False

    with st.spinner("Retrieving and answering…"):
        try:
            api_response = _call_api("/policy-question", payload)
        except RuntimeError as api_error:
            st.error(str(api_error))
            return

    st.divider()
    if api_response["insufficient_information"]:
        st.warning(api_response["answer"])
    else:
        st.write(api_response["answer"])
    _render_citations({"citations": api_response["citations"]})
    _render_retrieval_diagnostics(api_response["retrieval"], api_response.get("retrieved_chunks"))


def main() -> None:
    st.set_page_config(page_title="Home Loan Approval Assistant", page_icon="🏠", layout="wide")

    with st.sidebar:
        st.header("Home Loan RAG")
        page = st.radio("View", ["Assess an application", "Policy corpus", "Ask a question"])
        st.divider()
        try:
            health = _call_api("/health")
            st.success(f"Backend: {health['status']}")
            st.caption(
                f"{health['llm_provider']}/{health['llm_model']} · "
                f"{health['indexed_chunk_count']} chunks indexed"
            )
            for warning in health["warnings"]:
                st.warning(warning, icon="⚠️")
        except RuntimeError as api_error:
            st.error(str(api_error))
        st.divider()
        st.caption(
            "This is a demonstration system. It does not make binding credit "
            "decisions, and the Meridian policy in the corpus is illustrative."
        )

    if page == "Assess an application":
        _render_assessment_page()
    elif page == "Policy corpus":
        _render_corpus_page()
    else:
        _render_question_page()


if __name__ == "__main__":
    main()
