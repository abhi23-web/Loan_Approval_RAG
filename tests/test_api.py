"""API contract tests.

They exercise the real dependency graph through FastAPI's TestClient, so a
service-layer change that breaks the HTTP contract fails here rather than in the
browser.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.services.container import ApplicationContainer

API = "/api/v1"

_VALID_APPLICATION = {
    "applicant_name": "Asha Menon",
    "age_years": 34,
    "employment_type": "salaried",
    "employment_experience_months": 72,
    "monthly_income_inr": 200000.0,
    "credit_score": 780,
    "existing_monthly_emi_inr": 0.0,
    "number_of_existing_loans": 0,
    "loan_amount_required_inr": 4000000.0,
    "property_value_inr": 8000000.0,
    "loan_tenure_years": 20,
}


@pytest.fixture
def client(ingested_container: ApplicationContainer) -> TestClient:
    # The container fixture has already been installed as the process singleton,
    # so the app's dependencies resolve to the test wiring.
    return TestClient(create_app())


@pytest.fixture
def client_without_index(container: ApplicationContainer) -> TestClient:
    return TestClient(create_app())


def test_health_reports_configuration_and_warnings(client: TestClient) -> None:
    response = client.get(f"{API}/health")
    assert response.status_code == 200

    body = response.json()
    assert body["indexed_chunk_count"] > 0
    assert body["llm_provider"] == "deterministic"
    # The offline stub must never be reported as a healthy production setup.
    assert any("offline stub" in warning for warning in body["warnings"])
    assert body["status"] == "degraded"


def test_loan_assessment_returns_a_decision_and_a_grounded_explanation(
    client: TestClient,
) -> None:
    response = client.post(
        f"{API}/loan-assessment",
        json={"application": _VALID_APPLICATION, "include_retrieved_chunks": True},
    )
    assert response.status_code == 200

    body = response.json()["assessment"]
    assert body["decision"] in {"ELIGIBLE", "CONDITIONALLY_ELIGIBLE", "NOT_ELIGIBLE"}
    assert body["rule_assessment"]["checks"], "the decision must show its working"
    assert body["retrieval"]["retrieved_count"] > 0
    assert response.json()["retrieved_chunks"], "chunks were requested and must be returned"


def test_the_decision_comes_from_the_rules_not_the_model(client: TestClient) -> None:
    """A rejected applicant must be rejected regardless of what the model writes."""
    rejected_application = {**_VALID_APPLICATION, "credit_score": 590}
    response = client.post(f"{API}/loan-assessment", json={"application": rejected_application})

    body = response.json()["assessment"]
    assert body["decision"] == "NOT_ELIGIBLE"
    assert any(
        check["check_name"] == "credit_score" and check["outcome"] == "FAIL"
        for check in body["rule_assessment"]["checks"]
    )


def test_invalid_application_is_rejected_with_422(client: TestClient) -> None:
    invalid_application = {**_VALID_APPLICATION, "loan_amount_required_inr": 90_000_000.0}
    response = client.post(f"{API}/loan-assessment", json={"application": invalid_application})
    assert response.status_code == 422


def test_unknown_fields_are_rejected(client: TestClient) -> None:
    """extra='forbid' turns a typo in a client into an error instead of a silent default."""
    response = client.post(
        f"{API}/loan-assessment",
        json={"application": {**_VALID_APPLICATION, "annual_bonus": 100000}},
    )
    assert response.status_code == 422


def test_policy_question_endpoint_answers_from_the_corpus(client: TestClient) -> None:
    response = client.post(
        f"{API}/policy-question",
        json={"question": "What is the minimum CIBIL score?", "include_retrieved_chunks": True},
    )
    assert response.status_code == 200

    body = response.json()
    assert body["retrieval"]["retrieved_count"] > 0
    assert body["retrieved_chunks"]


def test_document_status_lists_versions(client: TestClient) -> None:
    response = client.get(f"{API}/documents/status")
    assert response.status_code == 200

    body = response.json()
    assert body["chunk_count_active_strategy"] > 0
    source_status = body["sources"][0]
    assert source_status["active_version_number"] == 1
    assert source_status["version_history"][0]["is_active"] is True


def test_document_refresh_is_a_no_op_when_nothing_changed(client: TestClient) -> None:
    response = client.post(f"{API}/documents/refresh", json={})
    assert response.status_code == 200
    assert "unchanged" in response.json()["summary"]


def test_an_empty_index_returns_503_not_a_confident_answer(
    client_without_index: TestClient,
) -> None:
    """An un-ingested system must fail loudly, not answer 'insufficient information'."""
    response = client_without_index.post(
        f"{API}/loan-assessment", json={"application": _VALID_APPLICATION}
    )
    assert response.status_code == 503
    assert response.json()["error"] == "KnowledgeBaseEmptyError"
    assert "scripts/ingest.py" in response.json()["detail"]
