"""Evaluation metrics and the golden-dataset harness."""

from __future__ import annotations

import pytest

from app.core.exceptions import EvaluationError
from app.models.assessment import Citation
from app.models.documents import ChunkMetadata, RetrievedChunk
from app.services.container import ApplicationContainer
from evaluation import metrics
from evaluation.dataset import GoldenDataset
from evaluation.runner import EvaluationRunner


def _chunk(text: str, version_number: int = 3, source: str = "meridian_home_loan_policy") -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        similarity=0.8,
        rank=1,
        metadata=ChunkMetadata(
            chunk_id=f"{source}::v{version_number}::s::00000",
            chunk_index=0,
            source_name=source,
            version_id=f"{source}::v{version_number}",
            version_number=version_number,
            institution="Meridian",
            document_title="Meridian Retail Home Loan Credit Policy",
            document_type="credit_policy",
            authority="primary",
            url="file://x",
            chunking_strategy="recursive_800_100",
            embedding_model="deterministic-hashing-256",
            ingested_at="2026-08-14T00:00:00+00:00",
            character_count=len(text),
        ),
    )


def _citation(version_number: int, source: str = "meridian_home_loan_policy") -> Citation:
    return Citation(
        marker="S1",
        source_name=source,
        institution="Meridian",
        document_title="Meridian Retail Home Loan Credit Policy",
        version_number=version_number,
        version_label=f"Version {version_number}",
        url="file://x",
        excerpt="…",
        similarity=0.8,
    )


# ------------------------------------------------------------------- retrieval


def test_relevance_requires_source_version_and_content() -> None:
    keywords = ["725"]
    right = _chunk("The minimum acceptable CIBIL score is 725.")
    wrong_version = _chunk("The minimum acceptable CIBIL score is 700.", version_number=1)
    wrong_content = _chunk("The processing fee is 0.40 percent.")

    assert metrics.is_chunk_relevant(right, "meridian_home_loan_policy", 3, keywords)
    assert not metrics.is_chunk_relevant(wrong_version, "meridian_home_loan_policy", 3, keywords)
    assert not metrics.is_chunk_relevant(wrong_content, "meridian_home_loan_policy", 3, keywords)


def test_precision_recall_mrr_and_ndcg() -> None:
    assert metrics.context_precision([True, False, True, False]) == 0.5
    assert metrics.reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)
    assert metrics.reciprocal_rank([False, False, False]) == 0.0
    # A relevant chunk first must score better than the same chunk last.
    assert metrics.normalised_discounted_cumulative_gain(
        [True, False, False]
    ) > metrics.normalised_discounted_cumulative_gain([False, False, True])


def test_recall_tolerates_digit_grouping() -> None:
    """'INR 15,000' in the policy must match a '15000' expectation and vice versa."""
    chunks = [_chunk("The processing fee is capped at INR 15,000.")]
    assert metrics.context_recall(chunks, ["15000"]) == 1.0
    assert metrics.context_recall(chunks, ["nothing here"]) == 0.0


# ------------------------------------------------------------------ generation


def test_forbidden_pattern_catches_a_stale_figure() -> None:
    """An answer that also quotes the superseded number is not correct."""
    assert metrics.answer_matches_expectation("The score is 725.", [r"\b725\b"], [r"\b700\b"], [])
    assert not metrics.answer_matches_expectation(
        "The score was 700 and is now 725.", [r"\b725\b"], [r"\b700\b"], []
    )


def test_lexical_faithfulness_separates_supported_from_invented() -> None:
    chunks = [_chunk("The maximum permitted FOIR is 50 percent for all applicants.")]
    supported = metrics.lexical_faithfulness("The maximum permitted FOIR is 50 percent.", chunks)
    invented = metrics.lexical_faithfulness(
        "Applicants receive a complimentary insurance rider worth two lakh rupees.", chunks
    )
    assert supported > invented


def test_no_context_means_zero_faithfulness() -> None:
    assert metrics.lexical_faithfulness("anything at all", []) == 0.0


# ------------------------------------------------------------------- citations


def test_version_correctness_is_strict() -> None:
    assert metrics.version_correct([_citation(3)], "meridian_home_loan_policy", 3)
    assert not metrics.version_correct(
        [_citation(3), _citation(1)], "meridian_home_loan_policy", 3
    )
    assert not metrics.version_correct([], "meridian_home_loan_policy", 3)


def test_citation_correctness_checks_the_document() -> None:
    assert metrics.citation_correct([_citation(3)], "meridian_home_loan_policy")
    assert not metrics.citation_correct([_citation(3, source="other")], "meridian_home_loan_policy")


# --------------------------------------------------------------------- dataset


def test_the_shipped_golden_dataset_is_valid() -> None:
    dataset = GoldenDataset.load()
    assert len(dataset.cases) == 10
    assert dataset.reproducibility.repeat_runs >= 2

    for case in dataset.cases:
        assert case.expected_context_keywords, f"{case.case_id} has no gradable keywords"
        assert case.expected_answer_patterns or case.acceptable_answer_variations
        assert case.expected_document_version >= 1

    historical_cases = [case for case in dataset.cases if case.is_historical_lookup]
    assert historical_cases, "the suite must cover retrieval from a superseded version"


# --------------------------------------------------------------------- harness


def test_precondition_blocks_a_run_against_the_wrong_corpus_state(
    ingested_container: ApplicationContainer,
) -> None:
    """Only version 1 is ingested here; the dataset grades against version 3."""
    runner = EvaluationRunner(ingested_container, GoldenDataset.load())
    problems = runner.check_precondition()
    assert problems

    with pytest.raises(EvaluationError, match="not in the state"):
        runner.run()


def test_the_harness_executes_and_scores_every_case(
    ingested_container: ApplicationContainer,
) -> None:
    """Structure and reproducibility, not answer quality: the LLM here is a stub."""
    dataset = GoldenDataset.load()
    runner = EvaluationRunner(ingested_container, dataset)

    evaluation_run = runner.run(repeat_runs=2, enforce_precondition=False)

    assert evaluation_run.aggregate.case_count == len(dataset.cases)
    assert evaluation_run.aggregate.execution_count == len(dataset.cases) * 2
    assert evaluation_run.warnings, "a mismatched corpus state must be recorded as a warning"
    assert any("offline stub" in note for note in evaluation_run.aggregate.notes)

    consistency = evaluation_run.aggregate.consistency
    assert consistency is not None
    # With a fixed corpus, fixed retrieval order and a deterministic provider,
    # repeated executions must agree completely. Anything less is a bug in the
    # reproducibility machinery rather than a property of the model.
    assert consistency.retrieval_consistency == 1.0
    assert consistency.answer_consistency == 1.0


def test_run_results_serialise_for_the_report(ingested_container: ApplicationContainer) -> None:
    from evaluation.report import render_run_summary

    runner = EvaluationRunner(ingested_container, GoldenDataset.load())
    evaluation_run = runner.run(repeat_runs=1, enforce_precondition=False)

    serialised = evaluation_run.to_dict()
    assert serialised["aggregate"]["case_count"] == 10
    assert "Evaluation run" in render_run_summary(evaluation_run)
