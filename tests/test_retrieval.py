"""Ingestion, ChromaDB persistence and version-aware retrieval.

Integration tests: they run the real pipeline over the real policy fixture, with
offline embedding and generation providers. What they prove is the part that
cannot be proven by unit tests — that metadata written at ingestion time is
still there and still filterable at retrieval time.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from app.core.config import PROJECT_ROOT
from app.core.exceptions import VectorStoreError
from app.ingestion.registry import SourceRegistry
from app.rag.retriever import RetrievalRequest
from app.services.container import ApplicationContainer

ARCHIVE_DIR = PROJECT_ROOT / "documents" / "local_policies" / "versions"
# Lives under data/ so it is gitignored; file:// sources must resolve inside the
# repository, which rules out pytest's tmp_path for the document itself.
MUTABLE_FIXTURE_DIR = PROJECT_ROOT / "data" / "test_fixtures"

_MUTABLE_REGISTRY_TEMPLATE = """
sources:
  - source_name: meridian_home_loan_policy
    url: file://data/test_fixtures/{filename}
    institution: Meridian Housing Finance Limited (illustrative)
    document_type: credit_policy
    document_title: Meridian Retail Home Loan Credit Policy
    version: null
    effective_date: null
    last_checked: null
    enabled: true
    authority: primary
"""


@pytest.fixture
def mutable_policy(tmp_path: Path) -> Iterator[tuple[Path, SourceRegistry]]:
    """A policy document the test can rewrite, plus a registry pointing at it."""
    MUTABLE_FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    document_path = MUTABLE_FIXTURE_DIR / f"policy_{tmp_path.name}.md"
    shutil.copyfile(ARCHIVE_DIR / "meridian_home_loan_policy_v1.md", document_path)

    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        _MUTABLE_REGISTRY_TEMPLATE.format(filename=document_path.name), encoding="utf-8"
    )
    try:
        yield document_path, SourceRegistry.load(registry_path)
    finally:
        document_path.unlink(missing_ok=True)


def test_ingestion_indexes_the_policy_with_full_metadata(
    ingested_container: ApplicationContainer,
) -> None:
    settings = ingested_container.settings
    strategy = settings.chunking.active_strategy

    assert ingested_container.vector_store.count(strategy) > 0

    version = ingested_container.version_store.active_version("meridian_home_loan_policy")
    assert version is not None
    assert version.version_number == 1
    assert version.chunk_counts[strategy] > 0
    assert version.effective_date is not None, "the effective date must be parsed from the document"


def test_repeat_ingestion_does_no_work(ingested_container: ApplicationContainer) -> None:
    """The three change gates must make a second run a no-op."""
    second_report = ingested_container.ingestion_pipeline.run()
    assert second_report.count_with_outcome("indexed") == 0
    assert second_report.count_with_outcome("unchanged") == 1


def test_retrieval_finds_the_clause_that_answers_the_question(
    ingested_container: ApplicationContainer,
) -> None:
    outcome = ingested_container.retriever.retrieve(
        RetrievalRequest(query="minimum acceptable CIBIL score for a standard sanction")
    )
    assert outcome.chunks
    assert any("700" in chunk.text for chunk in outcome.chunks)
    assert outcome.diagnostics.retrieved_count == len(outcome.chunks)


def test_retrieval_ordering_is_reproducible(
    ingested_container: ApplicationContainer,
) -> None:
    """Same query, same knowledge state, same ordered chunk ids — every time."""
    request = RetrievalRequest(query="maximum loan to value ratio")
    first = ingested_container.retriever.retrieve(request)
    second = ingested_container.retriever.retrieve(request)

    assert [chunk.metadata.chunk_id for chunk in first.chunks] == [
        chunk.metadata.chunk_id for chunk in second.chunks
    ]


def test_retrieval_respects_the_similarity_threshold(
    ingested_container: ApplicationContainer,
) -> None:
    outcome = ingested_container.retriever.retrieve(
        RetrievalRequest(query="quantum chromodynamics lattice gauge theory", min_similarity=0.95)
    )
    assert outcome.chunks == []
    assert outcome.diagnostics.dropped_below_threshold_count > 0


def test_a_new_version_supersedes_the_old_one_for_current_questions(
    container: ApplicationContainer, mutable_policy: tuple[Path, SourceRegistry]
) -> None:
    document_path, registry = mutable_policy
    container.__dict__["registry"] = registry
    strategy = container.settings.chunking.active_strategy

    container.ingestion_pipeline.run()
    shutil.copyfile(ARCHIVE_DIR / "meridian_home_loan_policy_v2.md", document_path)
    second_report = container.ingestion_pipeline.run()

    assert second_report.count_with_outcome("indexed") == 1
    versions = container.version_store.versions_for("meridian_home_loan_policy")
    assert [version.version_number for version in versions] == [1, 2]
    assert container.version_store.active_version("meridian_home_loan_policy").version_number == 2

    # Version 1's chunks are still in ChromaDB — retained, not overwritten.
    all_chunks = container.vector_store.collection_for(strategy).get(
        where={"version_id": "meridian_home_loan_policy::v1"}
    )
    assert all_chunks["ids"], "the superseded version's chunks must be retained"

    current_answer = container.retriever.retrieve(
        RetrievalRequest(query="minimum acceptable CIBIL score for a standard sanction")
    )
    assert current_answer.chunks
    assert all(chunk.metadata.version_number == 2 for chunk in current_answer.chunks)
    assert any("720" in chunk.text for chunk in current_answer.chunks)


def test_a_historical_lookup_reaches_the_superseded_version(
    container: ApplicationContainer, mutable_policy: tuple[Path, SourceRegistry]
) -> None:
    document_path, registry = mutable_policy
    container.__dict__["registry"] = registry

    container.ingestion_pipeline.run()
    shutil.copyfile(ARCHIVE_DIR / "meridian_home_loan_policy_v2.md", document_path)
    container.ingestion_pipeline.run()

    historical = container.retriever.retrieve(
        RetrievalRequest(
            query="minimum acceptable CIBIL score for a standard sanction",
            version_numbers_by_source={"meridian_home_loan_policy": 1},
            restrict_to_active_versions=False,
        )
    )
    assert historical.chunks
    assert all(chunk.metadata.version_number == 1 for chunk in historical.chunks)
    assert any("700" in chunk.text for chunk in historical.chunks)


def test_per_source_cap_limits_one_document_dominating(
    ingested_container: ApplicationContainer,
) -> None:
    cap = ingested_container.settings.retrieval.max_chunks_per_source
    outcome = ingested_container.retriever.retrieve(
        RetrievalRequest(query="loan policy clause requirement", top_k=10)
    )
    assert len(outcome.chunks) <= cap


def test_vector_store_rejects_mismatched_embeddings(
    container: ApplicationContainer,
) -> None:
    with pytest.raises(VectorStoreError, match="refusing to write"):
        container.vector_store.upsert_chunks("recursive_800_100", [], [[0.1, 0.2]])


def test_index_survives_a_new_process(ingested_container: ApplicationContainer) -> None:
    """A restart must not require re-embedding: that is why Chroma persists."""
    from app.ingestion.vector_store import ChromaVectorStore

    settings = ingested_container.settings
    reopened = ChromaVectorStore(
        persist_directory=settings.paths.chroma_dir,
        collection_prefix=settings.vector_store.collection_prefix,
        distance=settings.vector_store.distance,
    )
    assert reopened.count(settings.chunking.active_strategy) > 0
