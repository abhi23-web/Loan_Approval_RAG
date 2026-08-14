"""Shared test fixtures.

Every test runs fully offline. The deterministic embedding and LLM providers make
the pipeline exercisable with no Ollama server, and each test gets its own
temporary data directory so runs cannot contaminate each other or the developer's
real index.

The one thing tests do share with production is the corpus: they ingest the
committed Meridian policy fixture through the real pipeline. Testing chunking and
retrieval against a synthetic in-memory string would pass while the actual
extraction path was broken.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from app.core.config import (
    ChunkingConfig,
    PathsSection,
    Settings,
    load_chunking_config,
    load_settings,
)
from app.core.logging_config import configure_logging
from app.ingestion.registry import SourceRegistry
from app.services import container as container_module
from app.services.container import ApplicationContainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# A registry containing only the controlled local fixture: tests must never
# depend on a bank's website being up.
#
# It points at the *archived* version 1 rather than the working copy under
# documents/local_policies/current/. The working copy is what
# scripts/simulate_policy_update.py rewrites during a demo, so tests that read it
# would pass or fail depending on which demo step someone last ran.
_LOCAL_ONLY_REGISTRY = """
sources:
  - source_name: meridian_home_loan_policy
    url: file://documents/local_policies/versions/meridian_home_loan_policy_v1.md
    institution: Meridian Housing Finance Limited (illustrative)
    document_type: credit_policy
    document_title: Meridian Retail Home Loan Credit Policy
    version: null
    effective_date: null
    last_checked: null
    enabled: true
    authority: primary
"""


@pytest.fixture(scope="session", autouse=True)
def _quiet_logging() -> None:
    configure_logging("WARNING", force=True)


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Real settings, redirected to a temporary directory and offline providers."""
    settings = load_settings()
    return settings.model_copy(
        update={
            "paths": PathsSection(
                raw_dir=tmp_path / "raw",
                processed_dir=tmp_path / "processed",
                chroma_dir=tmp_path / "chroma",
                metadata_dir=tmp_path / "metadata",
                experiment_results_dir=tmp_path / "results",
            ),
            "llm": settings.llm.model_copy(update={"provider": "deterministic"}),
            "embeddings": settings.embeddings.model_copy(update={"provider": "deterministic"}),
        }
    )


@pytest.fixture
def chunking_config() -> ChunkingConfig:
    return load_chunking_config()


@pytest.fixture
def local_registry(tmp_path: Path) -> SourceRegistry:
    registry_path = tmp_path / "source_registry.yaml"
    registry_path.write_text(_LOCAL_ONLY_REGISTRY, encoding="utf-8")
    return SourceRegistry.load(registry_path)


@pytest.fixture
def container(
    test_settings: Settings, chunking_config: ChunkingConfig, local_registry: SourceRegistry
) -> Iterator[ApplicationContainer]:
    """A container wired for tests, installed as the process singleton.

    Installing it globally is what lets the FastAPI tests exercise the real
    dependency graph instead of a parallel one built just for tests.
    """
    application_container = ApplicationContainer(test_settings, chunking_config)
    # cached_property is backed by the instance __dict__, so assigning here
    # pre-populates the cache with the test registry.
    application_container.__dict__["registry"] = local_registry

    previous_container = container_module._container
    container_module._container = application_container
    try:
        yield application_container
    finally:
        container_module._container = previous_container


@pytest.fixture
def ingested_container(container: ApplicationContainer) -> ApplicationContainer:
    """A container whose corpus already holds version 1 of the local policy."""
    report = container.ingestion_pipeline.run()
    assert report.count_with_outcome("indexed") == 1, report.summary()
    return container
