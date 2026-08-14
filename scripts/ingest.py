#!/usr/bin/env python3
"""Run document ingestion once and print what happened.

    python scripts/ingest.py
    python scripts/ingest.py --source meridian_home_loan_policy
    python scripts/ingest.py --strategy recursive_800_100 --strategy semantic
    python scripts/ingest.py --all-strategies          # build every experiment index
    python scripts/ingest.py --force                   # re-embed unchanged documents
"""

from __future__ import annotations

import argparse
import sys

# Running this file directly puts its own directory on sys.path rather than the
# repository root, which would hide the "app" package. `pip install -e .` makes
# this redundant but never harmful; keeping it means a fresh clone runs with no
# install step at all.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from app.core.config import get_chunking_config, get_settings
from app.core.logging_config import configure_logging, get_logger
from app.core.tracing import configure_tracing
from app.services.container import get_container
from app.services.document_service import DocumentService

_logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest registered policy documents")
    parser.add_argument("--source", action="append", dest="sources", default=None)
    parser.add_argument("--strategy", action="append", dest="strategies", default=None)
    parser.add_argument(
        "--all-strategies",
        action="store_true",
        help="index every strategy in config/chunking.yaml (needed before the chunking experiment)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-chunk and re-embed even when documents are unchanged",
    )
    arguments = parser.parse_args()

    load_dotenv()
    settings = get_settings()
    configure_logging(settings.app.log_level)
    configure_tracing(settings.observability.langsmith_project)

    strategies = arguments.strategies
    if arguments.all_strategies:
        strategies = sorted(get_chunking_config().strategies)

    report = DocumentService(get_container()).refresh(
        source_names=arguments.sources,
        strategy_names=strategies,
        force_reindex=arguments.force,
    )

    print("\nIngestion report")
    print("-" * 78)
    for result in report.results:
        chunk_summary = (
            ", ".join(f"{strategy}={count}" for strategy, count in sorted(result.chunks_written.items()))
            or "-"
        )
        version_label = f"v{result.version_number}" if result.version_number else "-"
        print(
            f"{result.source_name:<44} {result.outcome:<10} {version_label:<5} {chunk_summary}"
        )
        if result.outcome == "failed":
            print(f"{'':<44} reason: {result.detail}")
    print("-" * 78)
    print(report.summary())

    # Non-zero only if nothing at all succeeded: one unreachable public website
    # should not fail an otherwise healthy ingest in CI.
    everything_failed = report.count_with_outcome("failed") == len(report.results)
    return 1 if everything_failed and report.results else 0


if __name__ == "__main__":
    sys.exit(main())
