#!/usr/bin/env python3
"""Promote the controlled local policy to a different version.

Version 1 is what the repository ships with. This script copies one of the
archived versions over the working copy that the registry points at, so the next
ingestion run sees a genuinely changed document and creates version 2 or 3
through the ordinary change-detection path. Nothing about versioning is faked or
special-cased for the demo.

    python scripts/simulate_policy_update.py --version 2
    python scripts/ingest.py
    python scripts/simulate_policy_update.py --status
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Running this file directly puts its own directory on sys.path rather than the
# repository root, which would hide the "app" package. `pip install -e .` makes
# this redundant but never harmful; keeping it means a fresh clone runs with no
# install step at all.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import PROJECT_ROOT, get_settings
from app.core.logging_config import configure_logging, get_logger
from app.ingestion.versioning import VersionStore

_logger = get_logger(__name__)

ARCHIVE_DIR = PROJECT_ROOT / "documents" / "local_policies" / "versions"
WORKING_COPY = (
    PROJECT_ROOT / "documents" / "local_policies" / "current" / "meridian_home_loan_policy.md"
)
SOURCE_NAME = "meridian_home_loan_policy"


def _archive_path_for(version_number: int) -> Path:
    return ARCHIVE_DIR / f"meridian_home_loan_policy_v{version_number}.md"


def _print_status() -> None:
    settings = get_settings()
    version_store = VersionStore(settings.paths.metadata_dir)
    versions = version_store.versions_for(SOURCE_NAME)
    if not versions:
        print(f"No versions of '{SOURCE_NAME}' have been ingested yet.")
        return

    print(f"\nVersion history for '{SOURCE_NAME}'")
    print("-" * 78)
    print(f"{'Version':<9}{'Active':<8}{'Effective':<13}{'Declared':<10}{'Chunks'}")
    for version in versions:
        chunk_summary = ", ".join(
            f"{strategy}={count}" for strategy, count in sorted(version.chunk_counts.items())
        )
        print(
            f"{version.version_number:<9}"
            f"{'yes' if version.is_active else 'no':<8}"
            f"{version.effective_date or '-'!s:<13}"
            f"{version.declared_version or '-'!s:<10}"
            f"{chunk_summary}"
        )
    print("-" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote the local policy to another version")
    parser.add_argument("--version", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--status", action="store_true", help="show ingested version history")
    arguments = parser.parse_args()

    configure_logging("INFO")

    if arguments.status or arguments.version is None:
        _print_status()
        if arguments.version is None and not arguments.status:
            parser.print_help()
        return 0

    archive_path = _archive_path_for(arguments.version)
    if not archive_path.exists():
        print(f"error: {archive_path} does not exist", file=sys.stderr)
        return 1

    WORKING_COPY.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(archive_path, WORKING_COPY)
    print(f"Working copy now holds version {arguments.version} ({archive_path.name}).")
    print("Run 'python scripts/ingest.py' to let the pipeline detect and index it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
