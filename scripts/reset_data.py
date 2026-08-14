#!/usr/bin/env python3
"""Delete local state so ingestion can start from nothing.

    python scripts/reset_data.py --dry-run
    python scripts/reset_data.py --yes

Destructive, so it refuses to run without ``--yes``. Version history is the part
that actually matters: deleting it discards the audit trail of which policy
version answered which question, which is exactly what the versioning design
exists to preserve.
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

from app.core.config import get_settings
from app.core.logging_config import configure_logging


def _describe(path: Path) -> str:
    if not path.exists():
        return "absent"
    file_count = sum(1 for _ in path.rglob("*") if _.is_file())
    return f"{file_count} file(s)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset local data directories")
    parser.add_argument("--yes", action="store_true", help="actually delete")
    parser.add_argument("--dry-run", action="store_true", help="show what would be deleted")
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="keep downloaded originals so a rebuild needs no network",
    )
    arguments = parser.parse_args()

    configure_logging("INFO")
    paths = get_settings().paths

    targets = [paths.chroma_dir, paths.metadata_dir, paths.processed_dir]
    if not arguments.keep_raw:
        targets.append(paths.raw_dir)

    for target in targets:
        print(f"{target}: {_describe(target)}")

    if arguments.dry_run or not arguments.yes:
        print("\nNothing deleted. Re-run with --yes to confirm.")
        return 0

    for target in targets:
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
    print("\nLocal data reset. Run 'python scripts/ingest.py' to rebuild.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
