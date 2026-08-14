#!/usr/bin/env python3
"""Run the long-lived document update watcher.

    python scripts/run_watcher.py
    python scripts/run_watcher.py --interval 60      # override the poll interval

Stop it with Ctrl-C; the current cycle finishes and version history is saved
before the process exits.
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

from app.core.config import get_settings
from app.core.logging_config import configure_logging
from app.core.tracing import configure_tracing
from app.services.container import get_container
from app.watcher.monitor import DocumentUpdateWatcher


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch registered sources for updates")
    parser.add_argument("--interval", type=int, default=None, help="seconds between polls")
    parser.add_argument(
        "--no-run-on-start",
        action="store_true",
        help="wait one full interval before the first poll",
    )
    arguments = parser.parse_args()

    load_dotenv()
    settings = get_settings()
    configure_logging(settings.app.log_level)
    configure_tracing(settings.observability.langsmith_project)

    watcher_settings = settings.watcher.model_copy(
        update={
            "poll_interval_seconds": arguments.interval or settings.watcher.poll_interval_seconds,
            "run_on_start": not arguments.no_run_on_start,
        }
    )

    watcher = DocumentUpdateWatcher(get_container(), watcher_settings)
    watcher.install_signal_handlers()
    return watcher.run_forever()


if __name__ == "__main__":
    sys.exit(main())
