"""The long-running document update process.

Requirements this satisfies, and how:

* **Not a busy loop.** The interval is waited on a ``threading.Event``, so the
  process consumes no CPU between polls and wakes instantly on a shutdown signal.
* **Graceful shutdown.** SIGINT and SIGTERM set the stop event. An ingestion run
  already in flight finishes and saves its version store before the process
  exits, so a container restart cannot corrupt version history.
* **No unnecessary work.** The pipeline's own three change gates mean a poll over
  unchanged sources costs one conditional HTTP request each and no embedding.
* **Failure containment.** Consecutive whole-run failures are counted; the
  process exits non-zero once the configured limit is reached so a supervisor can
  restart or alert, rather than looping silently forever.
* **Jitter.** A small random offset keeps several instances, or a restart loop,
  from synchronising their polls onto the same second at the origin.
"""

from __future__ import annotations

import random
import signal
import threading
from types import FrameType

from app.core.config import WatcherSection
from app.core.logging_config import get_logger
from app.services.container import ApplicationContainer
from app.services.document_service import DocumentService

_logger = get_logger(__name__)


class DocumentUpdateWatcher:
    """Polls the registry on an interval and re-indexes what changed."""

    def __init__(self, container: ApplicationContainer, settings: WatcherSection) -> None:
        self._container = container
        self._settings = settings
        self._document_service = DocumentService(container)
        self._stop_event = threading.Event()
        self._consecutive_failures = 0

    def request_stop(self) -> None:
        self._stop_event.set()

    def install_signal_handlers(self) -> None:
        def _handle_signal(signal_number: int, _frame: FrameType | None) -> None:
            _logger.info(
                "received signal %s; finishing the current cycle and stopping",
                signal.Signals(signal_number).name,
            )
            self.request_stop()

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

    def run_forever(self) -> int:
        """Poll until stopped. Returns a process exit code."""
        _logger.info(
            "watcher started: interval=%ss jitter=±%ss sources=%d strategy='%s'",
            self._settings.poll_interval_seconds,
            self._settings.jitter_seconds,
            len(self._container.registry.enabled_sources()),
            self._container.settings.chunking.active_strategy,
        )

        if self._settings.run_on_start:
            self._run_one_cycle()

        while not self._stop_event.is_set():
            if self._consecutive_failures >= self._settings.max_consecutive_failures:
                _logger.critical(
                    "%d consecutive failed cycles; exiting for the supervisor to handle",
                    self._consecutive_failures,
                )
                return 1
            if self._stop_event.wait(self._next_sleep_seconds()):
                break
            self._run_one_cycle()

        _logger.info("watcher stopped cleanly")
        return 0

    def _next_sleep_seconds(self) -> float:
        jitter = self._settings.jitter_seconds
        offset = random.uniform(-jitter, jitter) if jitter else 0.0
        # Never sleep less than a second, however the jitter lands.
        return max(1.0, self._settings.poll_interval_seconds + offset)

    def _run_one_cycle(self) -> None:
        try:
            report = self._document_service.refresh()
        except Exception as cycle_error:
            # A whole-cycle failure is infrastructure (disk, ChromaDB, config).
            # Per-source failures are already handled inside the pipeline.
            self._consecutive_failures += 1
            _logger.exception(
                "ingestion cycle failed (%d consecutive): %s",
                self._consecutive_failures,
                cycle_error,
            )
            return

        self._consecutive_failures = 0
        if report.has_changes:
            _logger.info("corpus updated: %s", report.summary())
        else:
            _logger.info("no document changes: %s", report.summary())
