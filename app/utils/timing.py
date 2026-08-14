"""Latency measurement.

Latency is a first-class metric here — retrieval and generation timings feed the
experiment comparison table — so it is measured with a monotonic clock and
reported in milliseconds everywhere, with no ad-hoc ``time.time()`` deltas
scattered through the pipeline.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Stopwatch:
    """Elapsed-time holder filled in by :func:`measure_latency`."""

    elapsed_ms: float = field(default=0.0)


@contextmanager
def measure_latency() -> Iterator[Stopwatch]:
    """Time a block with a monotonic clock, unaffected by system clock changes."""
    stopwatch = Stopwatch()
    started_at = time.perf_counter()
    try:
        yield stopwatch
    finally:
        stopwatch.elapsed_ms = (time.perf_counter() - started_at) * 1000.0
