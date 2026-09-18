"""Timing helpers for the benchmark suite.

Benchmarks on a developer laptop measure the laptop as much as the code, so the
budgets asserted here are deliberately loose - roughly an order of magnitude
above what the system actually takes. They are there to catch a change that makes
something accidentally quadratic, not to police milliseconds. A tight budget on
shared or virtualised hardware produces a suite that fails for reasons nobody can
act on, and a suite that fails for no reason is a suite people learn to ignore.

Medians are reported rather than means. One scheduler hiccup should not decide
whether a benchmark passes, and the p95 is recorded alongside so a genuinely
erratic operation is still visible.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class Timing:
    """The measured cost of one operation."""
    label: str
    samples: List[float] = field(default_factory=list)
    unit: str = "ms"

    @property
    def runs(self) -> int:
        return len(self.samples)

    @property
    def median_ms(self) -> float:
        return round(statistics.median(self.samples) * 1000, 3) if self.samples else 0.0

    @property
    def mean_ms(self) -> float:
        return round(statistics.fmean(self.samples) * 1000, 3) if self.samples else 0.0

    @property
    def min_ms(self) -> float:
        return round(min(self.samples) * 1000, 3) if self.samples else 0.0

    @property
    def max_ms(self) -> float:
        return round(max(self.samples) * 1000, 3) if self.samples else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        return round(ordered[index] * 1000, 3)

    @property
    def total_ms(self) -> float:
        return round(sum(self.samples) * 1000, 3)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "runs": self.runs,
            "median_ms": self.median_ms,
            "mean_ms": self.mean_ms,
            "p95_ms": self.p95_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "total_ms": self.total_ms,
        }

    def __str__(self) -> str:
        return "{}: median {:.1f} ms, p95 {:.1f} ms over {} run(s)".format(
            self.label, self.median_ms, self.p95_ms, self.runs)


def measure(label: str, fn: Callable[[], Any], runs: int = 5,
            warmup: int = 1) -> Timing:
    """Time `fn`, discarding warmup runs.

    The warmup matters here: the first call to almost anything in this system
    touches an import, a lazily-built index or a cold branch predictor, and
    including it would report first-call cost as steady-state cost.
    """
    for _ in range(max(0, warmup)):
        fn()
    timing = Timing(label=label)
    for _ in range(max(1, runs)):
        started = time.perf_counter()
        fn()
        timing.samples.append(time.perf_counter() - started)
    return timing


class BenchmarkRecorder:
    """Collects timings across the benchmark session so they can be published.

    Written to `tests/benchmark/results.json` at the end of the session and read
    by `scripts/build_test_report.py`, so `docs/TEST_REPORT.md` quotes measured
    numbers rather than remembered ones.
    """

    def __init__(self):
        self.timings: List[Timing] = []
        self.notes: Dict[str, Any] = {}

    def record(self, timing: Timing) -> Timing:
        self.timings.append(timing)
        return timing

    def note(self, key: str, value: Any) -> None:
        self.notes[key] = value

    def as_dict(self) -> Dict[str, Any]:
        return {
            "timings": [t.as_dict() for t in self.timings],
            "notes": self.notes,
        }
