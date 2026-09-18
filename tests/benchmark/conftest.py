"""Benchmark fixtures, and the results file the test report is built from."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.benchmark.harness import BenchmarkRecorder

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"


@pytest.fixture(scope="session")
def bench() -> BenchmarkRecorder:
    """Collects every timing so the run can publish measured numbers."""
    return BenchmarkRecorder()


@pytest.fixture(scope="session", autouse=True)
def _publish_results(bench):
    """Write the timings after the benchmark session finishes.

    `docs/TEST_REPORT.md` quotes these figures. Writing them from the run rather
    than transcribing them by hand is the only way the document stays true.
    """
    yield
    if not bench.timings:
        return
    import platform
    import sys

    payload = bench.as_dict()
    payload["environment"] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "machine": platform.machine(),
    }
    try:
        import pandas
        payload["environment"]["pandas"] = pandas.__version__
    except Exception:  # pragma: no cover - pandas is a hard dependency
        pass
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@pytest.fixture(scope="session")
def bench_executor(dataset):
    """A cached executor, as the application uses."""
    import src.analysis.advanced_patterns  # noqa: F401
    from src.ps.executor import PSExecutor

    return PSExecutor(dataset=dataset)


@pytest.fixture(scope="session")
def bench_chatbot(dataset):
    """A chatbot wired to the deterministic composer.

    The composer is the production fallback, so this measures the real
    application path minus the network. LLM latency is measured separately and
    deliberately - mixing the two would report the API's round trip as this
    system's cost.
    """
    from src.api.chatbot import Chatbot
    from src.llm.client import DeterministicClient

    return Chatbot(dataset=dataset, llm_client=DeterministicClient())
