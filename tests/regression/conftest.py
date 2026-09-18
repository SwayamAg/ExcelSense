"""Fixtures for the regression suite.

The suite compares what the system produces now against `baseline.json`, which
was produced by running the system (`scripts/build_regression_baseline.py`) and
is committed. Regenerating it is a deliberate, reviewable act; a figure that
moves without an explanation in the diff is the regression these tests exist to
catch.

Everything here executes fresh. A regression test served from the result cache
would be checking that the cache still holds what it was given, which is a
different and much weaker claim than that the system still computes it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

BASELINE_PATH = Path(__file__).resolve().parent / "baseline.json"


@pytest.fixture(scope="session")
def baseline() -> dict:
    assert BASELINE_PATH.exists(), (
        "tests/regression/baseline.json is missing - run "
        "python scripts/build_regression_baseline.py")
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def fresh_executor(dataset):
    """An executor with the result cache switched off."""
    import src.analysis.advanced_patterns  # noqa: F401  registers extra patterns
    from src.ps.executor import PSExecutor

    return PSExecutor(dataset=dataset, use_cache=False)


@pytest.fixture(scope="session")
def current_ps_results(fresh_executor, baseline):
    """Every executable PS, recomputed now, digested the same way as the baseline."""
    from scripts.build_regression_baseline import REQUIRED_PARAMETERS, ps_entry

    return {
        ps.id: ps_entry(fresh_executor.execute_fresh(
            ps.id, REQUIRED_PARAMETERS.get(ps.id)))
        for ps in fresh_executor.registry.executable()
    }
