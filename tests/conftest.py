"""Shared pytest fixtures. Keeps the project root importable and the CSVs cached."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "data" / "csv"
CONFIG_DIR = ROOT / "config"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_FILES = {
    "policy": "Policy_Details.csv",
    "owner": "Owner_Details.csv",
    "sales": "Sales_Details.csv",
    "claims": "Claims_Details.csv",
    "persistency": "Persistency_Details.csv",
}


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def csv_dir() -> Path:
    return CSV_DIR


@pytest.fixture(scope="session")
def contract() -> dict:
    path = CONFIG_DIR / "data_contract.json"
    assert path.exists(), "config/data_contract.json missing - run scripts/profile_data.py"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def raw_frames() -> dict:
    """Every CSV read as pure text, so nothing is coerced before we assert on it."""
    return {
        key: pd.read_csv(CSV_DIR / name, dtype=str, keep_default_na=True,
                         na_values=["", "NA", "N/A", "null", "NULL", "nan"])
        for key, name in EXPECTED_FILES.items()
    }


@pytest.fixture(scope="session")
def dataset():
    """The five CSVs loaded under the Phase 0 parse contract (no DQ-001 repair)."""
    from src.data import load_dataset
    return load_dataset()


@pytest.fixture(scope="session")
def ps_executor(dataset):
    """One executor for the whole session.

    The executor caches deterministic results itself (`src/ps/cache.py`), which
    is what keeps the suite fast. Insufficient results are not cached there, so
    this fixture memoises those too - a refusal is as deterministic as an answer
    and several tests ask for the same one repeatedly.

    `execute_uncached` bypasses both layers: the reproducibility test must
    observe the computation, not a record of an earlier one.
    """
    import src.analysis.advanced_patterns  # noqa: F401  registers extra patterns
    from src.ps.executor import PSExecutor

    executor = PSExecutor(dataset=dataset)
    refusals = {}
    raw_execute = executor.execute

    def cached(problem_id, parameters=None, question=""):
        key = (problem_id, repr(sorted((parameters or {}).items())), question)
        if key not in refusals:
            refusals[key] = raw_execute(problem_id, parameters, question)
        return refusals[key]

    executor.execute = cached
    executor.execute_uncached = executor.execute_fresh
    return executor


@pytest.fixture(scope="session")
def query_router(dataset):
    """One router for the session, with decisions cached per question.

    Routing is deterministic, so re-routing the same question can only produce the
    same decision. The determinism test bypasses the cache deliberately.
    """
    from src.router import QueryRouter

    router = QueryRouter(dataset=dataset)
    cache = {}
    raw_route = router.route

    def cached(question):
        if question not in cache:
            cache[question] = raw_route(question)
        return cache[question]

    router.route = cached
    router.route_uncached = raw_route
    return router
