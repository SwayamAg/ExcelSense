"""The result cache cannot serve an answer the data no longer supports.

A cache is the one performance optimisation in this system that can produce a
wrong answer, and it produces a particularly bad kind: the figures arrive with
full provenance, a formula id and a population size, and read exactly like a
fresh computation. Everything downstream - the evidence tables, the chart, the
validator - would confirm a stale number as grounded, because against the cached
evidence it is.

So the cache is tested against the failure it could cause, not the speedup it
provides. The speedup is measured in tests/benchmark/; correctness is here.
"""
from __future__ import annotations

import pytest

from src.ps.cache import ResultCache, fingerprint_dataset


PARAMETERISED = ("PS-031", {"app_id": "15011000000"})


@pytest.fixture(scope="module")
def cached_executor(dataset):
    import src.analysis.advanced_patterns  # noqa: F401
    from src.ps.executor import PSExecutor

    return PSExecutor(dataset=dataset)


# --------------------------------------------------------------------------
# A hit must equal a computation
# --------------------------------------------------------------------------

def test_a_cache_hit_matches_a_fresh_computation(cached_executor, fresh_executor):
    """The claim the whole cache rests on, checked across the catalogue."""
    differing = []
    for ps in list(cached_executor.registry.executable())[:25]:
        cached_executor.execute(ps.id)                       # populate
        hit = cached_executor.execute(ps.id)                 # serve from cache
        fresh = fresh_executor.execute_fresh(ps.id)
        if hit.aggregates != fresh.aggregates:
            differing.append("{}: aggregates differ".format(ps.id))
        elif hit.result_table != fresh.result_table:
            differing.append("{}: result table differs".format(ps.id))
        elif hit.population_size != fresh.population_size:
            differing.append("{}: population {} vs {}".format(
                ps.id, hit.population_size, fresh.population_size))
    assert not differing, "cached results differ from fresh ones:\n  {}".format(
        "\n  ".join(differing))


def test_a_hit_is_disclosed_in_provenance(cached_executor):
    """A reader must be able to tell a replay from a computation."""
    cache = ResultCache.for_dataset(cached_executor.dataset,
                                    registry=cached_executor.registry)
    cached_executor.cache, original = cache, cached_executor.cache
    try:
        first = cached_executor.execute("PS-001")
        second = cached_executor.execute("PS-001")
        assert "cache" not in first.provenance, (
            "the first computation was reported as a cache hit")
        assert second.provenance.get("cache", {}).get("hit") is True, (
            "a cached answer does not disclose that it was served from cache")
        assert second.provenance["cache"]["computed_at_utc"] == first.generated_at_utc, (
            "the cache does not record when the figures were actually computed")
    finally:
        cached_executor.cache = original


def test_the_evidence_keeps_the_time_it_was_computed(cached_executor):
    """Re-stamping the timestamp on a hit would misreport when the work happened."""
    cache = ResultCache.for_dataset(cached_executor.dataset)
    first = cached_executor.execute_fresh("PS-002")
    cache.put("PS-002", {}, first)
    hit = cache.get("PS-002", {})
    assert hit.generated_at_utc == first.generated_at_utc


# --------------------------------------------------------------------------
# Keys: what must and must not produce a separate entry
# --------------------------------------------------------------------------

def test_different_parameters_do_not_share_an_entry(cached_executor):
    """The failure that would return one policy's record for another."""
    ps_id, params = PARAMETERISED
    other = {"app_id": "15011000001"}
    cache = ResultCache.for_dataset(cached_executor.dataset)
    assert cache.key(ps_id, params) != cache.key(ps_id, other), (
        "two different parameter values produce the same cache key")


def test_parameter_order_does_not_produce_two_entries(cached_executor):
    cache = ResultCache.for_dataset(cached_executor.dataset)
    a = cache.key("PS-001", {"x": 1, "y": 2})
    b = cache.key("PS-001", {"y": 2, "x": 1})
    assert a == b, "argument order changes the cache key"


def test_different_problem_statements_do_not_share_an_entry(cached_executor):
    cache = ResultCache.for_dataset(cached_executor.dataset)
    assert cache.key("PS-001", None) != cache.key("PS-002", None)


def test_the_question_is_not_part_of_the_key(cached_executor):
    """Two phrasings of one analysis are one computation.

    No analytical code reads the question - it is display metadata - so keying on
    it would multiply entries without protecting anything.
    """
    cache = ResultCache.for_dataset(cached_executor.dataset)
    assert cache.key("PS-001", None) == cache.key("PS-001", None)
    evidence = cached_executor.execute_fresh("PS-001", None, "How many policies?")
    cache.put("PS-001", None, evidence)
    hit = cache.get("PS-001", None, question="Size of the portfolio")
    assert hit is not None, "a differently-phrased question missed the cache"
    assert hit.question == "Size of the portfolio", (
        "the cached answer reports the wrong question; it would be shown to the "
        "user attached to a question they did not ask")


# --------------------------------------------------------------------------
# Fingerprinting: changed data invalidates everything
# --------------------------------------------------------------------------

def test_the_fingerprint_reflects_the_loaded_data(dataset):
    assert fingerprint_dataset(dataset) == fingerprint_dataset(dataset), (
        "the fingerprint is not stable for one dataset")


class _ShimDataset:
    """The three things `fingerprint_dataset` reads, over substituted frames."""

    def __init__(self, dataset, frames):
        self.contract = dataset.contract
        self.table_keys = list(dataset.table_keys)
        self._frames = frames

    def frame(self, key):
        return self._frames[key]


def test_removing_a_row_changes_the_fingerprint(dataset):
    """The guard against serving a figure the current data no longer supports."""
    frames = {k: dataset.frame(k) for k in dataset.table_keys}
    before = fingerprint_dataset(_ShimDataset(dataset, frames))

    key = sorted(dataset.table_keys)[0]
    frames[key] = frames[key].iloc[:-1]
    assert fingerprint_dataset(_ShimDataset(dataset, frames)) != before, (
        "removing a row did not change the fingerprint, so entries computed over "
        "the old data would still be served")


def test_changing_one_cell_changes_the_fingerprint(dataset):
    """Row counts alone are not enough: a corrected value must invalidate too."""
    frames = {k: dataset.frame(k) for k in dataset.table_keys}
    before = fingerprint_dataset(_ShimDataset(dataset, frames))

    key = sorted(dataset.table_keys)[0]
    edited = frames[key].copy()
    column = edited.columns[0]
    # Cast first: assigning a string into a typed column warns about the dtype
    # change, and the dtype is not what this test is about.
    edited[column] = edited[column].astype(object)
    edited.iloc[0, edited.columns.get_loc(column)] = "changed-for-this-test"
    frames[key] = edited
    assert fingerprint_dataset(_ShimDataset(dataset, frames)) != before, (
        "a changed value did not change the fingerprint, so an answer computed "
        "before the correction would still be served afterwards")


def test_a_different_fingerprint_makes_prior_entries_unreachable(cached_executor):
    cache = ResultCache.for_dataset(cached_executor.dataset)
    cache.put("PS-001", None, cached_executor.execute_fresh("PS-001"))
    assert cache.get("PS-001", None) is not None

    stale = ResultCache(fingerprint="a-different-dataset")
    stale._entries = dict(cache._entries)
    assert stale.get("PS-001", None) is None, (
        "an entry computed over other data was served under a new fingerprint")


def test_the_catalogue_version_is_part_of_the_fingerprint(dataset):
    """A changed PS means a changed analysis, even over identical data."""
    class _Registry:
        version = "9.9.9"

        def __len__(self):
            return 1

    a = ResultCache.for_dataset(dataset)
    b = ResultCache.for_dataset(dataset, registry=_Registry())
    assert a.fingerprint != b.fingerprint


# --------------------------------------------------------------------------
# Isolation: a caller cannot corrupt the cache
# --------------------------------------------------------------------------

def test_mutating_a_returned_package_does_not_change_the_cache(cached_executor):
    """The summarizer annotates evidence. It must not reach back into the cache."""
    cache = ResultCache.for_dataset(cached_executor.dataset)
    cache.put("PS-001", None, cached_executor.execute_fresh("PS-001"))

    first = cache.get("PS-001", None)
    first.aggregates["fabricated"] = 999999
    first.limitations.append("invented limitation")
    first.result_table.append({"invented": True})

    second = cache.get("PS-001", None)
    assert "fabricated" not in second.aggregates, (
        "a caller's edit reached into the cache and would be served to the next user")
    assert "invented limitation" not in second.limitations
    assert {"invented": True} not in second.result_table


def test_mutating_the_stored_package_after_put_does_not_change_the_cache(
        cached_executor):
    evidence = cached_executor.execute_fresh("PS-001")
    cache = ResultCache.for_dataset(cached_executor.dataset)
    cache.put("PS-001", None, evidence)
    evidence.aggregates["fabricated"] = 1
    assert "fabricated" not in cache.get("PS-001", None).aggregates


# --------------------------------------------------------------------------
# What is not cached
# --------------------------------------------------------------------------

def test_insufficient_results_are_not_cached(cached_executor):
    """An insufficient result is cheap to recompute and expensive to get wrong."""
    cache = ResultCache.for_dataset(cached_executor.dataset)
    executor_cache, cached_executor.cache = cached_executor.cache, cache
    try:
        cached_executor.execute("PS-031", {"app_id": "00000000000"})
        assert len(cache) == 0, (
            "a refusal was cached; a later data change that makes the question "
            "answerable would keep returning the refusal")
    finally:
        cached_executor.cache = executor_cache


def test_an_invalid_parameter_is_still_rejected_by_the_ps(cached_executor):
    """A cache must not become a way around parameter validation."""
    evidence = cached_executor.execute("PS-031", {"app_id": ["not", "an", "id"]})
    assert not evidence.is_sufficient
    assert evidence.insufficient is not None


def test_disabling_the_cache_stops_all_caching(dataset):
    import src.analysis.advanced_patterns  # noqa: F401
    from src.ps.executor import PSExecutor

    executor = PSExecutor(dataset=dataset, use_cache=False)
    executor.execute("PS-001")
    executor.execute("PS-001")
    assert len(executor.cache) == 0
    assert executor.cache.hits == 0


def test_execute_fresh_never_populates_the_cache(cached_executor):
    cache = ResultCache.for_dataset(cached_executor.dataset)
    executor_cache, cached_executor.cache = cached_executor.cache, cache
    try:
        cached_executor.execute_fresh("PS-003")
        assert len(cache) == 0
    finally:
        cached_executor.cache = executor_cache


# --------------------------------------------------------------------------
# Housekeeping
# --------------------------------------------------------------------------

def test_the_cache_evicts_rather_than_growing_without_bound(cached_executor):
    cache = ResultCache.for_dataset(cached_executor.dataset, max_entries=3)
    evidence = cached_executor.execute_fresh("PS-001")
    for i in range(10):
        cache.put("PS-{:03d}".format(i), None, evidence)
    assert len(cache) == 3
    assert cache.evictions == 7


def test_eviction_is_least_recently_used(cached_executor):
    cache = ResultCache.for_dataset(cached_executor.dataset, max_entries=2)
    evidence = cached_executor.execute_fresh("PS-001")
    cache.put("A", None, evidence)
    cache.put("B", None, evidence)
    cache.get("A", None)          # A is now the most recent
    cache.put("C", None, evidence)
    assert cache.get("A", None) is not None
    assert cache.get("B", None) is None, "the most recently used entry was evicted"


def test_stats_report_honestly(cached_executor):
    cache = ResultCache.for_dataset(cached_executor.dataset)
    cache.put("PS-001", None, cached_executor.execute_fresh("PS-001"))
    cache.get("PS-001", None)
    cache.get("PS-002", None)
    stats = cache.stats()
    assert stats["hits"] == 1 and stats["misses"] == 1
    assert stats["hit_rate"] == 0.5
    assert stats["entries"] == 1


def test_clearing_the_cache_removes_everything(cached_executor):
    cache = ResultCache.for_dataset(cached_executor.dataset)
    cache.put("PS-001", None, cached_executor.execute_fresh("PS-001"))
    cache.clear()
    assert len(cache) == 0
    assert cache.get("PS-001", None) is None
