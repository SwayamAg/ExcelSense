"""Every problem statement still produces the result it produced.

This is the test the phase gate names: "all regression tests pass after every
subsequent code change". Its value is entirely in the failure message. A diff of
two hashes tells a maintainer that something moved; these assertions tell them
which PS, which figure, from what to what - which is the difference between a
five-minute fix and an afternoon of bisecting.

The catalogue is expected to grow. A new PS is not a regression, so a PS absent
from the baseline is reported as new rather than failed; a PS that has
*disappeared* is a failure, because something the system used to answer it no
longer can.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def entries(baseline):
    return baseline["problem_statements"]["entries"]


# --------------------------------------------------------------------------
# Coverage: the catalogue has not shrunk
# --------------------------------------------------------------------------

def test_no_problem_statement_has_disappeared(entries, current_ps_results):
    missing = sorted(set(entries) - set(current_ps_results))
    assert not missing, (
        "{} problem statement(s) in the baseline no longer execute: {}. A PS that "
        "used to produce an answer and now does not is a regression, even if every "
        "other figure is unchanged.".format(len(missing), missing))


def test_new_problem_statements_are_reported_not_failed(entries, current_ps_results):
    """A growing catalogue is the point. New PSs are listed, not treated as errors."""
    added = sorted(set(current_ps_results) - set(entries))
    if added:
        pytest.skip("{} new problem statement(s) not yet in the baseline: {}. "
                    "Regenerate with scripts/build_regression_baseline.py.".format(
                        len(added), added))


def test_every_baselined_ps_is_still_sufficient(entries, current_ps_results):
    broken = []
    for ps_id, recorded in entries.items():
        current = current_ps_results.get(ps_id)
        if current is None or not recorded["sufficient"]:
            continue
        if not current["sufficient"]:
            broken.append("{}: {}".format(
                ps_id, current["insufficient_reason"] or "no reason given"))
    assert not broken, (
        "{} problem statement(s) that used to answer now report insufficient "
        "evidence:\n  {}".format(len(broken), "\n  ".join(broken)))


# --------------------------------------------------------------------------
# Figures: nothing moved
# --------------------------------------------------------------------------

def test_headline_figures_are_unchanged(entries, current_ps_results):
    """The figures an answer would actually quote."""
    moved = []
    for ps_id, recorded in sorted(entries.items()):
        current = current_ps_results.get(ps_id)
        if current is None:
            continue
        before, after = recorded["headline"], current["headline"]
        for key in sorted(set(before) | set(after)):
            if before.get(key) != after.get(key):
                moved.append("{}.{}: {!r} -> {!r}".format(
                    ps_id, key, before.get(key, "<absent>"), after.get(key, "<absent>")))
    assert not moved, (
        "{} figure(s) changed against the baseline:\n  {}\n\nIf the change is "
        "intended, regenerate the baseline and let the diff carry the "
        "explanation.".format(len(moved), "\n  ".join(moved[:40])))


def test_population_sizes_are_unchanged(entries, current_ps_results):
    """A moved population means the filters changed, whatever the figures say."""
    moved = []
    for ps_id, recorded in sorted(entries.items()):
        current = current_ps_results.get(ps_id)
        if current is None:
            continue
        if recorded["population_size"] != current["population_size"]:
            moved.append("{}: {} -> {} rows ({})".format(
                ps_id, recorded["population_size"], current["population_size"],
                recorded["population"]))
    assert not moved, "population size changed for {} PS(s):\n  {}".format(
        len(moved), "\n  ".join(moved))


def test_result_tables_are_unchanged(entries, current_ps_results):
    """Row counts, column lists and the full contents of every result table."""
    changed = []
    for ps_id, recorded in sorted(entries.items()):
        current = current_ps_results.get(ps_id)
        if current is None:
            continue
        if recorded["result_rows"] != current["result_rows"]:
            changed.append("{}: {} -> {} rows".format(
                ps_id, recorded["result_rows"], current["result_rows"]))
        elif recorded["result_columns"] != current["result_columns"]:
            changed.append("{}: columns {} -> {}".format(
                ps_id, recorded["result_columns"], current["result_columns"]))
        elif recorded["table_digest"] != current["table_digest"]:
            changed.append("{}: {} rows x {} columns, same shape but different "
                           "contents".format(ps_id, current["result_rows"],
                                             len(current["result_columns"])))
    assert not changed, "{} result table(s) changed:\n  {}".format(
        len(changed), "\n  ".join(changed[:40]))


# --------------------------------------------------------------------------
# Provenance: the same figures, computed the same way
# --------------------------------------------------------------------------

def test_the_same_formulas_are_applied(entries, current_ps_results):
    """A right answer reached by a different formula is still worth knowing about."""
    changed = []
    for ps_id, recorded in sorted(entries.items()):
        current = current_ps_results.get(ps_id)
        if current is None:
            continue
        if recorded["formulas_applied"] != current["formulas_applied"]:
            changed.append("{}: {} -> {}".format(
                ps_id, recorded["formulas_applied"], current["formulas_applied"]))
    assert not changed, "formula usage changed for {} PS(s):\n  {}".format(
        len(changed), "\n  ".join(changed))


def test_the_same_business_rules_are_applied(entries, current_ps_results):
    changed = []
    for ps_id, recorded in sorted(entries.items()):
        current = current_ps_results.get(ps_id)
        if current is None:
            continue
        if recorded["business_rules_applied"] != current["business_rules_applied"]:
            changed.append("{}: {} -> {}".format(
                ps_id, recorded["business_rules_applied"],
                current["business_rules_applied"]))
    assert not changed, "business rule usage changed for {} PS(s):\n  {}".format(
        len(changed), "\n  ".join(changed))


def test_the_same_tables_are_read(entries, current_ps_results):
    changed = []
    for ps_id, recorded in sorted(entries.items()):
        current = current_ps_results.get(ps_id)
        if current is None:
            continue
        if recorded["tables_used"] != current["tables_used"]:
            changed.append("{}: {} -> {}".format(
                ps_id, recorded["tables_used"], current["tables_used"]))
    assert not changed, "table usage changed for {} PS(s):\n  {}".format(
        len(changed), "\n  ".join(changed))


# --------------------------------------------------------------------------
# Framing: the caveats have not quietly been dropped
# --------------------------------------------------------------------------

def test_data_quality_notes_are_unchanged(entries, current_ps_results):
    """Losing a data-quality note is a regression in the answer's honesty."""
    dropped = []
    for ps_id, recorded in sorted(entries.items()):
        current = current_ps_results.get(ps_id)
        if current is None:
            continue
        lost = sorted(set(recorded["data_quality_notes"]) -
                      set(current["data_quality_notes"]))
        if lost:
            dropped.append("{}: no longer carries {}".format(ps_id, lost))
    assert not dropped, "{} PS(s) dropped a data-quality note:\n  {}".format(
        len(dropped), "\n  ".join(dropped))


def test_unsupported_conclusions_are_still_declared(entries, current_ps_results):
    """The list of what cannot be concluded is part of the answer, not decoration."""
    weakened = []
    for ps_id, recorded in sorted(entries.items()):
        current = current_ps_results.get(ps_id)
        if current is None:
            continue
        if current["unsupported_count"] < recorded["unsupported_count"]:
            weakened.append("{}: {} -> {} declared limits on what the data can "
                            "establish".format(ps_id, recorded["unsupported_count"],
                                               current["unsupported_count"]))
    assert not weakened, "{} PS(s) now claim more than before:\n  {}".format(
        len(weakened), "\n  ".join(weakened))


def test_answer_types_are_unchanged(entries, current_ps_results):
    """A changed answer type breaks the composer and the validator downstream."""
    changed = []
    for ps_id, recorded in sorted(entries.items()):
        current = current_ps_results.get(ps_id)
        if current is None:
            continue
        if recorded["expected_answer_type"] != current["expected_answer_type"]:
            changed.append("{}: {} -> {}".format(
                ps_id, recorded["expected_answer_type"],
                current["expected_answer_type"]))
    assert not changed, "answer type changed for {} PS(s):\n  {}".format(
        len(changed), "\n  ".join(changed))


# --------------------------------------------------------------------------
# The catch-all
# --------------------------------------------------------------------------

def test_the_full_aggregate_digest_is_unchanged(entries, current_ps_results):
    """Everything the named assertions above do not reach.

    If this fails alone, something changed in a nested or non-scalar aggregate.
    The named tests are the useful ones; this is the net underneath them.
    """
    changed = [ps_id for ps_id, recorded in sorted(entries.items())
               if ps_id in current_ps_results
               and recorded["aggregate_digest"] != current_ps_results[ps_id][
                   "aggregate_digest"]]
    assert not changed, (
        "{} PS(s) have a changed aggregate structure that the named assertions did "
        "not catch: {}. Compare the evidence packages directly.".format(
            len(changed), changed[:20]))


def test_the_baseline_covers_the_whole_catalogue(baseline, current_ps_results):
    """A baseline that covers half the catalogue protects half the catalogue."""
    recorded_count = baseline["problem_statements"]["problem_statement_count"]
    assert len(baseline["problem_statements"]["entries"]) == recorded_count, (
        "the baseline records {} PSs but claims a catalogue of {}".format(
            len(baseline["problem_statements"]["entries"]), recorded_count))
