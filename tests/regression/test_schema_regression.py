"""The data has not changed underneath the system.

Every figure in this project is a statement about five CSVs. If a column is
renamed, a row count moves, or a date column starts parsing as text, the PS
regression tests will fail in dozens of places at once and none of those failures
will say why. These tests fail first and say exactly what changed about the data,
so the cause is read off the failure rather than inferred from its blast radius.

They also guard the parse contract, which is the part most likely to shift
silently: pandas changing how it infers a dtype does not announce itself, and
the difference between an identifier read as text and the same identifier read
as a number is a wrong answer with full provenance attached.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def schema(baseline):
    return baseline["schema"]


# --------------------------------------------------------------------------
# The files
# --------------------------------------------------------------------------

def test_the_same_five_tables_are_loaded(schema, dataset):
    assert sorted(dataset.table_keys) == sorted(schema["tables"]), (
        "loaded tables {} no longer match the baseline {}".format(
            sorted(dataset.table_keys), sorted(schema["tables"])))


def test_row_counts_are_unchanged(schema, dataset):
    moved = []
    for key, recorded in sorted(schema["tables"].items()):
        if key not in dataset.table_keys:
            continue
        actual = len(dataset.frame(key))
        if actual != recorded["rows"]:
            moved.append("{}: {:,} -> {:,} rows".format(key, recorded["rows"], actual))
    assert not moved, (
        "row counts changed:\n  {}\n\nEvery figure in the catalogue is computed over "
        "these rows.".format("\n  ".join(moved)))


def test_column_names_and_order_are_unchanged(schema, dataset):
    """Order matters: a positional read anywhere would silently shift."""
    changed = []
    for key, recorded in sorted(schema["tables"].items()):
        if key not in dataset.table_keys:
            continue
        actual = list(dataset.frame(key).columns)
        if actual == recorded["columns"]:
            continue
        added = sorted(set(actual) - set(recorded["columns"]))
        removed = sorted(set(recorded["columns"]) - set(actual))
        if added or removed:
            changed.append("{}: added {}, removed {}".format(key, added, removed))
        else:
            changed.append("{}: same columns in a different order".format(key))
    assert not changed, "column layout changed:\n  {}".format("\n  ".join(changed))


def test_source_files_are_unchanged(schema, dataset):
    wrong = []
    for key, recorded in sorted(schema["tables"].items()):
        if key not in dataset.table_keys:
            continue
        actual = dataset.contract.table(key).file
        if actual != recorded["file"]:
            wrong.append("{}: {} -> {}".format(key, recorded["file"], actual))
    assert not wrong, "table sources changed:\n  {}".format("\n  ".join(wrong))


# --------------------------------------------------------------------------
# The parse contract
# --------------------------------------------------------------------------

def test_every_column_still_parses_to_the_same_type(schema, dataset):
    """The regression most likely to happen and least likely to be noticed."""
    changed = []
    for key, recorded in sorted(schema["tables"].items()):
        if key not in dataset.table_keys:
            continue
        frame = dataset.frame(key)
        for column, dtype in sorted(recorded["dtypes"].items()):
            if column not in frame.columns:
                continue
            actual = str(frame[column].dtype)
            if actual != dtype:
                changed.append("{}.{}: {} -> {}".format(key, column, dtype, actual))
    assert not changed, (
        "{} column(s) now parse to a different type:\n  {}\n\nAn identifier read as "
        "a number, or a date read as text, produces wrong answers that still carry "
        "full provenance.".format(len(changed), "\n  ".join(changed)))


def test_null_counts_are_unchanged(schema, dataset):
    """A moved null count means parsing changed, even when the dtype did not."""
    moved = []
    for key, recorded in sorted(schema["tables"].items()):
        if key not in dataset.table_keys:
            continue
        frame = dataset.frame(key)
        for column, nulls in sorted(recorded["null_counts"].items()):
            if column not in frame.columns:
                continue
            actual = int(frame[column].isna().sum())
            if actual != nulls:
                moved.append("{}.{}: {:,} -> {:,} nulls".format(
                    key, column, nulls, actual))
    assert not moved, (
        "{} column(s) parse a different number of values:\n  {}\n\nMore nulls means "
        "values are being discarded; fewer means something is being coerced that was "
        "not before.".format(len(moved), "\n  ".join(moved[:30])))


def test_primary_keys_are_unchanged(schema, dataset):
    changed = []
    for key, recorded in sorted(schema["tables"].items()):
        if key not in dataset.table_keys:
            continue
        actual = dataset.contract.table(key).primary_key
        if actual != recorded["primary_key"]:
            changed.append("{}: {} -> {}".format(key, recorded["primary_key"], actual))
    assert not changed, "primary keys changed:\n  {}".format("\n  ".join(changed))


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------

def test_the_contract_version_is_unchanged(schema, dataset):
    assert dataset.contract.version == schema["contract_version"], (
        "the data contract moved from {} to {}. If that was deliberate, regenerate "
        "the regression baseline so the diff records what changed.".format(
            schema["contract_version"], dataset.contract.version))


def test_no_data_quality_finding_has_been_dropped(schema, dataset):
    """A finding disappearing means either the data changed or a caveat was lost."""
    current = {f["id"] for f in dataset.contract.data_quality_findings}
    lost = sorted(set(schema["finding_ids"]) - current)
    assert not lost, (
        "data-quality finding(s) {} are no longer declared. Every PS that cites one "
        "would silently stop warning about it.".format(lost))


def test_the_hub_table_and_joins_are_unchanged(schema, dataset):
    assert dataset.contract.hub_table == schema["hub_table"], (
        "the hub table moved from {} to {}".format(
            schema["hub_table"], dataset.contract.hub_table))
    assert len(dataset.contract.joins) == schema["join_count"], (
        "the contract declares {} joins, the baseline recorded {}".format(
            len(dataset.contract.joins), schema["join_count"]))


def test_unanswerable_domains_are_still_declared(schema, dataset):
    """These are what the router refuses on. Losing one turns a refusal into a guess."""
    assert len(dataset.contract.unanswerable_domains) >= \
        schema["unanswerable_domain_count"], (
        "the contract declares {} unanswerable domains, down from {}. Each one is a "
        "question the router would stop refusing.".format(
            len(dataset.contract.unanswerable_domains),
            schema["unanswerable_domain_count"]))


def test_load_warnings_are_unchanged(schema, dataset):
    """New load warnings mean the data or the loader changed."""
    current = sorted(dataset.warnings)
    recorded = sorted(schema["load_warnings"])
    new = [w for w in current if w not in recorded]
    assert not new, "the loader now reports {} new warning(s):\n  {}".format(
        len(new), "\n  ".join(new))


def test_no_data_is_silently_lost_on_load(dataset):
    """The Phase 2 guarantee, re-asserted here because it is easy to regress.

    A loader change that starts coercing something it used to preserve shows up
    as unparsed values long before it shows up as a wrong figure.
    """
    report = dataset.report
    assert report.unparsed_total == 0, (
        "{} value(s) failed to parse and were dropped:\n  {}".format(
            report.unparsed_total,
            "\n  ".join(str(i.as_dict()) for i in report.issues[:10])))
