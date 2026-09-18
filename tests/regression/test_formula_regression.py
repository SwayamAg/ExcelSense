"""The formulas still compute what they computed.

Formulas are the smallest unit of arithmetic in the system and the one place a
change is most expensive: every PS that cites a formula inherits its answer, so
a shifted rounding rule or a changed zero-denominator convention propagates to
dozens of figures at once.

Each formula declares worked examples, and `verify_examples()` already checks
them. These tests add what that cannot: the examples themselves are part of the
source and could be edited to match a broken implementation. The baseline is a
separate record of what the examples were and what they produced, so changing
both the code and its example no longer passes quietly.
"""
from __future__ import annotations

import pytest

from src.logic.formula_engine import get_engine


@pytest.fixture(scope="module")
def formulas():
    return get_engine()


@pytest.fixture(scope="module")
def recorded(baseline):
    return baseline["formulas"]["formulas"]


# --------------------------------------------------------------------------
# The registry itself
# --------------------------------------------------------------------------

def test_no_formula_has_disappeared(recorded, formulas):
    missing = sorted(set(recorded) - set(formulas.formula_ids))
    assert not missing, (
        "formula(s) {} no longer exist. Every PS citing one would fail to "
        "execute.".format(missing))


def test_formula_count_has_not_shrunk(baseline, formulas):
    assert len(formulas.formula_ids) >= baseline["formulas"]["formula_count"], (
        "the engine registers {} formulas, down from {}".format(
            len(formulas.formula_ids), baseline["formulas"]["formula_count"]))


# --------------------------------------------------------------------------
# The declared semantics
# --------------------------------------------------------------------------

def test_expressions_are_unchanged(recorded, formulas):
    """The expression is what the evidence package shows a reader. It must be true."""
    changed = []
    for formula_id, spec in sorted(recorded.items()):
        if formula_id not in formulas.formula_ids:
            continue
        actual = formulas.get(formula_id).expression
        if actual != spec["expression"]:
            changed.append("{}: {!r} -> {!r}".format(
                formula_id, spec["expression"], actual))
    assert not changed, "formula expressions changed:\n  {}".format(
        "\n  ".join(changed))


def test_null_handling_is_unchanged(recorded, formulas):
    """Whether nulls are excluded or treated as zero changes almost every average."""
    changed = []
    for formula_id, spec in sorted(recorded.items()):
        if formula_id not in formulas.formula_ids:
            continue
        actual = formulas.get(formula_id).null_handling
        if actual != spec["null_handling"]:
            changed.append("{}: {!r} -> {!r}".format(
                formula_id, spec["null_handling"], actual))
    assert not changed, (
        "null handling changed:\n  {}\n\nEvery figure computed over a column with "
        "missing values is affected.".format("\n  ".join(changed)))


def test_zero_denominator_handling_is_unchanged(recorded, formulas):
    """Returning None rather than 0 for an empty denominator is the whole point."""
    changed = []
    for formula_id, spec in sorted(recorded.items()):
        if formula_id not in formulas.formula_ids:
            continue
        actual = formulas.get(formula_id).zero_denominator
        if actual != spec["zero_denominator"]:
            changed.append("{}: {!r} -> {!r}".format(
                formula_id, spec["zero_denominator"], actual))
    assert not changed, (
        "zero-denominator handling changed:\n  {}\n\nA rate over an empty population "
        "reported as 0% is a fabricated fact.".format("\n  ".join(changed)))


def test_rounding_is_unchanged(recorded, formulas):
    changed = []
    for formula_id, spec in sorted(recorded.items()):
        if formula_id not in formulas.formula_ids:
            continue
        actual = formulas.get(formula_id).rounding
        if actual != spec["rounding"]:
            changed.append("{}: {} -> {} decimals".format(
                formula_id, spec["rounding"], actual))
    assert not changed, "rounding changed:\n  {}".format("\n  ".join(changed))


# --------------------------------------------------------------------------
# The arithmetic
# --------------------------------------------------------------------------

def test_every_worked_example_still_produces_its_recorded_value(recorded, formulas):
    """The core assertion: same inputs, same output, against a separate record."""
    from scripts.build_regression_baseline import _round

    wrong = []
    for formula_id, spec in sorted(recorded.items()):
        if formula_id not in formulas.formula_ids:
            continue
        for example in spec["examples"]:
            produced = _round(formulas.compute(formula_id, **example["inputs"]).value)
            if produced != example["produced"]:
                wrong.append("{}({}): {!r} -> {!r}".format(
                    formula_id,
                    ", ".join("{}={!r}".format(k, v)
                              for k, v in sorted(example["inputs"].items())),
                    example["produced"], produced))
    assert not wrong, (
        "{} worked example(s) now produce a different value:\n  {}".format(
            len(wrong), "\n  ".join(wrong)))


def test_the_examples_themselves_have_not_been_edited(recorded, formulas):
    """Guards against the fix that changes the test instead of the code.

    If an implementation breaks and its declared example is edited to match,
    `verify_examples()` passes again. The baseline holds the inputs and expected
    values as they were, so that edit surfaces here.
    """
    from scripts.build_regression_baseline import _round

    edited = []
    for formula_id, spec in sorted(recorded.items()):
        if formula_id not in formulas.formula_ids:
            continue
        current = formulas.get(formula_id).examples
        if len(current) < len(spec["examples"]):
            edited.append("{}: {} examples -> {}".format(
                formula_id, len(spec["examples"]), len(current)))
            continue
        for was, now in zip(spec["examples"], current):
            if _round(was["inputs"]) != _round(dict(now["inputs"])):
                edited.append("{}: example inputs changed from {!r} to {!r}".format(
                    formula_id, was["inputs"], now["inputs"]))
            elif _round(was["expected"]) != _round(now["expected"]):
                edited.append("{}: expected value changed from {!r} to {!r}".format(
                    formula_id, was["expected"], now["expected"]))
    assert not edited, (
        "{} declared example(s) were edited:\n  {}\n\nIf a formula genuinely needed "
        "to change, regenerate the baseline so the diff records it.".format(
            len(edited), "\n  ".join(edited)))


def test_declared_examples_still_verify_against_the_engine(formulas):
    """The engine's own self-check, re-run here so a regression run covers it."""
    failures = formulas.verify_examples()
    assert not failures, "{} formula example(s) do not verify:\n  {}".format(
        len(failures), "\n  ".join(failures))


def test_the_exported_formula_config_matches_the_engine(root, formulas):
    """The published config is what a reviewer reads. It must not drift."""
    import json

    path = root / "config" / "formulas.json"
    assert path.exists(), "config/formulas.json missing - run scripts/build_logic_config.py"
    exported = json.loads(path.read_text(encoding="utf-8"))
    exported_ids = sorted(f["id"] for f in exported["formulas"])
    assert exported_ids == sorted(formulas.formula_ids), (
        "config/formulas.json lists {} but the engine registers {}. Regenerate with "
        "scripts/build_logic_config.py.".format(exported_ids, sorted(formulas.formula_ids)))
