"""Phase 3 - formula engine tests.

Covers each formula's happy path, its edge cases, and the three places calculations
usually go quietly wrong: nulls, zero denominators and rounding.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.logic import FormulaError, get_engine

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def engine():
    return get_engine()


# --------------------------------------------------------------------------
# Registry integrity
# --------------------------------------------------------------------------

def test_every_formula_referenced_by_a_ps_is_implemented(engine):
    """This is the Phase 3 gate."""
    catalogue = json.loads(
        (ROOT / "config" / "problem_statements.json").read_text(encoding="utf-8"))
    referenced = {f for ps in catalogue["problem_statements"] for f in ps["formula_ids"]}
    missing = referenced - set(engine.formula_ids)
    assert not missing, "PSs reference formulas with no implementation: {}".format(sorted(missing))


def test_no_formula_is_registered_without_being_used(engine):
    catalogue = json.loads(
        (ROOT / "config" / "problem_statements.json").read_text(encoding="utf-8"))
    referenced = {f for ps in catalogue["problem_statements"] for f in ps["formula_ids"]}
    unused = set(engine.formula_ids) - referenced
    assert not unused, "formulas exist but no PS uses them: {}".format(sorted(unused))


def test_every_formula_declares_its_semantics(engine):
    for formula in engine.describe():
        for field in ("id", "name", "expression", "description", "null_handling",
                      "zero_denominator", "version", "returns"):
            assert formula.get(field), "{} missing {}".format(formula.get("id"), field)


def test_every_formula_has_worked_examples(engine):
    for formula in engine.describe():
        assert formula["examples"], "{} has no worked example".format(formula["id"])


def test_all_declared_examples_verify(engine):
    """Cross-check against the manually verified examples in the registry."""
    assert engine.verify_examples() == []


def test_unknown_formula_is_rejected(engine):
    with pytest.raises(FormulaError, match="unknown formula"):
        engine.compute("F-MAGIC", values=[1])


def test_unexpected_parameter_is_rejected(engine):
    with pytest.raises(FormulaError, match="does not accept"):
        engine.compute("F-SUM", values=[1], sneaky=True)


def test_results_carry_the_engine_version(engine):
    result = engine.compute("F-SUM", values=[1, 2])
    assert result.as_evidence()["engine_version"] == engine.version
    assert result.semantics["formula_version"]


# --------------------------------------------------------------------------
# Counting
# --------------------------------------------------------------------------

def test_count_excludes_nulls(engine):
    result = engine.compute("F-COUNT", values=[1, 2, None, 4])
    assert result.value == 3
    assert result.detail["nulls_excluded"] == 1


def test_count_from_a_population_size(engine):
    assert engine.compute("F-COUNT", population=25000).value == 25000


def test_count_needs_an_input(engine):
    with pytest.raises(FormulaError, match="needs either"):
        engine.compute("F-COUNT")


def test_count_distinct_ignores_nulls(engine):
    result = engine.compute("F-COUNT-DISTINCT", values=["a", "b", "a", None, "b"])
    assert result.value == 2
    assert result.detail["nulls_excluded"] == 1


def test_count_of_empty_is_zero(engine):
    assert engine.compute("F-COUNT", values=[]).value == 0


# --------------------------------------------------------------------------
# Sum, mean, median
# --------------------------------------------------------------------------

def test_sum_excludes_nulls(engine):
    assert engine.compute("F-SUM", values=[1.5, 2.5, None]).value == 4.0


def test_sum_of_empty_is_zero_but_flagged(engine):
    result = engine.compute("F-SUM", values=[])
    assert result.value == 0.0
    assert "population is empty" in result.warnings


def test_mean_divides_by_present_values_not_row_count(engine):
    """The classic error: [2, 4, None] averages to 3, not 2."""
    result = engine.compute("F-AVG", values=[2, 4, None])
    assert result.value == 3.0
    assert result.inputs["n"] == 2


def test_mean_of_empty_is_undefined_not_zero(engine):
    result = engine.compute("F-AVG", values=[])
    assert result.value is None
    assert not result.is_defined
    assert "population is empty" in result.warnings


def test_mean_of_all_nulls_is_undefined(engine):
    assert engine.compute("F-AVG", values=[None, None]).value is None


def test_median_interpolates_between_middle_values(engine):
    assert engine.compute("F-MEDIAN", values=[1, 2, 3, 4]).value == 2.5
    assert engine.compute("F-MEDIAN", values=[1, 2, 3]).value == 2.0


def test_median_of_single_value(engine):
    assert engine.compute("F-MEDIAN", values=[7]).value == 7.0


def test_median_of_empty_is_undefined(engine):
    assert engine.compute("F-MEDIAN", values=[]).value is None


def test_median_resists_an_outlier_that_moves_the_mean(engine):
    values = [1, 2, 3, 4, 1000000]
    assert engine.compute("F-MEDIAN", values=values).value == 3.0
    assert engine.compute("F-AVG", values=values).value > 100000


# --------------------------------------------------------------------------
# Rates, ratios, shares - zero denominators
# --------------------------------------------------------------------------

def test_rate_matches_a_hand_calculation(engine):
    """2737 lapsed of 25000 policies = 10.948%."""
    assert engine.compute("F-RATE", numerator=2737, denominator=25000).value == 10.948


def test_rate_with_zero_denominator_is_undefined(engine):
    result = engine.compute("F-RATE", numerator=5, denominator=0)
    assert result.value is None
    assert "denominator is zero" in result.warnings[0]


def test_rate_zero_over_positive_is_zero_not_undefined(engine):
    result = engine.compute("F-RATE", numerator=0, denominator=100)
    assert result.value == 0.0
    assert result.is_defined


def test_rate_with_null_input_is_undefined(engine):
    assert engine.compute("F-RATE", numerator=None, denominator=100).value is None
    assert engine.compute("F-RATE", numerator=10, denominator=None).value is None


def test_rate_above_one_hundred_is_flagged(engine):
    """A numerator larger than its denominator means mismatched populations."""
    result = engine.compute("F-RATE", numerator=150, denominator=100)
    assert result.value == 150.0
    assert any("exceeds denominator" in w for w in result.warnings)


def test_rate_boundaries(engine):
    assert engine.compute("F-RATE", numerator=100, denominator=100).value == 100.0
    assert engine.compute("F-RATE", numerator=1, denominator=3).value == 33.3333


def test_ratio_with_zero_denominator_is_undefined(engine):
    result = engine.compute("F-RATIO", numerator=1, denominator=0)
    assert result.value is None
    assert not result.is_defined


def test_ratio_computes_cover_multiple(engine):
    assert engine.compute("F-RATIO", numerator=1000000, denominator=50000).value == 20.0


def test_share_of_total(engine):
    result = engine.compute("F-SHARE", value=25, total=200)
    assert result.value == 12.5
    assert result.formula_id == "F-SHARE"
    assert "value" in result.inputs and "total" in result.inputs


def test_share_of_zero_total_is_undefined(engine):
    assert engine.compute("F-SHARE", value=5, total=0).value is None


def test_shares_sum_to_one_hundred(engine):
    parts = [21805, 2737, 324, 134]
    total = sum(parts)
    shares = [engine.compute("F-SHARE", value=p, total=total).value for p in parts]
    assert round(sum(shares), 6) == 100.0


# --------------------------------------------------------------------------
# Cumulative share and concentration
# --------------------------------------------------------------------------

def test_cumshare_accumulates_to_one_hundred(engine):
    result = engine.compute("F-CUMSHARE", values=[50, 30, 20])
    assert result.value == [50.0, 80.0, 100.0]
    assert result.detail["final_share"] == 100.0


def test_cumshare_of_zero_total_is_undefined(engine):
    result = engine.compute("F-CUMSHARE", values=[0, 0])
    assert result.value == []
    assert "denominator is zero" in result.warnings[0]


def test_cumshare_of_empty(engine):
    assert engine.compute("F-CUMSHARE", values=[]).value == []


def test_hhi_of_a_monopoly_is_ten_thousand(engine):
    assert engine.compute("F-HHI", shares=[100.0]).value == 10000.0


def test_hhi_of_an_even_split(engine):
    assert engine.compute("F-HHI", shares=[50.0, 50.0]).value == 5000.0
    assert engine.compute("F-HHI", shares=[25.0] * 4).value == 2500.0


def test_hhi_flags_shares_that_do_not_sum_to_one_hundred(engine):
    result = engine.compute("F-HHI", shares=[30.0, 30.0])
    assert any("not 100" in w for w in result.warnings)


def test_hhi_of_empty_is_undefined(engine):
    assert engine.compute("F-HHI", shares=[]).value is None


# --------------------------------------------------------------------------
# Deltas
# --------------------------------------------------------------------------

def test_delta_is_a_plain_difference(engine):
    assert engine.compute("F-DELTA", value=26.07, baseline=10.948).value == 15.122


def test_delta_can_be_negative(engine):
    assert engine.compute("F-DELTA", value=5, baseline=10).value == -5.0


def test_delta_with_null_is_undefined(engine):
    assert engine.compute("F-DELTA", value=None, baseline=10).value is None


def test_pct_delta_matches_a_hand_calculation(engine):
    assert engine.compute("F-PCT-DELTA", value=110, baseline=100).value == 10.0
    assert engine.compute("F-PCT-DELTA", value=90, baseline=100).value == -10.0


def test_pct_delta_from_zero_baseline_is_undefined_not_infinite(engine):
    result = engine.compute("F-PCT-DELTA", value=5, baseline=0)
    assert result.value is None
    assert not math.isinf(result.value or 0)
    assert "denominator is zero" in result.warnings[0]


def test_pct_delta_from_negative_baseline_is_flagged(engine):
    result = engine.compute("F-PCT-DELTA", value=5, baseline=-10)
    assert any("negative" in w for w in result.warnings)


def test_pct_delta_of_no_change_is_zero(engine):
    assert engine.compute("F-PCT-DELTA", value=100, baseline=100).value == 0.0


# --------------------------------------------------------------------------
# Percentiles and fences
# --------------------------------------------------------------------------

def test_percentile_boundaries(engine):
    values = [1, 2, 3, 4]
    assert engine.compute("F-PERCENTILE", values=values, percentile=0).value == 1.0
    assert engine.compute("F-PERCENTILE", values=values, percentile=100).value == 4.0
    assert engine.compute("F-PERCENTILE", values=values, percentile=50).value == 2.5


def test_percentile_out_of_range_is_rejected(engine):
    with pytest.raises(FormulaError, match="between 0 and 100"):
        engine.compute("F-PERCENTILE", values=[1, 2], percentile=101)
    with pytest.raises(FormulaError, match="between 0 and 100"):
        engine.compute("F-PERCENTILE", values=[1, 2], percentile=-1)


def test_percentile_of_empty_is_undefined(engine):
    assert engine.compute("F-PERCENTILE", values=[], percentile=50).value is None


def test_iqr_fences(engine):
    result = engine.compute("F-IQR", values=[1, 2, 3, 4, 5, 6, 7, 8])
    assert result.value["q1"] == 2.75
    assert result.value["q3"] == 6.25
    assert result.value["iqr"] == 3.5
    assert result.value["lower_fence"] == pytest.approx(-2.5)
    assert result.value["upper_fence"] == pytest.approx(11.5)


def test_iqr_needs_at_least_four_values(engine):
    result = engine.compute("F-IQR", values=[1, 2, 3])
    assert result.value is None
    assert "fewer than 4" in result.warnings[0]


def test_iqr_of_identical_values_is_flagged(engine):
    result = engine.compute("F-IQR", values=[5, 5, 5, 5])
    assert result.value["iqr"] == 0.0
    assert any("zero" in w for w in result.warnings)


def test_iqr_multiplier_widens_the_fence(engine):
    narrow = engine.compute("F-IQR", values=range(1, 9), multiplier=1.5).value
    wide = engine.compute("F-IQR", values=range(1, 9), multiplier=3.0).value
    assert wide["upper_fence"] > narrow["upper_fence"]


def test_zscore(engine):
    assert engine.compute("F-ZSCORE", value=12, mean=10, std=2).value == 1.0
    assert engine.compute("F-ZSCORE", value=8, mean=10, std=2).value == -1.0


def test_zscore_with_zero_deviation_is_undefined(engine):
    result = engine.compute("F-ZSCORE", value=12, mean=10, std=0)
    assert result.value is None
    assert "standard deviation is zero" in result.warnings[0]


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------

def test_datediff_counts_whole_days(engine):
    assert engine.compute("F-DATEDIFF", end="2024-01-11", start="2024-01-01").value == 10


def test_datediff_keeps_negative_durations(engine):
    result = engine.compute("F-DATEDIFF", end="2024-01-01", start="2024-01-11")
    assert result.value == -10
    assert "negative duration" in result.warnings[0]


def test_datediff_of_the_same_day_is_zero(engine):
    assert engine.compute("F-DATEDIFF", end="2024-01-01", start="2024-01-01").value == 0


def test_datediff_with_a_missing_date_is_undefined(engine):
    assert engine.compute("F-DATEDIFF", end=None, start="2024-01-01").value is None
    assert engine.compute("F-DATEDIFF", end="2024-01-01", start="WIP").value is None


def test_datediff_crosses_a_leap_day(engine):
    assert engine.compute("F-DATEDIFF", end="2024-03-01", start="2024-02-28").value == 2


def test_datebucket_keys_sort_chronologically(engine):
    keys = [engine.compute("F-DATEBUCKET", value=d, grain="month").value
            for d in ("2023-10-01", "2023-01-15", "2023-02-01")]
    assert sorted(keys) == ["2023-01", "2023-02", "2023-10"]


def test_datebucket_quarter_and_year(engine):
    assert engine.compute("F-DATEBUCKET", value="2023-11-15", grain="quarter").value == "2023-Q4"
    assert engine.compute("F-DATEBUCKET", value="2023-01-01", grain="quarter").value == "2023-Q1"
    assert engine.compute("F-DATEBUCKET", value="2023-06-15", grain="year").value == "2023"


def test_datebucket_quarter_boundaries(engine):
    for month, quarter in ((3, 1), (4, 2), (6, 2), (7, 3), (9, 3), (10, 4), (12, 4)):
        value = "2023-{:02d}-15".format(month)
        assert engine.compute("F-DATEBUCKET", value=value,
                              grain="quarter").value == "2023-Q{}".format(quarter)


def test_datebucket_rejects_an_unknown_grain(engine):
    with pytest.raises(FormulaError, match="month, quarter or year"):
        engine.compute("F-DATEBUCKET", value="2023-01-01", grain="fortnight")


def test_datebucket_of_a_missing_date(engine):
    assert engine.compute("F-DATEBUCKET", value=None).value is None


# --------------------------------------------------------------------------
# Composite
# --------------------------------------------------------------------------

def test_composite_is_a_weighted_mean(engine):
    result = engine.compute("F-COMPOSITE",
                            components={"a": 80.0, "b": 60.0},
                            weights={"a": 0.5, "b": 0.5})
    assert result.value == 70.0


def test_composite_respects_unequal_weights(engine):
    result = engine.compute("F-COMPOSITE",
                            components={"a": 100.0, "b": 0.0},
                            weights={"a": 0.75, "b": 0.25})
    assert result.value == 75.0


def test_composite_renormalises_when_a_component_is_missing(engine):
    """A missing component must not drag the score toward zero."""
    result = engine.compute("F-COMPOSITE",
                            components={"a": 80.0},
                            weights={"a": 0.5, "b": 0.5})
    assert result.value == 80.0
    assert any("missing" in w for w in result.warnings)
    assert result.detail["effective_weight_total"] == 0.5


def test_composite_ignores_null_components(engine):
    result = engine.compute("F-COMPOSITE",
                            components={"a": 80.0, "b": None},
                            weights={"a": 0.5, "b": 0.5})
    assert result.value == 80.0


def test_composite_with_no_usable_component_is_undefined(engine):
    result = engine.compute("F-COMPOSITE", components={}, weights={"a": 1.0})
    assert result.value is None
    assert not result.is_defined


def test_composite_with_zero_weights_is_undefined(engine):
    result = engine.compute("F-COMPOSITE", components={"a": 50.0}, weights={"a": 0.0})
    assert result.value is None


def test_composite_reports_which_components_were_used(engine):
    result = engine.compute("F-COMPOSITE",
                            components={"a": 1.0, "b": 2.0},
                            weights={"a": 0.5, "b": 0.5})
    assert result.detail["components_used"] == ["a", "b"]


# --------------------------------------------------------------------------
# Rounding
# --------------------------------------------------------------------------

def test_rounding_is_applied_at_the_declared_precision(engine):
    assert engine.compute("F-RATE", numerator=1, denominator=3).value == 33.3333
    assert engine.compute("F-RATE", numerator=1, denominator=3, decimals=2).value == 33.33
    assert engine.compute("F-SUM", values=[1.005, 2.005]).value == 3.01


def test_rounding_does_not_compound_across_steps(engine):
    """Rounding once at the end must beat rounding at each step."""
    values = [1.004, 1.004, 1.004]
    summed_then_rounded = engine.compute("F-SUM", values=values, decimals=2).value
    rounded_then_summed = sum(round(v, 2) for v in values)
    assert summed_then_rounded == 3.01
    assert rounded_then_summed == 3.0


def test_nan_input_is_undefined(engine):
    assert engine.compute("F-AVG", values=[float("nan")]).value is None


def test_infinity_is_undefined_and_says_so(engine):
    """An infinite input means an earlier division already failed; it must not propagate."""
    result = engine.compute("F-RATIO", numerator=float("inf"), denominator=1)
    assert result.value is None
    assert any("non-finite" in w for w in result.warnings)

    rate = engine.compute("F-RATE", numerator=1, denominator=float("inf"))
    assert rate.value is None
    assert any("non-finite" in w for w in rate.warnings)


def test_non_numeric_input_is_undefined_not_an_exception(engine):
    result = engine.compute("F-DELTA", value="not a number", baseline=10)
    assert result.value is None
    assert result.warnings


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------

def test_result_is_serialisable_as_evidence(engine):
    evidence = engine.compute("F-RATE", numerator=2737, denominator=25000).as_evidence()
    json.dumps(evidence)
    for field in ("formula_id", "value", "inputs", "semantics_applied",
                  "warnings", "engine_version"):
        assert field in evidence


def test_evidence_records_the_inputs_that_produced_the_value(engine):
    evidence = engine.compute("F-RATE", numerator=2737, denominator=25000).as_evidence()
    assert evidence["inputs"] == {"numerator": 2737.0, "denominator": 25000.0}
    assert evidence["value"] == 10.948


# --------------------------------------------------------------------------
# Cross-check against the real datasets
# --------------------------------------------------------------------------

def test_formulas_reproduce_known_portfolio_figures(engine, dataset):
    """Hand-verifiable numbers computed through the engine, not around it."""
    policy = dataset.table("policy")
    status = policy.col("POLICY.STATUS")

    lapse = engine.compute("F-RATE", numerator=int((status == "Lapsed").sum()),
                           denominator=len(policy))
    assert lapse.value == 10.948

    total_ape = engine.compute("F-SUM", values=policy.num("APE"))
    assert total_ape.value == 2306202661.89

    mean_ape = engine.compute("F-AVG", values=policy.num("APE"))
    assert mean_ape.value == 92248.1065

    median_ape = engine.compute("F-MEDIAN", values=policy.num("APE"))
    assert median_ape.value == 50000.0

    inforce = engine.compute("F-RATE",
                             numerator=int((status == "Premium paying (regular)").sum()),
                             denominator=len(policy))
    assert inforce.value == 87.22


def test_status_shares_reconcile_to_one_hundred(engine, dataset):
    policy = dataset.table("policy")
    counts = policy.col("POLICY.STATUS").value_counts()
    shares = [engine.compute("F-SHARE", value=int(c), total=len(policy)).value
              for c in counts]
    assert round(sum(shares), 4) == 100.0


def test_claim_payout_ratio_excludes_the_unsettled_claim(engine, dataset):
    claims = dataset.table("claims")
    settled = claims.frame[claims.col("Status.of.Claim") == "Paid"]
    claimed = engine.compute("F-SUM", values=settled["Claim.Amount"])
    paid = engine.compute("F-SUM", values=settled["Amount.Paid"])
    ratio = engine.compute("F-RATIO", numerator=paid.value, denominator=claimed.value)
    assert paid.inputs["n"] == 133
    assert 0.99 < ratio.value < 1.01


def test_renewal_rate_uses_the_persistency_population(engine, dataset):
    pers = dataset.table("persistency")
    paid = int((pers.col("Renewal_Status") == "Paid").sum())
    rate = engine.compute("F-RATE", numerator=paid, denominator=len(pers))
    assert paid == 12439
    assert rate.value == pytest.approx(50.6846, abs=0.001)
