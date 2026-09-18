"""Deterministic formula engine.

Every authoritative number in the system comes from here. The LLM never computes
a figure; it only explains figures this engine produced.

Three things are explicit for every formula, because they are where calculations
usually go quietly wrong:

  - null semantics      what is excluded, and from which side of a fraction
  - zero denominators   undefined, never silently zero or infinity
  - rounding            applied once, at the end, to a declared number of places
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

ENGINE_VERSION = "1.0.0"

UNDEFINED_ZERO_DENOMINATOR = "undefined: denominator is zero"
NULL_OR_NON_FINITE = "undefined: an input was null or non-finite"
EMPTY_POPULATION = "population is empty"


class FormulaError(Exception):
    """Raised when a formula is called in a way it cannot honour."""


@dataclass
class FormulaResult:
    """A computed value plus everything needed to audit it."""
    formula_id: str
    value: Any
    inputs: Dict[str, Any]
    semantics: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_defined(self) -> bool:
        return self.value is not None

    def as_evidence(self) -> Dict[str, Any]:
        return {
            "formula_id": self.formula_id,
            "value": self.value,
            "inputs": self.inputs,
            "semantics_applied": self.semantics,
            "warnings": list(self.warnings),
            "detail": self.detail,
            "engine_version": ENGINE_VERSION,
        }


@dataclass
class Formula:
    """A registered, inspectable calculation."""
    id: str
    name: str
    expression: str
    description: str
    fn: Callable[..., "FormulaResult"]
    parameters: List[str]
    returns: str
    null_handling: str
    zero_denominator: str
    rounding: Optional[int]
    version: str = "1.0.0"
    examples: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "expression": self.expression,
            "description": self.description, "parameters": self.parameters,
            "returns": self.returns, "null_handling": self.null_handling,
            "zero_denominator": self.zero_denominator,
            "rounding_decimals": self.rounding, "version": self.version,
            "examples": self.examples,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(values: Iterable) -> pd.Series:
    """Coerce to a numeric Series with nulls dropped."""
    series = values if isinstance(values, pd.Series) else pd.Series(list(values))
    return pd.to_numeric(series, errors="coerce").dropna()


def _round(value: Optional[float], places: Optional[int]) -> Optional[float]:
    if value is None or places is None:
        return value
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), places)


def _num(value: Any) -> Optional[float]:
    """Coerce to float, mapping null and non-finite values to None.

    Infinity is treated as undefined rather than propagated: an infinite input
    means an earlier division already went wrong, and carrying it forward would
    produce a confident-looking answer built on it.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


# ---------------------------------------------------------------------------
# Formula implementations
# ---------------------------------------------------------------------------

def f_count(values: Iterable = None, population: int = None) -> FormulaResult:
    if values is None and population is None:
        raise FormulaError("F-COUNT needs either values or population")
    if values is not None:
        series = values if isinstance(values, pd.Series) else pd.Series(list(values))
        count = int(series.notna().sum())
        nulls = int(series.isna().sum())
    else:
        count, nulls = int(population), 0
    return FormulaResult(
        "F-COUNT", count, {"count": count},
        {"null_handling": "null values are not counted", "rounding": None},
        detail={"nulls_excluded": nulls})


def f_count_distinct(values: Iterable) -> FormulaResult:
    series = values if isinstance(values, pd.Series) else pd.Series(list(values))
    count = int(series.nunique(dropna=True))
    return FormulaResult(
        "F-COUNT-DISTINCT", count, {"count": count},
        {"null_handling": "nulls are not a distinct value", "rounding": None},
        detail={"rows": int(len(series)), "nulls_excluded": int(series.isna().sum())})


def f_sum(values: Iterable, decimals: int = 2) -> FormulaResult:
    clean = _clean(values)
    warnings = []
    if clean.empty:
        warnings.append(EMPTY_POPULATION)
        value = 0.0
    else:
        value = float(clean.sum())
    return FormulaResult(
        "F-SUM", _round(value, decimals), {"n": int(len(clean))},
        {"null_handling": "nulls excluded from the sum",
         "empty_population": "sums to 0.0, flagged as empty", "rounding": decimals},
        warnings, {"values_summed": int(len(clean))})


def f_avg(values: Iterable, decimals: int = 4) -> FormulaResult:
    clean = _clean(values)
    warnings = []
    if clean.empty:
        warnings.append(EMPTY_POPULATION)
        value = None
    else:
        value = float(clean.mean())
    return FormulaResult(
        "F-AVG", _round(value, decimals), {"n": int(len(clean))},
        {"null_handling": "nulls excluded from BOTH numerator and denominator",
         "empty_population": "undefined, not zero", "rounding": decimals},
        warnings, {"values_averaged": int(len(clean))})


def f_median(values: Iterable, decimals: int = 4) -> FormulaResult:
    clean = _clean(values)
    warnings = []
    value = None if clean.empty else float(clean.median())
    if clean.empty:
        warnings.append(EMPTY_POPULATION)
    return FormulaResult(
        "F-MEDIAN", _round(value, decimals), {"n": int(len(clean))},
        {"null_handling": "nulls excluded", "interpolation": "linear between the two middle values",
         "empty_population": "undefined", "rounding": decimals},
        warnings, {"values_used": int(len(clean))})


def f_rate(numerator: float, denominator: float, decimals: int = 4) -> FormulaResult:
    num, den = _num(numerator), _num(denominator)
    warnings = []
    if num is None or den is None:
        warnings.append(NULL_OR_NON_FINITE)
        value = None
    elif den == 0:
        warnings.append(UNDEFINED_ZERO_DENOMINATOR)
        value = None
    elif num > den:
        warnings.append("numerator exceeds denominator; the populations may not match")
        value = 100.0 * num / den
    else:
        value = 100.0 * num / den
    return FormulaResult(
        "F-RATE", _round(value, decimals),
        {"numerator": num, "denominator": den},
        {"null_handling": "a null or non-finite input makes the result undefined",
         "zero_denominator": "undefined, never 0 and never infinity",
         "units": "percent", "rounding": decimals},
        warnings)


def f_ratio(numerator: float, denominator: float, decimals: int = 6) -> FormulaResult:
    num, den = _num(numerator), _num(denominator)
    warnings = []
    if num is None or den is None:
        warnings.append(NULL_OR_NON_FINITE)
        value = None
    elif den == 0:
        warnings.append(UNDEFINED_ZERO_DENOMINATOR)
        value = None
    else:
        value = num / den
    return FormulaResult(
        "F-RATIO", _round(value, decimals),
        {"numerator": num, "denominator": den},
        {"null_handling": "a null or non-finite input makes the result undefined",
         "zero_denominator": "undefined", "units": "ratio", "rounding": decimals},
        warnings)


def f_share(value: float, total: float, decimals: int = 4) -> FormulaResult:
    result = f_rate(value, total, decimals)
    result.formula_id = "F-SHARE"
    result.inputs = {"value": result.inputs["numerator"], "total": result.inputs["denominator"]}
    result.semantics["units"] = "percent of total"
    return result


def f_cumshare(values: Sequence[float], decimals: int = 4) -> FormulaResult:
    clean = _clean(values)
    warnings = []
    total = float(clean.sum())
    if clean.empty:
        warnings.append(EMPTY_POPULATION)
        cumulative = []
    elif total == 0:
        warnings.append(UNDEFINED_ZERO_DENOMINATOR)
        cumulative = []
    else:
        cumulative = [_round(v, decimals) for v in (100.0 * clean.cumsum() / total).tolist()]
    return FormulaResult(
        "F-CUMSHARE", cumulative, {"n": int(len(clean)), "total": _round(total, 2)},
        {"null_handling": "nulls excluded before accumulating",
         "ordering": "the caller must sort first; order is preserved as given",
         "zero_denominator": "undefined", "rounding": decimals},
        warnings, {"final_share": cumulative[-1] if cumulative else None})


def f_delta(value: float, baseline: float, decimals: int = 4) -> FormulaResult:
    val, base = _num(value), _num(baseline)
    warnings = []
    if val is None or base is None:
        warnings.append(NULL_OR_NON_FINITE)
        result = None
    else:
        result = val - base
    return FormulaResult(
        "F-DELTA", _round(result, decimals), {"value": val, "baseline": base},
        {"null_handling": "a null or non-finite input makes the result undefined",
         "units": "same units as the inputs", "rounding": decimals},
        warnings)


def f_pct_delta(value: float, baseline: float, decimals: int = 4) -> FormulaResult:
    val, base = _num(value), _num(baseline)
    warnings = []
    if val is None or base is None:
        warnings.append(NULL_OR_NON_FINITE)
        result = None
    elif base == 0:
        warnings.append(UNDEFINED_ZERO_DENOMINATOR)
        result = None
    else:
        result = 100.0 * (val - base) / base
        if base < 0:
            warnings.append("baseline is negative; the sign of the change may mislead")
    return FormulaResult(
        "F-PCT-DELTA", _round(result, decimals), {"value": val, "baseline": base},
        {"null_handling": "a null or non-finite input makes the result undefined",
         "zero_denominator": "undefined, never infinity",
         "units": "percent change", "rounding": decimals},
        warnings)


def f_percentile(values: Iterable, percentile: float, decimals: int = 4) -> FormulaResult:
    if not 0 <= percentile <= 100:
        raise FormulaError("percentile must be between 0 and 100, got {}".format(percentile))
    clean = _clean(values)
    warnings = []
    value = None if clean.empty else float(np.percentile(clean.to_numpy(), percentile))
    if clean.empty:
        warnings.append(EMPTY_POPULATION)
    return FormulaResult(
        "F-PERCENTILE", _round(value, decimals),
        {"percentile": percentile, "n": int(len(clean))},
        {"null_handling": "nulls excluded", "interpolation": "linear",
         "empty_population": "undefined", "rounding": decimals},
        warnings)


def f_datediff(end: Any, start: Any) -> FormulaResult:
    warnings = []
    end_ts, start_ts = pd.to_datetime(end, errors="coerce"), pd.to_datetime(start, errors="coerce")
    if pd.isna(end_ts) or pd.isna(start_ts):
        warnings.append("undefined: a date could not be parsed or was missing")
        value = None
    else:
        value = int((end_ts - start_ts).days)
        if value < 0:
            warnings.append("negative duration: the end date precedes the start date")
    return FormulaResult(
        "F-DATEDIFF", value,
        {"end": str(end_ts.date()) if not pd.isna(end_ts) else None,
         "start": str(start_ts.date()) if not pd.isna(start_ts) else None},
        {"null_handling": "a missing date makes the result undefined",
         "negative_values": "reported, never clipped to zero",
         "units": "whole days", "rounding": None},
        warnings)


def f_datebucket(value: Any, grain: str = "month") -> FormulaResult:
    if grain not in ("month", "quarter", "year"):
        raise FormulaError("grain must be month, quarter or year, got {!r}".format(grain))
    warnings = []
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        warnings.append("undefined: the date could not be parsed or was missing")
        key = None
    elif grain == "year":
        key = "{:04d}".format(ts.year)
    elif grain == "quarter":
        key = "{:04d}-Q{}".format(ts.year, (ts.month - 1) // 3 + 1)
    else:
        key = "{:04d}-{:02d}".format(ts.year, ts.month)
    return FormulaResult(
        "F-DATEBUCKET", key, {"grain": grain},
        {"null_handling": "a missing date yields no bucket",
         "ordering": "keys are zero-padded so lexical order equals chronological order",
         "rounding": None},
        warnings)


def f_iqr(values: Iterable, multiplier: float = 1.5, decimals: int = 4) -> FormulaResult:
    clean = _clean(values)
    warnings = []
    if len(clean) < 4:
        warnings.append("fewer than 4 values: quartiles are not meaningful")
        return FormulaResult(
            "F-IQR", None, {"n": int(len(clean)), "multiplier": multiplier},
            {"null_handling": "nulls excluded", "minimum_n": 4, "rounding": decimals}, warnings)
    q1 = float(np.percentile(clean.to_numpy(), 25))
    q3 = float(np.percentile(clean.to_numpy(), 75))
    iqr = q3 - q1
    bounds = {"q1": _round(q1, decimals), "q3": _round(q3, decimals),
              "iqr": _round(iqr, decimals),
              "lower_fence": _round(q1 - multiplier * iqr, decimals),
              "upper_fence": _round(q3 + multiplier * iqr, decimals)}
    if iqr == 0:
        warnings.append("interquartile range is zero: the fences collapse onto the quartile")
    return FormulaResult(
        "F-IQR", bounds, {"n": int(len(clean)), "multiplier": multiplier},
        {"null_handling": "nulls excluded", "interpolation": "linear",
         "minimum_n": 4, "rounding": decimals},
        warnings)


def f_zscore(value: float, mean: float, std: float, decimals: int = 4) -> FormulaResult:
    val, mu, sigma = _num(value), _num(mean), _num(std)
    warnings = []
    if val is None or mu is None or sigma is None:
        warnings.append(NULL_OR_NON_FINITE)
        result = None
    elif sigma == 0:
        warnings.append("undefined: standard deviation is zero, so no value is an outlier")
        result = None
    else:
        result = (val - mu) / sigma
    return FormulaResult(
        "F-ZSCORE", _round(result, decimals), {"value": val, "mean": mu, "std": sigma},
        {"null_handling": "a null or non-finite input makes the result undefined",
         "zero_denominator": "undefined when the deviation is zero",
         "units": "standard deviations", "rounding": decimals},
        warnings)


def f_hhi(shares: Iterable, decimals: int = 4) -> FormulaResult:
    """Herfindahl-style index over percentage shares."""
    clean = _clean(shares)
    warnings = []
    if clean.empty:
        warnings.append(EMPTY_POPULATION)
        value = None
    else:
        total = float(clean.sum())
        if abs(total - 100.0) > 0.5:
            warnings.append(
                "shares sum to {:.2f}, not 100; the index assumes percentage shares".format(total))
        value = float((clean ** 2).sum())
    return FormulaResult(
        "F-HHI", _round(value, decimals), {"n": int(len(clean))},
        {"null_handling": "nulls excluded",
         "scale": "0 to 10,000; 10,000 means a single group holds everything",
         "input_units": "percentage shares", "rounding": decimals},
        warnings, {"groups": int(len(clean))})


def f_composite(components: Dict[str, float], weights: Dict[str, float],
                decimals: int = 4) -> FormulaResult:
    """Weighted sum of already-normalised components."""
    warnings = []
    missing = [k for k in weights if k not in components]
    if missing:
        warnings.append("components missing for {}; they are excluded and weights "
                        "renormalised".format(sorted(missing)))
    usable = {k: _num(components[k]) for k in weights if k in components}
    usable = {k: v for k, v in usable.items() if v is not None}
    if not usable:
        warnings.append("no usable components")
        return FormulaResult(
            "F-COMPOSITE", None, {"components": {}, "weights": dict(weights)},
            {"null_handling": "components that are null are excluded and weights renormalised",
             "rounding": decimals}, warnings)

    weight_total = sum(weights[k] for k in usable)
    if weight_total == 0:
        warnings.append(UNDEFINED_ZERO_DENOMINATOR)
        return FormulaResult(
            "F-COMPOSITE", None, {"components": usable, "weights": dict(weights)},
            {"null_handling": "weights renormalised", "rounding": decimals}, warnings)

    value = sum(usable[k] * weights[k] for k in usable) / weight_total
    return FormulaResult(
        "F-COMPOSITE", _round(value, decimals),
        {"components": usable, "weights": dict(weights)},
        {"null_handling": "components that are null are excluded and weights renormalised",
         "normalisation": "weights are renormalised to sum to 1 over usable components",
         "precondition": "components must already be on a comparable scale",
         "rounding": decimals},
        warnings,
        {"components_used": sorted(usable), "effective_weight_total": _round(weight_total, 6)})


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Formula] = {}


def _register(formula: Formula) -> None:
    if formula.id in _REGISTRY:
        raise FormulaError("duplicate formula id {}".format(formula.id))
    _REGISTRY[formula.id] = formula


_register(Formula(
    id="F-COUNT", name="Count", expression="count of non-null values",
    description="Number of rows or values matching the declared population.",
    fn=f_count, parameters=["values", "population"], returns="integer",
    null_handling="null values are not counted",
    zero_denominator="not applicable", rounding=None,
    examples=[{"inputs": {"values": [1, 2, None, 4]}, "expected": 3}]))

_register(Formula(
    id="F-COUNT-DISTINCT", name="Distinct count", expression="count of distinct non-null values",
    description="Number of distinct values, used for entity counts such as clients or agents.",
    fn=f_count_distinct, parameters=["values"], returns="integer",
    null_handling="null is not treated as a distinct value",
    zero_denominator="not applicable", rounding=None,
    examples=[{"inputs": {"values": ["a", "b", "a", None]}, "expected": 2}]))

_register(Formula(
    id="F-SUM", name="Sum", expression="sum(values)",
    description="Total of a numeric column over the declared population.",
    fn=f_sum, parameters=["values", "decimals"], returns="number",
    null_handling="nulls excluded from the sum",
    zero_denominator="not applicable", rounding=2,
    examples=[{"inputs": {"values": [1.5, 2.5, None]}, "expected": 4.0}]))

_register(Formula(
    id="F-AVG", name="Arithmetic mean", expression="sum(values) / count(non-null values)",
    description="Mean, dividing by present values rather than by row count.",
    fn=f_avg, parameters=["values", "decimals"], returns="number or null",
    null_handling="nulls excluded from BOTH numerator and denominator",
    zero_denominator="undefined when the population is empty, never zero", rounding=4,
    examples=[{"inputs": {"values": [2, 4, None]}, "expected": 3.0},
              {"inputs": {"values": []}, "expected": None}]))

_register(Formula(
    id="F-MEDIAN", name="Median", expression="50th percentile",
    description="Middle value; reported alongside the mean wherever a distribution is skewed.",
    fn=f_median, parameters=["values", "decimals"], returns="number or null",
    null_handling="nulls excluded",
    zero_denominator="undefined when the population is empty", rounding=4,
    examples=[{"inputs": {"values": [1, 2, 3, 4]}, "expected": 2.5}]))

_register(Formula(
    id="F-RATE", name="Rate", expression="100 * numerator / denominator",
    description="Percentage of a population meeting a condition.",
    fn=f_rate, parameters=["numerator", "denominator", "decimals"], returns="percent or null",
    null_handling="a null input makes the result undefined",
    zero_denominator="undefined, never 0 and never infinity", rounding=4,
    examples=[{"inputs": {"numerator": 2737, "denominator": 25000}, "expected": 10.948},
              {"inputs": {"numerator": 5, "denominator": 0}, "expected": None}]))

_register(Formula(
    id="F-RATIO", name="Ratio", expression="numerator / denominator",
    description="Unitless ratio such as sum assured to premium.",
    fn=f_ratio, parameters=["numerator", "denominator", "decimals"], returns="number or null",
    null_handling="a null input makes the result undefined",
    zero_denominator="undefined", rounding=6,
    examples=[{"inputs": {"numerator": 1000000, "denominator": 50000}, "expected": 20.0}]))

_register(Formula(
    id="F-SHARE", name="Share of total", expression="100 * value / total",
    description="A group's percentage share of an overall total.",
    fn=f_share, parameters=["value", "total", "decimals"], returns="percent or null",
    null_handling="a null input makes the result undefined",
    zero_denominator="undefined", rounding=4,
    examples=[{"inputs": {"value": 25, "total": 200}, "expected": 12.5}]))

_register(Formula(
    id="F-CUMSHARE", name="Cumulative share", expression="running sum / total, as a percentage",
    description="Concentration curve over an ordered series.",
    fn=f_cumshare, parameters=["values", "decimals"], returns="list of percents",
    null_handling="nulls excluded before accumulating",
    zero_denominator="undefined when the total is zero", rounding=4,
    examples=[{"inputs": {"values": [50, 30, 20]}, "expected": [50.0, 80.0, 100.0]}]))

_register(Formula(
    id="F-DELTA", name="Difference vs baseline", expression="value - baseline",
    description="Absolute difference in the inputs' own units.",
    fn=f_delta, parameters=["value", "baseline", "decimals"], returns="number or null",
    null_handling="a null input makes the result undefined",
    zero_denominator="not applicable", rounding=4,
    examples=[{"inputs": {"value": 26.07, "baseline": 10.948}, "expected": 15.122}]))

_register(Formula(
    id="F-PCT-DELTA", name="Percentage change vs baseline",
    expression="100 * (value - baseline) / baseline",
    description="Relative change against a stated baseline.",
    fn=f_pct_delta, parameters=["value", "baseline", "decimals"], returns="percent or null",
    null_handling="a null input makes the result undefined",
    zero_denominator="undefined, never infinity", rounding=4,
    examples=[{"inputs": {"value": 110, "baseline": 100}, "expected": 10.0},
              {"inputs": {"value": 5, "baseline": 0}, "expected": None}]))

_register(Formula(
    id="F-PERCENTILE", name="Percentile", expression="nth percentile with linear interpolation",
    description="Distribution position, used for bands and outlier fences.",
    fn=f_percentile, parameters=["values", "percentile", "decimals"], returns="number or null",
    null_handling="nulls excluded", zero_denominator="undefined when empty", rounding=4,
    examples=[{"inputs": {"values": [1, 2, 3, 4], "percentile": 50}, "expected": 2.5}]))

_register(Formula(
    id="F-DATEDIFF", name="Date difference", expression="end - start, in whole days",
    description="Duration between two parsed dates.",
    fn=f_datediff, parameters=["end", "start"], returns="integer days or null",
    null_handling="a missing date makes the result undefined",
    zero_denominator="not applicable", rounding=None,
    examples=[{"inputs": {"end": "2024-01-11", "start": "2024-01-01"}, "expected": 10},
              {"inputs": {"end": "2024-01-01", "start": "2024-01-11"}, "expected": -10}]))

_register(Formula(
    id="F-DATEBUCKET", name="Date bucket", expression="zero-padded period key from a date",
    description="Chronologically sortable period key at month, quarter or year grain.",
    fn=f_datebucket, parameters=["value", "grain"], returns="string key or null",
    null_handling="a missing date yields no bucket",
    zero_denominator="not applicable", rounding=None,
    examples=[{"inputs": {"value": "2023-01-15", "grain": "month"}, "expected": "2023-01"},
              {"inputs": {"value": "2023-11-15", "grain": "quarter"}, "expected": "2023-Q4"}]))

_register(Formula(
    id="F-IQR", name="Interquartile fence", expression="Q1 - k*IQR and Q3 + k*IQR",
    description="Outlier fence, preferred over standard deviations on skewed data.",
    fn=f_iqr, parameters=["values", "multiplier", "decimals"], returns="bounds object or null",
    null_handling="nulls excluded",
    zero_denominator="fences collapse when the IQR is zero; this is flagged", rounding=4,
    examples=[{"inputs": {"values": [1, 2, 3, 4, 5, 6, 7, 8]},
               "expected": {"q1": 2.75, "q3": 6.25}}]))

_register(Formula(
    id="F-ZSCORE", name="Z-score", expression="(value - mean) / standard deviation",
    description="Distance from the mean in standard deviations.",
    fn=f_zscore, parameters=["value", "mean", "std", "decimals"], returns="number or null",
    null_handling="a null input makes the result undefined",
    zero_denominator="undefined when the deviation is zero", rounding=4,
    examples=[{"inputs": {"value": 12, "mean": 10, "std": 2}, "expected": 1.0},
              {"inputs": {"value": 12, "mean": 10, "std": 0}, "expected": None}]))

_register(Formula(
    id="F-HHI", name="Concentration index", expression="sum of squared percentage shares",
    description="Herfindahl-style concentration measure over group shares.",
    fn=f_hhi, parameters=["shares", "decimals"], returns="number or null",
    null_handling="nulls excluded",
    zero_denominator="undefined when empty", rounding=4,
    examples=[{"inputs": {"shares": [100.0]}, "expected": 10000.0},
              {"inputs": {"shares": [50.0, 50.0]}, "expected": 5000.0}]))

_register(Formula(
    id="F-COMPOSITE", name="Weighted composite",
    expression="sum(component * weight) / sum(weight)",
    description=("Weighted score over components already normalised to a comparable scale. "
                 "Weights are declared in config and reported with the result."),
    fn=f_composite, parameters=["components", "weights", "decimals"], returns="number or null",
    null_handling="null components are excluded and the weights renormalised",
    zero_denominator="undefined when no usable component has weight", rounding=4,
    examples=[{"inputs": {"components": {"a": 80.0, "b": 60.0},
                          "weights": {"a": 0.5, "b": 0.5}}, "expected": 70.0}]))


class FormulaEngine:
    """The single authority for computed values."""

    version = ENGINE_VERSION

    def __init__(self, registry: Optional[Dict[str, Formula]] = None):
        self._registry = registry if registry is not None else _REGISTRY

    @property
    def formula_ids(self) -> List[str]:
        return sorted(self._registry)

    def get(self, formula_id: str) -> Formula:
        if formula_id not in self._registry:
            raise FormulaError(
                "unknown formula {!r}; registered: {}".format(formula_id, self.formula_ids))
        return self._registry[formula_id]

    def compute(self, formula_id: str, **kwargs) -> FormulaResult:
        formula = self.get(formula_id)
        unexpected = set(kwargs) - set(formula.parameters)
        if unexpected:
            raise FormulaError(
                "{} does not accept {}; parameters are {}".format(
                    formula_id, sorted(unexpected), formula.parameters))
        result = formula.fn(**kwargs)
        result.semantics.setdefault("formula_version", formula.version)
        return result

    def describe(self) -> List[Dict[str, Any]]:
        return [self._registry[i].as_dict() for i in self.formula_ids]

    def verify_examples(self) -> List[str]:
        """Run every declared example; returns a list of mismatch descriptions."""
        failures = []
        for formula_id in self.formula_ids:
            formula = self._registry[formula_id]
            for example in formula.examples:
                result = self.compute(formula_id, **example["inputs"])
                expected = example["expected"]
                actual = result.value
                if isinstance(expected, dict) and isinstance(actual, dict):
                    ok = all(actual.get(k) == v for k, v in expected.items())
                else:
                    ok = actual == expected
                if not ok:
                    failures.append("{}: expected {!r}, got {!r} for {!r}".format(
                        formula_id, expected, actual, example["inputs"]))
        return failures


def get_engine() -> FormulaEngine:
    return FormulaEngine()
