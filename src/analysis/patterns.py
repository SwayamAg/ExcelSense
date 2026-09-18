"""Analytical patterns.

Every PS executes through one of these. They are deliberately few: a PS that needs
a fourteenth pattern is usually a PS that has not been thought through, and keeping
the set small means each one can be tested hard rather than each of 103 PSs being
tested once.

Each handler receives a resolved plan and an EvidencePackage, and is responsible
for filling the package. No handler computes a number itself: every value comes
from the formula engine, so every figure carries its formula id and inputs.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.data import Dataset, query
from src.data.query_engine import Query
from src.logic import get_engine, get_rule_engine
from src.ps.evidence import EvidencePackage

FORMULAS = get_engine()
RULES = get_rule_engine()

MAX_RECORDS = 50           # never return an unbounded record set
MAX_TABLE_ROWS = 200


class PatternError(Exception):
    """Raised when a plan cannot be executed as written."""


# ---------------------------------------------------------------------------
# Plan helpers
# ---------------------------------------------------------------------------

def _base_query(dataset: Dataset, plan: Dict[str, Any], evidence: EvidencePackage) -> Query:
    """Start from the base table, apply joins and filters, record everything."""
    base = plan["base"]
    q = query(dataset, base)

    for relationship in plan.get("joins", []):
        how = plan.get("join_how", {}).get(relationship, "left")
        q = q.join(relationship, how=how)

    for spec in plan.get("filters", []):
        column, comparator, value = spec[0], spec[1], (spec[2] if len(spec) > 2 else None)
        q = q.filter(column, comparator, value)

    # Derivations come before rules: a rule such as BR-005 filters on a derived
    # duration, so the column has to exist before the rule can be applied.
    for spec in plan.get("derive", []):
        q = _apply_derivation(q, spec)

    for rule_id, kwargs in plan.get("rules", []):
        frame, application = RULES.apply(rule_id, q.frame, **kwargs)
        evidence.add_rule(application)
        q = q.filter_rows(
            pd.Series(True, index=frame.index), "{}: {}".format(rule_id, application.note))

    return q


def _apply_derivation(q: Query, spec: Dict[str, Any]) -> Query:
    kind = spec["kind"]
    if kind == "date_diff":
        return q.date_diff(spec["name"], spec["end"], spec["start"])
    if kind == "date_bucket":
        return q.date_bucket(spec["name"], spec["column"], spec.get("grain", "month"))
    if kind == "ratio":
        numerator = q.frame[q.resolve(spec["numerator"])]
        denominator = q.frame[q.resolve(spec["denominator"])]
        safe = denominator.replace(0, np.nan)
        return q.derive(spec["name"], numerator / safe,
                        "{} / {} (zero denominators excluded)".format(
                            spec["numerator"], spec["denominator"]))
    if kind == "band":
        column = q.resolve(spec["column"])
        series = pd.cut(q.frame[column], bins=spec["bins"], labels=spec["labels"],
                        include_lowest=True)
        return q.derive(spec["name"], series.astype("object"),
                        "{} banded into {}".format(column, spec["labels"]))
    if kind == "flag":
        column = q.resolve(spec["column"])
        mask = q.frame[column].isin(spec["values"]) if "values" in spec \
            else q.frame[column].notna()
        return q.derive(spec["name"], mask.map({True: 1, False: 0}),
                        "1 when {} matches {}".format(column, spec.get("values", "not null")))
    raise PatternError("unknown derivation {!r}".format(kind))


def _record_provenance(q: Query, evidence: EvidencePackage) -> None:
    prov = q.provenance
    evidence.provenance = prov.as_dict()
    for table in prov.tables_used:
        if table not in evidence.tables_used:
            evidence.tables_used.append(table)
    evidence.join_path = list(prov.join_path)
    evidence.filters = list(prov.filters)
    for note in prov.notes:
        if note not in evidence.warnings:
            evidence.warnings.append(note)


def _measure_values(q: Query, measures: Dict[str, Sequence]) -> Dict[str, tuple]:
    """Translate a plan's measure spec into query-engine aggregations."""
    out = {}
    for alias, spec in measures.items():
        column = spec[0] if spec[0] else None
        out[alias] = (column, spec[1])
    return out


MEASURE_FORMULAS = {"count_all": "F-COUNT", "count": "F-COUNT", "distinct": "F-COUNT-DISTINCT",
                    "sum": "F-SUM", "mean": "F-AVG", "median": "F-MEDIAN"}


def _record_measure_formulas(q: Query, measures: Dict[str, Sequence],
                             evidence: EvidencePackage) -> None:
    """Name the formula behind every grouped statistic, with its whole-population value.

    Grouped aggregation runs in the query engine for speed, so without this the
    reported numbers would carry no formula id. Recording the same statistic over the
    whole population both names the formula and gives the reader the portfolio figure
    that the per-group values should be read against.
    """
    for alias, spec in measures.items():
        column = spec[0] if spec[0] else None
        func = spec[1]
        formula_id = MEASURE_FORMULAS.get(func)
        if formula_id is None:
            continue
        if func == "count_all" or column is None:
            result = FORMULAS.compute("F-COUNT", population=len(q))
        elif func == "distinct":
            result = FORMULAS.compute("F-COUNT-DISTINCT", values=q.frame[q.resolve(column)])
        else:
            result = FORMULAS.compute(formula_id, values=q.frame[q.resolve(column)])
        evidence.add_calculation(
            "{} (whole population)".format(alias), result,
            note="{} over every row in scope, for comparison with the per-group values".format(
                func))


def _add_rates(rows: List[Dict[str, Any]], rates: Dict[str, Dict[str, str]],
               evidence: EvidencePackage) -> None:
    """Compute declared rates through the formula engine, row by row."""
    for alias, spec in rates.items():
        for row in rows:
            result = FORMULAS.compute(
                "F-RATE", numerator=row.get(spec["numerator"]),
                denominator=row.get(spec["denominator"]))
            row[alias] = result.value
        if rows:
            evidence.add_calculation(
                "{} (per group)".format(alias),
                FORMULAS.compute("F-RATE",
                                 numerator=rows[0].get(spec["numerator"]),
                                 denominator=rows[0].get(spec["denominator"])),
                note="100 * {} / {}, applied to every group".format(
                    spec["numerator"], spec["denominator"]))


def _add_shares(rows: List[Dict[str, Any]], shares: Dict[str, str],
                evidence: EvidencePackage) -> None:
    """Each group's share of the column total, recorded as a calculation."""
    for alias, source in shares.items():
        total = sum(r.get(source) or 0 for r in rows)
        for row in rows:
            row[alias] = FORMULAS.compute(
                "F-SHARE", value=row.get(source), total=total).value
        evidence.supporting_metrics["{}_total".format(source)] = total
        if rows:
            evidence.add_calculation(
                "{} (per group)".format(alias),
                FORMULAS.compute("F-SHARE", value=rows[0].get(source), total=total),
                note="100 * {} / {}, applied to every group".format(source, total))


def _trim(rows: List[Dict[str, Any]], limit: int = MAX_TABLE_ROWS) -> List[Dict[str, Any]]:
    return rows[:limit]


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

def pattern_aggregate(dataset: Dataset, plan: Dict[str, Any],
                      evidence: EvidencePackage) -> EvidencePackage:
    """Whole-population metrics. Answers scalar and small-table questions."""
    q = _base_query(dataset, plan, evidence)
    evidence.population_size = len(q)
    if len(q) == 0:
        evidence.mark_insufficient("the declared population contains no rows")
        _record_provenance(q, evidence)
        return evidence

    for alias, spec in plan.get("measures", {}).items():
        column, func = spec[0], spec[1]
        if func == "count_all":
            value = evidence.add_calculation(alias, FORMULAS.compute("F-COUNT", population=len(q)))
        elif func == "distinct":
            value = evidence.add_calculation(
                alias, FORMULAS.compute("F-COUNT-DISTINCT", values=q.frame[q.resolve(column)]))
        else:
            formula = {"sum": "F-SUM", "mean": "F-AVG", "median": "F-MEDIAN"}.get(func)
            if formula is None:
                raise PatternError("aggregate pattern cannot use {!r}".format(func))
            value = evidence.add_calculation(
                alias, FORMULAS.compute(formula, values=q.frame[q.resolve(column)]))
        evidence.aggregates[alias] = value

    for alias, spec in plan.get("condition_counts", {}).items():
        column = q.resolve(spec["column"])
        mask = q.frame[column].isin(spec["values"])
        evidence.aggregates[alias] = evidence.add_calculation(
            alias, FORMULAS.compute("F-COUNT", population=int(mask.sum())),
            note="{} in {}".format(column, spec["values"]))

    for alias, spec in plan.get("rates", {}).items():
        evidence.aggregates[alias] = evidence.add_calculation(
            alias, FORMULAS.compute(
                "F-RATE", numerator=evidence.aggregates.get(spec["numerator"]),
                denominator=evidence.aggregates.get(spec["denominator"])),
            note="100 * {} / {}".format(spec["numerator"], spec["denominator"]))

    for alias, spec in plan.get("percentiles", {}).items():
        evidence.aggregates[alias] = evidence.add_calculation(
            alias, FORMULAS.compute("F-PERCENTILE",
                                    values=q.frame[q.resolve(spec["column"])],
                                    percentile=spec["percentile"]))

    _record_provenance(q, evidence)
    return evidence


def pattern_group_metrics(dataset: Dataset, plan: Dict[str, Any],
                          evidence: EvidencePackage) -> EvidencePackage:
    """Group by one or more columns, measure, rate, rank and compare. The workhorse."""
    q = _base_query(dataset, plan, evidence)
    evidence.population_size = len(q)
    if len(q) == 0:
        evidence.mark_insufficient("the declared population contains no rows")
        _record_provenance(q, evidence)
        return evidence

    grouped = q.group_by(plan["group_by"], _measure_values(q, plan.get("measures", {})),
                         dropna=plan.get("dropna", False))

    minimum = plan.get("min_sample")
    if minimum:
        before = len(grouped)
        grouped = grouped.min_group_size(minimum)
        excluded = before - len(grouped)
        if excluded:
            evidence.add_limitation(
                "{} {} fell below the {}-row minimum sample and {} excluded from "
                "ranking".format(excluded, "group" if excluded == 1 else "groups",
                                 minimum, "is" if excluded == 1 else "are"))
        evidence.sample_sizes["minimum_group_size"] = minimum
        evidence.sample_sizes["groups_excluded"] = excluded

    rows = grouped.to_records()
    if not rows:
        evidence.mark_insufficient(
            "no group met the minimum sample size",
            minimum_group_size=minimum, groups_before=evidence.sample_sizes.get("groups_excluded"))
        _record_provenance(q, evidence)
        return evidence

    _record_measure_formulas(q, plan.get("measures", {}), evidence)
    _add_rates(rows, plan.get("rates", {}), evidence)
    _add_shares(rows, plan.get("shares", {}), evidence)

    baseline_spec = plan.get("baseline")
    if baseline_spec:
        _apply_baseline(q, rows, baseline_spec, evidence)

    rank_by = plan.get("rank_by")
    if rank_by:
        ascending = plan.get("rank_ascending", False)
        rows = [r for r in rows if r.get(rank_by) is not None] + \
               [r for r in rows if r.get(rank_by) is None]
        ranked = sorted([r for r in rows if r.get(rank_by) is not None],
                        key=lambda r: r[rank_by], reverse=not ascending)
        unranked = [r for r in rows if r.get(rank_by) is None]
        for position, row in enumerate(ranked, start=1):
            row["rank"] = position
        rows = ranked + unranked
        evidence.ranking = {"measure": rank_by, "ascending": ascending,
                            "entities_ranked": len(ranked),
                            "entities_unranked": len(unranked)}
        if unranked:
            evidence.add_limitation(
                "{} groups have no value for {} and are listed without a rank".format(
                    len(unranked), rank_by))

    sort_key = plan.get("sort_by_group")
    if sort_key:
        rows.sort(key=lambda r: (r.get(sort_key) is None, r.get(sort_key)))

    tail = plan.get("tail_n")
    if tail:
        rows = rows[-tail:]
        evidence.supporting_metrics["groups_returned"] = len(rows)

    top = plan.get("top_n")
    if top:
        rows = rows[:top]

    evidence.result_table = _trim(rows)
    evidence.result_columns = list(rows[0].keys()) if rows else []
    evidence.sample_sizes["groups"] = len(rows)
    _record_provenance(q, evidence)
    return evidence


def _apply_baseline(q: Query, rows: List[Dict[str, Any]], spec: Dict[str, Any],
                    evidence: EvidencePackage) -> None:
    """Compute a whole-population baseline and each group's difference from it."""
    measure = spec["measure"]
    if spec["kind"] == "rate":
        numerator = q.frame[q.resolve(spec["numerator_column"])].isin(spec["numerator_values"]).sum()
        baseline_result = FORMULAS.compute("F-RATE", numerator=int(numerator),
                                           denominator=len(q))
    elif spec["kind"] == "mean":
        baseline_result = FORMULAS.compute("F-AVG", values=q.frame[q.resolve(spec["column"])])
    elif spec["kind"] == "median":
        baseline_result = FORMULAS.compute("F-MEDIAN", values=q.frame[q.resolve(spec["column"])])
    else:
        raise PatternError("unknown baseline kind {!r}".format(spec["kind"]))

    baseline = evidence.add_calculation(
        "baseline_{}".format(measure), baseline_result,
        note=spec.get("description", "whole-population baseline"))
    evidence.comparison_baseline = {
        "measure": measure, "value": baseline,
        "description": spec.get("description", "whole-population baseline"),
        "population_size": len(q),
    }
    for row in rows:
        row["delta_vs_baseline"] = FORMULAS.compute(
            "F-DELTA", value=row.get(measure), baseline=baseline).value
        row["pct_delta_vs_baseline"] = FORMULAS.compute(
            "F-PCT-DELTA", value=row.get(measure), baseline=baseline).value


def pattern_distribution(dataset: Dataset, plan: Dict[str, Any],
                         evidence: EvidencePackage) -> EvidencePackage:
    """Band a numeric column and describe its shape."""
    q = _base_query(dataset, plan, evidence)
    evidence.population_size = len(q)
    column = q.resolve(plan["column"])
    values = q.frame[column]

    if values.notna().sum() == 0:
        evidence.mark_insufficient("the column has no values to describe", column=column)
        _record_provenance(q, evidence)
        return evidence

    for label, percentile in (("p25", 25), ("median", 50), ("p75", 75),
                              ("p90", 90), ("p99", 99)):
        evidence.aggregates[label] = evidence.add_calculation(
            label, FORMULAS.compute("F-PERCENTILE", values=values, percentile=percentile))
    evidence.aggregates["mean"] = evidence.add_calculation(
        "mean", FORMULAS.compute("F-AVG", values=values))
    evidence.aggregates["minimum"] = float(values.min())
    evidence.aggregates["maximum"] = float(values.max())
    evidence.aggregates["values_present"] = int(values.notna().sum())
    evidence.aggregates["values_missing"] = int(values.isna().sum())

    bins, labels = plan.get("bins"), plan.get("labels")
    if bins:
        banded = pd.cut(values, bins=bins, labels=labels, include_lowest=True)
        counts = banded.value_counts().reindex(labels).fillna(0)
        total = int(counts.sum())
        rows = []
        for label in labels:
            count = int(counts[label])
            rows.append({
                "band": label, "policies": count,
                "share_pct": FORMULAS.compute("F-SHARE", value=count, total=total).value,
            })
        evidence.result_table = rows
        evidence.result_columns = ["band", "policies", "share_pct"]
    else:
        counts = values.value_counts().sort_index()
        total = int(counts.sum())
        rows = [{"value": _plain(v), "count": int(c),
                 "share_pct": FORMULAS.compute("F-SHARE", value=int(c), total=total).value}
                for v, c in counts.items()]
        evidence.result_table = _trim(rows)
        evidence.result_columns = ["value", "count", "share_pct"]

    _record_provenance(q, evidence)
    return evidence


def pattern_timeseries(dataset: Dataset, plan: Dict[str, Any],
                       evidence: EvidencePackage) -> EvidencePackage:
    """Bucket by period and measure, in chronological order."""
    grain = plan.get("grain", "month")
    plan = dict(plan)
    plan.setdefault("derive", [])
    plan["derive"] = list(plan["derive"]) + [
        {"kind": "date_bucket", "name": "period", "column": plan["date_column"], "grain": grain}]

    q = _base_query(dataset, plan, evidence)
    evidence.population_size = len(q)
    if len(q) == 0:
        evidence.mark_insufficient("the declared population contains no rows")
        _record_provenance(q, evidence)
        return evidence

    grouped = q.group_by("period", _measure_values(q, plan.get("measures", {})), dropna=True)
    rows = sorted(grouped.to_records(), key=lambda r: r["period"])
    _add_rates(rows, plan.get("rates", {}), evidence)

    if len(rows) >= 2 and plan.get("trend_measure"):
        measure = plan["trend_measure"]
        first, last = rows[0], rows[-1]
        evidence.supporting_metrics["first_period"] = first["period"]
        evidence.supporting_metrics["last_period"] = last["period"]
        evidence.supporting_metrics["change"] = evidence.add_calculation(
            "change_over_series",
            FORMULAS.compute("F-DELTA", value=last.get(measure), baseline=first.get(measure)),
            note="{} from {} to {}".format(measure, first["period"], last["period"]))
        evidence.supporting_metrics["pct_change"] = evidence.add_calculation(
            "pct_change_over_series",
            FORMULAS.compute("F-PCT-DELTA", value=last.get(measure), baseline=first.get(measure)))

    evidence.result_table = _trim(rows)
    evidence.result_columns = list(rows[0].keys()) if rows else []
    evidence.add_limitation(
        "The extract ends in December 2025, so the final period may be incomplete.")
    _record_provenance(q, evidence)
    return evidence


def pattern_record_lookup(dataset: Dataset, plan: Dict[str, Any],
                          evidence: EvidencePackage) -> EvidencePackage:
    """Retrieve one entity and everything the schema can attach to it."""
    key_column = plan["key_column"]
    key_value = evidence.parameters.get(plan["key_parameter"])
    if key_value is None:
        evidence.mark_insufficient(
            "no identifier was supplied", required_parameter=plan["key_parameter"])
        return evidence

    q = query(dataset, plan["base"]).filter(key_column, "eq", str(key_value))
    if len(q) == 0:
        evidence.mark_insufficient(
            "no record matches the supplied identifier",
            identifier=str(key_value), column=key_column)
        _record_provenance(q, evidence)
        return evidence

    for relationship in plan.get("joins", []):
        q = q.join(relationship, how="left")

    q, application = _withhold_pii(q)
    evidence.add_rule(application)

    records = q.to_records(limit=1)
    evidence.retrieved_records = records
    evidence.population_size = 1

    absent = [table for table in plan.get("optional_tables", [])
              if all(v is None for k, v in records[0].items() if k.startswith(table + "_"))]
    if absent:
        evidence.supporting_metrics["related_records_absent"] = absent
        evidence.add_limitation(
            "No record exists for this policy in: {}. Absence is not a zero value.".format(
                ", ".join(absent)))

    _record_provenance(q, evidence)
    return evidence


def _withhold_pii(q: Query):
    """Apply BR-006 and actually remove the columns from the result.

    The rule reports which columns it withheld; dropping them from the query is what
    makes the result safe to return, so the two must happen together.
    """
    _, application = RULES.apply("BR-006", q.frame)
    withheld = application.detail.get("columns_withheld", [])
    return q.drop_columns(withheld, "BR-006: personal data withheld"), application


def pattern_top_n(dataset: Dataset, plan: Dict[str, Any],
                  evidence: EvidencePackage) -> EvidencePackage:
    """Rank individual rows and return the leading ones with their share."""
    q = _base_query(dataset, plan, evidence)
    evidence.population_size = len(q)
    if len(q) == 0:
        evidence.mark_insufficient("the declared population contains no rows")
        _record_provenance(q, evidence)
        return evidence

    measure = q.resolve(plan["measure"])
    n = int(evidence.parameters.get("n", plan.get("default_n", 10)))
    ascending = plan.get("ascending", False)

    total = evidence.add_calculation(
        "total_{}".format(plan["measure"]), FORMULAS.compute("F-SUM", values=q.frame[measure]),
        note="total across the whole population, for the share calculation")

    ranked = q.sort(measure, ascending=ascending).limit(n)
    q2, application = _withhold_pii(ranked)
    evidence.add_rule(application)

    columns = [c for c in plan.get("columns", []) if c]
    records = q2.to_records(columns=columns or None, limit=n)
    for position, record in enumerate(records, start=1):
        record["rank"] = position

    selected_total = evidence.add_calculation(
        "selected_total", FORMULAS.compute("F-SUM", values=ranked.frame[measure]),
        note="total across the returned rows only")
    evidence.supporting_metrics["share_of_total_pct"] = evidence.add_calculation(
        "share_of_total", FORMULAS.compute("F-SHARE", value=selected_total, total=total))

    evidence.result_table = records
    evidence.result_columns = list(records[0].keys()) if records else []
    evidence.ranking = {"measure": plan["measure"], "ascending": ascending, "n": n}
    _record_provenance(q2, evidence)
    return evidence


def pattern_concentration(dataset: Dataset, plan: Dict[str, Any],
                          evidence: EvidencePackage) -> EvidencePackage:
    """How much of a total sits in the largest groups."""
    q = _base_query(dataset, plan, evidence)
    evidence.population_size = len(q)
    measure_spec = plan.get("measure", (None, "count_all"))
    grouped = q.group_by(plan["group_by"], {"value": (measure_spec[0], measure_spec[1])},
                         dropna=False)
    rows = sorted(grouped.to_records(), key=lambda r: r["value"] or 0, reverse=True)
    if not rows:
        evidence.mark_insufficient("no groups to measure concentration over")
        _record_provenance(q, evidence)
        return evidence

    values = [r["value"] or 0 for r in rows]
    total = evidence.add_calculation("total", FORMULAS.compute("F-SUM", values=values))
    cumulative = evidence.add_calculation(
        "cumulative_share", FORMULAS.compute("F-CUMSHARE", values=values))

    for row, cum in zip(rows, cumulative or []):
        row["share_pct"] = FORMULAS.compute("F-SHARE", value=row["value"], total=total).value
        row["cumulative_share_pct"] = cum

    shares = [r["share_pct"] for r in rows if r["share_pct"] is not None]
    evidence.aggregates["group_count"] = len(rows)
    evidence.aggregates["hhi"] = evidence.add_calculation(
        "hhi", FORMULAS.compute("F-HHI", shares=shares),
        note="Herfindahl-style index over group percentage shares")

    for label, count in plan.get("top_points", {"top_1": 1, "top_3": 3,
                                                "top_5": 5, "top_10": 10}).items():
        if count <= len(cumulative or []):
            evidence.aggregates["{}_share_pct".format(label)] = cumulative[count - 1]

    for pct in plan.get("top_percentiles", []):
        count = max(1, int(round(len(rows) * pct / 100.0)))
        if count <= len(cumulative or []):
            evidence.aggregates["top_{}pct_share".format(pct)] = cumulative[count - 1]

    evidence.result_table = _trim(rows, plan.get("table_rows", 20))
    evidence.result_columns = list(rows[0].keys())
    _record_provenance(q, evidence)
    return evidence


def pattern_outliers(dataset: Dataset, plan: Dict[str, Any],
                     evidence: EvidencePackage) -> EvidencePackage:
    """Flag rows outside an interquartile fence, within their peer group."""
    q = _base_query(dataset, plan, evidence)
    evidence.population_size = len(q)
    measure = q.resolve(plan["measure"])
    multiplier = float(evidence.parameters.get("iqr_multiplier",
                                               RULES.parameter("BR-007", "multiplier")))
    within = plan.get("within")

    frame = q.frame
    flagged: List[Dict[str, Any]] = []
    groups = frame.groupby(q.resolve(within)) if within else [("all", frame)]

    fences_recorded = 0
    for group_key, chunk in groups:
        values = chunk[measure].dropna()
        fence = FORMULAS.compute("F-IQR", values=values, multiplier=multiplier)
        if fence.value is None:
            evidence.warnings.append(
                "{}: too few values to compute a fence, so it was not assessed".format(
                    _plain(group_key)))
            continue

        # Record the fence for the first few groups: the reader must be able to see
        # the threshold a row was judged against, not just that it was flagged.
        if fences_recorded < 12:
            evidence.add_calculation(
                "iqr_fence[{}]".format(_plain(group_key)), fence,
                note="outlier fence for {} within {}".format(
                    measure, _plain(group_key)))
            fences_recorded += 1

        median = FORMULAS.compute("F-MEDIAN", values=values)
        lower, upper = fence.value["lower_fence"], fence.value["upper_fence"]
        mask = (chunk[measure] < lower) | (chunk[measure] > upper)
        if mask.any() and fences_recorded <= 12:
            evidence.add_calculation(
                "group_median[{}]".format(_plain(group_key)), median,
                note="the central value each flagged row is compared against")
        for _, row in chunk[mask].iterrows():
            flagged.append({
                "group": _plain(group_key),
                "value": _plain(row[measure]),
                "lower_fence": lower, "upper_fence": upper,
                "group_median": median.value,
                **{c: _plain(row[q.resolve(c)]) for c in plan.get("columns", [])},
            })

    evidence.aggregates["outliers_flagged"] = len(flagged)
    evidence.aggregates["groups_assessed"] = len(list(groups)) if within else 1
    evidence.result_table = _trim(sorted(
        flagged, key=lambda r: abs((r["value"] or 0) - (r["group_median"] or 0)), reverse=True))
    evidence.result_columns = list(evidence.result_table[0].keys()) if evidence.result_table else []
    evidence.add_unsupported(
        "An outlier flag is a statistical signal, not evidence of an error or of wrongdoing. "
        "The data holds no rule against which correctness could be verified.")
    evidence.business_rules_applied.append({"rule_id": "BR-007", "applied": True,
                                            "detail": {"multiplier": multiplier},
                                            "note": RULES.get("BR-007").statement})
    _record_provenance(q, evidence)
    return evidence


def pattern_cohort_compare(dataset: Dataset, plan: Dict[str, Any],
                           evidence: EvidencePackage) -> EvidencePackage:
    """Profile two cohorts side by side across every declared attribute."""
    q = _base_query(dataset, plan, evidence)
    evidence.population_size = len(q)
    split = plan["split"]
    column = q.resolve(split["column"])

    cohorts = {}
    for name, values in split["cohorts"].items():
        cohorts[name] = q.frame[q.frame[column].isin(values)]

    sizes = {name: len(frame) for name, frame in cohorts.items()}
    evidence.sample_sizes.update(sizes)
    empty = [name for name, size in sizes.items() if size == 0]
    if empty:
        evidence.mark_insufficient(
            "a cohort is empty, so no comparison is possible", empty_cohorts=empty, sizes=sizes)
        _record_provenance(q, evidence)
        return evidence

    rows = []
    names = list(cohorts)
    for attribute in plan.get("numeric_attributes", []):
        col = q.resolve(attribute)
        row = {"attribute": attribute, "kind": "numeric"}
        for name, frame in cohorts.items():
            mean = FORMULAS.compute("F-AVG", values=frame[col])
            median = FORMULAS.compute("F-MEDIAN", values=frame[col])
            row["{}_mean".format(name)] = evidence.add_calculation(
                "{} mean [{}]".format(attribute, name), mean,
                note="cohort mean, nulls excluded from the denominator")
            row["{}_median".format(name)] = evidence.add_calculation(
                "{} median [{}]".format(attribute, name), median)
        if len(names) == 2:
            a, b = names
            row["delta_mean"] = evidence.add_calculation(
                "{} delta ({} vs {})".format(attribute, a, b),
                FORMULAS.compute("F-DELTA", value=row["{}_mean".format(a)],
                                 baseline=row["{}_mean".format(b)]))
            row["pct_delta_mean"] = evidence.add_calculation(
                "{} pct delta ({} vs {})".format(attribute, a, b),
                FORMULAS.compute("F-PCT-DELTA", value=row["{}_mean".format(a)],
                                 baseline=row["{}_mean".format(b)]))
        rows.append(row)

    for attribute in plan.get("categorical_attributes", []):
        col = q.resolve(attribute)
        for value in sorted(set(q.frame[col].dropna())):
            row = {"attribute": "{} = {}".format(attribute, value), "kind": "categorical"}
            for name, frame in cohorts.items():
                count = int((frame[col] == value).sum())
                row["{}_share_pct".format(name)] = FORMULAS.compute(
                    "F-SHARE", value=count, total=len(frame)).value
                row["{}_count".format(name)] = count
            if len(names) == 2:
                a, b = names
                row["delta_mean"] = FORMULAS.compute(
                    "F-DELTA", value=row["{}_share_pct".format(a)],
                    baseline=row["{}_share_pct".format(b)]).value
            rows.append(row)

    categorical = [a for a in plan.get("categorical_attributes", [])]
    if categorical and names:
        first = names[0]
        evidence.add_calculation(
            "categorical share [{}]".format(first),
            FORMULAS.compute("F-SHARE", value=len(cohorts[first]),
                             total=sum(len(f) for f in cohorts.values())),
            note="each categorical level is expressed as a share of its own cohort")

    if plan.get("rank_by_gap", True):
        rows.sort(key=lambda r: abs(r.get("pct_delta_mean") or r.get("delta_mean") or 0),
                  reverse=True)

    evidence.result_table = _trim(rows)
    evidence.result_columns = list(rows[0].keys()) if rows else []
    evidence.comparison_baseline = {
        "measure": "cohort comparison",
        "description": "cohorts defined by {}".format(split["column"]),
        "cohorts": sizes,
    }
    evidence.add_unsupported(
        "Differences between the cohorts are associations. The data records no mechanism, "
        "so it cannot establish that any attribute caused the difference.")
    _record_provenance(q, evidence)
    return evidence


def pattern_driver_scan(dataset: Dataset, plan: Dict[str, Any],
                        evidence: EvidencePackage) -> EvidencePackage:
    """Find the top entity on a KPI, then compare it against its peers."""
    q = _base_query(dataset, plan, evidence)
    evidence.population_size = len(q)

    entity = plan["entity_column"]
    kpi = plan["kpi"]
    minimum = int(evidence.parameters.get("min_policies", plan.get("min_sample", 30)))

    grouped = q.group_by(entity, _measure_values(q, plan.get("measures", {})), dropna=False)
    grouped = grouped.min_group_size(minimum)
    rows = grouped.to_records()
    if not rows:
        evidence.mark_insufficient(
            "no entity met the minimum sample size", minimum=minimum)
        _record_provenance(q, evidence)
        return evidence

    _add_rates(rows, plan.get("rates", {}), evidence)
    candidates = [r for r in rows if r.get(kpi) is not None]
    if not candidates:
        evidence.mark_insufficient("no entity has a value for the ranking measure", kpi=kpi)
        _record_provenance(q, evidence)
        return evidence

    ascending = plan.get("ascending", False)
    candidates.sort(key=lambda r: r[kpi], reverse=not ascending)
    target = candidates[0]
    peers = candidates[1:]

    evidence.ranking = {"measure": kpi, "ascending": ascending,
                        "entities_ranked": len(candidates),
                        "leader": _plain(target[q.resolve(entity)]),
                        "leader_value": target[kpi]}
    evidence.sample_sizes["minimum_entity_size"] = minimum
    evidence.sample_sizes["entities_eligible"] = len(candidates)

    peer_value = evidence.add_calculation(
        "peer_average_{}".format(kpi),
        FORMULAS.compute("F-AVG", values=[p[kpi] for p in peers]),
        note="mean of every eligible entity except the leader")
    evidence.comparison_baseline = {
        "measure": kpi, "value": peer_value,
        "description": "peer average across {} eligible entities".format(len(peers)),
        "population_size": len(peers),
    }

    target_id = target[q.resolve(entity)]
    target_rows = q.frame[q.frame[q.resolve(entity)] == target_id]
    peer_rows = q.frame[q.frame[q.resolve(entity)] != target_id]

    drivers = []
    for attribute in plan.get("numeric_attributes", []):
        col = q.resolve(attribute)
        target_mean = FORMULAS.compute("F-AVG", values=target_rows[col]).value
        peer_mean = FORMULAS.compute("F-AVG", values=peer_rows[col]).value
        drivers.append({
            "attribute": attribute, "kind": "numeric",
            "leader": target_mean, "peers": peer_mean,
            "delta": FORMULAS.compute("F-DELTA", value=target_mean, baseline=peer_mean).value,
            "pct_delta": FORMULAS.compute(
                "F-PCT-DELTA", value=target_mean, baseline=peer_mean).value,
        })

    for attribute in plan.get("categorical_attributes", []):
        col = q.resolve(attribute)
        for value in sorted(set(q.frame[col].dropna()))[:10]:
            leader_share = FORMULAS.compute(
                "F-SHARE", value=int((target_rows[col] == value).sum()),
                total=len(target_rows)).value
            peer_share = FORMULAS.compute(
                "F-SHARE", value=int((peer_rows[col] == value).sum()),
                total=len(peer_rows)).value
            drivers.append({
                "attribute": "{} = {}".format(attribute, value), "kind": "categorical",
                "leader": leader_share, "peers": peer_share,
                "delta": FORMULAS.compute(
                    "F-DELTA", value=leader_share, baseline=peer_share).value,
                "pct_delta": FORMULAS.compute(
                    "F-PCT-DELTA", value=leader_share, baseline=peer_share).value,
            })

    drivers.sort(key=lambda d: abs(d.get("pct_delta") or 0), reverse=True)
    evidence.result_table = _trim(drivers)
    evidence.result_columns = list(drivers[0].keys()) if drivers else []
    evidence.supporting_metrics["leader"] = _plain(target_id)
    evidence.supporting_metrics["leader_rows"] = len(target_rows)
    evidence.supporting_metrics["peer_rows"] = len(peer_rows)
    evidence.add_unsupported(
        "The attributes listed differ between the leader and their peers. That is an "
        "association. The data records nothing about behaviour, process or intent, so it "
        "cannot establish why the leader performs differently.")
    _record_provenance(q, evidence)
    return evidence


def pattern_reconcile(dataset: Dataset, plan: Dict[str, Any],
                      evidence: EvidencePackage) -> EvidencePackage:
    """Run declared cross-table integrity checks and report pass or fail."""
    rows = []
    for check in plan["checks"]:
        outcome = _run_check(dataset, check)
        rows.append(outcome)
        evidence.tables_used.extend(
            t for t in check.get("tables", []) if t not in evidence.tables_used)

    passed = sum(1 for r in rows if r["passed"])
    evidence.aggregates["checks_total"] = evidence.add_calculation(
        "checks_total", FORMULAS.compute("F-COUNT", population=len(rows)))
    evidence.aggregates["checks_passed"] = evidence.add_calculation(
        "checks_passed", FORMULAS.compute("F-COUNT", population=passed))
    evidence.aggregates["checks_failed"] = evidence.add_calculation(
        "checks_failed", FORMULAS.compute("F-COUNT", population=len(rows) - passed))
    evidence.aggregates["pass_rate"] = evidence.add_calculation(
        "pass_rate", FORMULAS.compute("F-RATE", numerator=passed, denominator=len(rows)),
        note="share of declared integrity checks that pass")
    evidence.result_table = rows
    evidence.result_columns = list(rows[0].keys()) if rows else []
    evidence.population_size = len(rows)
    return evidence


def _run_check(dataset: Dataset, check: Dict[str, Any]) -> Dict[str, Any]:
    kind = check["kind"]
    if kind == "subset":
        left = set(dataset.table(check["left_table"]).col(check["left_column"]).dropna())
        right = set(dataset.table(check["right_table"]).col(check["right_column"]).dropna())
        orphans = sorted(left - right)
        return {"check": check["name"], "passed": not orphans,
                "detail": "{} of {} keys unmatched".format(len(orphans), len(left)),
                "examples": orphans[:5]}
    if kind == "set_equal":
        left_table = dataset.table(check["left_table"])
        mask = left_table.col(check["left_filter_column"]).isin(check["left_filter_values"])
        left = set(left_table.col(check["left_column"])[mask].dropna())
        right = set(dataset.table(check["right_table"]).col(check["right_column"]).dropna())
        return {"check": check["name"], "passed": left == right,
                "detail": "{} vs {} keys; {} differ".format(
                    len(left), len(right), len(left ^ right)),
                "examples": sorted(left ^ right)[:5]}
    if kind == "column_agreement":
        left_table = dataset.table(check["left_table"])
        right_table = dataset.table(check["right_table"])
        merged = left_table.frame[[check["key"], check["left_column"]]].merge(
            right_table.frame[[check["key"], check["right_column"]]],
            on=check["key"], suffixes=("_l", "_r"))
        left_col = check["left_column"] + ("_l" if check["left_column"] == check["right_column"] else "")
        right_col = check["right_column"] + ("_r" if check["left_column"] == check["right_column"] else "")
        agree = int((merged[left_col] == merged[right_col]).sum())
        return {"check": check["name"], "passed": agree == len(merged),
                "detail": "{} of {} joined rows agree".format(agree, len(merged)),
                "examples": []}
    if kind == "equivalence":
        table = dataset.table(check["table"])
        left = table.col(check["left_column"]).eq(check["left_value"])
        right = table.col(check["right_column"]).notna()
        return {"check": check["name"], "passed": bool(left.equals(right)),
                "detail": "{} rows disagree".format(int((left != right).sum())),
                "examples": []}
    raise PatternError("unknown reconciliation check {!r}".format(kind))


def pattern_scope_check(dataset: Dataset, plan: Dict[str, Any],
                        evidence: EvidencePackage) -> EvidencePackage:
    """Report what the datasets cannot answer, and why."""
    contract = dataset.contract
    rows = [{"domain": d["domain"], "reason": d["reason"]}
            for d in contract.unanswerable_domains]
    evidence.result_table = rows
    evidence.result_columns = ["domain", "reason"]
    evidence.aggregates["unanswerable_domain_count"] = evidence.add_calculation(
        "unanswerable_domain_count", FORMULAS.compute("F-COUNT", population=len(rows)),
        note="domains the five datasets cannot support")
    evidence.population_size = len(rows)
    for row in rows:
        evidence.add_unsupported("{}: {}".format(row["domain"], row["reason"]))
    evidence.tables_used = sorted(contract.tables)
    return evidence


def pattern_quality_report(dataset: Dataset, plan: Dict[str, Any],
                           evidence: EvidencePackage) -> EvidencePackage:
    """Report the recorded data-quality findings."""
    findings = dataset.contract.data_quality_findings
    rows = [{"id": f["id"], "severity": f["severity"],
             "tables": ", ".join(f["tables"]), "finding": f["finding"],
             "impact": f["impact"], "handling": f["handling"]}
            for f in findings]
    evidence.result_table = rows
    evidence.result_columns = ["id", "severity", "tables", "finding", "impact", "handling"]
    evidence.aggregates["finding_count"] = evidence.add_calculation(
        "finding_count", FORMULAS.compute("F-COUNT", population=len(rows)))
    for severity in ("high", "medium", "low"):
        count = sum(1 for f in findings if f["severity"] == severity)
        evidence.aggregates["{}_severity".format(severity)] = evidence.add_calculation(
            "{}_severity".format(severity), FORMULAS.compute("F-COUNT", population=count))
    evidence.data_quality_notes = [f["id"] for f in findings]
    evidence.population_size = len(rows)
    evidence.tables_used = sorted(dataset.contract.tables)
    return evidence


def pattern_executive(dataset: Dataset, plan: Dict[str, Any],
                      evidence: EvidencePackage) -> EvidencePackage:
    """Compose several sub-analyses into one briefing, each keeping its own population."""
    sections = []
    for section in plan["sections"]:
        sub = EvidencePackage(
            question=evidence.question,
            detected_problem_id=evidence.detected_problem_id,
            problem_statement=section["title"])
        sub.parameters = dict(evidence.parameters)
        handler = PATTERNS[section["pattern"]]
        handler(dataset, section["plan"], sub)

        sections.append({
            "title": section["title"],
            "population": section.get("population", ""),
            "population_size": sub.population_size,
            "aggregates": sub.aggregates,
            "table": sub.result_table[:section.get("rows", 5)],
            "sufficient": sub.is_sufficient,
        })
        for key, value in sub.aggregates.items():
            evidence.aggregates["{}__{}".format(section["key"], key)] = value
        evidence.calculations.extend(sub.calculations)
        for formula_id in sub.formulas_applied:
            if formula_id not in evidence.formulas_applied:
                evidence.formulas_applied.append(formula_id)
        evidence.business_rules_applied.extend(sub.business_rules_applied)
        for table in sub.tables_used:
            if table not in evidence.tables_used:
                evidence.tables_used.append(table)
        for limitation in sub.limitations:
            evidence.add_limitation(limitation)
        for text in sub.unsupported_conclusions:
            evidence.add_unsupported(text)

    evidence.result_table = sections
    evidence.result_columns = ["title", "population", "population_size", "aggregates"]
    evidence.supporting_metrics["section_count"] = len(sections)
    evidence.add_limitation(
        "Sections use different populations and denominators. Figures must not be combined "
        "across sections without checking the population each was computed over.")
    return evidence


def _plain(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.bool_):
        return bool(value)
    return value


PATTERNS: Dict[str, Callable] = {
    "aggregate": pattern_aggregate,
    "group_metrics": pattern_group_metrics,
    "distribution": pattern_distribution,
    "timeseries": pattern_timeseries,
    "record_lookup": pattern_record_lookup,
    "top_n": pattern_top_n,
    "concentration": pattern_concentration,
    "outliers": pattern_outliers,
    "cohort_compare": pattern_cohort_compare,
    "driver_scan": pattern_driver_scan,
    "reconcile": pattern_reconcile,
    "scope_check": pattern_scope_check,
    "quality_report": pattern_quality_report,
    "executive": pattern_executive,
}
