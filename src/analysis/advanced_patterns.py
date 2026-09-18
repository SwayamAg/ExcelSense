"""Patterns whose unit of analysis is not the row.

Three analyses need more than a group-and-measure pass:

  composite_score   combine metrics that sit on different, undocumented scales
  two_stage_group   roll rows up to an entity, then band the entities
  rule_breach       apply declared plausibility rules and rank by breach count

They live here rather than in patterns.py to keep each file readable.
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from src.analysis.patterns import (
    FORMULAS,
    PATTERNS,
    RULES,
    PatternError,
    _add_rates,
    _base_query,
    _measure_values,
    _plain,
    _record_provenance,
    _trim,
)
from src.data import Dataset
from src.ps.evidence import EvidencePackage


def pattern_composite_score(dataset: Dataset, plan: Dict[str, Any],
                            evidence: EvidencePackage) -> EvidencePackage:
    """Score entities on several metrics that sit on different scales.

    Raw agent rates cannot simply be added: their maxima range from 1.28 to 25.96
    and their meanings differ. Each component is converted to a percentile rank
    within the eligible population, oriented so that higher is always better, and
    combined with weights declared in BR-012 and returned alongside the score.
    """
    q = _base_query(dataset, plan, evidence)
    evidence.population_size = len(q)

    entity = plan["entity_column"]
    minimum = int(evidence.parameters.get("min_policies", plan.get("min_sample", 30)))

    grouped = q.group_by(entity, _measure_values(q, plan.get("measures", {})), dropna=False)
    grouped = grouped.min_group_size(minimum)
    rows = grouped.to_records()
    if not rows:
        evidence.mark_insufficient("no entity met the minimum sample size", minimum=minimum)
        _record_provenance(q, evidence)
        return evidence

    _add_rates(rows, plan.get("rates", {}), evidence)

    components = plan["components"]
    weights = dict(evidence.parameters.get("weights")
                   or RULES.parameter("BR-012", "weights"))
    weights = {k: v for k, v in weights.items() if k in components}
    if not weights:
        evidence.mark_insufficient("no weight applies to any declared component")
        _record_provenance(q, evidence)
        return evidence

    for alias, spec in components.items():
        present = [r.get(spec["column"]) for r in rows if r.get(spec["column"]) is not None]
        for row in rows:
            value = row.get(spec["column"])
            if value is None or not present:
                row["{}_percentile".format(alias)] = None
                continue
            below = sum(1 for v in present if v < value)
            percentile = FORMULAS.compute(
                "F-RATE", numerator=below, denominator=len(present)).value
            if percentile is not None and not spec.get("higher_is_better", True):
                percentile = round(100.0 - percentile, 4)
            row["{}_percentile".format(alias)] = percentile

    for row in rows:
        parts = {alias: row.get("{}_percentile".format(alias)) for alias in components}
        result = FORMULAS.compute("F-COMPOSITE", components=parts, weights=weights)
        row["composite_score"] = result.value
        row["components_scored"] = len(result.detail.get("components_used", []))

    scored = sorted([r for r in rows if r["composite_score"] is not None],
                    key=lambda r: r["composite_score"], reverse=True)
    for position, row in enumerate(scored, start=1):
        row["rank"] = position

    if not scored:
        evidence.mark_insufficient("no entity could be scored on any component")
        _record_provenance(q, evidence)
        return evidence

    evidence.ranking = {"measure": "composite_score", "ascending": False,
                        "entities_ranked": len(scored)}
    evidence.sample_sizes["minimum_entity_size"] = minimum
    evidence.sample_sizes["entities_scored"] = len(scored)
    evidence.supporting_metrics["weights"] = weights
    evidence.supporting_metrics["normalisation"] = (
        "percentile rank within the eligible population; every component oriented so a "
        "higher percentile is better")
    evidence.add_limitation(
        "The weights are a declared analytical convention recorded in BR-012, not a company "
        "standard. They are returned with the score so the ranking can be recomputed.")

    partial = [r for r in scored if r["components_scored"] < len(components)]
    if partial:
        evidence.add_limitation(
            "{} entities were scored on fewer than {} components because a supplied metric "
            "was missing; their weights were renormalised.".format(len(partial), len(components)))

    evidence.result_table = _trim(scored, plan.get("table_rows", 50))
    evidence.result_columns = list(scored[0].keys())
    evidence.business_rules_applied.append({
        "rule_id": "BR-012", "applied": True, "detail": {"weights": weights},
        "note": RULES.get("BR-012").statement})
    _record_provenance(q, evidence)
    return evidence


def pattern_two_stage_group(dataset: Dataset, plan: Dict[str, Any],
                            evidence: EvidencePackage) -> EvidencePackage:
    """Aggregate rows to an entity, band the entities, then profile each band.

    Needed wherever the unit of analysis is not the row: customer value is a
    property of a client, not of a policy, so policies must be rolled up to the
    client before any tiering can be honest.
    """
    q = _base_query(dataset, plan, evidence)
    evidence.population_size = len(q)

    entity = plan["entity_column"]
    stage_one = q.group_by(entity, _measure_values(q, plan.get("entity_measures", {})),
                           dropna=False)
    entities = stage_one.to_records()
    if not entities:
        evidence.mark_insufficient("no entities to tier")
        _record_provenance(q, evidence)
        return evidence

    measure = plan["tier_measure"]
    values = [e.get(measure) for e in entities if e.get(measure) is not None]
    if not values:
        evidence.mark_insufficient("no entity has a value to tier on", measure=measure)
        _record_provenance(q, evidence)
        return evidence

    cuts = plan.get("tier_percentiles", [50, 80, 95])
    thresholds = [
        evidence.add_calculation(
            "tier_threshold_p{}".format(percentile),
            FORMULAS.compute("F-PERCENTILE", values=values, percentile=percentile),
            note="tier boundary on {}".format(measure))
        for percentile in cuts
    ]

    labels = plan.get("tier_labels", ["tier_4_lowest", "tier_3", "tier_2", "tier_1_highest"])
    for row in entities:
        value = row.get(measure)
        if value is None:
            row["tier"] = None
            continue
        index = sum(1 for t in thresholds if t is not None and value >= t)
        row["tier"] = labels[min(index, len(labels) - 1)]

    total_value = sum(e.get(measure) or 0 for e in entities)
    rows = []
    for label in labels:
        members = [e for e in entities if e.get("tier") == label]
        if not members:
            continue
        tier_total = FORMULAS.compute(
            "F-SUM", values=[m.get(measure) for m in members]).value
        row = {
            "tier": label,
            "entities": len(members),
            "entity_share_pct": FORMULAS.compute(
                "F-SHARE", value=len(members), total=len(entities)).value,
            "{}_total".format(measure): tier_total,
            "{}_share_pct".format(measure): FORMULAS.compute(
                "F-SHARE", value=tier_total, total=total_value).value,
        }
        for alias, source in plan.get("tier_profile", {}).items():
            row[alias] = FORMULAS.compute(
                "F-AVG", values=[m.get(source) for m in members]).value
        rows.append(row)

    rows.reverse()      # highest tier first
    evidence.result_table = rows
    evidence.result_columns = list(rows[0].keys()) if rows else []
    evidence.aggregates["entities"] = len(entities)
    evidence.sample_sizes["entities"] = len(entities)
    evidence.supporting_metrics["tier_boundaries"] = dict(
        zip(["p{}".format(p) for p in cuts], thresholds))
    evidence.add_limitation(
        "Tier boundaries are percentiles of the observed distribution, not externally "
        "defined value bands.")
    _record_provenance(q, evidence)
    return evidence


def pattern_rule_breach(dataset: Dataset, plan: Dict[str, Any],
                        evidence: EvidencePackage) -> EvidencePackage:
    """Apply declared plausibility rules and rank rows by how many they breach."""
    q = _base_query(dataset, plan, evidence)
    evidence.population_size = len(q)
    frame = q.frame

    breaches = pd.Series(0, index=frame.index)
    per_rule: Dict[str, int] = {}
    labels: Dict[Any, list] = {idx: [] for idx in frame.index}

    for check in plan["checks"]:
        series = frame[q.resolve(check["column"])]
        test = check["test"]
        if test == "lt":
            mask = series < check["value"]
        elif test == "gt":
            mask = series > check["value"]
        elif test == "outside":
            mask = (series < check["low"]) | (series > check["high"])
        elif test == "greater_than_column":
            mask = series > frame[q.resolve(check["other"])]
        elif test == "is_null":
            mask = series.isna()
        else:
            raise PatternError("unknown plausibility test {!r}".format(test))
        mask = mask.fillna(False).astype(bool)
        per_rule[check["name"]] = int(mask.sum())
        breaches = breaches + mask.astype(int)
        for idx in frame.index[mask]:
            labels[idx].append(check["name"])

    rows = []
    for idx in frame.index[breaches > 0]:
        row = {"breaches": int(breaches[idx]), "rules": ", ".join(labels[idx])}
        for column in plan.get("columns", []):
            row[column] = _plain(frame.loc[idx, q.resolve(column)])
        rows.append(row)
    rows.sort(key=lambda r: r["breaches"], reverse=True)

    evidence.aggregates["rows_flagged"] = evidence.add_calculation(
        "rows_flagged", FORMULAS.compute("F-COUNT", population=len(rows)))
    evidence.aggregates["rows_assessed"] = evidence.add_calculation(
        "rows_assessed", FORMULAS.compute("F-COUNT", population=len(frame)))
    evidence.aggregates["breach_rate"] = evidence.add_calculation(
        "breach_rate",
        FORMULAS.compute("F-RATE", numerator=len(rows), denominator=len(frame)),
        note="share of assessed rows breaching at least one plausibility rule")
    evidence.aggregates["breaches_by_rule"] = per_rule
    evidence.result_table = _trim(rows)
    evidence.result_columns = list(rows[0].keys()) if rows else []
    evidence.supporting_metrics["rule_set"] = [c["name"] for c in plan["checks"]]
    evidence.add_unsupported(
        "A flag marks a record worth checking, not an error. The rule set is a declared "
        "analytical convention, not a company validation standard, and no source file is "
        "modified by this analysis.")
    _record_provenance(q, evidence)
    return evidence


PATTERNS["composite_score"] = pattern_composite_score
PATTERNS["two_stage_group"] = pattern_two_stage_group
PATTERNS["rule_breach"] = pattern_rule_breach
