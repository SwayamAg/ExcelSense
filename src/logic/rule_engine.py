"""Business rule engine.

A business rule is a decision about *which rows count* or *how a result must be
qualified*. They are separated from formulas because they are judgement calls
grounded in the data's quirks, and an executive is entitled to see and challenge
them. Every rule carries its statement, its rationale, and the finding or
observation that motivated it.

Rules that can be executed are executed here, so no analysis can quietly apply a
different population definition from the one it reports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

ENGINE_VERSION = "1.0.0"


class RuleError(Exception):
    """Raised when a rule is missing, misconfigured, or applied to the wrong shape of data."""


@dataclass
class RuleApplication:
    """The record of a rule having been applied."""
    rule_id: str
    applied: bool
    rows_before: Optional[int] = None
    rows_after: Optional[int] = None
    detail: Dict[str, Any] = field(default_factory=dict)
    note: str = ""

    @property
    def rows_excluded(self) -> Optional[int]:
        if self.rows_before is None or self.rows_after is None:
            return None
        return self.rows_before - self.rows_after

    def as_evidence(self) -> Dict[str, Any]:
        payload = {"rule_id": self.rule_id, "applied": self.applied,
                   "detail": self.detail, "note": self.note,
                   "engine_version": ENGINE_VERSION}
        if self.rows_before is not None:
            payload.update({"rows_before": self.rows_before, "rows_after": self.rows_after,
                            "rows_excluded": self.rows_excluded})
        return payload


@dataclass
class BusinessRule:
    """A declared, inspectable decision about the analysis."""
    id: str
    name: str
    statement: str
    rationale: str
    category: str
    applies_to: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    grounded_in: List[str] = field(default_factory=list)   # DQ finding ids, where relevant
    executable: bool = False
    fn: Optional[Callable] = None
    version: str = "1.0.0"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "statement": self.statement,
            "rationale": self.rationale, "category": self.category,
            "applies_to": self.applies_to, "parameters": self.parameters,
            "grounded_in": self.grounded_in, "executable": self.executable,
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# Status vocabulary, read from the data rather than assumed
# ---------------------------------------------------------------------------

STATUS_INFORCE = "Premium paying (regular)"
STATUS_LAPSED = "Lapsed"
STATUS_SURRENDERED = "Surrendered"
STATUS_DEATH = "Death"
ALL_STATUSES = (STATUS_INFORCE, STATUS_LAPSED, STATUS_SURRENDERED, STATUS_DEATH)

PERSISTENCY_ELIGIBLE_STATUSES = (STATUS_INFORCE, STATUS_LAPSED)

PII_COLUMNS = ("Owner.Given.Name", "Owner.Comm.Postal.Code")


# ---------------------------------------------------------------------------
# Executable rule implementations
# ---------------------------------------------------------------------------

def _persistency_population(frame: pd.DataFrame, status_column: str = "POLICY.STATUS",
                            **_) -> "tuple":
    if status_column not in frame.columns:
        raise RuleError("BR-001 needs column {!r}".format(status_column))
    before = len(frame)
    mask = frame[status_column].isin(PERSISTENCY_ELIGIBLE_STATUSES)
    out = frame[mask]
    excluded = frame.loc[~mask, status_column].value_counts().to_dict()
    return out, {"eligible_statuses": list(PERSISTENCY_ELIGIBLE_STATUSES),
                 "excluded_by_status": {k: int(v) for k, v in excluded.items()},
                 "rows_before": before, "rows_after": len(out)}


def _min_sample(frame: pd.DataFrame, size_column: str, minimum: int, **_) -> "tuple":
    if size_column not in frame.columns:
        raise RuleError("BR-002 needs column {!r}".format(size_column))
    before = len(frame)
    keep = frame[size_column] >= minimum
    out = frame[keep]
    return out, {"minimum": minimum, "size_column": size_column,
                 "groups_before": before, "groups_after": len(out),
                 "groups_excluded": int((~keep).sum()),
                 "rows_in_excluded_groups": int(frame.loc[~keep, size_column].sum())}


def _exclude_unsettled_claims(frame: pd.DataFrame, status_column: str = "Status.of.Claim",
                              **_) -> "tuple":
    if status_column not in frame.columns:
        raise RuleError("BR-003 needs column {!r}".format(status_column))
    before = len(frame)
    mask = frame[status_column] == "Paid"
    out = frame[mask]
    return out, {"settled_status": "Paid", "rows_before": before, "rows_after": len(out),
                 "unsettled_excluded": int((~mask).sum())}


def _exclude_null_metric(frame: pd.DataFrame, column: str, **_) -> "tuple":
    if column not in frame.columns:
        raise RuleError("BR-004 needs column {!r}".format(column))
    before = len(frame)
    mask = frame[column].notna()
    out = frame[mask]
    return out, {"column": column, "rows_before": before, "rows_after": len(out),
                 "null_rows_excluded": int((~mask).sum())}


def _exclude_negative_durations(frame: pd.DataFrame, column: str, **_) -> "tuple":
    if column not in frame.columns:
        raise RuleError("BR-005 needs column {!r}".format(column))
    before = len(frame)
    mask = frame[column].isna() | (frame[column] >= 0)
    out = frame[mask]
    return out, {"column": column, "rows_before": before, "rows_after": len(out),
                 "negative_rows_excluded": int((~mask).sum())}


def _withhold_pii(frame: pd.DataFrame, **_) -> "tuple":
    present = [c for c in frame.columns
               if any(c == p or c.endswith("_" + p) for p in PII_COLUMNS)]
    out = frame.drop(columns=present) if present else frame
    return out, {"columns_withheld": present, "pii_columns": list(PII_COLUMNS)}


def _small_cell_suppression(frame: pd.DataFrame, count_column: str, minimum: int,
                            rate_columns: Optional[List[str]] = None, **_) -> "tuple":
    if count_column not in frame.columns:
        raise RuleError("BR-008 needs column {!r}".format(count_column))
    out = frame.copy()
    thin = out[count_column] < minimum
    rate_columns = rate_columns or [c for c in out.columns if c.endswith(("_rate", "_pct"))]
    for column in rate_columns:
        if column in out.columns:
            out.loc[thin, column] = None
    return out, {"count_column": count_column, "minimum": minimum,
                 "cells_suppressed": int(thin.sum()),
                 "rate_columns_suppressed": rate_columns}


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------

_RULES: Dict[str, BusinessRule] = {}


def _register(rule: BusinessRule) -> None:
    if rule.id in _RULES:
        raise RuleError("duplicate rule id {}".format(rule.id))
    _RULES[rule.id] = rule


_register(BusinessRule(
    id="BR-001", name="Persistency eligible population",
    statement=("Renewal and persistency metrics are computed only over policies with status "
               "'Premium paying (regular)' or 'Lapsed'."),
    rationale=("Persistency_Details contains no row for surrendered or death policies. Using "
               "the full policy count as a denominator would understate every renewal rate by "
               "including 458 policies that could never have a renewal row."),
    category="population", applies_to=["persistency", "policy"],
    parameters={"eligible_statuses": list(PERSISTENCY_ELIGIBLE_STATUSES)},
    grounded_in=["DQ-006"], executable=True, fn=_persistency_population))

_register(BusinessRule(
    id="BR-002", name="Minimum sample for entity ranking",
    statement=("An entity is ranked on a rate only when its population meets the minimum "
               "sample size. Excluded entities are counted and reported, never hidden."),
    rationale=("The median agent has 8 policies and the median branch 3. Without a threshold an "
               "agent with 2 policies and 1 lapse would top a lapse ranking at 50%."),
    category="statistical_guard", applies_to=["sales", "policy", "persistency"],
    parameters={"default_minimum_agent_policies": 30, "default_minimum_branch_policies": 10,
                "default_minimum_segment_size": 100},
    grounded_in=[], executable=True, fn=_min_sample))

_register(BusinessRule(
    id="BR-003", name="Unsettled claims excluded from settlement metrics",
    statement=("Settlement-lag and payout metrics cover settled claims only. The claim carrying "
               "the 'WIP' sentinel is excluded and the exclusion is reported."),
    rationale=("'WIP' appears in Status.of.Claim, Amount.Paid and Settlement.Date at once. "
               "Treating it as a zero payment would understate the payout ratio; treating it as "
               "a settled claim would corrupt the lag distribution."),
    category="population", applies_to=["claims"],
    parameters={"settled_status": "Paid", "sentinel": "WIP"},
    grounded_in=["DQ-004"], executable=True, fn=_exclude_unsettled_claims))

_register(BusinessRule(
    id="BR-004", name="Null supplied metrics excluded, never zeroed",
    statement=("An entity whose supplied metric is null is excluded from rankings on that "
               "metric rather than treated as zero, and the excluded count is reported."),
    rationale=("369 of 1,956 agents have no First Year Persistency Rate. Reading null as zero "
               "would place every one of them at the bottom of a persistency ranking on the "
               "strength of missing data."),
    category="null_handling", applies_to=["sales"],
    parameters={"affected_columns": ["First Year Persistency Rate"]},
    grounded_in=["DQ-005"], executable=True, fn=_exclude_null_metric))

_register(BusinessRule(
    id="BR-005", name="Negative durations reported and excluded",
    statement=("A computed duration below zero is excluded from duration statistics and the "
               "excluded count is reported. It is never clipped to zero."),
    rationale=("One policy has Login.Date after Inforce.Date. Clipping would silently invent a "
               "same-day turnaround; dropping it without saying so would hide a data defect."),
    category="statistical_guard", applies_to=["policy", "claims", "persistency"],
    parameters={}, grounded_in=["DQ-011"], executable=True, fn=_exclude_negative_durations))

_register(BusinessRule(
    id="BR-006", name="Personal data withheld from analytical output",
    statement=("Owner.Given.Name and Owner.Comm.Postal.Code are removed from any result "
               "returned to a user, in aggregate and record output alike."),
    rationale=("Neither field is needed to answer a portfolio question. Postal code may be read "
               "for a data-quality check but is never returned."),
    category="governance", applies_to=["owner"],
    parameters={"pii_columns": list(PII_COLUMNS)},
    grounded_in=[], executable=True, fn=_withhold_pii))

_register(BusinessRule(
    id="BR-007", name="Outlier fence uses the interquartile range",
    statement=("Outliers are flagged using an interquartile fence at Q1 - k*IQR and Q3 + k*IQR "
               "with k = 1.5, not standard deviations."),
    rationale=("The supplied agent rates are heavily skewed and mostly zero, so a mean and "
               "standard deviation would be dragged by the very values being tested. A flag is "
               "a prompt to look, never a finding."),
    category="statistical_guard", applies_to=["sales", "policy"],
    parameters={"multiplier": 1.5, "minimum_values": 4},
    grounded_in=[], executable=False))

_register(BusinessRule(
    id="BR-008", name="Small-cell rate suppression",
    statement=("A rate computed from fewer than the minimum cell count is suppressed; the count "
               "is still shown so the reader sees the population without a misleading rate."),
    rationale=("134 claims spread across 10 products, 34 states and many age bands leaves cells "
               "of one or two. A rate from such a cell is noise presented as a measurement."),
    category="statistical_guard", applies_to=["claims", "policy", "owner"],
    parameters={"default_minimum_cell": 5},
    grounded_in=["DQ-007"], executable=True, fn=_small_cell_suppression))

_register(BusinessRule(
    id="BR-009", name="Policy status definitions",
    statement=("Lapse, in-force, surrender and death are read from POLICY.STATUS and from "
               "nothing else. No status is inferred from another field."),
    rationale=("POLICY.STATUS has exactly four values and no status-change date exists. Any "
               "attempt to infer a lapse from renewal data would invent a transition the data "
               "does not record."),
    category="definition", applies_to=["policy", "persistency"],
    parameters={"inforce": STATUS_INFORCE, "lapsed": STATUS_LAPSED,
                "surrendered": STATUS_SURRENDERED, "death": STATUS_DEATH},
    grounded_in=[], executable=False))

_register(BusinessRule(
    id="BR-010", name="Early claim window",
    statement=("An early claim is a death occurring within the configured window of the policy "
               "going in force. The window is stated with every result that uses it."),
    rationale=("There is no contestability flag in the data, so 'early' is an analytical "
               "convention rather than a contractual status. The default is 365 days."),
    category="definition", applies_to=["claims", "policy"],
    parameters={"default_window_days": 365, "minimum": 30, "maximum": 1095},
    grounded_in=[], executable=False))

_register(BusinessRule(
    id="BR-011", name="Rounding and presentation",
    statement=("Calculations are carried at full precision and rounded once, at presentation: "
               "money to 2 decimals, rates and percentages to 2, ratios to 4, counts as integers."),
    rationale=("Rounding intermediate values compounds error across a multi-step analysis and "
               "can make a table's parts disagree with its total."),
    category="presentation", applies_to=["all"],
    parameters={"money_decimals": 2, "rate_decimals": 2, "ratio_decimals": 4,
                "count_decimals": 0},
    grounded_in=[], executable=False))

_register(BusinessRule(
    id="BR-012", name="Composite score weights are declared",
    statement=("A composite score is computed from percentile-normalised components using the "
               "weights declared here, and the weights are returned with the score."),
    rationale=("The seven supplied agent rates sit on different and undocumented scales, with "
               "maxima from 1.28 to 25.96, so they cannot be combined raw. The weighting is an "
               "analytical convention, not a company standard, and must be challengeable."),
    category="definition", applies_to=["sales"],
    parameters={"weights": {
        "agent_first_year_persistency": 0.30,
        "agent_complaint_rate": 0.15,
        "agent_rcu_rejection_rate": 0.15,
        "agent_surrender_rate": 0.10,
        "agent_early_claim_rate": 0.10,
        "agent_freelook_cancellation_rate": 0.10,
        "agent_pivc_concern_rate": 0.05,
        "agent_pivc_mismatch_rate": 0.05,
    }, "normalisation": "percentile rank within the eligible population",
       "orientation": "every component is oriented so that higher is better"},
    grounded_in=["DQ-005"], executable=False))

_register(BusinessRule(
    id="BR-013", name="Denominator must be declared",
    statement=("Every rate reports the population it was computed over. Figures from different "
               "populations are never combined without stating that they differ."),
    rationale=("Policy metrics run over 25,000 rows, renewal metrics over 24,542 and owner "
               "metrics over 24,999. Mixing them silently produces a number that is wrong in a "
               "way no reader can detect."),
    category="governance", applies_to=["all"],
    parameters={}, grounded_in=["DQ-001", "DQ-006"], executable=False))

_register(BusinessRule(
    id="BR-014", name="No causal claim without evidence",
    statement=("An observed difference between groups is reported as an association. A causal "
               "claim is made only where the data establishes a mechanism, which these five "
               "datasets never do."),
    rationale=("There is no lapse reason, surrender reason, payment history, contact log, "
               "underwriting record or intervention history. Every diagnostic is therefore a "
               "comparison of observable attributes."),
    category="governance", applies_to=["all"],
    parameters={}, grounded_in=[], executable=False))

_register(BusinessRule(
    id="BR-015", name="Corrupted AppID is not silently repaired",
    statement=("The AppID stored as '1.50E+10' is left as found by default and surfaces as an "
               "orphan. A repair is opt-in and is recorded with its justification."),
    rationale=("The mapping to policy 15011000000 is strongly evidenced but still an inference. "
               "Applying it silently would put an inference into the data itself."),
    category="governance", applies_to=["owner", "persistency"],
    parameters={"corrupted_value": "1.50E+10", "inferred_target": "15011000000",
                "default_enabled": False},
    grounded_in=["DQ-001"], executable=False))


class RuleEngine:
    """Applies and reports business rules."""

    version = ENGINE_VERSION

    def __init__(self, registry: Optional[Dict[str, BusinessRule]] = None):
        self._rules = registry if registry is not None else _RULES

    @property
    def rule_ids(self) -> List[str]:
        return sorted(self._rules)

    def get(self, rule_id: str) -> BusinessRule:
        if rule_id not in self._rules:
            raise RuleError("unknown rule {!r}; registered: {}".format(rule_id, self.rule_ids))
        return self._rules[rule_id]

    def parameter(self, rule_id: str, name: str) -> Any:
        params = self.get(rule_id).parameters
        if name not in params:
            raise RuleError("{} has no parameter {!r}".format(rule_id, name))
        return params[name]

    def apply(self, rule_id: str, frame: pd.DataFrame, **kwargs):
        """Execute a rule against a frame; returns (frame, RuleApplication)."""
        rule = self.get(rule_id)
        if not rule.executable or rule.fn is None:
            raise RuleError(
                "{} is a declared rule, not an executable one; read its parameters "
                "instead".format(rule_id))
        out, detail = rule.fn(frame, **kwargs)
        application = RuleApplication(
            rule_id=rule_id, applied=True,
            rows_before=detail.get("rows_before", len(frame)),
            rows_after=detail.get("rows_after", len(out)),
            detail=detail, note=rule.statement)
        return out, application

    def rules_for_category(self, category: str) -> List[BusinessRule]:
        return [self._rules[i] for i in self.rule_ids
                if self._rules[i].category == category]

    def describe(self) -> List[Dict[str, Any]]:
        return [self._rules[i].as_dict() for i in self.rule_ids]


def get_rule_engine() -> RuleEngine:
    return RuleEngine()
