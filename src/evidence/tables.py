"""Evidence tables.

Turns an evidence package into tables an executive can read and an auditor can
check. Nothing is computed here: every value already exists in the package, and
this layer only chooses which values to show, how to format them, and what to say
about where they came from.

Formatting is where numbers quietly mislead, so units are explicit. A column named
`lapse_rate` is a percentage and is shown with a % sign; `total_ape` is money and is
shown grouped; `policies` is a count and never gains a decimal point.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.ps.evidence import EvidencePackage

MAX_DISPLAY_ROWS = 25

# Column naming conventions, used to infer unit and formatting.
RATE_PATTERN = re.compile(r"(_rate|_pct|percentage|percent|share)$|^rate_", re.IGNORECASE)
MONEY_PATTERN = re.compile(
    r"(ape|premium|amount|sum_assured|sum\.assured|exposure|value|income|stake|paid)",
    re.IGNORECASE)
COUNT_PATTERN = re.compile(
    r"(count|policies|claims|renewals|agents|branches|clients|entities|groups|rows|"
    r"owners|breaches|outliers|lapsed|paid|pending|settled|inforce|surrendered)",
    re.IGNORECASE)
DAYS_PATTERN = re.compile(r"(_days|days_)", re.IGNORECASE)
IDENTIFIER_PATTERN = re.compile(
    r"^(appid|clientid|sales_id|claim_no|branch_no|prod_no|__row_id)$", re.IGNORECASE)

# Columns that exist for bookkeeping and should not be shown by default.
INTERNAL_PREFIXES = ("__",)


@dataclass
class Column:
    """One column of an evidence table, with its unit made explicit."""
    key: str
    label: str
    unit: str           # 'money' | 'rate' | 'count' | 'days' | 'identifier' | 'text' | 'number'
    decimals: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "label": self.label, "unit": self.unit,
                "decimals": self.decimals}


@dataclass
class EvidenceTable:
    """A presentation-ready table, with a note saying where it came from."""
    key: str
    title: str
    columns: List[Column]
    rows: List[Dict[str, Any]]
    source_note: str = ""
    footnotes: List[str] = field(default_factory=list)
    total_rows: int = 0
    truncated: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "columns": [c.as_dict() for c in self.columns],
            "rows": self.rows,
            "source_note": self.source_note,
            "footnotes": list(self.footnotes),
            "total_rows": self.total_rows,
            "truncated": self.truncated,
        }

    def values_for(self, key: str) -> List[Any]:
        return [row.get(key) for row in self.rows]


def humanize(key: str) -> str:
    """A column label a reader can parse without knowing the schema."""
    text = key.replace("_", " ").replace(".", " ")
    text = re.sub(r"\bpct\b", "%", text)
    text = re.sub(r"\bape\b", "APE", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhhi\b", "HHI", text, flags=re.IGNORECASE)
    text = re.sub(r"\bulip\b", "ULIP", text, flags=re.IGNORECASE)
    text = re.sub(r"\bid\b", "ID", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1].upper() + text[1:] if text else key


def infer_unit(key: str, values: List[Any]) -> str:
    """Decide a column's unit from its name, falling back to its values."""
    if IDENTIFIER_PATTERN.match(key):
        return "identifier"
    if RATE_PATTERN.search(key):
        return "rate"
    if DAYS_PATTERN.search(key):
        return "days"
    if MONEY_PATTERN.search(key) and not COUNT_PATTERN.search(key):
        return "money"
    if COUNT_PATTERN.search(key):
        return "count"
    present = [v for v in values if v is not None]
    if present and all(isinstance(v, bool) for v in present):
        return "text"
    if present and all(isinstance(v, (int, float)) for v in present):
        return "number"
    return "text"


def _decimals_for(unit: str) -> Optional[int]:
    # BR-011: money and rates to 2, ratios to 4, counts as integers.
    return {"money": 2, "rate": 2, "count": 0, "days": 1, "number": 4}.get(unit)


def _visible_keys(rows: List[Dict[str, Any]]) -> List[str]:
    seen: List[str] = []
    for row in rows:
        for key in row:
            if key.startswith(INTERNAL_PREFIXES):
                continue
            if key not in seen:
                seen.append(key)
    return seen


def build_columns(rows: List[Dict[str, Any]],
                  keys: Optional[List[str]] = None) -> List[Column]:
    keys = keys or _visible_keys(rows)
    columns = []
    for key in keys:
        values = [row.get(key) for row in rows]
        unit = infer_unit(key, values)
        columns.append(Column(key=key, label=humanize(key), unit=unit,
                              decimals=_decimals_for(unit)))
    return columns


def _source_note(evidence: EvidencePackage) -> str:
    parts = []
    if evidence.tables_used:
        parts.append("from {}".format(", ".join(evidence.tables_used)))
    if evidence.join_path:
        parts.append("joined via {}".format("; ".join(evidence.join_path)))
    if evidence.population:
        parts.append("population: {}".format(evidence.population))
    return " | ".join(parts)


def _footnotes(evidence: EvidencePackage) -> List[str]:
    notes: List[str] = []
    if evidence.comparison_baseline:
        baseline = evidence.comparison_baseline
        if baseline.get("value") is not None:
            notes.append("Baseline: {} = {} ({}).".format(
                humanize(str(baseline.get("measure", "value"))),
                baseline["value"], baseline.get("description", "declared baseline")))
    if evidence.sample_sizes.get("minimum_group_size"):
        excluded = evidence.sample_sizes.get("groups_excluded", 0)
        notes.append(
            "Groups with fewer than {} rows are excluded from ranking ({} excluded).".format(
                evidence.sample_sizes["minimum_group_size"], excluded))
    for limitation in evidence.limitations:
        notes.append(limitation)
    for warning in evidence.warnings:
        notes.append(warning)
    return notes


def build_result_table(evidence: EvidencePackage,
                       limit: int = MAX_DISPLAY_ROWS) -> Optional[EvidenceTable]:
    """The main analytical table, if the analysis produced one."""
    if not evidence.result_table:
        return None
    rows = evidence.result_table
    shown = rows[:limit]
    return EvidenceTable(
        key="result",
        title=evidence.problem_statement,
        columns=build_columns(shown),
        rows=shown,
        source_note=_source_note(evidence),
        footnotes=_footnotes(evidence),
        total_rows=len(rows),
        truncated=len(rows) > limit,
    )


def build_aggregate_table(evidence: EvidencePackage) -> Optional[EvidenceTable]:
    """Whole-population figures, one metric per row."""
    if not evidence.aggregates:
        return None
    rows = []
    for key, value in evidence.aggregates.items():
        if isinstance(value, dict):
            continue          # nested breakdowns belong in their own table
        unit = infer_unit(key, [value])
        rows.append({"metric": humanize(key), "value": value, "unit": unit})
    if not rows:
        return None
    return EvidenceTable(
        key="aggregates",
        title="Key figures",
        columns=[Column("metric", "Metric", "text"),
                 Column("value", "Value", "number", 4),
                 Column("unit", "Unit", "text")],
        rows=rows,
        source_note=_source_note(evidence),
        footnotes=_footnotes(evidence),
        total_rows=len(rows),
    )


def build_records_table(evidence: EvidencePackage,
                        limit: int = MAX_DISPLAY_ROWS) -> Optional[EvidenceTable]:
    """Individual source records, with their row ids preserved for traceability."""
    if not evidence.retrieved_records:
        return None
    rows = evidence.retrieved_records[:limit]
    keys = [k for k in _visible_keys(rows)]
    return EvidenceTable(
        key="records",
        title="Source records",
        columns=build_columns(rows, keys),
        rows=rows,
        source_note=_source_note(evidence),
        footnotes=_footnotes(evidence),
        total_rows=len(evidence.retrieved_records),
        truncated=len(evidence.retrieved_records) > limit,
    )


def build_calculation_table(evidence: EvidencePackage) -> Optional[EvidenceTable]:
    """Every computed value with the formula and inputs behind it.

    This is the audit table. It is what makes a figure checkable rather than merely
    presented, so it lists the formula id and the exact inputs for each calculation.
    """
    if not evidence.calculations:
        return None
    rows = []
    for calc in evidence.calculations:
        inputs = calc.get("inputs", {})
        rendered = ", ".join("{}={}".format(k, v) for k, v in inputs.items()
                             if not isinstance(v, (dict, list)))
        rows.append({
            "label": calc.get("label", ""),
            "formula": calc.get("formula_id", ""),
            "value": calc.get("value"),
            "inputs": rendered,
            "warnings": "; ".join(calc.get("warnings", [])),
        })
    return EvidenceTable(
        key="calculations",
        title="How each figure was calculated",
        columns=[Column("label", "Figure", "text"),
                 Column("formula", "Formula", "identifier"),
                 Column("value", "Value", "number", 4),
                 Column("inputs", "Inputs", "text"),
                 Column("warnings", "Warnings", "text")],
        rows=rows,
        source_note="every value produced by the formula engine",
        total_rows=len(rows),
    )


def build_rule_table(evidence: EvidencePackage) -> Optional[EvidenceTable]:
    """Which business rules were applied, and what each one excluded."""
    if not evidence.business_rules_applied:
        return None
    rows = []
    for rule in evidence.business_rules_applied:
        rows.append({
            "rule": rule.get("rule_id", ""),
            "statement": rule.get("note", ""),
            "rows_before": rule.get("rows_before"),
            "rows_after": rule.get("rows_after"),
            "rows_excluded": rule.get("rows_excluded"),
        })
    return EvidenceTable(
        key="business_rules",
        title="Business rules applied",
        columns=[Column("rule", "Rule", "identifier"),
                 Column("statement", "What it does", "text"),
                 Column("rows_before", "Rows before", "count", 0),
                 Column("rows_after", "Rows after", "count", 0),
                 Column("rows_excluded", "Excluded", "count", 0)],
        rows=rows,
        source_note="rules declared in config/business_rules.json",
        total_rows=len(rows),
    )


def build_limitations_table(evidence: EvidencePackage) -> Optional[EvidenceTable]:
    """What the analysis cannot establish, stated as its own table."""
    rows = []
    for text in evidence.unsupported_conclusions:
        rows.append({"kind": "Cannot be concluded", "statement": text})
    for text in evidence.limitations:
        rows.append({"kind": "Limitation", "statement": text})
    for note in evidence.data_quality_notes:
        detail = evidence.supporting_metrics.get("data_quality_detail", {}).get(note, {})
        rows.append({"kind": "Data quality ({})".format(note),
                     "statement": detail.get("finding", note)})
    if not rows:
        return None
    return EvidenceTable(
        key="limitations",
        title="What this analysis cannot establish",
        columns=[Column("kind", "Kind", "text"), Column("statement", "Statement", "text")],
        rows=rows,
        total_rows=len(rows),
    )


def build_tables(evidence: EvidencePackage,
                 limit: int = MAX_DISPLAY_ROWS) -> List[EvidenceTable]:
    """Every table this evidence supports, in reading order."""
    if not evidence.is_sufficient:
        return []
    builders = (
        lambda e: build_result_table(e, limit),
        build_aggregate_table,
        lambda e: build_records_table(e, limit),
        build_calculation_table,
        build_rule_table,
        build_limitations_table,
    )
    return [table for table in (builder(evidence) for builder in builders) if table]


def tables_as_dict(evidence: EvidencePackage,
                   limit: int = MAX_DISPLAY_ROWS) -> List[Dict[str, Any]]:
    return [table.as_dict() for table in build_tables(evidence, limit)]
