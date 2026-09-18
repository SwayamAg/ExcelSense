"""Chart specifications derived from evidence.

A chart is the most persuasive thing this system produces and therefore the easiest
place to mislead. Two rules hold it down:

  1. Every value plotted must already exist in the evidence package. Nothing is
     computed, interpolated, smoothed or extrapolated here.
  2. A chart is emitted only when the evidence actually supports the shape the PS
     asked for. A scalar cannot become a trend line; a suppressed small-sample rate
     cannot become a bar.

`verify_chart` re-checks the first rule against the evidence and is asserted in the
test suite, so a chart that drifted from its evidence would fail the build rather
than reach a reader.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.evidence.tables import humanize, infer_unit
from src.ps.evidence import EvidencePackage

MAX_CATEGORIES = 25

# Chart types and the evidence shape each one needs.
CATEGORY_CHARTS = {"bar", "horizontal_bar", "grouped_bar", "stacked_bar", "pie", "donut"}
SERIES_CHARTS = {"line", "multi_line"}
DISTRIBUTION_CHARTS = {"histogram", "box"}
POINT_CHARTS = {"scatter"}
TILE_CHARTS = {"kpi_tiles"}
NO_CHART = {None, "table_only"}


@dataclass
class Series:
    """One plotted series, named and with its unit declared."""
    name: str
    key: str
    values: List[Any]
    unit: str = "number"

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "key": self.key, "values": list(self.values),
                "unit": self.unit}


@dataclass
class Annotation:
    """A reference line, such as the baseline a ranking is measured against."""
    kind: str
    label: str
    value: Any

    def as_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "label": self.label, "value": self.value}


@dataclass
class ChartSpec:
    """Graph-ready structured data. No rendering, no library, no styling."""
    chart_type: str
    title: str
    category_key: str = ""
    category_label: str = ""
    categories: List[Any] = field(default_factory=list)
    series: List[Series] = field(default_factory=list)
    annotations: List[Annotation] = field(default_factory=list)
    tiles: List[Dict[str, Any]] = field(default_factory=list)
    footnotes: List[str] = field(default_factory=list)
    source_note: str = ""
    problem_id: str = ""
    truncated: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "chart_type": self.chart_type,
            "title": self.title,
            "problem_id": self.problem_id,
            "category_key": self.category_key,
            "category_label": self.category_label,
            "categories": list(self.categories),
            "series": [s.as_dict() for s in self.series],
            "annotations": [a.as_dict() for a in self.annotations],
            "tiles": list(self.tiles),
            "footnotes": list(self.footnotes),
            "source_note": self.source_note,
            "truncated": self.truncated,
        }

    @property
    def all_values(self) -> List[Any]:
        return [v for s in self.series for v in s.values]


class ChartError(Exception):
    """Raised when a chart is requested that the evidence cannot support."""


def _numeric_columns(rows: List[Dict[str, Any]], exclude: Sequence[str]) -> List[str]:
    """Columns that can be plotted, in the order the evidence presents them."""
    if not rows:
        return []
    excluded = set(exclude) | {"rank", "group_rows"}
    keys = []
    for key in rows[0]:
        if key in excluded or key.startswith("__"):
            continue
        values = [r.get(key) for r in rows]
        present = [v for v in values if v is not None]
        if present and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                           for v in present):
            keys.append(key)
    return keys


def _category_key(evidence: EvidencePackage, rows: List[Dict[str, Any]]) -> Optional[str]:
    """The label column: the first non-numeric column the evidence produced."""
    if not rows:
        return None
    for key in rows[0]:
        if key.startswith("__") or key in ("rank", "group_rows"):
            continue
        values = [r.get(key) for r in rows]
        present = [v for v in values if v is not None]
        if not present:
            continue
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present):
            return key
    return None


def _preferred_measure(evidence: EvidencePackage, candidates: List[str]) -> List[str]:
    """Plot what the analysis was about: its ranking measure, or its rates."""
    if not candidates:
        return []
    ranked = evidence.ranking.get("measure") if evidence.ranking else None
    if ranked and ranked in candidates:
        return [ranked]
    rates = [c for c in candidates if c.endswith(("_rate", "_pct"))]
    if rates:
        return rates[:2]
    return candidates[:2]


def _footnotes(evidence: EvidencePackage) -> List[str]:
    notes = list(evidence.limitations)
    if evidence.sample_sizes.get("groups_excluded"):
        excluded = evidence.sample_sizes["groups_excluded"]
        notes.append(
            "{} {} fell below the minimum sample size and {} not plotted.".format(
                excluded, "group" if excluded == 1 else "groups",
                "is" if excluded == 1 else "are"))
    for text in evidence.unsupported_conclusions:
        notes.append(text)
    return notes


def _source_note(evidence: EvidencePackage) -> str:
    parts = []
    if evidence.tables_used:
        parts.append("computed from {}".format(", ".join(evidence.tables_used)))
    if evidence.formulas_applied:
        parts.append("formulas: {}".format(", ".join(evidence.formulas_applied[:6])))
    return " | ".join(parts)


def _baseline_annotations(evidence: EvidencePackage) -> List[Annotation]:
    baseline = evidence.comparison_baseline
    if not baseline or baseline.get("value") is None:
        return []
    return [Annotation(
        kind="baseline",
        label=baseline.get("description", "baseline"),
        value=baseline["value"])]


def build_chart(evidence: EvidencePackage) -> Optional[ChartSpec]:
    """The chart this evidence supports, or None.

    Returning None is a legitimate and common outcome: insufficient evidence, a
    scalar answer, or a PS that asked for a table.
    """
    if not evidence.is_sufficient:
        return None

    requested = (evidence.visualization or {}).get("type")
    if requested in NO_CHART:
        return None

    if requested in TILE_CHARTS:
        return _build_tiles(evidence, requested)

    rows = evidence.result_table
    if not rows:
        # The PS asked for a chart but the analysis produced no table to draw from.
        return None

    category_key = _category_key(evidence, rows)
    if category_key is None and requested not in DISTRIBUTION_CHARTS:
        return None

    shown = rows[:MAX_CATEGORIES]
    numeric = _numeric_columns(shown, exclude=[category_key] if category_key else [])
    if not numeric:
        return None

    if requested in SERIES_CHARTS:
        return _build_series(evidence, requested, shown, category_key, numeric, len(rows))
    if requested in DISTRIBUTION_CHARTS:
        return _build_distribution(evidence, requested, shown, category_key, numeric, len(rows))
    if requested in POINT_CHARTS:
        return _build_scatter(evidence, requested, shown, category_key, numeric, len(rows))
    if requested in CATEGORY_CHARTS or requested == "waterfall" or requested == "heatmap":
        return _build_category(evidence, requested, shown, category_key, numeric, len(rows))
    return None


def _common(evidence: EvidencePackage, chart_type: str, total_rows: int,
            shown: int) -> Dict[str, Any]:
    return {
        "chart_type": chart_type,
        "title": evidence.problem_statement,
        "problem_id": evidence.detected_problem_id,
        "footnotes": _footnotes(evidence),
        "source_note": _source_note(evidence),
        "truncated": total_rows > shown,
    }


def _build_category(evidence, chart_type, rows, category_key, numeric,
                    total_rows) -> ChartSpec:
    keys = _preferred_measure(evidence, numeric)
    if chart_type in ("pie", "donut"):
        keys = keys[:1]                      # a pie can only carry one measure
    series = [Series(name=humanize(k), key=k, values=[r.get(k) for r in rows],
                     unit=infer_unit(k, [r.get(k) for r in rows]))
              for k in keys]
    return ChartSpec(
        category_key=category_key,
        category_label=humanize(category_key),
        categories=[rows[i].get(category_key) for i in range(len(rows))],
        series=series,
        annotations=_baseline_annotations(evidence),
        **_common(evidence, chart_type, total_rows, len(rows)))


def _build_series(evidence, chart_type, rows, category_key, numeric,
                  total_rows) -> ChartSpec:
    keys = numeric[:3] if chart_type == "multi_line" else _preferred_measure(evidence, numeric)
    series = [Series(name=humanize(k), key=k, values=[r.get(k) for r in rows],
                     unit=infer_unit(k, [r.get(k) for r in rows]))
              for k in keys]
    return ChartSpec(
        category_key=category_key,
        category_label=humanize(category_key),
        categories=[r.get(category_key) for r in rows],
        series=series,
        annotations=_baseline_annotations(evidence),
        **_common(evidence, chart_type, total_rows, len(rows)))


def _build_distribution(evidence, chart_type, rows, category_key, numeric,
                        total_rows) -> ChartSpec:
    """A histogram plots the banded counts the evidence already produced."""
    count_keys = [k for k in numeric if not k.endswith(("_pct", "_rate"))]
    keys = count_keys[:1] or numeric[:1]
    series = [Series(name=humanize(k), key=k, values=[r.get(k) for r in rows],
                     unit=infer_unit(k, [r.get(k) for r in rows]))
              for k in keys]
    annotations = []
    for label in ("median", "mean"):
        if evidence.aggregates.get(label) is not None:
            annotations.append(Annotation(kind=label, label=humanize(label),
                                          value=evidence.aggregates[label]))
    return ChartSpec(
        category_key=category_key or "",
        category_label=humanize(category_key) if category_key else "",
        categories=[r.get(category_key) for r in rows] if category_key else [],
        series=series,
        annotations=annotations,
        **_common(evidence, chart_type, total_rows, len(rows)))


def _build_scatter(evidence, chart_type, rows, category_key, numeric,
                   total_rows) -> ChartSpec:
    if len(numeric) < 2:
        return None
    x_key, y_key = numeric[0], numeric[1]
    return ChartSpec(
        category_key=x_key,
        category_label=humanize(x_key),
        categories=[r.get(x_key) for r in rows],
        series=[Series(name=humanize(y_key), key=y_key,
                       values=[r.get(y_key) for r in rows],
                       unit=infer_unit(y_key, [r.get(y_key) for r in rows]))],
        annotations=_baseline_annotations(evidence),
        **_common(evidence, chart_type, total_rows, len(rows)))


def _build_tiles(evidence: EvidencePackage, chart_type: str) -> Optional[ChartSpec]:
    """KPI tiles, drawn from the aggregates rather than a table."""
    tiles = []
    for key, value in evidence.aggregates.items():
        if isinstance(value, (dict, list)) or value is None:
            continue
        tiles.append({"label": humanize(key), "key": key, "value": value,
                      "unit": infer_unit(key, [value])})
    if not tiles:
        return None
    return ChartSpec(
        tiles=tiles[:12],
        **_common(evidence, chart_type, len(tiles), len(tiles)))


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_chart(chart: Optional[ChartSpec],
                 evidence: EvidencePackage) -> List[str]:
    """Check that every plotted value exists in the evidence.

    Returns a list of discrepancies; an empty list means the chart is reproducible
    from the evidence package alone. This is the Phase 6 gate, expressed as code.
    """
    if chart is None:
        return []

    problems: List[str] = []

    if not evidence.is_sufficient:
        problems.append("a chart was produced from insufficient evidence")
        return problems

    if chart.problem_id != evidence.detected_problem_id:
        problems.append("chart problem_id {!r} does not match the evidence {!r}".format(
            chart.problem_id, evidence.detected_problem_id))

    if chart.tiles:
        for tile in chart.tiles:
            key = tile["key"]
            if key not in evidence.aggregates:
                problems.append("tile {!r} is not in the evidence aggregates".format(key))
            elif evidence.aggregates[key] != tile["value"]:
                problems.append(
                    "tile {!r} shows {!r} but the evidence holds {!r}".format(
                        key, tile["value"], evidence.aggregates[key]))
        return problems

    rows = evidence.result_table
    if not rows:
        problems.append("a chart was produced with no evidence table behind it")
        return problems

    shown = rows[:len(chart.categories)] if chart.categories else rows

    if chart.category_key:
        expected = [r.get(chart.category_key) for r in shown]
        if expected != chart.categories:
            problems.append(
                "categories do not match the evidence column {!r}".format(chart.category_key))

    for series in chart.series:
        if not rows or series.key not in rows[0]:
            problems.append("series {!r} is not a column of the evidence table".format(
                series.key))
            continue
        expected = [r.get(series.key) for r in shown]
        if expected != series.values:
            problems.append(
                "series {!r} does not match the evidence column".format(series.key))
        if len(series.values) != len(chart.categories):
            problems.append(
                "series {!r} has {} values but there are {} categories".format(
                    series.key, len(series.values), len(chart.categories)))

    for annotation in chart.annotations:
        if annotation.kind == "baseline":
            baseline = evidence.comparison_baseline or {}
            if baseline.get("value") != annotation.value:
                problems.append("baseline annotation does not match the evidence baseline")
        elif annotation.kind in ("median", "mean"):
            if evidence.aggregates.get(annotation.kind) != annotation.value:
                problems.append(
                    "{} annotation does not match the evidence aggregate".format(
                        annotation.kind))

    return problems


def chart_as_dict(evidence: EvidencePackage) -> Optional[Dict[str, Any]]:
    chart = build_chart(evidence)
    return chart.as_dict() if chart else None
