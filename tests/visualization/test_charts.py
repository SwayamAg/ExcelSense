"""Phase 6 - chart tests.

The gate is that every supported visualization is reproducible from the evidence
package. `verify_chart` re-derives each plotted value from the evidence, so these
tests fail if a chart ever drifts from the numbers behind it.
"""
from __future__ import annotations

import json

import pytest

import src.analysis.advanced_patterns  # noqa: F401
from src.visualization import build_chart, chart_as_dict, verify_chart
from src.visualization.charts import MAX_CATEGORIES

REQUIRED_PARAMETERS = {
    "PS-031": {"app_id": "11000000015"},
    "PS-069": {"agent_id": "977337"},
}


@pytest.fixture(scope="module")
def executor(ps_executor):
    return ps_executor


@pytest.fixture(scope="module")
def all_charts(executor):
    out = {}
    for ps in executor.registry.executable():
        evidence = executor.execute(ps.id, REQUIRED_PARAMETERS.get(ps.id))
        out[ps.id] = (evidence, build_chart(evidence))
    return out


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------

def test_every_chart_is_reproducible_from_its_evidence(all_charts):
    """The Phase 6 gate, in one assertion."""
    failures = {}
    for pid, (evidence, chart) in all_charts.items():
        problems = verify_chart(chart, evidence)
        if problems:
            failures[pid] = problems
    assert not failures, "charts that do not match their evidence: {}".format(failures)


def test_every_ps_recommending_a_chart_produces_one(all_charts, executor):
    missing = []
    for pid, (evidence, chart) in all_charts.items():
        ps = executor.registry.get(pid)
        if ps.record.get("visualization_recommended") and chart is None:
            missing.append((pid, ps.record.get("visualization_type")))
    assert not missing, "PSs recommend a chart but produce none: {}".format(missing)


def test_chart_count_matches_the_catalogue(all_charts, executor):
    built = sum(1 for _, chart in all_charts.values() if chart is not None)
    recommended = sum(1 for ps in executor.registry.executable()
                      if ps.record.get("visualization_recommended"))
    assert built == recommended == 79


def test_charts_are_json_serialisable(all_charts):
    for pid, (evidence, chart) in all_charts.items():
        if chart is not None:
            json.dumps(chart.as_dict())


# --------------------------------------------------------------------------
# No graph from unsupported data
# --------------------------------------------------------------------------

def test_no_chart_from_insufficient_evidence(executor):
    evidence = executor.execute("PS-031", {"app_id": "00000000000"})
    assert not evidence.is_sufficient
    assert build_chart(evidence) is None


def test_verification_rejects_a_chart_built_on_insufficient_evidence(executor):
    good = executor.execute("PS-013")
    chart = build_chart(good)
    broken = executor.execute("PS-031", {"app_id": "00000000000"})
    problems = verify_chart(chart, broken)
    assert problems and "insufficient evidence" in problems[0]


def test_no_chart_for_a_table_only_ps(executor):
    """A PS that asked for a table must not be given a chart."""
    for pid in ("PS-020", "PS-073", "PS-089", "PS-095", "PS-096", "PS-098"):
        evidence = executor.execute(pid)
        assert evidence.visualization is None or \
            evidence.visualization.get("type") in (None, "table_only")
        assert build_chart(evidence) is None, "{} was given a chart".format(pid)


def test_no_chart_for_a_scalar_answer(executor):
    """A single number cannot honestly become a bar chart."""
    for pid in ("PS-001", "PS-003", "PS-004", "PS-005", "PS-011", "PS-035", "PS-054"):
        evidence = executor.execute(pid)
        assert evidence.expected_answer_type == "scalar"
        assert build_chart(evidence) is None


def test_verification_catches_a_tampered_value(executor):
    """The check must actually fail when a value is changed."""
    evidence = executor.execute("PS-013")
    chart = build_chart(evidence)
    assert verify_chart(chart, evidence) == []
    chart.series[0].values[0] = 999.0
    assert verify_chart(chart, evidence), "a tampered value went undetected"


def test_verification_catches_a_tampered_category(executor):
    evidence = executor.execute("PS-002")
    chart = build_chart(evidence)
    chart.categories[0] = "Invented status"
    problems = verify_chart(chart, evidence)
    assert any("categories" in p for p in problems)


def test_verification_catches_an_invented_series(executor):
    from src.visualization.charts import Series
    evidence = executor.execute("PS-002")
    chart = build_chart(evidence)
    chart.series.append(Series(name="Invented", key="not_a_column", values=[1, 2, 3, 4]))
    problems = verify_chart(chart, evidence)
    assert any("not a column" in p for p in problems)


def test_verification_catches_a_tampered_baseline(executor):
    evidence = executor.execute("PS-013")
    chart = build_chart(evidence)
    assert chart.annotations
    chart.annotations[0].value = 99.9
    problems = verify_chart(chart, evidence)
    assert any("baseline" in p for p in problems)


def test_verification_catches_a_mismatched_problem_id(executor):
    evidence = executor.execute("PS-013")
    chart = build_chart(evidence)
    chart.problem_id = "PS-999"
    assert any("problem_id" in p for p in verify_chart(chart, evidence))


# --------------------------------------------------------------------------
# Chart shape follows the evidence shape
# --------------------------------------------------------------------------

def test_series_length_always_matches_categories(all_charts):
    for pid, (_, chart) in all_charts.items():
        if chart is None or chart.tiles:
            continue
        for series in chart.series:
            assert len(series.values) == len(chart.categories), (
                "{}: series {} has {} values for {} categories".format(
                    pid, series.key, len(series.values), len(chart.categories)))


def test_ranking_chart_plots_the_ranked_measure(executor):
    evidence = executor.execute("PS-013")
    chart = build_chart(evidence)
    assert chart.chart_type == "bar"
    assert chart.series[0].key == evidence.ranking["measure"] == "lapse_rate"
    assert chart.categories[0] == "P91"


def test_ranking_chart_carries_the_baseline_as_an_annotation(executor):
    evidence = executor.execute("PS-013")
    chart = build_chart(evidence)
    baseline = next(a for a in chart.annotations if a.kind == "baseline")
    assert baseline.value == evidence.comparison_baseline["value"] == 10.948


def test_timeseries_chart_is_chronological(executor):
    evidence = executor.execute("PS-032")
    chart = build_chart(evidence)
    assert chart.chart_type == "line"
    assert chart.categories == sorted(chart.categories)
    assert chart.categories[0] == "2023-01"


def test_distribution_chart_plots_counts_with_central_annotations(executor):
    evidence = executor.execute("PS-022")
    chart = build_chart(evidence)
    assert chart.chart_type == "histogram"
    kinds = {a.kind for a in chart.annotations}
    assert {"median", "mean"} <= kinds
    for annotation in chart.annotations:
        assert annotation.value == evidence.aggregates[annotation.kind]


def test_pie_chart_carries_exactly_one_series(executor):
    evidence = executor.execute("PS-002")
    chart = build_chart(evidence)
    assert chart.chart_type == "donut"
    assert len(chart.series) == 1


def test_kpi_tiles_come_from_the_aggregates(executor):
    evidence = executor.execute("PS-009")
    chart = build_chart(evidence)
    assert chart.chart_type == "kpi_tiles"
    assert chart.tiles
    for tile in chart.tiles:
        assert evidence.aggregates[tile["key"]] == tile["value"]


def test_charts_are_capped_and_declare_truncation(executor):
    evidence = executor.execute("PS-034")
    chart = build_chart(evidence)
    assert len(chart.categories) <= MAX_CATEGORIES
    if len(evidence.result_table) > MAX_CATEGORIES:
        assert chart.truncated is True


def test_chart_units_are_declared(all_charts):
    for pid, (_, chart) in all_charts.items():
        if chart is None:
            continue
        for series in chart.series:
            assert series.unit, "{}: series {} has no unit".format(pid, series.key)


# --------------------------------------------------------------------------
# Warnings survive into the chart
# --------------------------------------------------------------------------

def test_small_sample_warning_reaches_the_chart(executor):
    """A chart that omits excluded groups must say that it did."""
    evidence = executor.execute("PS-013")
    chart = build_chart(evidence)
    joined = " ".join(chart.footnotes).lower()
    assert "minimum sample" in joined or "not plotted" in joined


def test_diagnostic_charts_carry_the_causation_disclaimer(executor):
    evidence = executor.execute("PS-040")
    chart = build_chart(evidence)
    assert chart is not None
    joined = " ".join(chart.footnotes).lower()
    assert "caus" in joined or "association" in joined


def test_timeseries_chart_warns_about_the_final_period(executor):
    evidence = executor.execute("PS-032")
    chart = build_chart(evidence)
    assert any("incomplete" in note for note in chart.footnotes)


def test_every_chart_names_its_source(all_charts):
    for pid, (_, chart) in all_charts.items():
        if chart is None:
            continue
        assert chart.source_note, "{}: chart has no source note".format(pid)
        assert "computed from" in chart.source_note


def test_every_chart_names_its_problem_statement(all_charts):
    for pid, (evidence, chart) in all_charts.items():
        if chart is None:
            continue
        assert chart.problem_id == pid
        assert chart.title == evidence.problem_statement


# --------------------------------------------------------------------------
# Helper
# --------------------------------------------------------------------------

def test_chart_as_dict_matches_build_chart(executor):
    evidence = executor.execute("PS-013")
    assert chart_as_dict(evidence) == build_chart(evidence).as_dict()


def test_chart_as_dict_is_none_when_no_chart(executor):
    assert chart_as_dict(executor.execute("PS-001")) is None
