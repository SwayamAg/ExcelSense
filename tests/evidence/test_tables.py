"""Phase 6 - evidence table tests.

The tables are the audit trail made readable. These tests check that what a reader
sees matches the source data, that every figure can be traced to a formula, and that
warnings which qualify a number travel with it rather than being dropped in
presentation.
"""
from __future__ import annotations

import json

import pytest

import src.analysis.advanced_patterns  # noqa: F401
from src.evidence import (
    build_aggregate_table,
    build_calculation_table,
    build_limitations_table,
    build_records_table,
    build_result_table,
    build_rule_table,
    build_tables,
    humanize,
    infer_unit,
    tables_as_dict,
)

REQUIRED_PARAMETERS = {
    "PS-031": {"app_id": "11000000015"},
    "PS-069": {"agent_id": "977337"},
}


@pytest.fixture(scope="module")
def executor(ps_executor):
    return ps_executor


@pytest.fixture(scope="module")
def all_tables(executor):
    out = {}
    for ps in executor.registry.executable():
        evidence = executor.execute(ps.id, REQUIRED_PARAMETERS.get(ps.id))
        out[ps.id] = (evidence, build_tables(evidence))
    return out


# --------------------------------------------------------------------------
# Every analysis yields readable evidence
# --------------------------------------------------------------------------

def test_every_ps_produces_at_least_one_table(all_tables):
    empty = [pid for pid, (_, tables) in all_tables.items() if not tables]
    assert not empty, "PSs produced no evidence table: {}".format(empty)


def test_every_ps_produces_a_calculation_table(all_tables):
    """A figure with no visible derivation cannot be audited."""
    missing = [pid for pid, (ev, tables) in all_tables.items()
               if ev.calculations and "calculations" not in {t.key for t in tables}]
    assert not missing, "PSs hide their calculations: {}".format(missing)


def test_tables_are_json_serialisable(all_tables):
    for pid, (evidence, _) in all_tables.items():
        json.dumps(tables_as_dict(evidence))


def test_insufficient_evidence_produces_no_tables(executor):
    evidence = executor.execute("PS-031", {"app_id": "00000000000"})
    assert not evidence.is_sufficient
    assert build_tables(evidence) == []


# --------------------------------------------------------------------------
# Evidence matches the source data
# --------------------------------------------------------------------------

def test_result_table_matches_the_evidence_exactly(all_tables):
    """Presentation must not alter a value on its way to the reader."""
    for pid, (evidence, tables) in all_tables.items():
        result = next((t for t in tables if t.key == "result"), None)
        if result is None:
            continue
        assert result.rows == evidence.result_table[:len(result.rows)]
        assert result.total_rows == len(evidence.result_table)


def test_status_counts_match_the_source_csv(executor, dataset):
    evidence = executor.execute("PS-002")
    table = build_result_table(evidence)
    shown = {row["POLICY.STATUS"]: row["policies"] for row in table.rows}
    actual = dataset.table("policy").col("POLICY.STATUS").value_counts().to_dict()
    assert shown == actual


def test_claim_totals_match_the_source_csv(executor, dataset):
    evidence = executor.execute("PS-052")
    table = build_aggregate_table(evidence)
    values = {row["metric"]: row["value"] for row in table.rows}
    claims = dataset.table("claims")
    assert values["Claim count"] == len(claims)
    assert values["Total claim amount"] == round(float(claims.num("Claim.Amount").sum()), 2)


def test_record_table_preserves_the_source_row_id(executor):
    """Traceability: a record must still point back at its CSV row."""
    from src.data.contract import ROW_ID
    evidence = executor.execute("PS-031", {"app_id": "11000000015"})
    table = build_records_table(evidence)
    assert table is not None
    assert ROW_ID in evidence.retrieved_records[0]
    assert table.rows[0]["AppID"] == "11000000015"


def test_internal_columns_are_not_shown(all_tables):
    for pid, (_, tables) in all_tables.items():
        for table in tables:
            for column in table.columns:
                assert not column.key.startswith("__"), (
                    "{}: internal column {} is visible".format(pid, column.key))


def test_personal_data_never_reaches_a_table(all_tables):
    for pid, (_, tables) in all_tables.items():
        for table in tables:
            leaked = [c.key for c in table.columns
                      if "Given.Name" in c.key or "Postal.Code" in c.key]
            assert not leaked, "{} exposes {}".format(pid, leaked)


# --------------------------------------------------------------------------
# Calculated values match the formula engine
# --------------------------------------------------------------------------

def test_calculation_table_names_a_formula_for_every_figure(all_tables):
    for pid, (evidence, tables) in all_tables.items():
        table = next((t for t in tables if t.key == "calculations"), None)
        if table is None:
            continue
        for row in table.rows:
            assert row["formula"], "{}: a calculation has no formula id".format(pid)


def test_calculations_use_only_registered_formulas(all_tables):
    from src.logic import get_engine
    known = set(get_engine().formula_ids)
    for pid, (_, tables) in all_tables.items():
        table = next((t for t in tables if t.key == "calculations"), None)
        if table is None:
            continue
        for row in table.rows:
            assert row["formula"] in known, "{}: unknown formula {}".format(pid, row["formula"])


def test_calculated_values_are_reproducible_from_their_inputs(executor):
    """Recompute a rate from the inputs shown and confirm it lands on the same number."""
    from src.logic import get_engine
    engine = get_engine()
    evidence = executor.execute("PS-035")
    calc = next(c for c in evidence.calculations if c["formula_id"] == "F-RATE")
    recomputed = engine.compute("F-RATE", **calc["inputs"])
    assert recomputed.value == calc["value"]


def test_rate_calculations_shown_in_the_table_match_the_evidence(executor):
    evidence = executor.execute("PS-035")
    table = build_calculation_table(evidence)
    rate_row = next(r for r in table.rows if r["formula"] == "F-RATE")
    assert rate_row["value"] == evidence.aggregates["lapse_rate"] == 10.948
    assert "numerator=2737" in rate_row["inputs"]


# --------------------------------------------------------------------------
# Warnings and limitations survive presentation
# --------------------------------------------------------------------------

def test_small_sample_warning_is_preserved(executor):
    """A rate from a thin group must not lose the caveat that qualifies it."""
    evidence = executor.execute("PS-013")
    table = build_result_table(evidence)
    joined = " ".join(table.footnotes).lower()
    assert "minimum sample" in joined or "threshold" in joined
    assert evidence.sample_sizes["groups_excluded"] >= 1


def test_baseline_is_stated_as_a_footnote(executor):
    evidence = executor.execute("PS-013")
    table = build_result_table(evidence)
    assert any("Baseline" in note for note in table.footnotes)
    assert str(evidence.comparison_baseline["value"]) in " ".join(table.footnotes)


def test_limitations_table_states_what_cannot_be_concluded(executor):
    evidence = executor.execute("PS-040")
    table = build_limitations_table(evidence)
    assert table is not None
    kinds = {row["kind"] for row in table.rows}
    assert "Cannot be concluded" in kinds
    assert any("caus" in row["statement"].lower() for row in table.rows)


def test_every_diagnostic_ps_publishes_its_limitations(all_tables, executor):
    for pid, (evidence, tables) in all_tables.items():
        if executor.registry.get(pid).analysis_tier != "diagnostic":
            continue
        assert "limitations" in {t.key for t in tables}, (
            "{} is diagnostic but publishes no limitations table".format(pid))


def test_business_rules_table_reports_what_was_excluded(executor):
    evidence = executor.execute("PS-029")
    table = build_rule_table(evidence)
    assert table is not None
    rule = next(r for r in table.rows if r["rule"] == "BR-005")
    assert rule["rows_excluded"] == 1
    assert rule["statement"]


def test_data_quality_notes_reach_the_reader(executor):
    evidence = executor.execute("PS-036")
    table = build_limitations_table(evidence)
    assert any("DQ-001" in row["kind"] for row in table.rows)


# --------------------------------------------------------------------------
# Formatting and units
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key,expected", [
    ("lapse_rate", "rate"),
    ("share_pct", "rate"),
    ("total_ape", "money"),
    ("avg_sum_assured", "money"),
    ("policies", "count"),
    ("claim_count", "count"),
    ("renewal_delay_days", "days"),
    ("AppID", "identifier"),
    ("SALES_ID", "identifier"),
])
def test_units_are_inferred_from_column_names(key, expected):
    assert infer_unit(key, [1.0]) == expected


def test_counts_never_carry_decimals(all_tables):
    for pid, (_, tables) in all_tables.items():
        for table in tables:
            for column in table.columns:
                if column.unit == "count":
                    assert column.decimals == 0, "{}: {} would show decimals".format(
                        pid, column.key)


def test_money_and_rates_round_to_two_places(all_tables):
    for pid, (_, tables) in all_tables.items():
        for table in tables:
            for column in table.columns:
                if column.unit in ("money", "rate"):
                    assert column.decimals == 2


def test_labels_are_readable():
    assert humanize("total_ape") == "Total APE"
    assert humanize("lapse_rate") == "Lapse rate"
    assert humanize("share_pct") == "Share %"
    assert humanize("POLICY.STATUS") == "POLICY STATUS"


def test_every_column_has_a_label_and_unit(all_tables):
    for pid, (_, tables) in all_tables.items():
        for table in tables:
            for column in table.columns:
                assert column.label, "{}: column {} has no label".format(pid, column.key)
                assert column.unit, "{}: column {} has no unit".format(pid, column.key)


# --------------------------------------------------------------------------
# Truncation is declared
# --------------------------------------------------------------------------

def test_large_tables_are_truncated_and_say_so(executor):
    evidence = executor.execute("PS-034")
    table = build_result_table(evidence, limit=10)
    assert len(table.rows) == 10
    assert table.truncated is True
    assert table.total_rows > 10


def test_small_tables_are_not_marked_truncated(executor):
    evidence = executor.execute("PS-002")
    table = build_result_table(evidence)
    assert table.truncated is False
    assert table.total_rows == len(table.rows) == 4


def test_source_note_names_the_tables_used(all_tables):
    for pid, (evidence, tables) in all_tables.items():
        result = next((t for t in tables if t.key == "result"), None)
        if result is None:
            continue
        for table_key in evidence.tables_used[:1]:
            assert table_key in result.source_note, (
                "{}: source note omits {}".format(pid, table_key))
