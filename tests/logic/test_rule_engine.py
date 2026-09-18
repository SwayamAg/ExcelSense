"""Phase 3 - business rule engine tests.

A business rule decides which rows count. These tests check that each executable
rule selects the right population, reports what it excluded, and that every rule
carries the rationale that makes it challengeable.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.logic import RuleError, get_rule_engine
from src.logic.rule_engine import PII_COLUMNS, PERSISTENCY_ELIGIBLE_STATUSES

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def rules():
    return get_rule_engine()


# --------------------------------------------------------------------------
# Registry integrity
# --------------------------------------------------------------------------

def test_rules_are_registered(rules):
    assert len(rules.rule_ids) >= 15
    assert rules.rule_ids == sorted(rules.rule_ids)


def test_every_rule_states_itself_and_its_reason(rules):
    """A rule an executive cannot challenge is a hidden assumption."""
    for rule in rules.describe():
        for field in ("id", "name", "statement", "rationale", "category", "applies_to", "version"):
            assert rule.get(field), "{} missing {}".format(rule.get("id"), field)
        assert len(rule["rationale"].split()) >= 12, (
            "{} rationale is too thin to be challengeable".format(rule["id"]))


def test_rule_ids_are_unique_and_well_formed(rules):
    ids = rules.rule_ids
    assert len(ids) == len(set(ids))
    for rule_id in ids:
        assert rule_id.startswith("BR-") and rule_id[3:].isdigit()


def test_rules_grounded_in_findings_reference_real_ones(rules, contract):
    known = {f["id"] for f in contract["data_quality_findings"]}
    for rule in rules.describe():
        for finding in rule["grounded_in"]:
            assert finding in known, "{} cites unknown finding {}".format(rule["id"], finding)


def test_unknown_rule_is_rejected(rules):
    with pytest.raises(RuleError, match="unknown rule"):
        rules.get("BR-999")


def test_declared_rules_cannot_be_executed(rules):
    """A non-executable rule must not silently do nothing when applied."""
    with pytest.raises(RuleError, match="not an executable one"):
        rules.apply("BR-014", pd.DataFrame())


def test_parameter_access(rules):
    assert rules.parameter("BR-002", "default_minimum_agent_policies") == 30
    assert rules.parameter("BR-010", "default_window_days") == 365
    with pytest.raises(RuleError, match="no parameter"):
        rules.parameter("BR-002", "nonexistent")


def test_rule_categories_are_from_a_known_set(rules):
    allowed = {"population", "statistical_guard", "null_handling",
               "governance", "definition", "presentation"}
    for rule in rules.describe():
        assert rule["category"] in allowed, (
            "{} has category {}".format(rule["id"], rule["category"]))


# --------------------------------------------------------------------------
# BR-001 persistency population
# --------------------------------------------------------------------------

def test_br001_selects_only_the_eligible_population(rules, dataset):
    frame = dataset.table("policy").frame
    out, application = rules.apply("BR-001", frame)
    assert len(out) == 24542
    assert set(out["POLICY.STATUS"]) == set(PERSISTENCY_ELIGIBLE_STATUSES)


def test_br001_reports_exactly_what_it_excluded(rules, dataset):
    _, application = rules.apply("BR-001", dataset.table("policy").frame)
    assert application.rows_excluded == 458
    assert application.detail["excluded_by_status"] == {"Surrendered": 324, "Death": 134}


def test_br001_denominator_matches_the_persistency_table(rules, dataset):
    """The whole point: the eligible count must equal the actual renewal rows."""
    out, _ = rules.apply("BR-001", dataset.table("policy").frame)
    assert len(out) == len(dataset.table("persistency"))


def test_br001_needs_the_status_column(rules):
    with pytest.raises(RuleError, match="needs column"):
        rules.apply("BR-001", pd.DataFrame({"other": [1]}))


# --------------------------------------------------------------------------
# BR-002 minimum sample
# --------------------------------------------------------------------------

def test_br002_excludes_thin_groups(rules):
    frame = pd.DataFrame({"agent": list("abcd"), "policies": [100, 50, 5, 2]})
    out, application = rules.apply("BR-002", frame, size_column="policies", minimum=30)
    assert list(out["agent"]) == ["a", "b"]
    assert application.detail["groups_excluded"] == 2
    assert application.detail["rows_in_excluded_groups"] == 7


def test_br002_on_real_agents(rules, dataset):
    counts = (dataset.table("policy").frame.groupby("SALES_ID").size()
              .reset_index(name="policies"))
    out, application = rules.apply("BR-002", counts, size_column="policies", minimum=30)
    assert len(out) == 242
    assert application.detail["groups_before"] == 1956


def test_br002_keeps_everything_when_the_threshold_is_one(rules):
    frame = pd.DataFrame({"g": ["a", "b"], "n": [1, 1]})
    out, _ = rules.apply("BR-002", frame, size_column="n", minimum=1)
    assert len(out) == 2


def test_br002_boundary_is_inclusive(rules):
    frame = pd.DataFrame({"g": ["a", "b"], "n": [30, 29]})
    out, _ = rules.apply("BR-002", frame, size_column="n", minimum=30)
    assert list(out["g"]) == ["a"]


# --------------------------------------------------------------------------
# BR-003 unsettled claims
# --------------------------------------------------------------------------

def test_br003_excludes_the_unsettled_claim(rules, dataset):
    out, application = rules.apply("BR-003", dataset.table("claims").frame)
    assert len(out) == 133
    assert application.detail["unsettled_excluded"] == 1
    assert set(out["Status.of.Claim"]) == {"Paid"}


def test_br003_removes_the_row_whose_amount_is_a_sentinel(rules, dataset):
    out, _ = rules.apply("BR-003", dataset.table("claims").frame)
    assert out["Amount.Paid"].notna().all(), "a sentinel row survived the settled filter"


# --------------------------------------------------------------------------
# BR-004 null metrics
# --------------------------------------------------------------------------

def test_br004_excludes_null_rates_rather_than_zeroing_them(rules, dataset):
    frame = dataset.table("sales").frame
    out, application = rules.apply("BR-004", frame, column="First Year Persistency Rate")
    assert len(out) == 1587
    assert application.detail["null_rows_excluded"] == 369
    assert out["First Year Persistency Rate"].notna().all()


def test_br004_excluded_agents_are_not_ranked_at_zero(rules, dataset):
    """Reading null as zero would put 369 agents at the bottom on missing data alone."""
    frame = dataset.table("sales").frame
    out, _ = rules.apply("BR-004", frame, column="First Year Persistency Rate")
    assert (out["First Year Persistency Rate"] == 0).sum() < 369


# --------------------------------------------------------------------------
# BR-005 negative durations
# --------------------------------------------------------------------------

def test_br005_excludes_negative_durations(rules):
    frame = pd.DataFrame({"days": [10, 0, -3, None, 5]})
    out, application = rules.apply("BR-005", frame, column="days")
    assert application.detail["negative_rows_excluded"] == 1
    assert len(out) == 4


def test_br005_keeps_zero_and_null(rules):
    frame = pd.DataFrame({"days": [0, None]})
    out, _ = rules.apply("BR-005", frame, column="days")
    assert len(out) == 2, "zero and null are not negative and must survive"


def test_br005_does_not_clip_to_zero(rules):
    frame = pd.DataFrame({"days": [-3]})
    out, _ = rules.apply("BR-005", frame, column="days")
    assert len(out) == 0
    assert (out["days"] == 0).sum() == 0


def test_br005_on_real_underwriting_durations(rules, dataset):
    from src.data import query
    result = query(dataset, "policy").date_diff("uw_days", "Inforce.Date", "Login.Date")
    out, application = rules.apply("BR-005", result.frame, column="uw_days")
    assert application.detail["negative_rows_excluded"] == 1
    assert len(out) == 24999


# --------------------------------------------------------------------------
# BR-006 personal data
# --------------------------------------------------------------------------

def test_br006_removes_personal_columns(rules, dataset):
    frame = dataset.table("owner").frame
    out, application = rules.apply("BR-006", frame)
    for column in PII_COLUMNS:
        assert column not in out.columns
    assert set(application.detail["columns_withheld"]) == set(PII_COLUMNS)


def test_br006_removes_prefixed_columns_after_a_join(rules, dataset):
    from src.data import query
    joined = query(dataset, "policy").join("policy_owner", how="left")
    out, application = rules.apply("BR-006", joined.frame)
    assert not [c for c in out.columns if "Given.Name" in c or "Postal.Code" in c]
    assert application.detail["columns_withheld"]


def test_br006_leaves_analytical_columns_alone(rules, dataset):
    out, _ = rules.apply("BR-006", dataset.table("owner").frame)
    for column in ("Owner.Age", "Owner.Gender", "Owner.Comm.State", "ClientID"):
        assert column in out.columns


def test_br006_is_a_no_op_when_no_personal_data_is_present(rules, dataset):
    frame = dataset.table("policy").frame
    out, application = rules.apply("BR-006", frame)
    assert len(out.columns) == len(frame.columns)
    assert application.detail["columns_withheld"] == []


# --------------------------------------------------------------------------
# BR-008 small cells
# --------------------------------------------------------------------------

def test_br008_suppresses_rates_from_thin_cells(rules):
    frame = pd.DataFrame({
        "band": ["a", "b", "c"],
        "claims": [50, 3, 10],
        "claim_rate": [1.5, 66.7, 2.0],
    })
    out, application = rules.apply("BR-008", frame, count_column="claims", minimum=5)
    assert pd.isna(out.loc[1, "claim_rate"]), "a rate from 3 claims survived"
    assert out.loc[0, "claim_rate"] == 1.5
    assert application.detail["cells_suppressed"] == 1


def test_br008_keeps_the_count_visible(rules):
    """Suppressing the rate must not hide the population it came from."""
    frame = pd.DataFrame({"band": ["a"], "claims": [2], "claim_rate": [50.0]})
    out, _ = rules.apply("BR-008", frame, count_column="claims", minimum=5)
    assert out.loc[0, "claims"] == 2
    assert pd.isna(out.loc[0, "claim_rate"])


def test_br008_targets_named_rate_columns(rules):
    frame = pd.DataFrame({"n": [2], "lapse_rate": [50.0], "total_ape": [1000.0]})
    out, _ = rules.apply("BR-008", frame, count_column="n", minimum=5,
                         rate_columns=["lapse_rate"])
    assert pd.isna(out.loc[0, "lapse_rate"])
    assert out.loc[0, "total_ape"] == 1000.0


# --------------------------------------------------------------------------
# Application records
# --------------------------------------------------------------------------

def test_application_is_serialisable_as_evidence(rules, dataset):
    _, application = rules.apply("BR-001", dataset.table("policy").frame)
    evidence = application.as_evidence()
    json.dumps(evidence)
    for field in ("rule_id", "applied", "detail", "note", "rows_before",
                  "rows_after", "rows_excluded", "engine_version"):
        assert field in evidence


def test_application_note_is_the_rule_statement(rules, dataset):
    _, application = rules.apply("BR-003", dataset.table("claims").frame)
    assert application.note == rules.get("BR-003").statement


# --------------------------------------------------------------------------
# Config export and PS mapping
# --------------------------------------------------------------------------

def test_exported_config_matches_the_registry(rules):
    payload = json.loads((ROOT / "config" / "business_rules.json").read_text(encoding="utf-8"))
    exported = {r["id"]: r for r in payload["rules"]}
    assert set(exported) == set(rules.rule_ids)
    for rule in rules.describe():
        for field in ("statement", "rationale", "category", "parameters"):
            assert exported[rule["id"]][field] == rule[field], (
                "{} drifted between code and config".format(rule["id"]))


def test_exported_formula_config_matches_the_engine():
    from src.logic import get_engine
    engine = get_engine()
    payload = json.loads((ROOT / "config" / "formulas.json").read_text(encoding="utf-8"))
    exported = {f["id"]: f for f in payload["formulas"]}
    assert set(exported) == set(engine.formula_ids)
    for formula in engine.describe():
        for field in ("expression", "null_handling", "zero_denominator", "rounding_decimals"):
            assert exported[formula["id"]][field] == formula[field], (
                "{} drifted between code and config".format(formula["id"]))


def test_every_ps_is_mapped_to_its_logic():
    mapping = json.loads((ROOT / "config" / "ps_logic_mapping.json").read_text(encoding="utf-8"))
    catalogue = json.loads(
        (ROOT / "config" / "problem_statements.json").read_text(encoding="utf-8"))
    mapped = {m["problem_id"] for m in mapping["mappings"]}
    expected = {ps["problem_id"] for ps in catalogue["problem_statements"]}
    assert mapped == expected


def test_mapped_formulas_and_rules_all_exist(rules):
    from src.logic import get_engine
    engine = get_engine()
    mapping = json.loads((ROOT / "config" / "ps_logic_mapping.json").read_text(encoding="utf-8"))
    for entry in mapping["mappings"]:
        for formula_id in entry["formula_ids"]:
            assert formula_id in engine.formula_ids, (
                "{} maps to unknown formula {}".format(entry["problem_id"], formula_id))
        for rule_id in entry["business_rule_ids"]:
            assert rule_id in rules.rule_ids, (
                "{} maps to unknown rule {}".format(entry["problem_id"], rule_id))


def test_persistency_ps_are_all_bound_to_br001():
    """Every PS touching Persistency must carry the population rule."""
    mapping = json.loads((ROOT / "config" / "ps_logic_mapping.json").read_text(encoding="utf-8"))
    catalogue = json.loads(
        (ROOT / "config" / "problem_statements.json").read_text(encoding="utf-8"))
    rules_by_ps = {m["problem_id"]: m["business_rule_ids"] for m in mapping["mappings"]}
    for ps in catalogue["problem_statements"]:
        if "persistency" in ps["relevant_tables"]:
            assert "BR-001" in rules_by_ps[ps["problem_id"]], (
                "{} uses Persistency but is not bound to BR-001".format(ps["problem_id"]))


def test_diagnostic_ps_are_bound_to_the_no_causation_rule():
    mapping = json.loads((ROOT / "config" / "ps_logic_mapping.json").read_text(encoding="utf-8"))
    catalogue = json.loads(
        (ROOT / "config" / "problem_statements.json").read_text(encoding="utf-8"))
    rules_by_ps = {m["problem_id"]: m["business_rule_ids"] for m in mapping["mappings"]}
    for ps in catalogue["problem_statements"]:
        if ps["analysis_tier"] == "diagnostic":
            assert "BR-014" in rules_by_ps[ps["problem_id"]], (
                "{} is diagnostic but not bound to BR-014".format(ps["problem_id"]))


def test_every_executable_rule_is_used_by_at_least_one_ps(rules):
    payload = json.loads((ROOT / "config" / "business_rules.json").read_text(encoding="utf-8"))
    for rule in payload["rules"]:
        if rule["executable"]:
            assert rule["applies_to_ps_count"] > 0, (
                "{} is executable but no PS triggers it".format(rule["id"]))
