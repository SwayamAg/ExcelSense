"""Phase 4 - execution engine tests.

The gate is that a PS produces reproducible evidence from the CSVs without any LLM
involvement. These tests execute every PS, check the numbers against figures derived
independently, and check that failure is reported rather than disguised.
"""
from __future__ import annotations

import json

import pytest

import src.analysis.advanced_patterns  # noqa: F401  (registers the extra patterns)
from src.ps.evidence import EvidencePackage
from src.ps.executor import PSExecutor
from src.ps.registry import ParameterError, PSNotFound

# Parameters for the PSs that require one.
REQUIRED_PARAMETERS = {
    "PS-031": {"app_id": "15011000000"},
    "PS-069": {"agent_id": "977337"},
}


@pytest.fixture(scope="module")
def executor(ps_executor):
    return ps_executor


@pytest.fixture(scope="module")
def all_evidence(executor):
    return {ps.id: executor.execute(ps.id, REQUIRED_PARAMETERS.get(ps.id))
            for ps in executor.registry.executable()}


# --------------------------------------------------------------------------
# The gate: every PS executes and produces evidence
# --------------------------------------------------------------------------

def test_every_ps_has_an_execution_plan(executor):
    total = len(executor.registry)
    executable = len(executor.registry.executable())
    assert executable == total, "{} of {} PSs have no execution plan".format(
        total - executable, total)


def test_every_ps_executes_successfully(all_evidence):
    """The Phase 4 gate, in one assertion."""
    failed = {pid: ev.insufficient.reason
              for pid, ev in all_evidence.items() if not ev.is_sufficient}
    assert not failed, "PSs failed to produce evidence: {}".format(failed)


def test_every_ps_produces_a_result_or_aggregates(all_evidence):
    empty = [pid for pid, ev in all_evidence.items()
             if not ev.result_table and not ev.aggregates and not ev.retrieved_records]
    assert not empty, "PSs produced no findings at all: {}".format(empty)


def test_every_ps_records_the_tables_it_used(all_evidence):
    for pid, ev in all_evidence.items():
        assert ev.tables_used, "{} records no tables".format(pid)


def test_every_computed_ps_records_a_formula(all_evidence, executor):
    """A number with no formula behind it is a number nobody can audit."""
    for pid, ev in all_evidence.items():
        ps = executor.registry.get(pid)
        if not ps.formula_ids:
            continue
        assert ev.formulas_applied, "{} declares formulas but applied none".format(pid)


def test_every_ps_carries_its_caveat(all_evidence):
    for pid, ev in all_evidence.items():
        assert ev.caveat, "{} lost its caveat".format(pid)
        assert ev.population, "{} lost its population definition".format(pid)


def test_evidence_is_json_serialisable(all_evidence):
    for pid, ev in all_evidence.items():
        json.dumps(ev.as_dict())


def test_evidence_carries_version_provenance(all_evidence):
    for pid, ev in all_evidence.items():
        for field in ("executor_version", "catalogue_version", "contract_version"):
            assert field in ev.provenance, "{} missing {}".format(pid, field)


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

def test_execution_is_reproducible(executor):
    """Same data, same parameters, same evidence. No randomness, no ordering drift."""
    for pid in ("PS-013", "PS-035", "PS-045", "PS-066", "PS-073", "PS-101"):
        # Bypass the fixture cache: two genuinely separate executions.
        first = executor.execute_uncached(pid, REQUIRED_PARAMETERS.get(pid)).as_dict()
        second = executor.execute_uncached(pid, REQUIRED_PARAMETERS.get(pid)).as_dict()
        for key in ("aggregates", "result_table", "comparison_baseline", "ranking"):
            assert first[key] == second[key], "{} is not reproducible on {}".format(pid, key)


def test_execution_does_not_mutate_the_dataset(executor, dataset):
    before = {k: len(dataset.table(k)) for k in dataset.table_keys}
    executor.execute("PS-013")
    executor.execute("PS-040")
    after = {k: len(dataset.table(k)) for k in dataset.table_keys}
    assert before == after


# --------------------------------------------------------------------------
# Correctness against independently derived figures
# --------------------------------------------------------------------------

def test_portfolio_count(all_evidence):
    assert all_evidence["PS-001"].aggregates["policy_count"] == 25000


def test_lapse_rate_matches_the_hand_calculation(all_evidence):
    ev = all_evidence["PS-035"]
    assert ev.aggregates["lapsed"] == 2737
    assert ev.aggregates["lapse_rate"] == 10.948


def test_total_ape(all_evidence):
    assert all_evidence["PS-003"].aggregates["total_ape"] == 2306202661.89


def test_status_distribution_covers_every_policy(all_evidence):
    ev = all_evidence["PS-002"]
    assert sum(r["policies"] for r in ev.result_table) == 25000
    assert round(sum(r["share_pct"] for r in ev.result_table), 4) == 100.0


def test_worst_product_on_lapse(all_evidence):
    """P91 lapses at 26.07%, the highest of any product with 50+ policies."""
    ev = all_evidence["PS-013"]
    top = ev.result_table[0]
    assert top["Prod_No"] == "P91"
    assert top["lapse_rate"] == pytest.approx(26.075, abs=0.01)
    assert top["rank"] == 1
    assert ev.comparison_baseline["value"] == 10.948
    assert top["delta_vs_baseline"] == pytest.approx(15.127, abs=0.01)


def test_low_sample_product_is_excluded_from_ranking(all_evidence):
    """P18 has 11 policies; ranking it would be noise."""
    ev = all_evidence["PS-013"]
    assert "P18" not in {r["Prod_No"] for r in ev.result_table}
    assert any("minimum sample" in lim for lim in ev.limitations)


def test_claims_totals(all_evidence):
    ev = all_evidence["PS-052"]
    assert ev.aggregates["claim_count"] == 134
    assert ev.aggregates["total_claim_amount"] == 93030740.33
    assert ev.aggregates["settled"] == 133


def test_persistency_buckets(all_evidence):
    ev = all_evidence["PS-045"]
    by_bucket = {r["Persistency_Bucket"]: r for r in ev.result_table}
    assert by_bucket["13th Month"]["renewals"] == 6943
    assert by_bucket["25th Month"]["renewals"] == 8189
    assert by_bucket["37th Month"]["renewals"] == 9410
    assert by_bucket["37th Month"]["renewal_paid_rate"] < by_bucket["25th Month"]["renewal_paid_rate"]


def test_renewal_population_uses_the_persistency_denominator(all_evidence):
    ev = all_evidence["PS-044"]
    assert sum(r["renewals"] for r in ev.result_table) == 24542


def test_top_policies_by_ape(all_evidence):
    ev = all_evidence["PS-023"]
    assert len(ev.result_table) == 10
    assert ev.result_table[0]["APE"] == 2500000.0
    assert [r["rank"] for r in ev.result_table] == list(range(1, 11))


def test_largest_claim(all_evidence):
    ev = all_evidence["PS-061"]
    assert ev.result_table[0]["Claim.Amount"] == 5000000.0


def test_claims_stay_the_population_when_joining_to_policy(all_evidence):
    """A reversed join must not expand 134 claims into 25,000 policies."""
    assert all_evidence["PS-061"].population_size == 134
    assert all_evidence["PS-060"].population_size == 134


def test_agent_ranking_respects_the_sample_threshold(all_evidence):
    ev = all_evidence["PS-039"]
    assert all(r["policies"] >= 30 for r in ev.result_table)
    assert ev.sample_sizes["minimum_group_size"] == 30


def test_reconciliation_reports_every_check(all_evidence):
    ev = all_evidence["PS-095"]
    assert ev.aggregates["checks_total"] == 6
    by_name = {r["check"]: r for r in ev.result_table}
    assert by_name["every claim resolves to a policy"]["passed"] is True
    assert by_name["death-status policies match the claims set exactly"]["passed"] is True
    assert by_name["every policy resolves to an agent"]["passed"] is True
    # The corrupted AppID (DQ-001) is real and must show as a genuine failure.
    assert by_name["every policy resolves to an owner"]["passed"] is False


def test_quality_report_lists_every_finding(all_evidence, contract):
    ev = all_evidence["PS-096"]
    assert ev.aggregates["finding_count"] == len(contract["data_quality_findings"])
    assert ev.aggregates["high_severity"] == 2


def test_scope_check_names_unanswerable_domains(all_evidence, contract):
    ev = all_evidence["PS-098"]
    assert ev.aggregates["unanswerable_domain_count"] == len(contract["unanswerable_domains"])
    assert ev.unsupported_conclusions


def test_executive_summary_keeps_section_populations_separate(all_evidence):
    ev = all_evidence["PS-101"]
    sections = {s["title"]: s for s in ev.result_table}
    assert sections["Portfolio"]["population_size"] == 25000
    assert sections["Renewal collection"]["population_size"] == 24542
    assert sections["Claims"]["population_size"] == 134
    assert any("different populations" in lim for lim in ev.limitations)


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------

def test_record_lookup_returns_the_requested_policy(executor):
    ev = executor.execute("PS-031", {"app_id": "15011000000"})
    assert ev.is_sufficient
    assert len(ev.retrieved_records) == 1
    assert ev.retrieved_records[0]["AppID"] == "15011000000"


def test_record_lookup_reports_absent_related_records(executor):
    """This policy is the DQ-001 orphan: it has no owner or persistency row."""
    ev = executor.execute("PS-031", {"app_id": "15011000000"})
    assert any("Absence is not a zero value" in lim for lim in ev.limitations)


def test_record_lookup_withholds_personal_data(executor):
    ev = executor.execute("PS-031", {"app_id": "11000000015"})
    record = ev.retrieved_records[0]
    assert not [k for k in record if "Given.Name" in k or "Postal.Code" in k]
    assert any(r["rule_id"] == "BR-006" for r in ev.business_rules_applied)


def test_unknown_identifier_fails_safely(executor):
    ev = executor.execute("PS-031", {"app_id": "00000000000"})
    assert not ev.is_sufficient
    assert "no record matches" in ev.insufficient.reason


def test_missing_required_parameter_fails_safely(executor):
    ev = executor.execute("PS-031")
    assert not ev.is_sufficient
    assert "requires parameter" in ev.insufficient.reason


def test_undeclared_parameter_is_rejected(executor):
    ev = executor.execute("PS-001", {"nonsense": 1})
    assert not ev.is_sufficient
    assert "does not accept" in ev.insufficient.reason


def test_parameter_bounds_are_enforced(executor):
    ev = executor.execute("PS-023", {"n": 500})
    assert not ev.is_sufficient
    assert "at most" in ev.insufficient.reason


def test_parameter_changes_the_result(executor):
    small = executor.execute("PS-023", {"n": 3})
    large = executor.execute("PS-023", {"n": 20})
    assert len(small.result_table) == 3
    assert len(large.result_table) == 20
    assert small.parameters["n"] == 3


def test_defaults_are_applied_when_no_parameter_is_given(executor):
    ev = executor.execute("PS-023")
    assert ev.parameters["n"] == 10
    assert len(ev.result_table) == 10


def test_enum_parameter_is_validated(executor):
    ev = executor.execute("PS-072", {"metric": "not_a_metric"})
    assert not ev.is_sufficient
    assert "must be one of" in ev.insufficient.reason


def test_unknown_problem_id_raises(executor):
    with pytest.raises(PSNotFound):
        executor.execute("PS-999")


# --------------------------------------------------------------------------
# Grounding and honesty
# --------------------------------------------------------------------------

def test_diagnostic_ps_always_disclaim_causation(all_evidence, executor):
    for pid, ev in all_evidence.items():
        if executor.registry.get(pid).analysis_tier != "diagnostic":
            continue
        joined = " ".join(ev.unsupported_conclusions).lower()
        assert "caus" in joined, "{} makes no causal disclaimer".format(pid)


def test_persistency_ps_record_the_restricted_population(all_evidence, executor):
    for pid, ev in all_evidence.items():
        ps = executor.registry.get(pid)
        if "persistency" not in ps.relevant_tables:
            continue
        text = (ev.population + ev.caveat).lower()
        assert any(k in text for k in ("persistency", "dq-006", "eligible", "active"))


def test_no_result_table_leaks_personal_data(all_evidence):
    for pid, ev in all_evidence.items():
        for row in ev.result_table + ev.retrieved_records:
            leaked = [k for k in row if "Given.Name" in k or "Postal.Code" in k]
            assert not leaked, "{} leaked {}".format(pid, leaked)


def test_outlier_analyses_refuse_to_call_a_flag_an_error(all_evidence):
    for pid in ("PS-025", "PS-075", "PS-097"):
        joined = " ".join(all_evidence[pid].unsupported_conclusions).lower()
        assert "not evidence" in joined or "not an error" in joined


def test_data_quality_notes_are_expanded_with_detail(all_evidence):
    ev = all_evidence["PS-036"]
    assert "DQ-001" in ev.data_quality_notes
    detail = ev.supporting_metrics.get("data_quality_detail", {})
    assert "DQ-001" in detail and detail["DQ-001"]["severity"] == "high"


def test_insufficient_evidence_is_never_an_empty_success(executor):
    ev = executor.execute("PS-031", {"app_id": "does-not-exist"})
    assert not ev.is_sufficient
    assert ev.as_dict()["sufficient"] is False
    assert ev.insufficient.reason
    assert not ev.aggregates


# --------------------------------------------------------------------------
# Calculations are traceable
# --------------------------------------------------------------------------

def test_every_calculation_records_its_inputs(all_evidence):
    for pid, ev in all_evidence.items():
        for calc in ev.calculations:
            assert "formula_id" in calc and "inputs" in calc, pid
            assert "semantics_applied" in calc, pid


def test_rate_calculations_carry_their_numerator_and_denominator(all_evidence):
    ev = all_evidence["PS-035"]
    rate = next(c for c in ev.calculations if c["formula_id"] == "F-RATE")
    assert rate["inputs"]["numerator"] == 2737
    assert rate["inputs"]["denominator"] == 25000
    assert rate["value"] == 10.948


def test_provenance_records_the_full_query_chain(all_evidence):
    ev = all_evidence["PS-036"]
    ops = [s["op"] for s in ev.provenance["steps"]]
    assert ops[0] == "load"
    assert "join" in ops and "group_by" in ops


def test_row_counts_are_continuous_through_the_chain(all_evidence):
    for pid, ev in all_evidence.items():
        steps = ev.provenance.get("steps", [])
        for previous, current in zip(steps, steps[1:]):
            if current["op"] in ("group_by", "aggregate"):
                continue
            assert current["rows_in"] == previous["rows_out"], (
                "{}: row discontinuity at {}".format(pid, current["op"]))
