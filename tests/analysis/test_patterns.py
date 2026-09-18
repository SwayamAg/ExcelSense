"""Phase 4 - analytical pattern tests.

The specification asks for coverage of every PS category. Each category is checked
here against a PS that exercises it, plus the edge cases that matter for that shape
of analysis.
"""
from __future__ import annotations

import pytest

import src.analysis.advanced_patterns  # noqa: F401  (registers the extra patterns)
from src.analysis.patterns import PATTERNS
from src.ps.executor import PSExecutor

# One PS per category the specification names.
CATEGORY_EXAMPLES = {
    "direct lookup": ("PS-031", {"app_id": "11000000015"}),
    "aggregation": ("PS-003", None),
    "ratio": ("PS-035", None),
    "ranking": ("PS-013", None),
    "comparison": ("PS-016", None),
    "time series": ("PS-032", None),
    "multi-table": ("PS-020", None),
    "diagnostic": ("PS-040", None),
    "anomaly": ("PS-025", None),
    "concentration": ("PS-019", None),
    "distribution": ("PS-022", None),
    "reconciliation": ("PS-095", None),
    "composite scoring": ("PS-073", None),
    "entity tiering": ("PS-083", None),
    "rule breach": ("PS-097", None),
    "scope refusal": ("PS-098", None),
    "executive analysis": ("PS-101", None),
}


@pytest.fixture(scope="module")
def executor(ps_executor):
    return ps_executor


# --------------------------------------------------------------------------
# Category coverage
# --------------------------------------------------------------------------

@pytest.mark.parametrize("category,spec", sorted(CATEGORY_EXAMPLES.items()))
def test_every_category_executes(executor, category, spec):
    problem_id, parameters = spec
    evidence = executor.execute(problem_id, parameters)
    assert evidence.is_sufficient, "{} ({}) failed: {}".format(
        category, problem_id, evidence.insufficient.reason if evidence.insufficient else "")
    assert evidence.result_table or evidence.aggregates or evidence.retrieved_records


def test_every_registered_pattern_is_used_by_a_ps(executor):
    used = {ps.execution["pattern"] for ps in executor.registry.executable()}
    unused = set(PATTERNS) - used
    assert not unused, "patterns exist but no PS uses them: {}".format(sorted(unused))


def test_every_ps_pattern_is_registered(executor):
    for ps in executor.registry.executable():
        assert ps.execution["pattern"] in PATTERNS, ps.id


# --------------------------------------------------------------------------
# aggregate
# --------------------------------------------------------------------------

def test_aggregate_produces_scalars_not_tables(executor):
    evidence = executor.execute("PS-003")
    assert evidence.aggregates
    assert not evidence.result_table
    assert evidence.population_size == 25000


def test_aggregate_condition_counts_and_rates_agree(executor):
    evidence = executor.execute("PS-011")
    inforce = evidence.aggregates["inforce"]
    total = evidence.aggregates["policy_count"]
    assert evidence.aggregates["inforce_rate"] == pytest.approx(100.0 * inforce / total, abs=1e-4)


# --------------------------------------------------------------------------
# group_metrics
# --------------------------------------------------------------------------

def test_group_metrics_covers_every_row(executor):
    evidence = executor.execute("PS-002")
    assert sum(r["policies"] for r in evidence.result_table) == evidence.population_size


def test_group_metrics_ranks_descending(executor):
    evidence = executor.execute("PS-013")
    rates = [r["lapse_rate"] for r in evidence.result_table if r["lapse_rate"] is not None]
    assert rates == sorted(rates, reverse=True)
    assert evidence.ranking["measure"] == "lapse_rate"


def test_group_metrics_baseline_is_the_whole_population(executor, dataset):
    evidence = executor.execute("PS-013")
    policy = dataset.table("policy")
    expected = 100.0 * (policy.col("POLICY.STATUS") == "Lapsed").sum() / len(policy)
    assert evidence.comparison_baseline["value"] == pytest.approx(expected, abs=1e-4)
    assert evidence.comparison_baseline["population_size"] == 25000


def test_group_metrics_delta_matches_rate_minus_baseline(executor):
    evidence = executor.execute("PS-013")
    baseline = evidence.comparison_baseline["value"]
    for row in evidence.result_table:
        if row["lapse_rate"] is None:
            continue
        assert row["delta_vs_baseline"] == pytest.approx(row["lapse_rate"] - baseline, abs=1e-3)


def test_group_metrics_shares_sum_to_one_hundred(executor):
    evidence = executor.execute("PS-007")
    assert round(sum(r["share_pct"] for r in evidence.result_table), 4) == 100.0


def test_small_sample_guard_excludes_and_explains(executor):
    evidence = executor.execute("PS-013")
    assert evidence.sample_sizes["groups_excluded"] >= 1
    assert any("minimum sample" in lim for lim in evidence.limitations)


def test_sort_by_group_orders_chronologically(executor):
    evidence = executor.execute("PS-041")
    years = [r["cohort_year"] for r in evidence.result_table]
    assert years == sorted(years)


def test_tail_n_returns_the_latest_periods(executor):
    evidence = executor.execute("PS-093")
    assert len(evidence.result_table) == 2
    assert evidence.result_table[0]["period"] < evidence.result_table[1]["period"]


# --------------------------------------------------------------------------
# distribution
# --------------------------------------------------------------------------

def test_distribution_reports_percentiles_and_bands(executor):
    evidence = executor.execute("PS-022")
    for key in ("p25", "median", "p75", "p90", "p99", "mean", "minimum", "maximum"):
        assert key in evidence.aggregates
    assert round(sum(r["share_pct"] for r in evidence.result_table), 2) == 100.0


def test_distribution_bands_cover_every_value(executor):
    evidence = executor.execute("PS-022")
    banded = sum(r["policies"] for r in evidence.result_table)
    assert banded == evidence.aggregates["values_present"]


def test_distribution_median_is_below_the_mean_for_skewed_ape(executor):
    evidence = executor.execute("PS-022")
    assert evidence.aggregates["median"] < evidence.aggregates["mean"]


# --------------------------------------------------------------------------
# timeseries
# --------------------------------------------------------------------------

def test_timeseries_is_chronological(executor):
    evidence = executor.execute("PS-032")
    periods = [r["period"] for r in evidence.result_table]
    assert periods == sorted(periods)
    assert periods[0] == "2023-01"


def test_timeseries_covers_every_row(executor):
    evidence = executor.execute("PS-032")
    assert sum(r["policies"] for r in evidence.result_table) == 25000


def test_timeseries_flags_a_possibly_incomplete_final_period(executor):
    evidence = executor.execute("PS-032")
    assert any("final period may be incomplete" in lim for lim in evidence.limitations)


def test_timeseries_reports_the_change_across_the_series(executor):
    evidence = executor.execute("PS-033")
    assert "first_period" in evidence.supporting_metrics
    assert "last_period" in evidence.supporting_metrics
    assert "pct_change" in evidence.supporting_metrics


# --------------------------------------------------------------------------
# record_lookup
# --------------------------------------------------------------------------

def test_record_lookup_returns_exactly_one_record(executor):
    evidence = executor.execute("PS-031", {"app_id": "11000000015"})
    assert len(evidence.retrieved_records) == 1
    assert evidence.population_size == 1


def test_record_lookup_spans_every_related_table(executor):
    evidence = executor.execute("PS-031", {"app_id": "11000000015"})
    record = evidence.retrieved_records[0]
    assert any(k.startswith("owner_") for k in record)
    assert any(k.startswith("sales_") for k in record)


def test_record_lookup_keeps_the_source_row_id(executor):
    from src.data.contract import ROW_ID
    evidence = executor.execute("PS-031", {"app_id": "11000000015"})
    assert ROW_ID in evidence.retrieved_records[0]


# --------------------------------------------------------------------------
# top_n
# --------------------------------------------------------------------------

def test_top_n_is_ordered_and_ranked(executor):
    evidence = executor.execute("PS-023", {"n": 5})
    values = [r["APE"] for r in evidence.result_table]
    assert values == sorted(values, reverse=True)
    assert [r["rank"] for r in evidence.result_table] == [1, 2, 3, 4, 5]


def test_top_n_reports_its_share_of_the_total(executor):
    evidence = executor.execute("PS-023", {"n": 10})
    share = evidence.supporting_metrics["share_of_total_pct"]
    assert 0 < share < 100


# --------------------------------------------------------------------------
# concentration
# --------------------------------------------------------------------------

def test_concentration_cumulative_share_reaches_one_hundred(executor):
    evidence = executor.execute("PS-019")
    last = evidence.result_table[-1]
    assert last["cumulative_share_pct"] == pytest.approx(100.0, abs=0.01)


def test_concentration_is_monotonic(executor):
    evidence = executor.execute("PS-019")
    cumulative = [r["cumulative_share_pct"] for r in evidence.result_table]
    assert cumulative == sorted(cumulative)


def test_concentration_reports_an_index(executor):
    evidence = executor.execute("PS-019")
    hhi = evidence.aggregates["hhi"]
    assert 0 < hhi <= 10000


def test_concentration_top_points_are_reported(executor):
    evidence = executor.execute("PS-070")
    assert any(k.startswith("top_") for k in evidence.aggregates)


# --------------------------------------------------------------------------
# outliers
# --------------------------------------------------------------------------

def test_outliers_are_flagged_within_their_group(executor):
    evidence = executor.execute("PS-025")
    assert evidence.aggregates["outliers_flagged"] > 0
    for row in evidence.result_table:
        assert row["value"] < row["lower_fence"] or row["value"] > row["upper_fence"]


def test_outliers_record_the_fence_they_were_judged_against(executor):
    evidence = executor.execute("PS-025")
    assert any(c["formula_id"] == "F-IQR" for c in evidence.calculations)


def test_outliers_refuse_to_claim_an_error(executor):
    evidence = executor.execute("PS-025")
    joined = " ".join(evidence.unsupported_conclusions).lower()
    assert "not evidence of an error" in joined


def test_outlier_multiplier_is_configurable(executor):
    narrow = executor.execute("PS-025", {"iqr_multiplier": 1.5})
    wide = executor.execute("PS-025", {"iqr_multiplier": 3.0})
    assert wide.aggregates["outliers_flagged"] <= narrow.aggregates["outliers_flagged"]


# --------------------------------------------------------------------------
# cohort_compare
# --------------------------------------------------------------------------

def test_cohort_compare_reports_both_cohort_sizes(executor):
    evidence = executor.execute("PS-040")
    assert evidence.sample_sizes["lapsed"] == 2737
    assert evidence.sample_sizes["inforce"] == 21805


def test_cohort_compare_ranks_by_the_size_of_the_gap(executor):
    evidence = executor.execute("PS-040")
    gaps = [abs(r.get("pct_delta_mean") or r.get("delta_mean") or 0)
            for r in evidence.result_table]
    assert gaps == sorted(gaps, reverse=True)


def test_cohort_compare_disclaims_causation(executor):
    evidence = executor.execute("PS-043")
    joined = " ".join(evidence.unsupported_conclusions).lower()
    assert "association" in joined and "caus" in joined


# --------------------------------------------------------------------------
# driver_scan
# --------------------------------------------------------------------------

def test_driver_scan_identifies_a_leader_and_a_peer_benchmark(executor):
    evidence = executor.execute("PS-068")
    assert evidence.ranking["leader"]
    assert evidence.comparison_baseline["description"].startswith("peer average")
    assert evidence.supporting_metrics["peer_rows"] > 0


def test_driver_scan_ranks_attributes_by_gap(executor):
    evidence = executor.execute("PS-068")
    gaps = [abs(r.get("pct_delta") or 0) for r in evidence.result_table]
    assert gaps == sorted(gaps, reverse=True)


def test_driver_scan_respects_the_sample_threshold(executor):
    evidence = executor.execute("PS-068")
    assert evidence.sample_sizes["minimum_entity_size"] == 30
    assert evidence.sample_sizes["entities_eligible"] > 0


def test_driver_scan_states_what_it_cannot_establish(executor):
    evidence = executor.execute("PS-068")
    joined = " ".join(evidence.unsupported_conclusions).lower()
    assert "association" in joined


# --------------------------------------------------------------------------
# composite_score
# --------------------------------------------------------------------------

def test_composite_scores_are_ranked_and_bounded(executor):
    evidence = executor.execute("PS-073")
    scores = [r["composite_score"] for r in evidence.result_table]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= s <= 100 for s in scores)


def test_composite_returns_its_weights_so_it_can_be_challenged(executor):
    evidence = executor.execute("PS-073")
    weights = evidence.supporting_metrics["weights"]
    assert weights
    assert any(r["rule_id"] == "BR-012" for r in evidence.business_rules_applied)
    assert any("not a company standard" in lim for lim in evidence.limitations)


def test_composite_components_are_percentile_normalised(executor):
    evidence = executor.execute("PS-073")
    row = evidence.result_table[0]
    percentiles = [v for k, v in row.items() if k.endswith("_percentile") and v is not None]
    assert percentiles
    assert all(0 <= p <= 100 for p in percentiles)


# --------------------------------------------------------------------------
# two_stage_group
# --------------------------------------------------------------------------

def test_two_stage_group_tiers_entities_not_rows(executor, dataset):
    """Clients, not policies, are the unit of analysis.

    24,756 rather than the 24,757 clients in Owner_Details: the inner join drops the
    one policy whose owner row carries the corrupted AppID (DQ-001), and that owner
    is the only holder of their ClientID.
    """
    evidence = executor.execute("PS-083")
    assert evidence.population_size == 24999               # joined policies
    assert evidence.aggregates["entities"] == 24756        # distinct clients reachable
    assert sum(r["entities"] for r in evidence.result_table) == 24756
    assert dataset.table("owner").ident("ClientID").nunique() == 24757


def test_two_stage_group_shares_sum_to_one_hundred(executor):
    evidence = executor.execute("PS-083")
    assert round(sum(r["entity_share_pct"] for r in evidence.result_table), 2) == 100.0
    assert round(sum(r["client_ape_share_pct"] for r in evidence.result_table), 2) == 100.0


def test_two_stage_group_reports_its_tier_boundaries(executor):
    evidence = executor.execute("PS-083")
    boundaries = evidence.supporting_metrics["tier_boundaries"]
    assert set(boundaries) == {"p50", "p80", "p95"}
    assert boundaries["p50"] <= boundaries["p80"] <= boundaries["p95"]


# --------------------------------------------------------------------------
# rule_breach
# --------------------------------------------------------------------------

def test_rule_breach_reports_each_rule_separately(executor):
    evidence = executor.execute("PS-097")
    by_rule = evidence.aggregates["breaches_by_rule"]
    assert by_rule["negative underwriting duration"] == 1     # DQ-011
    assert by_rule["owner age below 18"] > 0                  # DQ-009
    assert evidence.aggregates["rows_flagged"] > 0


def test_rule_breach_ranks_by_number_of_breaches(executor):
    evidence = executor.execute("PS-097")
    counts = [r["breaches"] for r in evidence.result_table]
    assert counts == sorted(counts, reverse=True)


def test_rule_breach_names_the_rules_a_row_broke(executor):
    evidence = executor.execute("PS-097")
    assert all(r["rules"] for r in evidence.result_table)
    assert evidence.supporting_metrics["rule_set"]


# --------------------------------------------------------------------------
# reconcile, scope_check, quality_report
# --------------------------------------------------------------------------

def test_reconcile_reports_pass_and_fail_with_counts(executor):
    evidence = executor.execute("PS-095")
    assert evidence.aggregates["checks_total"] == 6
    assert evidence.aggregates["checks_passed"] + evidence.aggregates["checks_failed"] == 6
    for row in evidence.result_table:
        assert "detail" in row and isinstance(row["passed"], bool)


def test_reconcile_surfaces_the_known_orphan(executor):
    """DQ-001 is a real defect and must be reported as a failing check."""
    evidence = executor.execute("PS-095")
    owner_check = next(r for r in evidence.result_table
                       if r["check"] == "every policy resolves to an owner")
    assert owner_check["passed"] is False
    assert "15011000000" in owner_check["examples"]


def test_scope_check_refuses_rather_than_analysing(executor):
    evidence = executor.execute("PS-098")
    assert evidence.result_table
    assert evidence.unsupported_conclusions
    assert not evidence.calculations or all(
        c["formula_id"] == "F-COUNT" for c in evidence.calculations)


def test_quality_report_counts_by_severity(executor, contract):
    evidence = executor.execute("PS-096")
    findings = contract["data_quality_findings"]
    for severity in ("high", "medium", "low"):
        expected = sum(1 for f in findings if f["severity"] == severity)
        assert evidence.aggregates["{}_severity".format(severity)] == expected


# --------------------------------------------------------------------------
# executive
# --------------------------------------------------------------------------

def test_executive_composes_several_sections(executor):
    evidence = executor.execute("PS-101")
    assert evidence.supporting_metrics["section_count"] == 5
    assert all(s["sufficient"] for s in evidence.result_table)


def test_executive_prefixes_section_aggregates(executor):
    evidence = executor.execute("PS-101")
    assert "portfolio__total_ape" in evidence.aggregates
    assert "claims__claims" in evidence.aggregates


def test_executive_warns_that_populations_differ(executor):
    evidence = executor.execute("PS-099")
    assert any("different populations" in lim or "not additive" in lim
               for lim in evidence.limitations)


def test_executive_value_at_risk_is_not_called_a_loss(executor):
    evidence = executor.execute("PS-099")
    text = " ".join(evidence.limitations + [evidence.caveat]).lower()
    assert "premium" in text and ("not profit" in text or "not a loss" in text
                                  or "premium-at-risk" in text or "not losses" in text)
