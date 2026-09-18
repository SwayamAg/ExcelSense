"""Phase 2 - query engine tests.

Every operation is checked for correctness and for provenance: a step that drops
rows must record how many, which is what makes "no silent data loss" verifiable.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data import query
from src.data.query_engine import QueryError


@pytest.fixture()
def policy(dataset):
    return query(dataset, "policy")


# --------------------------------------------------------------------------
# Construction and provenance basics
# --------------------------------------------------------------------------

def test_query_starts_with_every_row(policy):
    assert len(policy) == 25000
    assert policy.provenance.steps[0].op == "load"
    assert policy.provenance.tables_used == ["policy"]


def test_query_does_not_mutate_the_dataset(dataset):
    before = len(dataset.table("policy"))
    query(dataset, "policy").filter("POLICY.STATUS", "eq", "Lapsed")
    assert len(dataset.table("policy")) == before


def test_unknown_table_is_rejected(dataset):
    from src.data.loaders import DataLoadError
    with pytest.raises(DataLoadError, match="unknown table"):
        query(dataset, "nope")


def test_every_step_records_row_counts(policy):
    result = (policy
              .filter("POLICY.STATUS", "eq", "Lapsed")
              .sort("APE", ascending=False)
              .limit(10))
    assert len(result) == 10
    ops = [s.op for s in result.provenance.steps]
    assert ops == ["load", "filter", "sort", "limit"]
    for step in result.provenance.steps:
        assert step.rows_in >= step.rows_out >= 0


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------

def test_filter_equality(policy):
    result = policy.filter("POLICY.STATUS", "eq", "Lapsed")
    assert len(result) == 2737
    assert set(result.frame["POLICY.STATUS"]) == {"Lapsed"}


def test_filter_by_normalized_alias(policy):
    assert len(policy.filter("policy_status", "eq", "Lapsed")) == 2737


@pytest.mark.parametrize("comparator,value,expected_all", [
    ("ne", "Lapsed", lambda s: (s != "Lapsed").all()),
    ("in", ["Lapsed", "Surrendered"], lambda s: s.isin(["Lapsed", "Surrendered"]).all()),
    ("not_in", ["Lapsed"], lambda s: (~s.isin(["Lapsed"])).all()),
    ("contains", "paying", lambda s: s.str.contains("paying", case=False).all()),
])
def test_filter_comparators_on_text(policy, comparator, value, expected_all):
    result = policy.filter("POLICY.STATUS", comparator, value)
    assert len(result) > 0
    assert expected_all(result.frame["POLICY.STATUS"])


def test_filter_numeric_comparators(policy):
    assert (policy.filter("APE", "gt", 100000).frame["APE"] > 100000).all()
    assert (policy.filter("APE", "lte", 50000).frame["APE"] <= 50000).all()
    between = policy.filter("APE", "between", (50000, 100000))
    assert between.frame["APE"].between(50000, 100000).all()


def test_filter_on_dates(policy):
    result = policy.filter("Inforce.Date", "gte", pd.Timestamp("2025-01-01"))
    assert (result.frame["Inforce.Date"] >= pd.Timestamp("2025-01-01")).all()
    assert 0 < len(result) < 25000


def test_unknown_comparator_is_rejected(policy):
    with pytest.raises(QueryError, match="unknown comparator"):
        policy.filter("APE", "approximately", 1)


def test_unknown_column_is_rejected(policy):
    with pytest.raises(QueryError, match="no column"):
        policy.filter("not_a_column", "eq", 1)


def test_filter_drops_nulls_by_default(dataset):
    """Null handling must be a decision, not an accident."""
    sales = query(dataset, "sales")
    kept = sales.filter("First Year Persistency Rate", "gte", 0)
    assert len(kept) == 1956 - 369
    assert kept.frame["First Year Persistency Rate"].notna().all()


def test_filter_can_keep_nulls_explicitly(dataset):
    sales = query(dataset, "sales")
    kept = sales.filter("First Year Persistency Rate", "gte", 0, keep_null=True)
    assert len(kept) == 1956


def test_filter_records_the_null_population(dataset):
    sales = query(dataset, "sales").filter("First Year Persistency Rate", "gte", 0)
    step = next(s for s in sales.provenance.steps if s.op == "filter")
    assert step.detail["null_rows_in_column"] == 369
    assert step.rows_dropped == 369


def test_is_null_and_not_null(dataset):
    sales = query(dataset, "sales")
    assert len(sales.filter("First Year Persistency Rate", "is_null")) == 369
    assert len(sales.filter("First Year Persistency Rate", "not_null")) == 1587


def test_filter_rows_with_a_custom_mask(policy):
    mask = policy.frame["APE"] > policy.frame["APE"].median()
    result = policy.filter_rows(mask, "APE above the portfolio median")
    assert len(result) > 0
    assert result.provenance.filters[-1]["description"].startswith("APE above")


# --------------------------------------------------------------------------
# Sorting, limiting, distinct
# --------------------------------------------------------------------------

def test_sort_descending(policy):
    result = policy.sort("APE", ascending=False)
    assert result.frame["APE"].iloc[0] == 2500000.0
    assert result.frame["APE"].is_monotonic_decreasing


def test_sort_is_stable_so_ties_keep_source_order(policy):
    from src.data.contract import ROW_ID
    result = policy.sort("POLICY.STATUS")
    lapsed = result.frame[result.frame["POLICY.STATUS"] == "Lapsed"][ROW_ID]
    assert lapsed.is_monotonic_increasing, "ties were reordered"


def test_sort_puts_nulls_last_by_default(dataset):
    result = query(dataset, "sales").sort("First Year Persistency Rate")
    tail = result.frame["First Year Persistency Rate"].tail(369)
    assert tail.isna().all()


def test_limit(policy):
    assert len(policy.limit(5)) == 5
    assert len(policy.limit(0)) == 0


def test_negative_limit_is_rejected(policy):
    with pytest.raises(QueryError, match="must not be negative"):
        policy.limit(-1)


def test_distinct(policy):
    assert len(policy.select(["POLICY.STATUS"]).distinct(["POLICY.STATUS"])) == 4


# --------------------------------------------------------------------------
# Derived columns and dates
# --------------------------------------------------------------------------

def test_derive_records_its_source(policy):
    result = policy.derive("double_ape", policy.frame["APE"] * 2, "APE * 2")
    assert (result.frame["double_ape"] == result.frame["APE"] * 2).all()
    step = next(s for s in result.provenance.steps if s.op == "derive")
    assert step.detail["source"] == "APE * 2"


def test_derive_refuses_to_overwrite(policy):
    with pytest.raises(QueryError, match="already exists"):
        policy.derive("APE", policy.frame["APE"], "clobber")


def test_date_diff_computes_whole_days(policy):
    result = policy.date_diff("uw_days", "Inforce.Date", "Login.Date")
    manual = (policy.frame["Inforce.Date"] - policy.frame["Login.Date"]).dt.days
    assert result.frame["uw_days"].equals(manual)


def test_date_diff_keeps_negative_durations_and_reports_them(policy):
    """DQ-011: one policy has Login.Date after Inforce.Date. It must not be hidden."""
    result = policy.date_diff("uw_days", "Inforce.Date", "Login.Date")
    negatives = int((result.frame["uw_days"] < 0).sum())
    assert negatives == 1
    assert any("negative duration" in n for n in result.provenance.notes)


def test_date_bucket_month_sorts_correctly(policy):
    """DQ-010: the source Inforce.month column sorts wrongly; the derived key must not."""
    result = policy.date_bucket("period", "Inforce.Date", grain="month")
    periods = sorted(result.frame["period"].dropna().unique())
    assert periods[0] == "2023-01"
    assert periods[-1] == "2025-12"
    assert periods == sorted(periods), "period keys do not sort chronologically"


def test_date_bucket_quarter_and_year(policy):
    quarters = policy.date_bucket("q", "Inforce.Date", grain="quarter")
    assert quarters.frame["q"].iloc[0].startswith("202")
    assert "-Q" in quarters.frame["q"].iloc[0]
    years = policy.date_bucket("y", "Inforce.Date", grain="year")
    assert set(years.frame["y"].dropna()) == {"2023", "2024", "2025"}


def test_unknown_grain_is_rejected(policy):
    with pytest.raises(QueryError, match="unknown grain"):
        policy.date_bucket("x", "Inforce.Date", grain="fortnight")


def test_case_when_applies_first_match(policy):
    frame = policy.frame
    result = policy.case_when("band", [
        (frame["APE"] >= 500000, "high"),
        (frame["APE"] >= 100000, "medium"),
    ], default="low")
    assert set(result.frame["band"]) == {"high", "medium", "low"}
    assert (result.frame.loc[result.frame["APE"] >= 500000, "band"] == "high").all()
    # A row above 500000 also satisfies the second condition; first match must win.
    assert (result.frame.loc[result.frame["band"] == "medium", "APE"] < 500000).all()


# --------------------------------------------------------------------------
# Aggregation and grouping
# --------------------------------------------------------------------------

def test_aggregate_whole_result(policy):
    result = policy.aggregate({
        "policies": (None, "count_all"),
        "total_ape": ("APE", "sum"),
        "mean_ape": ("APE", "mean"),
        "median_ape": ("APE", "median"),
    })
    assert result["policies"] == 25000
    assert round(result["total_ape"], 2) == 2306202661.89
    assert round(result["mean_ape"], 4) == 92248.1065
    assert result["median_ape"] == 50000.0


def test_unknown_aggregate_is_rejected(policy):
    with pytest.raises(QueryError, match="unknown aggregate"):
        policy.aggregate({"x": ("APE", "geometric_mean")})


def test_group_by_covers_every_row(policy):
    grouped = policy.group_by("POLICY.STATUS", {"policies": (None, "count_all")})
    assert len(grouped) == 4
    assert grouped.frame["policies"].sum() == 25000
    step = next(s for s in grouped.provenance.steps if s.op == "group_by")
    assert step.detail["rows_covered"] == 25000
    assert "rows_lost" not in step.detail


def test_group_by_known_values(policy):
    grouped = policy.group_by("POLICY.STATUS", {"n": (None, "count_all")})
    counts = dict(zip(grouped.frame["POLICY.STATUS"], grouped.frame["n"]))
    assert counts == {"Premium paying (regular)": 21805, "Lapsed": 2737,
                      "Surrendered": 324, "Death": 134}


def test_group_by_multiple_keys(policy):
    grouped = policy.group_by(["Trad.ULIP", "POLICY.STATUS"], {"n": (None, "count_all")})
    assert grouped.frame["n"].sum() == 25000
    assert set(grouped.frame["Trad.ULIP"]) == {"TRAD", "ULIP"}


def test_group_by_keeps_null_groups_so_rows_are_not_lost(dataset):
    """dropna=False is the default precisely so a null group cannot vanish."""
    grouped = query(dataset, "sales").group_by(
        "First Year Persistency Rate", {"n": (None, "count_all")})
    assert grouped.frame["n"].sum() == 1956


def test_group_by_can_drop_nulls_explicitly_and_reports_the_loss(dataset):
    grouped = query(dataset, "sales").group_by(
        "First Year Persistency Rate", {"n": (None, "count_all")}, dropna=True)
    assert grouped.frame["n"].sum() == 1956 - 369
    step = next(s for s in grouped.provenance.steps if s.op == "group_by")
    assert step.detail["rows_lost"] == 369


def test_group_by_aggregate_values_are_correct(policy):
    grouped = policy.group_by("Trad.ULIP", {
        "n": (None, "count_all"),
        "total_ape": ("APE", "sum"),
        "mean_sa": ("Sum.Assured", "mean"),
    })
    frame = grouped.frame.set_index("Trad.ULIP")
    assert int(frame.loc["ULIP", "n"]) == 4003
    assert int(frame.loc["TRAD", "n"]) == 20997
    assert round(float(frame["total_ape"].sum()), 2) == 2306202661.89


def test_aggregation_ignores_nulls_in_the_denominator(dataset):
    """mean must divide by present values, not by row count."""
    grouped = query(dataset, "sales").group_by("Type", {
        "rows": (None, "count_all"),
        "rated": ("First Year Persistency Rate", "count"),
        "mean_rate": ("First Year Persistency Rate", "mean"),
    })
    frame = grouped.frame.set_index("Type")
    assert int(frame["rows"].sum()) == 1956
    assert int(frame["rated"].sum()) == 1587


# --------------------------------------------------------------------------
# Grouped result operations
# --------------------------------------------------------------------------

def test_having_filters_groups(policy):
    grouped = policy.group_by("Prod_No", {"n": (None, "count_all")}).having("n", "gte", 2000)
    assert (grouped.frame["n"] >= 2000).all()
    assert len(grouped) < 10


def test_rank_assigns_positions(policy):
    ranked = policy.group_by("Prod_No", {"n": (None, "count_all")}).rank("n")
    assert ranked.frame["rank"].tolist() == list(range(1, len(ranked) + 1))
    assert ranked.frame["n"].is_monotonic_decreasing
    assert ranked.frame.iloc[0]["Prod_No"] == "P65"


def test_share_of_total_sums_to_one_hundred(policy):
    shared = policy.group_by("POLICY.STATUS", {"n": (None, "count_all")}).share_of_total("n")
    assert round(float(shared.frame["n_share_pct"].sum()), 6) == 100.0


def test_small_sample_guard_excludes_and_reports(policy):
    """P18 has 11 policies; a rate from it would be noise."""
    grouped = policy.group_by("Prod_No", {"n": (None, "count_all")})
    guarded = grouped.min_group_size(50)
    assert "P18" not in set(guarded.frame["Prod_No"])
    step = next(s for s in guarded.provenance.steps if s.op == "small_sample_guard")
    assert step.detail["groups_excluded"] == 1
    assert step.detail["rows_excluded"] == 11
    assert any("threshold" in n for n in guarded.provenance.notes)


def test_top_n(policy):
    top = policy.group_by("Prod_No", {"n": (None, "count_all")}).rank("n").top(3)
    assert len(top) == 3


# --------------------------------------------------------------------------
# Joins
# --------------------------------------------------------------------------

def test_inner_join_policy_owner(policy):
    joined = policy.join("policy_owner")
    assert len(joined) == 24999
    step = next(s for s in joined.provenance.steps if s.op == "join")
    assert step.detail["left_keys_unmatched"] == 1
    assert "inner join dropped 1 unmatched left keys" in step.detail["note"]


def test_left_join_keeps_every_left_row(policy, dataset):
    joined = policy.join("policy_owner", how="left")
    assert len(joined) == 25000
    # ClientID is never null in the source, so its only null here is the unmatched row.
    assert joined.frame["owner_ClientID"].isna().sum() == 1
    # A nullable column shows source nulls plus the unmatched row, never fewer.
    source_nulls = int(dataset.table("owner").col("Owner.Age").isna().sum())
    assert joined.frame["owner_Owner.Age"].isna().sum() == source_nulls + 1


def test_left_join_to_claims_keeps_the_whole_denominator(policy):
    """Claim incidence needs unclaimed policies in the denominator."""
    joined = policy.join("policy_claims", how="left")
    assert len(joined) == 25000
    assert joined.frame["claims_Claim_No"].notna().sum() == 134


def test_anti_join_finds_unmatched_rows(policy):
    orphans = policy.join("policy_persistency", how="anti")
    # 324 surrendered + 134 death + 1 corrupted AppID have no persistency row.
    assert len(orphans) == 459
    assert set(orphans.frame["POLICY.STATUS"]) == {"Surrendered", "Death",
                                                   "Premium paying (regular)"}


def test_join_records_coverage_and_fan_out(policy):
    joined = policy.join("policy_sales")
    step = next(s for s in joined.provenance.steps if s.op == "join")
    assert step.detail["matched_keys"] == 1956
    assert step.detail["left_keys_unmatched"] == 0
    assert step.detail["max_right_rows_per_key"] == 1
    assert "fan_out_warning" not in step.detail


def test_join_does_not_multiply_rows(policy):
    """Every documented join is 1:1 or N:1 here; a fan-out would corrupt every total."""
    for relationship in ("policy_owner", "policy_claims", "policy_persistency", "policy_sales"):
        joined = policy.join(relationship, how="left")
        assert len(joined) == 25000, "{} multiplied rows".format(relationship)


def test_join_preserves_aggregates(policy):
    """A left join must not change the left side's totals."""
    before = policy.aggregate({"total": ("APE", "sum")})["total"]
    after = policy.join("policy_owner", how="left").aggregate({"total": ("APE", "sum")})["total"]
    assert round(before, 2) == round(after, 2)


def test_join_records_the_path(policy):
    joined = policy.join("policy_owner")
    assert joined.provenance.join_path == ["policy.AppID = owner.AppID (inner)"]
    assert "owner" in joined.provenance.tables_used


def test_join_can_select_right_columns(policy):
    joined = policy.join("policy_owner", columns=["Owner.Age", "Owner.Gender"])
    assert "owner_Owner.Age" in joined.frame.columns
    assert "owner_Owner.Education" not in joined.frame.columns


def test_unknown_relationship_is_rejected(policy):
    with pytest.raises(QueryError, match="unknown relationship"):
        policy.join("policy_to_mars")


def test_unknown_join_type_is_rejected(policy):
    with pytest.raises(QueryError, match="inner, left or anti"):
        policy.join("policy_owner", how="outer")


def test_multi_table_chain(policy):
    """Policy -> owner -> claims, the shape a diagnostic PS needs."""
    result = (policy
              .join("policy_owner", how="left", columns=["Owner.Age", "Owner.Comm.State"])
              .join("policy_claims", how="left", columns=["Claim.Amount"]))
    assert len(result) == 25000
    assert result.frame["claims_Claim.Amount"].notna().sum() == 134
    assert len(result.provenance.join_path) == 2
    assert set(result.provenance.tables_used) == {"policy", "owner", "claims"}


def test_persistency_to_sales_join(dataset):
    result = query(dataset, "persistency").join("persistency_sales")
    assert len(result) == 24542
    assert "sales_Type" in result.frame.columns


# --------------------------------------------------------------------------
# Output and traceability
# --------------------------------------------------------------------------

def test_records_carry_their_source_row(policy):
    from src.data.contract import ROW_ID
    records = policy.filter("POLICY.STATUS", "eq", "Death").to_records(limit=3)
    assert len(records) == 3
    for record in records:
        assert ROW_ID in record and isinstance(record[ROW_ID], int)


def test_records_are_json_safe(policy):
    import json
    records = policy.sort("APE", ascending=False).to_records(
        columns=["AppID", "APE", "Inforce.Date"], limit=5)
    json.dumps(records)  # must not raise
    assert isinstance(records[0]["Inforce.Date"], str)
    assert records[0]["APE"] == 2500000.0


def test_grouped_records_are_json_safe(policy):
    import json
    records = policy.group_by("POLICY.STATUS", {"n": (None, "count_all")}).to_records()
    json.dumps(records)
    assert len(records) == 4


def test_provenance_serialises_completely(policy):
    import json
    result = (policy
              .filter("POLICY.STATUS", "eq", "Lapsed")
              .join("policy_owner", how="left")
              .date_bucket("period", "Inforce.Date"))
    payload = result.provenance.as_dict()
    json.dumps(payload)
    for field in ("tables_used", "columns_used", "filters", "join_path", "steps", "notes"):
        assert field in payload
    assert payload["steps"][0]["op"] == "load"


def test_full_chain_row_accounting(policy):
    """Row counts must reconcile across the whole chain, with nothing unexplained."""
    result = (policy
              .filter("POLICY.STATUS", "in", ["Lapsed", "Premium paying (regular)"])
              .join("policy_owner", how="left")
              .filter("owner_Owner.Age", "gte", 18))
    steps = result.provenance.steps
    for previous, current in zip(steps, steps[1:]):
        assert current.rows_in == previous.rows_out, (
            "row count discontinuity at step {}".format(current.op))
