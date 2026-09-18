"""Executable plans for every Problem Statement.

Kept separate from the PS definitions so the analytical intent and the mechanics
can each be read on their own. The builder merges them, and fails if any PS lacks
a plan or names a pattern that does not exist.

A plan names a pattern and supplies its inputs. Column references follow the query
engine: after a join, right-hand columns carry their table prefix, so Owner.Age
reached through policy_owner is `owner_Owner.Age`.
"""
from __future__ import annotations

# Status vocabulary, read from the data rather than assumed.
INFORCE = "Premium paying (regular)"
LAPSED = "Lapsed"
SURRENDERED = "Surrendered"
DEATH = "Death"

AGE_BINS = [0, 17, 29, 39, 49, 59, 69, 120]
AGE_LABELS = ["under_18_implausible", "18_29", "30_39", "40_49", "50_59", "60_69", "70_plus"]

INCOME_BINS = [-1, 0, 250000, 500000, 1000000, 2500000, 10_000_000_000]
INCOME_LABELS = ["zero_declared", "upto_2.5L", "2.5L_5L", "5L_10L", "10L_25L", "above_25L"]

TERM_BINS = [0, 10, 15, 20, 25, 100]
TERM_LABELS = ["upto_10y", "11_15y", "16_20y", "21_25y", "over_25y"]

APE_BINS = [0, 25000, 50000, 100000, 250000, 500000, 3_000_000]
APE_LABELS = ["upto_25k", "25k_50k", "50k_100k", "100k_250k", "250k_500k", "above_500k"]

DERIVE_AGE_BAND = {"kind": "band", "name": "age_band", "column": "owner_Owner.Age",
                   "bins": AGE_BINS, "labels": AGE_LABELS}
DERIVE_INCOME_BAND = {"kind": "band", "name": "income_band",
                      "column": "owner_Owner.Earned.Income",
                      "bins": INCOME_BINS, "labels": INCOME_LABELS}
DERIVE_UW_DAYS = {"kind": "date_diff", "name": "underwriting_days",
                  "end": "Inforce.Date", "start": "Login.Date"}
DERIVE_SA_MULTIPLE = {"kind": "ratio", "name": "sum_assured_multiple",
                      "numerator": "Sum.Assured", "denominator": "APE"}

LAPSE_BASELINE = {"kind": "rate", "measure": "lapse_rate",
                  "numerator_column": "POLICY.STATUS", "numerator_values": [LAPSED],
                  "description": "portfolio lapse rate across all policies"}

COUNT_AND_LAPSE = {"policies": (None, "count_all")}
LAPSE_CONDITION = {"lapsed": {"column": "POLICY.STATUS", "values": [LAPSED]}}


def _status_counts():
    """Measures that let a group's status mix be turned into rates."""
    return {"policies": (None, "count_all"), "total_ape": ("APE", "sum")}


EXECUTIONS = {

    # ---------------------------------------------------------------- A: portfolio
    "PS-001": dict(pattern="aggregate", base="policy",
                   measures={"policy_count": ("AppID", "distinct")}),

    "PS-002": dict(pattern="group_metrics", base="policy", group_by="POLICY.STATUS",
                   measures={"policies": (None, "count_all")},
                   shares={"share_pct": "policies"}, rank_by="policies"),

    "PS-003": dict(pattern="aggregate", base="policy",
                   measures={"total_ape": ("APE", "sum")}),

    "PS-004": dict(pattern="aggregate", base="policy",
                   measures={"total_sum_assured": ("Sum.Assured", "sum")}),

    "PS-005": dict(pattern="aggregate", base="policy",
                   measures={"avg_ape": ("APE", "mean"), "median_ape": ("APE", "median"),
                             "policy_count": (None, "count_all")}),

    "PS-006": dict(pattern="group_metrics", base="policy", group_by="Trad.ULIP",
                   measures={"policies": (None, "count_all"), "total_ape": ("APE", "sum"),
                             "avg_ape": ("APE", "mean")},
                   shares={"policy_share_pct": "policies", "ape_share_pct": "total_ape"},
                   rank_by="policies"),

    "PS-007": dict(pattern="group_metrics", base="policy", group_by=["Prod_No", "Plan.Name"],
                   measures={"policies": (None, "count_all")},
                   shares={"share_pct": "policies"}, rank_by="policies"),

    "PS-008": dict(pattern="group_metrics", base="policy", group_by=["Prod_No", "Plan.Name"],
                   measures={"policies": (None, "count_all"), "total_ape": ("APE", "sum"),
                             "avg_ape": ("APE", "mean")},
                   shares={"ape_share_pct": "total_ape"}, rank_by="total_ape"),

    "PS-009": dict(pattern="executive", sections=[
        {"key": "portfolio", "title": "Portfolio scale", "pattern": "aggregate",
         "population": "All 25,000 policies",
         "plan": dict(pattern="aggregate", base="policy",
                      measures={"policy_count": (None, "count_all"),
                                "total_ape": ("APE", "sum"),
                                "total_sum_assured": ("Sum.Assured", "sum")},
                      condition_counts={
                          "inforce": {"column": "POLICY.STATUS", "values": [INFORCE]},
                          "lapsed": {"column": "POLICY.STATUS", "values": [LAPSED]},
                          "surrendered": {"column": "POLICY.STATUS", "values": [SURRENDERED]}},
                      rates={"inforce_rate": {"numerator": "inforce",
                                              "denominator": "policy_count"},
                             "lapse_rate": {"numerator": "lapsed",
                                            "denominator": "policy_count"},
                             "surrender_rate": {"numerator": "surrendered",
                                                "denominator": "policy_count"}})},
        {"key": "renewal", "title": "Renewal collection", "pattern": "aggregate",
         "population": "The 24,542 policies present in Persistency_Details",
         "plan": dict(pattern="aggregate", base="persistency",
                      measures={"renewal_rows": (None, "count_all")},
                      condition_counts={"paid": {"column": "Renewal_Status",
                                                 "values": ["Paid"]}},
                      rates={"renewal_paid_rate": {"numerator": "paid",
                                                   "denominator": "renewal_rows"}})},
        {"key": "products", "title": "Product spread", "pattern": "group_metrics", "rows": 5,
         "population": "All policies grouped by product",
         "plan": dict(pattern="group_metrics", base="policy", group_by="Prod_No",
                      measures={"policies": (None, "count_all")},
                      shares={"share_pct": "policies"}, rank_by="policies")},
    ]),

    "PS-010": dict(pattern="aggregate", base="policy", joins=["policy_owner"],
                   measures={"agent_count": ("SALES_ID", "distinct"),
                             "branch_count": ("Branch_No", "distinct"),
                             "client_count": ("owner_ClientID", "distinct"),
                             "policy_count": (None, "count_all")}),

    "PS-011": dict(pattern="aggregate", base="policy",
                   measures={"policy_count": (None, "count_all")},
                   condition_counts={"inforce": {"column": "POLICY.STATUS",
                                                 "values": [INFORCE]}},
                   rates={"inforce_rate": {"numerator": "inforce",
                                           "denominator": "policy_count"}}),

    "PS-012": dict(pattern="distribution", base="policy", column="Policy.Term",
                   bins=TERM_BINS, labels=TERM_LABELS),

    # ---------------------------------------------------------------- B: product
    "PS-013": dict(pattern="group_metrics", base="policy", group_by=["Prod_No", "Plan.Name"],
                   measures={"policies": (None, "count_all"),
                             "lapsed": ("is_lapsed", "sum"),
                             "total_ape": ("APE", "sum")},
                   derive=[{"kind": "flag", "name": "is_lapsed",
                            "column": "POLICY.STATUS", "values": [LAPSED]}],
                   rates={"lapse_rate": {"numerator": "lapsed", "denominator": "policies"}},
                   min_sample=50, rank_by="lapse_rate", baseline=LAPSE_BASELINE),

    "PS-014": dict(pattern="driver_scan", base="policy",
                   joins=["policy_owner", "policy_sales"],
                   entity_column="Prod_No", kpi="lapse_rate", min_sample=50,
                   measures={"policies": (None, "count_all"),
                             "lapsed": ("is_lapsed", "sum")},
                   derive=[{"kind": "flag", "name": "is_lapsed",
                            "column": "POLICY.STATUS", "values": [LAPSED]}],
                   rates={"lapse_rate": {"numerator": "lapsed", "denominator": "policies"}},
                   numeric_attributes=["APE", "Sum.Assured", "Policy.Term",
                                       "PREMIUM.PAYING.TERM", "owner_Owner.Age",
                                       "owner_Owner.Earned.Income",
                                       "sales_First Year Persistency Rate"],
                   categorical_attributes=["Trad.ULIP", "owner_Owner.Occupation"]),

    "PS-015": dict(pattern="group_metrics", base="policy", group_by=["Prod_No", "Plan.Name"],
                   measures={"policies": (None, "count_all"),
                             "avg_sum_assured": ("Sum.Assured", "mean"),
                             "median_sum_assured": ("Sum.Assured", "median")},
                   rank_by="avg_sum_assured"),

    "PS-016": dict(pattern="group_metrics", base="policy", group_by="Trad.ULIP",
                   measures={"policies": (None, "count_all"), "total_ape": ("APE", "sum"),
                             "avg_ape": ("APE", "mean"),
                             "avg_sum_assured": ("Sum.Assured", "mean"),
                             "avg_term": ("Policy.Term", "mean"),
                             "lapsed": ("is_lapsed", "sum"), "inforce": ("is_inforce", "sum")},
                   derive=[{"kind": "flag", "name": "is_lapsed",
                            "column": "POLICY.STATUS", "values": [LAPSED]},
                           {"kind": "flag", "name": "is_inforce",
                            "column": "POLICY.STATUS", "values": [INFORCE]}],
                   rates={"lapse_rate": {"numerator": "lapsed", "denominator": "policies"},
                          "inforce_rate": {"numerator": "inforce", "denominator": "policies"}},
                   rank_by="policies", baseline=LAPSE_BASELINE),

    "PS-017": dict(pattern="group_metrics", base="policy", group_by=["Prod_No", "Plan.Name"],
                   measures={"policies": (None, "count_all"), "inforce": ("is_inforce", "sum"),
                             "lapsed": ("is_lapsed", "sum"),
                             "surrendered": ("is_surrendered", "sum")},
                   derive=[{"kind": "flag", "name": "is_inforce",
                            "column": "POLICY.STATUS", "values": [INFORCE]},
                           {"kind": "flag", "name": "is_lapsed",
                            "column": "POLICY.STATUS", "values": [LAPSED]},
                           {"kind": "flag", "name": "is_surrendered",
                            "column": "POLICY.STATUS", "values": [SURRENDERED]}],
                   rates={"inforce_rate": {"numerator": "inforce", "denominator": "policies"},
                          "lapse_rate": {"numerator": "lapsed", "denominator": "policies"},
                          "surrender_rate": {"numerator": "surrendered",
                                             "denominator": "policies"}},
                   min_sample=50, rank_by="inforce_rate"),

    "PS-018": dict(pattern="group_metrics", base="policy", group_by=["Prod_No", "Plan.Name"],
                   measures={"policies": (None, "count_all"),
                             "avg_premium_paying_term": ("PREMIUM.PAYING.TERM", "mean"),
                             "avg_policy_term": ("Policy.Term", "mean")},
                   rank_by="avg_premium_paying_term"),

    "PS-019": dict(pattern="concentration", base="policy", group_by="Prod_No",
                   measure=(None, "count_all"),
                   top_points={"top_1": 1, "top_3": 3, "top_5": 5}),

    "PS-020": dict(pattern="group_metrics", base="policy",
                   joins=["policy_claims", "policy_persistency"],
                   group_by=["Prod_No", "Plan.Name"],
                   measures={"policies": (None, "count_all"), "total_ape": ("APE", "sum"),
                             "avg_ape": ("APE", "mean"),
                             "avg_sum_assured": ("Sum.Assured", "mean"),
                             "inforce": ("is_inforce", "sum"), "lapsed": ("is_lapsed", "sum"),
                             "surrendered": ("is_surrendered", "sum"),
                             "claims": ("claims_Claim_No", "count"),
                             "renewal_rows": ("persistency_Renewal_Status", "count"),
                             "renewals_paid": ("is_renewal_paid", "sum")},
                   derive=[{"kind": "flag", "name": "is_inforce",
                            "column": "POLICY.STATUS", "values": [INFORCE]},
                           {"kind": "flag", "name": "is_lapsed",
                            "column": "POLICY.STATUS", "values": [LAPSED]},
                           {"kind": "flag", "name": "is_surrendered",
                            "column": "POLICY.STATUS", "values": [SURRENDERED]},
                           {"kind": "flag", "name": "is_renewal_paid",
                            "column": "persistency_Renewal_Status", "values": ["Paid"]}],
                   rates={"inforce_rate": {"numerator": "inforce", "denominator": "policies"},
                          "lapse_rate": {"numerator": "lapsed", "denominator": "policies"},
                          "surrender_rate": {"numerator": "surrendered",
                                             "denominator": "policies"},
                          "claim_incidence_rate": {"numerator": "claims",
                                                   "denominator": "policies"},
                          "renewal_paid_rate": {"numerator": "renewals_paid",
                                                "denominator": "renewal_rows"}},
                   shares={"portfolio_share_pct": "policies"},
                   min_sample=50, rank_by="inforce_rate"),

    "PS-021": dict(pattern="group_metrics", base="policy", group_by=["Prod_No", "Plan.Name"],
                   derive=[DERIVE_SA_MULTIPLE],
                   measures={"policies": (None, "count_all"),
                             "median_multiple": ("sum_assured_multiple", "median"),
                             "avg_sum_assured": ("Sum.Assured", "mean"),
                             "avg_ape": ("APE", "mean")},
                   rank_by="median_multiple"),

    # ---------------------------------------------------------------- C: financial
    "PS-022": dict(pattern="distribution", base="policy", column="APE",
                   bins=APE_BINS, labels=APE_LABELS),

    "PS-023": dict(pattern="top_n", base="policy", measure="APE", default_n=10,
                   columns=["AppID", "APE", "Sum.Assured", "Plan.Name",
                            "POLICY.STATUS", "SALES_ID"]),

    "PS-024": dict(pattern="concentration", base="policy", group_by="AppID",
                   measure=("APE", "sum"), table_rows=10,
                   top_points={"top_10": 10, "top_100": 100},
                   top_percentiles=[1, 5, 10, 20]),

    "PS-025": dict(pattern="outliers", base="policy", measure="sum_assured_multiple",
                   within="Prod_No", derive=[DERIVE_SA_MULTIPLE],
                   columns=["AppID", "Prod_No", "Sum.Assured", "APE"]),

    "PS-026": dict(pattern="group_metrics", base="policy", group_by="term_band",
                   derive=[{"kind": "band", "name": "term_band", "column": "Policy.Term",
                            "bins": TERM_BINS, "labels": TERM_LABELS}],
                   measures={"policies": (None, "count_all"), "avg_ape": ("APE", "mean"),
                             "median_ape": ("APE", "median"),
                             "avg_sum_assured": ("Sum.Assured", "mean")},
                   sort_by_group="term_band"),

    "PS-027": dict(pattern="aggregate", base="persistency",
                   measures={"total_renewal_premium": ("Renewal_Premium_Amount", "sum"),
                             "avg_renewal_premium": ("Renewal_Premium_Amount", "mean"),
                             "median_renewal_premium": ("Renewal_Premium_Amount", "median"),
                             "renewal_count": (None, "count_all")}),

    "PS-028": dict(pattern="group_metrics", base="policy", joins=["policy_persistency"],
                   join_how={"policy_persistency": "inner"},
                   group_by="renewal_vs_ape",
                   derive=[{"kind": "ratio", "name": "renewal_ratio",
                            "numerator": "persistency_Renewal_Premium_Amount",
                            "denominator": "APE"},
                           {"kind": "band", "name": "renewal_vs_ape", "column": "renewal_ratio",
                            "bins": [-1, 0.999, 1.001, 1000],
                            "labels": ["below_ape", "matches_ape", "above_ape"]}],
                   measures={"policies": (None, "count_all"), "avg_ape": ("APE", "mean"),
                             "avg_renewal": ("persistency_Renewal_Premium_Amount", "mean")},
                   shares={"share_pct": "policies"}, rank_by="policies"),

    # ---------------------------------------------------------------- D: lifecycle
    "PS-029": dict(pattern="distribution", base="policy", column="underwriting_days",
                   derive=[DERIVE_UW_DAYS],
                   rules=[("BR-005", {"column": "underwriting_days"})],
                   bins=[-1, 0, 7, 14, 30, 60, 400],
                   labels=["same_day", "1_7d", "8_14d", "15_30d", "31_60d", "over_60d"]),

    "PS-030": dict(pattern="group_metrics", base="policy", group_by=["Prod_No", "Plan.Name"],
                   derive=[DERIVE_UW_DAYS],
                   rules=[("BR-005", {"column": "underwriting_days"})],
                   measures={"policies": (None, "count_all"),
                             "median_days": ("underwriting_days", "median"),
                             "avg_days": ("underwriting_days", "mean")},
                   min_sample=50, rank_by="median_days",
                   baseline={"kind": "median", "measure": "median_days",
                             "column": "underwriting_days",
                             "description": "portfolio median underwriting turnaround"}),

    "PS-031": dict(pattern="record_lookup", base="policy", key_column="AppID",
                   key_parameter="app_id",
                   joins=["policy_owner", "policy_persistency", "policy_claims",
                          "policy_sales"],
                   optional_tables=["claims", "persistency"]),

    "PS-032": dict(pattern="timeseries", base="policy", date_column="Inforce.Date",
                   grain="month", trend_measure="policies",
                   measures={"policies": (None, "count_all"), "total_ape": ("APE", "sum")}),

    "PS-033": dict(pattern="timeseries", base="policy", date_column="Inforce.Date",
                   grain="month", trend_measure="total_ape",
                   measures={"total_ape": ("APE", "sum"), "policies": (None, "count_all"),
                             "avg_ape": ("APE", "mean")}),

    "PS-034": dict(pattern="group_metrics", base="policy", group_by="Branch_No",
                   measures={"policies": (None, "count_all"), "total_ape": ("APE", "sum"),
                             "avg_ape": ("APE", "mean"),
                             "total_sum_assured": ("Sum.Assured", "sum")},
                   shares={"share_pct": "policies"},
                   rank_by="policies", top_n=25),

    # ---------------------------------------------------------------- E: lapse
    "PS-035": dict(pattern="aggregate", base="policy",
                   measures={"policy_count": (None, "count_all")},
                   condition_counts={"lapsed": {"column": "POLICY.STATUS",
                                                "values": [LAPSED]}},
                   rates={"lapse_rate": {"numerator": "lapsed",
                                         "denominator": "policy_count"}}),

    "PS-036": dict(pattern="group_metrics", base="policy", joins=["policy_owner"],
                   join_how={"policy_owner": "inner"},
                   group_by="age_band",
                   derive=[DERIVE_AGE_BAND,
                           {"kind": "flag", "name": "is_lapsed",
                            "column": "POLICY.STATUS", "values": [LAPSED]}],
                   measures={"policies": (None, "count_all"), "lapsed": ("is_lapsed", "sum"),
                             "avg_age": ("owner_Owner.Age", "mean")},
                   rates={"lapse_rate": {"numerator": "lapsed", "denominator": "policies"}},
                   min_sample=100, rank_by="lapse_rate", baseline=LAPSE_BASELINE),

    "PS-037": dict(pattern="group_metrics", base="policy", joins=["policy_owner"],
                   join_how={"policy_owner": "inner"},
                   group_by="owner_Owner.Occupation",
                   derive=[{"kind": "flag", "name": "is_lapsed",
                            "column": "POLICY.STATUS", "values": [LAPSED]}],
                   measures={"policies": (None, "count_all"), "lapsed": ("is_lapsed", "sum")},
                   rates={"lapse_rate": {"numerator": "lapsed", "denominator": "policies"}},
                   shares={"share_pct": "policies"},
                   rank_by="lapse_rate", baseline=LAPSE_BASELINE),

    "PS-038": dict(pattern="group_metrics", base="policy", joins=["policy_owner"],
                   join_how={"policy_owner": "inner"},
                   group_by="income_band",
                   derive=[DERIVE_INCOME_BAND,
                           {"kind": "flag", "name": "is_lapsed",
                            "column": "POLICY.STATUS", "values": [LAPSED]}],
                   measures={"policies": (None, "count_all"), "lapsed": ("is_lapsed", "sum"),
                             "avg_ape": ("APE", "mean"),
                             "avg_income": ("owner_Owner.Earned.Income", "mean")},
                   rates={"lapse_rate": {"numerator": "lapsed", "denominator": "policies"}},
                   min_sample=100, rank_by="lapse_rate", baseline=LAPSE_BASELINE),

    "PS-039": dict(pattern="group_metrics", base="policy", joins=["policy_sales"],
                   group_by="SALES_ID",
                   derive=[{"kind": "flag", "name": "is_lapsed",
                            "column": "POLICY.STATUS", "values": [LAPSED]}],
                   measures={"policies": (None, "count_all"), "lapsed": ("is_lapsed", "sum"),
                             "total_ape": ("APE", "sum"),
                             "supplied_persistency": ("sales_First Year Persistency Rate",
                                                      "mean")},
                   rates={"lapse_rate": {"numerator": "lapsed", "denominator": "policies"}},
                   min_sample="$min_policies", rank_by="lapse_rate",
                   baseline=LAPSE_BASELINE, top_n="$n"),

    "PS-040": dict(pattern="cohort_compare", base="policy",
                   joins=["policy_owner", "policy_sales"],
                   split={"column": "POLICY.STATUS",
                          "cohorts": {"lapsed": [LAPSED], "inforce": [INFORCE]}},
                   numeric_attributes=["APE", "Sum.Assured", "Policy.Term",
                                       "PREMIUM.PAYING.TERM", "owner_Owner.Age",
                                       "owner_Owner.Earned.Income",
                                       "sales_First Year Persistency Rate",
                                       "sales_Sales Related Complaint",
                                       "sales_RCU Rejections Rate"],
                   categorical_attributes=["Trad.ULIP", "owner_Owner.Occupation",
                                           "owner_Owner.Education", "owner_Owner.Gender"]),

    "PS-041": dict(pattern="group_metrics", base="policy", group_by="cohort_year",
                   derive=[{"kind": "date_bucket", "name": "cohort_year",
                            "column": "Inforce.Date", "grain": "year"},
                           {"kind": "flag", "name": "is_lapsed",
                            "column": "POLICY.STATUS", "values": [LAPSED]},
                           {"kind": "flag", "name": "is_inforce",
                            "column": "POLICY.STATUS", "values": [INFORCE]}],
                   measures={"policies": (None, "count_all"), "lapsed": ("is_lapsed", "sum"),
                             "inforce": ("is_inforce", "sum")},
                   rates={"lapse_rate": {"numerator": "lapsed", "denominator": "policies"},
                          "inforce_rate": {"numerator": "inforce", "denominator": "policies"}},
                   sort_by_group="cohort_year", baseline=LAPSE_BASELINE),

    # ---------------------------------------------------------------- F: surrender
    "PS-042": dict(pattern="group_metrics", base="policy", group_by=["Prod_No", "Plan.Name"],
                   derive=[{"kind": "flag", "name": "is_surrendered",
                            "column": "POLICY.STATUS", "values": [SURRENDERED]}],
                   measures={"policies": (None, "count_all"),
                             "surrendered": ("is_surrendered", "sum"),
                             "total_ape": ("APE", "sum")},
                   rates={"surrender_rate": {"numerator": "surrendered",
                                             "denominator": "policies"}},
                   min_sample=50, rank_by="surrender_rate",
                   baseline={"kind": "rate", "measure": "surrender_rate",
                             "numerator_column": "POLICY.STATUS",
                             "numerator_values": [SURRENDERED],
                             "description": "portfolio surrender rate"}),

    "PS-043": dict(pattern="cohort_compare", base="policy", joins=["policy_owner"],
                   split={"column": "POLICY.STATUS",
                          "cohorts": {"surrendered": [SURRENDERED], "inforce": [INFORCE]}},
                   numeric_attributes=["APE", "Sum.Assured", "Policy.Term",
                                       "PREMIUM.PAYING.TERM", "owner_Owner.Age",
                                       "owner_Owner.Earned.Income"],
                   categorical_attributes=["Trad.ULIP", "owner_Owner.Occupation",
                                           "owner_Owner.Education"]),

    # ---------------------------------------------------------------- G: persistency
    "PS-044": dict(pattern="group_metrics", base="persistency", group_by="Renewal_Status",
                   measures={"renewals": (None, "count_all"),
                             "premium": ("Renewal_Premium_Amount", "sum")},
                   shares={"share_pct": "renewals"}, rank_by="renewals"),

    "PS-045": dict(pattern="group_metrics", base="persistency", group_by="Persistency_Bucket",
                   derive=[{"kind": "flag", "name": "is_paid",
                            "column": "Renewal_Status", "values": ["Paid"]}],
                   measures={"renewals": (None, "count_all"), "paid": ("is_paid", "sum"),
                             "premium_at_stake": ("Renewal_Premium_Amount", "sum")},
                   rates={"renewal_paid_rate": {"numerator": "paid",
                                                "denominator": "renewals"}},
                   sort_by_group="Persistency_Bucket",
                   baseline={"kind": "rate", "measure": "renewal_paid_rate",
                             "numerator_column": "Renewal_Status",
                             "numerator_values": ["Paid"],
                             "description": "overall renewal paid rate"}),

    "PS-046": dict(pattern="distribution", base="persistency", column="renewal_delay_days",
                   derive=[{"kind": "date_diff", "name": "renewal_delay_days",
                            "end": "Renewal_Paid_Date", "start": "Renewal_Due_Date"}],
                   filters=[("Renewal_Status", "eq", "Paid")],
                   bins=[-1, 0, 5, 10, 15, 400],
                   labels=["on_or_before_due", "1_5d", "6_10d", "11_15d", "over_15d"]),

    "PS-047": dict(pattern="group_metrics", base="persistency", group_by="Payment_Mode",
                   derive=[{"kind": "flag", "name": "is_paid",
                            "column": "Renewal_Status", "values": ["Paid"]},
                           {"kind": "date_diff", "name": "renewal_delay_days",
                            "end": "Renewal_Paid_Date", "start": "Renewal_Due_Date"}],
                   measures={"renewals": (None, "count_all"), "paid": ("is_paid", "sum"),
                             "median_delay_days": ("renewal_delay_days", "median")},
                   rates={"renewal_paid_rate": {"numerator": "paid",
                                                "denominator": "renewals"}},
                   rank_by="renewal_paid_rate",
                   baseline={"kind": "rate", "measure": "renewal_paid_rate",
                             "numerator_column": "Renewal_Status",
                             "numerator_values": ["Paid"],
                             "description": "overall renewal paid rate"}),

    "PS-048": dict(pattern="group_metrics", base="persistency",
                   filters=[("Renewal_Status", "eq", "Pending")],
                   group_by="Persistency_Bucket",
                   measures={"pending_renewals": (None, "count_all"),
                             "pending_premium": ("Renewal_Premium_Amount", "sum"),
                             "avg_premium": ("Renewal_Premium_Amount", "mean")},
                   shares={"share_of_pending_pct": "pending_premium"},
                   rank_by="pending_premium"),

    "PS-049": dict(pattern="group_metrics", base="persistency", joins=["persistency_sales"],
                   group_by="SALES_ID",
                   derive=[{"kind": "flag", "name": "is_paid",
                            "column": "Renewal_Status", "values": ["Paid"]}],
                   measures={"renewals": (None, "count_all"), "paid": ("is_paid", "sum"),
                             "premium": ("Renewal_Premium_Amount", "sum"),
                             "supplied_persistency": ("sales_First Year Persistency Rate",
                                                      "mean")},
                   rates={"renewal_paid_rate": {"numerator": "paid",
                                                "denominator": "renewals"}},
                   min_sample="$min_policies", rank_by="renewal_paid_rate", top_n="$n",
                   baseline={"kind": "rate", "measure": "renewal_paid_rate",
                             "numerator_column": "Renewal_Status",
                             "numerator_values": ["Paid"],
                             "description": "overall renewal paid rate"}),

    "PS-050": dict(pattern="cohort_compare", base="persistency",
                   joins=["policy_persistency"], base_override=None,
                   split={"column": "Persistency_Bucket",
                          "cohorts": {"month_37": ["37th Month"],
                                      "earlier_buckets": ["13th Month", "25th Month"]}},
                   numeric_attributes=["Renewal_Premium_Amount", "PREMIUM.PAYING.TERM"],
                   categorical_attributes=["Renewal_Status", "Payment_Mode",
                                           "POLICY.STATUS"]),

    "PS-051": dict(pattern="group_metrics", base="policy", joins=["policy_persistency"],
                   join_how={"policy_persistency": "inner"},
                   group_by=["Prod_No", "Plan.Name"],
                   derive=[{"kind": "flag", "name": "is_paid",
                            "column": "persistency_Renewal_Status", "values": ["Paid"]}],
                   measures={"renewals": (None, "count_all"), "paid": ("is_paid", "sum"),
                             "premium": ("persistency_Renewal_Premium_Amount", "sum")},
                   rates={"renewal_paid_rate": {"numerator": "paid",
                                                "denominator": "renewals"}},
                   min_sample=50, rank_by="renewal_paid_rate"),
}
