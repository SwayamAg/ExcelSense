"""Phase 1 - PS catalogue framework.

Defines the PS record shape, the controlled vocabularies, and a builder that
validates every table, column, relationship, operation and metric against
config/data_contract.json at construction time. Referencing a column the CSVs
do not have is a hard build failure, not a lint warning.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "config" / "data_contract.json"

GROUNDED_IN = "Talk_to_Data_PoC_Documentation_Metadata_V4"

# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

TIERS = {"descriptive", "analytical", "diagnostic", "multi_step_business"}

ANSWER_TYPES = {
    "scalar", "record", "table", "ranked_table", "comparison",
    "timeseries", "distribution", "diagnostic_narrative", "executive_summary",
}

# Intent grammar: <structure>/<measure>, reused from the reference taxonomy
# because it maps one-to-one onto deterministic execution primitives.
INTENT_STRUCTURES = {
    "Lookup", "Filtering", "Aggregation", "GroupBy", "Join", "Multi-Join",
    "Time-Series", "Temporal", "Ranking", "Distribution", "Conditional",
    "Derived-Metric", "Multi-Condition", "Anomaly", "Concentration",
    "Cohort", "Reconciliation", "Business", "Anti-Join", "Having",
}
INTENT_MEASURES = {
    "Count", "Sum", "Average", "Median", "Min", "Max", "Ratio", "Rate",
    "Comparison", "Trend", "TopN", "BottomN", "Distinct", "Bucketing",
    "Ranking", "Distribution", "Outliers", "Profile", "Diagnostic",
    "Driver", "Integrity", "Overview", "Concentration", "Seasonality", "Record",
    "Anomaly",
}

OPERATIONS = {
    # relational primitives
    "SELECT", "WHERE", "GROUP BY", "ORDER BY", "LIMIT", "JOIN", "LEFT JOIN",
    "ANTI JOIN", "DISTINCT", "HAVING", "CASE WHEN", "WINDOW",
    # aggregates
    "COUNT", "SUM", "AVG", "MEDIAN", "MIN", "MAX", "STDDEV", "PERCENTILE",
    # analytical macro-operations
    "derive_column", "date_diff", "date_bucket", "ratio", "share_of_total",
    "compare_to_baseline", "rank_entities", "multi_metric_table",
    "segment_profile", "crosstab_metric", "concentration", "outliers",
    "driver_scan", "portfolio_overview", "entity_shortlist", "cohort_split",
    "reconcile", "small_sample_guard",
}

METRICS = {
    # volume
    "policy_count", "owner_count", "client_count", "agent_count", "claim_count",
    "branch_count", "product_count", "distinct_count", "renewal_count",
    # money
    "total_ape", "avg_ape", "median_ape", "total_sum_assured", "avg_sum_assured",
    "median_sum_assured", "total_claim_amount", "avg_claim_amount",
    "total_amount_paid", "avg_renewal_premium", "total_renewal_premium",
    "avg_earned_income", "median_earned_income", "sum_assured_to_ape_ratio",
    "payout_ratio",
    # rates and shares
    "lapse_rate", "surrender_rate", "death_rate", "inforce_rate",
    "renewal_paid_rate", "renewal_pending_rate", "claim_incidence_rate",
    "share_of_portfolio", "ulip_share", "settlement_rate",
    # agent quality (Sales_Details rate columns)
    "agent_surrender_rate", "agent_early_claim_rate", "agent_pivc_mismatch_rate",
    "agent_pivc_concern_rate", "agent_rcu_rejection_rate",
    "agent_freelook_cancellation_rate", "agent_complaint_rate",
    "agent_first_year_persistency", "agent_quality_composite",
    # durations
    "underwriting_turnaround_days", "renewal_delay_days", "intimation_lag_days",
    "settlement_lag_days", "days_to_death_from_inforce", "policy_tenure_days",
    # distribution / concentration
    "percentile", "gini_like_concentration", "top_n_share", "herfindahl_index",
    "std_deviation", "z_score", "iqr_bounds",
    # demographics
    "avg_owner_age", "median_owner_age", "gender_share", "education_share",
    "occupation_share",
    # comparison
    "delta_vs_baseline", "pct_delta_vs_baseline", "rank_position", "peer_average",
}

# Formula identifiers a PS may reference. Phase 3 implements each of these
# deterministically; declaring one here without an implementation is caught by the
# Phase 3 gate, and referencing an id absent from this set fails the Phase 1 build.
FORMULA_IDS = {
    "F-COUNT": "count of rows matching the population definition",
    "F-COUNT-DISTINCT": "count of distinct values of a key column",
    "F-SUM": "sum of a numeric column over the population",
    "F-AVG": "arithmetic mean, null values excluded from both numerator and denominator",
    "F-MEDIAN": "50th percentile, reported wherever a distribution is skewed",
    "F-RATE": "100 * matching rows / population rows; undefined when the denominator is zero",
    "F-RATIO": "numerator / denominator with an explicit zero-denominator guard",
    "F-SHARE": "100 * group value / total value across all groups",
    "F-CUMSHARE": "running share of an ordered series, for concentration curves",
    "F-DELTA": "value minus baseline, in the value's own units",
    "F-PCT-DELTA": "100 * (value - baseline) / baseline; undefined when baseline is zero",
    "F-PERCENTILE": "nth percentile using a declared interpolation method",
    "F-DATEDIFF": "whole days between two parsed dates; negative results are reported, not clipped",
    "F-DATEBUCKET": "period key derived from a parsed date at month, quarter or year grain",
    "F-IQR": "interquartile fence at Q1 - k*IQR and Q3 + k*IQR",
    "F-ZSCORE": "(value - mean) / standard deviation; undefined when the deviation is zero",
    "F-HHI": "sum of squared percentage shares, a Herfindahl-style concentration index",
    "F-COMPOSITE": "weighted sum of percentile-normalised components, weights declared in config",
}

VIZ_TYPES = {
    None, "bar", "horizontal_bar", "grouped_bar", "stacked_bar", "line",
    "multi_line", "pie", "donut", "histogram", "box", "scatter", "heatmap",
    "waterfall", "table_only", "kpi_tiles",
}

RELATIONSHIPS = {
    "policy_owner": ("policy", "AppID", "owner", "AppID"),
    "policy_claims": ("policy", "AppID", "claims", "AppID"),
    "policy_sales": ("policy", "SALES_ID", "sales", "SALES_ID"),
    "policy_persistency": ("policy", "AppID", "persistency", "AppID"),
    "persistency_sales": ("persistency", "SALES_ID", "sales", "SALES_ID"),
}

EVIDENCE_KEYS = {
    "tables_used", "columns_used", "filters", "join_path", "aggregates",
    "records", "calculations", "formulas_applied", "business_rules_applied",
    "comparison_baseline", "ranking", "supporting_metrics", "data_quality_notes",
    "limitations", "population_definition", "sample_sizes",
}

PS_ID_RE = re.compile(r"^PS-\d{3}$")


class PSValidationError(Exception):
    """Raised when a PS references something the datasets cannot support."""


# ---------------------------------------------------------------------------
# Schema knowledge, loaded from the Phase 0 contract
# ---------------------------------------------------------------------------

class Schema:
    def __init__(self, contract: dict):
        self.contract = contract
        self.tables = {k: t["source_columns"] for k, t in contract["tables"].items()}
        self.columns = {
            "{}.{}".format(k, c) for k, cols in self.tables.items() for c in cols
        }
        # Columns the engine derives rather than reads; each must declare its source.
        self.derived = {
            "policy.inforce_year": "derived from policy.Inforce.Date",
            "policy.inforce_month": "derived from policy.Inforce.Date",
            "policy.inforce_quarter": "derived from policy.Inforce.Date",
            "policy.login_year": "derived from policy.Login.Date",
            "policy.underwriting_days": "policy.Inforce.Date - policy.Login.Date",
            "policy.sum_assured_multiple": "policy.Sum.Assured / policy.APE",
            "persistency.renewal_delay_days": "persistency.Renewal_Paid_Date - persistency.Renewal_Due_Date",
            "claims.intimation_lag_days": "claims.Date.of.Intimation - claims.Date.of.Death",
            "claims.settlement_lag_days": "claims.Settlement.Date - claims.Date.of.Intimation",
            "claims.days_from_inforce_to_death": "claims.Date.of.Death - policy.Inforce.Date",
            "claims.payout_ratio": "claims.Amount.Paid / claims.Claim.Amount",
            "owner.age_band": "banded from owner.Owner.Age",
            "owner.income_band": "banded from owner.Owner.Earned.Income",
        }
        self.single_valued = {
            "policy.CHANNEL": "Banca",
            "policy.Mode": "Annually",
        }

    def check_table(self, table: str, ps_id: str) -> None:
        if table not in self.tables:
            raise PSValidationError(
                "{}: unknown table '{}' (known: {})".format(
                    ps_id, table, sorted(self.tables)))

    def check_column(self, ref: str, ps_id: str) -> None:
        if ref in self.columns or ref in self.derived:
            return
        if "." not in ref:
            raise PSValidationError(
                "{}: column '{}' must be qualified as table.Column".format(ps_id, ref))
        table = ref.split(".", 1)[0]
        self.check_table(table, ps_id)
        raise PSValidationError(
            "{}: column '{}' does not exist in the CSVs and is not a declared "
            "derived column".format(ps_id, ref))


def load_schema() -> Schema:
    if not CONTRACT_PATH.exists():
        raise SystemExit("config/data_contract.json missing - run scripts/profile_data.py first")
    return Schema(json.loads(CONTRACT_PATH.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# PS construction
# ---------------------------------------------------------------------------

REQUIRED_TEXT_FIELDS = (
    "problem_statement", "business_intent", "business_objective",
    "answer_template", "population", "caveat",
)


def build_ps(schema: Schema, **kw) -> dict:
    """Construct and fully validate a single PS record."""
    ps_id = kw.get("id")
    if not ps_id or not PS_ID_RE.match(ps_id):
        raise PSValidationError("bad or missing problem_id: {!r}".format(ps_id))

    tables = list(kw["tables"])
    columns = list(kw["columns"])
    ops = list(kw["ops"])
    metrics = list(kw["metrics"])
    rels = list(kw.get("rels", []))
    queries = list(kw["queries"])
    steps = list(kw["steps"])

    for t in tables:
        schema.check_table(t, ps_id)
    for c in columns:
        schema.check_column(c, ps_id)

    # Every table a column touches must be declared, and vice versa.
    used_tables = {c.split(".", 1)[0] for c in columns}
    missing = used_tables - set(tables)
    if missing:
        raise PSValidationError(
            "{}: columns reference undeclared tables {}".format(ps_id, sorted(missing)))
    unused = set(tables) - used_tables
    if unused:
        raise PSValidationError(
            "{}: declares tables {} but uses no column from them".format(ps_id, sorted(unused)))

    # Multi-table PSs must declare a join path, unless they explicitly read each table
    # independently (a scope or schema question), which must be justified in writing.
    independent = kw.get("independent_tables")
    if len(tables) > 1 and not rels:
        if not independent:
            raise PSValidationError(
                "{}: spans {} tables but declares no relationship; set independent_tables "
                "with a justification if the tables are read separately".format(
                    ps_id, len(tables)))
        if not str(independent).strip():
            raise PSValidationError(
                "{}: independent_tables must carry a justification string".format(ps_id))
    for r in rels:
        if r not in RELATIONSHIPS:
            raise PSValidationError("{}: unknown relationship '{}'".format(ps_id, r))
        lt, _, rt, _ = RELATIONSHIPS[r]
        for side in (lt, rt):
            if side not in tables:
                raise PSValidationError(
                    "{}: relationship '{}' needs table '{}' which is not declared".format(
                        ps_id, r, side))

    # Intent grammar: <Structure>[/<Structure|Measure>...]/<Measure>. Structures may be
    # chained (Join/GroupBy/Ratio) but the intent must always terminate in a measure,
    # because that is what determines the shape of the answer.
    intent = kw["intent"]
    parts = intent.split("/")
    if len(parts) < 2:
        raise PSValidationError(
            "{}: intent_type '{}' must be <Structure>/<Measure>".format(ps_id, intent))
    if parts[0] not in INTENT_STRUCTURES:
        raise PSValidationError("{}: unknown intent structure '{}'".format(ps_id, parts[0]))
    if parts[-1] not in INTENT_MEASURES:
        raise PSValidationError(
            "{}: intent_type '{}' must end in a measure, got '{}'".format(
                ps_id, intent, parts[-1]))
    for token in parts[1:-1]:
        if token not in INTENT_STRUCTURES and token not in INTENT_MEASURES:
            raise PSValidationError(
                "{}: unknown intent token '{}'".format(ps_id, token))

    bad_ops = [o for o in ops if o not in OPERATIONS]
    if bad_ops:
        raise PSValidationError("{}: unknown operations {}".format(ps_id, bad_ops))
    bad_metrics = [m for m in metrics if m not in METRICS]
    if bad_metrics:
        raise PSValidationError("{}: unknown metrics {}".format(ps_id, bad_metrics))

    formulas = list(kw.get("formulas", []))
    bad_formulas = [f for f in formulas if f not in FORMULA_IDS]
    if bad_formulas:
        raise PSValidationError(
            "{}: undefined formula ids {}".format(ps_id, bad_formulas))

    tier = kw["tier"]
    if tier not in TIERS:
        raise PSValidationError("{}: unknown analysis_tier '{}'".format(ps_id, tier))

    answer = kw["answer"]
    if answer not in ANSWER_TYPES:
        raise PSValidationError("{}: unknown expected_answer_type '{}'".format(ps_id, answer))

    viz = kw.get("viz")
    if viz not in VIZ_TYPES:
        raise PSValidationError("{}: unknown visualization_type '{}'".format(ps_id, viz))

    if len(queries) < 3:
        raise PSValidationError(
            "{}: needs at least 3 example queries, has {}".format(ps_id, len(queries)))
    if len(steps) < 2:
        raise PSValidationError(
            "{}: needs at least 2 analytical steps, has {}".format(ps_id, len(steps)))

    evidence = dict(kw.get("evidence", {}))
    evidence.setdefault("tables_used", tables)
    evidence.setdefault("columns_used", columns)
    evidence.setdefault("population_definition", kw["population"])
    bad_ev = set(evidence) - EVIDENCE_KEYS
    if bad_ev:
        raise PSValidationError("{}: unknown evidence keys {}".format(ps_id, sorted(bad_ev)))

    record = {
        "problem_id": ps_id,
        "problem_statement": kw["q"],
        "intent_type": intent,
        "business_intent": kw["business_intent"],
        "business_domain": kw["domain"],
        "business_objective": kw["objective"],
        "analysis_tier": tier,
        "example_queries": queries,
        "relevant_tables": tables,
        "relevant_columns": columns,
        "required_operations": ops,
        "required_metrics": metrics,
        "relationships": rels,
        "independent_tables": kw.get("independent_tables"),
        "mentions_unsupported_concepts": kw.get("mentions_unsupported_concepts"),
        "execution": kw.get("execution"),
        "concepts": list(kw.get("concepts", [])),
        "analytical_steps": steps,
        "expected_answer_type": answer,
        "answer_template": kw["template"],
        "parameters": dict(kw.get("params", {})),
        "population": kw["population"],
        "caveat": kw["caveat"],
        "evidence_requirements": evidence,
        "visualization_recommended": viz is not None and viz != "table_only",
        "visualization_type": viz,
        "formula_ids": formulas,
        "limitations": list(kw.get("limits", [])),
        "grounded_in": GROUNDED_IN,
    }

    for field in REQUIRED_TEXT_FIELDS:
        key = {"problem_statement": "q", "answer_template": "template"}.get(field, field)
        value = kw.get(key) if key in kw else record.get(field)
        if not value or not str(value).strip():
            raise PSValidationError("{}: '{}' must not be empty".format(ps_id, field))

    return record
