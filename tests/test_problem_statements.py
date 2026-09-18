"""Phase 1 gate tests - the PS catalogue is grounded, complete and non-redundant.

Validation runs against the live CSVs as well as the catalogue metadata, so a PS
cannot claim a column, join or population the data does not actually provide.
"""
from __future__ import annotations

import difflib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE_PATH = ROOT / "config" / "problem_statements.json"
PS_DIR = ROOT / "data" / "PS"

RELATIONSHIPS = {
    "policy_owner": ("policy", "AppID", "owner", "AppID"),
    "policy_claims": ("policy", "AppID", "claims", "AppID"),
    "policy_sales": ("policy", "SALES_ID", "sales", "SALES_ID"),
    "policy_persistency": ("policy", "AppID", "persistency", "AppID"),
    "persistency_sales": ("persistency", "SALES_ID", "sales", "SALES_ID"),
}

REQUIRED_FIELDS = (
    "problem_id", "problem_statement", "intent_type", "business_intent",
    "business_domain", "business_objective", "analysis_tier", "example_queries",
    "relevant_tables", "relevant_columns", "required_operations", "required_metrics",
    "relationships", "concepts", "analytical_steps", "expected_answer_type",
    "answer_template", "parameters", "population", "caveat",
    "evidence_requirements", "visualization_recommended", "visualization_type",
    "formula_ids", "limitations", "grounded_in",
)

# Concepts the five CSVs cannot support. A PS whose text promises one of these is
# making a claim the data cannot honour.
UNSUPPORTED_TERMS = (
    "smoker", "bmi", "medical underwriting", "underwriting score", "risk score",
    "commission", "profitability", "profit margin", "expense ratio", "reserve",
    "nav", "fund performance", "unit price", "rider", "nominee", "maturity benefit",
    "zone", "target achievement", "agent tenure", "agent training",
    "surrender value", "surrender charge", "lapse date", "lapse reason",
    "premium paid to date", "payment history", "marital status", "complaint case",
)

# A causal assertion needs either supporting evidence or an explicit disclaimer.
CAUSAL_PATTERNS = re.compile(
    r"\b(causes?|caused by|because of|due to|leads? to|results? in|drives? the)\b",
    re.IGNORECASE)
DISCLAIMER_PATTERNS = re.compile(
    r"(association|not.{0,20}caus|cannot (be )?(establish|conclude|prove)|"
    r"no causal|not proven|observable difference|indicative only|"
    r"correlat|not evidence of|does not establish)",
    re.IGNORECASE)

VIZ_NEEDING_SERIES = {"line", "multi_line"}
VIZ_NEEDING_GROUPS = {"bar", "horizontal_bar", "grouped_bar", "stacked_bar",
                      "pie", "donut", "heatmap"}


@pytest.fixture(scope="session")
def catalogue():
    assert CATALOGUE_PATH.exists(), "run scripts/build_ps_catalogue.py"
    return json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def problems(catalogue):
    return catalogue["problem_statements"]


@pytest.fixture(scope="session")
def schema_columns(contract):
    return {"{}.{}".format(k, c)
            for k, t in contract["tables"].items() for c in t["source_columns"]}


@pytest.fixture(scope="session")
def derived_columns():
    from scripts.ps_framework import load_schema
    return set(load_schema().derived)


def ids(problems):
    return [p["problem_id"] for p in problems]


# --------------------------------------------------------------------------
# Catalogue integrity
# --------------------------------------------------------------------------

def test_catalogue_exists_and_is_substantial(problems):
    assert len(problems) >= 100, "catalogue has only {} PSs".format(len(problems))


def test_catalogue_metadata_is_present(catalogue, contract):
    for field in ("catalogue_version", "generated_at_utc", "data_contract_version",
                  "grounded_in", "total_problem_statements"):
        assert catalogue.get(field), "catalogue missing {}".format(field)
    assert catalogue["total_problem_statements"] == len(catalogue["problem_statements"])
    assert catalogue["data_contract_version"] == contract["contract_version"], (
        "catalogue was built against a different data contract - rebuild it")


def test_no_duplicate_problem_ids(problems):
    dupes = [i for i, n in Counter(ids(problems)).items() if n > 1]
    assert not dupes, "duplicate problem_ids: {}".format(dupes)


def test_problem_ids_are_well_formed_and_contiguous(problems):
    pattern = re.compile(r"^PS-\d{3}$")
    for p in problems:
        assert pattern.match(p["problem_id"]), "bad id {}".format(p["problem_id"])
    numbers = sorted(int(i.split("-")[1]) for i in ids(problems))
    assert numbers == list(range(numbers[0], numbers[0] + len(numbers))), (
        "problem ids have gaps: {}".format(numbers))


def test_every_ps_has_all_required_fields(problems):
    for p in problems:
        missing = [f for f in REQUIRED_FIELDS if f not in p]
        assert not missing, "{} missing fields {}".format(p["problem_id"], missing)


def test_required_text_fields_are_non_empty(problems):
    text_fields = ("problem_statement", "business_intent", "business_objective",
                   "answer_template", "population", "caveat", "business_domain")
    for p in problems:
        for f in text_fields:
            assert p[f] and str(p[f]).strip(), "{}: '{}' is empty".format(p["problem_id"], f)


def test_grounded_in_is_consistent(problems):
    values = {p["grounded_in"] for p in problems}
    assert values == {"Talk_to_Data_PoC_Documentation_Metadata_V4"}, values


# --------------------------------------------------------------------------
# Table, column and relationship validation
# --------------------------------------------------------------------------

def test_table_existence_validation(problems, contract):
    known = set(contract["tables"])
    for p in problems:
        assert p["relevant_tables"], "{} declares no tables".format(p["problem_id"])
        bad = [t for t in p["relevant_tables"] if t not in known]
        assert not bad, "{} references unknown tables {}".format(p["problem_id"], bad)


def test_column_existence_validation(problems, schema_columns, derived_columns):
    """Every column must exist in a CSV or be a declared derived column."""
    for p in problems:
        assert p["relevant_columns"], "{} declares no columns".format(p["problem_id"])
        for col in p["relevant_columns"]:
            assert col in schema_columns or col in derived_columns, (
                "{}: column '{}' exists in neither the CSVs nor the derived set".format(
                    p["problem_id"], col))


def test_columns_are_readable_from_the_actual_csvs(problems, raw_frames, derived_columns):
    """Strongest grounding check: read each declared source column off the real file."""
    for p in problems:
        for col in p["relevant_columns"]:
            if col in derived_columns:
                continue
            table, name = col.split(".", 1)
            assert name in raw_frames[table].columns, (
                "{}: {} is not a column of the {} CSV".format(p["problem_id"], col, table))


def test_declared_tables_match_the_columns_used(problems):
    for p in problems:
        used = {c.split(".", 1)[0] for c in p["relevant_columns"]}
        assert used == set(p["relevant_tables"]), (
            "{}: declares tables {} but uses columns from {}".format(
                p["problem_id"], sorted(p["relevant_tables"]), sorted(used)))


def test_relationship_validation(problems):
    for p in problems:
        for rel in p["relationships"]:
            assert rel in RELATIONSHIPS, (
                "{}: unknown relationship '{}'".format(p["problem_id"], rel))
            lt, _, rt, _ = RELATIONSHIPS[rel]
            for side in (lt, rt):
                assert side in p["relevant_tables"], (
                    "{}: relationship '{}' needs table '{}'".format(
                        p["problem_id"], rel, side))


def test_multi_table_ps_declares_a_join_path_or_justifies_none(problems):
    for p in problems:
        if len(p["relevant_tables"]) <= 1:
            continue
        if p["relationships"]:
            continue
        justification = p.get("independent_tables")
        assert justification and str(justification).strip(), (
            "{}: spans {} tables with no join path and no justification".format(
                p["problem_id"], len(p["relevant_tables"])))


def test_declared_joins_actually_work_on_the_data(problems, raw_frames):
    """Execute each declared relationship once and require a non-empty overlap."""
    exercised = {rel for p in problems for rel in p["relationships"]}
    assert exercised, "no relationships exercised by any PS"
    for rel in exercised:
        lt, lk, rt, rk = RELATIONSHIPS[rel]
        left = set(raw_frames[lt][lk].dropna())
        right = set(raw_frames[rt][rk].dropna())
        overlap = left & right
        assert overlap, "relationship {} yields no matching keys".format(rel)


def test_every_documented_relationship_is_used_somewhere(problems):
    used = {rel for p in problems for rel in p["relationships"]}
    documented = set(RELATIONSHIPS)
    unused = documented - used
    assert not unused, "documented relationships never used by any PS: {}".format(sorted(unused))


# --------------------------------------------------------------------------
# Operation and metric validation
# --------------------------------------------------------------------------

def test_operation_validation(problems):
    from scripts.ps_framework import OPERATIONS
    for p in problems:
        assert p["required_operations"], "{} declares no operations".format(p["problem_id"])
        bad = [o for o in p["required_operations"] if o not in OPERATIONS]
        assert not bad, "{}: unknown operations {}".format(p["problem_id"], bad)


def test_metric_validation(problems):
    from scripts.ps_framework import METRICS
    for p in problems:
        assert p["required_metrics"], "{} declares no metrics".format(p["problem_id"])
        bad = [m for m in p["required_metrics"] if m not in METRICS]
        assert not bad, "{}: unknown metrics {}".format(p["problem_id"], bad)


def test_intent_type_follows_the_grammar(problems):
    from scripts.ps_framework import INTENT_MEASURES, INTENT_STRUCTURES
    for p in problems:
        parts = p["intent_type"].split("/")
        assert len(parts) >= 2, "{}: malformed intent".format(p["problem_id"])
        assert parts[0] in INTENT_STRUCTURES, (
            "{}: bad intent structure {}".format(p["problem_id"], parts[0]))
        assert parts[-1] in INTENT_MEASURES, (
            "{}: intent must end in a measure, got {}".format(p["problem_id"], parts[-1]))


def test_analysis_tier_and_answer_type_are_valid(problems):
    from scripts.ps_framework import ANSWER_TYPES, TIERS
    for p in problems:
        assert p["analysis_tier"] in TIERS, p["problem_id"]
        assert p["expected_answer_type"] in ANSWER_TYPES, p["problem_id"]


def test_no_formula_is_undefined(problems):
    """A PS may not reference a formula the engine has no definition for."""
    from scripts.ps_framework import FORMULA_IDS
    for p in problems:
        bad = [f for f in p["formula_ids"] if f not in FORMULA_IDS]
        assert not bad, "{}: undefined formulas {}".format(p["problem_id"], bad)


def test_calculating_ps_declares_at_least_one_formula(problems):
    """Anything that computes a number must say which formula produces it."""
    computing = {"Ratio", "Rate", "Average", "Median", "Sum", "Count",
                 "Comparison", "Trend", "Concentration"}
    for p in problems:
        measures = set(p["intent_type"].split("/"))
        if measures & computing:
            assert p["formula_ids"], (
                "{} computes {} but declares no formula".format(
                    p["problem_id"], sorted(measures & computing)))


# --------------------------------------------------------------------------
# Example queries and analytical steps
# --------------------------------------------------------------------------

def test_example_query_coverage(problems):
    for p in problems:
        queries = p["example_queries"]
        assert len(queries) >= 3, (
            "{} has only {} example queries".format(p["problem_id"], len(queries)))
        assert len(set(q.lower().strip() for q in queries)) == len(queries), (
            "{} has duplicate example queries".format(p["problem_id"]))
        for q in queries:
            assert len(q.strip()) >= 10, "{}: query too short: {!r}".format(p["problem_id"], q)


def test_example_queries_are_unique_across_the_catalogue(problems):
    seen = defaultdict(list)
    for p in problems:
        for q in p["example_queries"]:
            seen[q.lower().strip()].append(p["problem_id"])
    clashes = {q: owners for q, owners in seen.items() if len(owners) > 1}
    assert not clashes, (
        "identical example queries would make routing ambiguous: {}".format(
            {q: o for q, o in list(clashes.items())[:5]}))


def test_analytical_step_consistency(problems):
    for p in problems:
        steps = p["analytical_steps"]
        assert len(steps) >= 2, "{} has {} steps".format(p["problem_id"], len(steps))
        for s in steps:
            assert len(s.strip()) >= 15, "{}: step too terse: {!r}".format(p["problem_id"], s)
        assert len(set(steps)) == len(steps), "{} repeats a step".format(p["problem_id"])


def test_multi_step_ps_have_proportionate_depth(problems):
    """A multi-step business PS must actually describe multiple steps."""
    for p in problems:
        if p["analysis_tier"] == "multi_step_business":
            assert len(p["analytical_steps"]) >= 5, (
                "{} is multi_step_business but has only {} steps".format(
                    p["problem_id"], len(p["analytical_steps"])))


def test_join_ps_describe_the_join_in_their_steps(problems):
    offenders = [p["problem_id"] for p in problems
                 if p["relationships"]
                 and "join" not in " ".join(p["analytical_steps"]).lower()]
    assert not offenders, (
        "PSs declare a relationship but never mention joining: {}".format(offenders))


# --------------------------------------------------------------------------
# Unsupported-domain and grounding checks
# --------------------------------------------------------------------------

def test_no_ps_promises_an_unsupported_concept(problems):
    """The PS's own promise (statement, objective, queries) must be deliverable."""
    offences = []
    for p in problems:
        # A PS may name unsupported concepts only to refuse them, and must say so.
        if p.get("mentions_unsupported_concepts"):
            continue
        promise = " ".join([p["problem_statement"], p["business_objective"],
                            p["business_intent"], " ".join(p["example_queries"])]).lower()
        for term in UNSUPPORTED_TERMS:
            if term in promise:
                offences.append((p["problem_id"], term))
    assert not offences, (
        "PSs promise concepts the data cannot supply: {}".format(offences))


def test_single_valued_columns_are_never_used_for_comparison(problems):
    """CHANNEL and Mode have one value each; no PS may compare across them."""
    for p in problems:
        for col in ("policy.CHANNEL", "policy.Mode"):
            if col not in p["relevant_columns"]:
                continue
            measures = set(p["intent_type"].split("/"))
            assert not (measures & {"Comparison", "Ranking", "TopN", "BottomN"}), (
                "{} uses single-valued {} for a comparison".format(p["problem_id"], col))


def test_unsupported_domains_from_the_contract_are_respected(problems, contract):
    """Whatever Phase 0 declared unanswerable must not reappear as a PS domain."""
    blocked = " ".join(d["domain"].lower() for d in contract["unanswerable_domains"])
    for p in problems:
        domain = p["business_domain"].lower()
        assert domain not in blocked or "analysis" in domain, (
            "{} sits in an unanswerable domain: {}".format(p["problem_id"], domain))


def test_causal_language_carries_a_disclaimer(problems):
    """Any PS using causal wording must also say what cannot be concluded."""
    offences = []
    for p in problems:
        body = " ".join([p["problem_statement"], p["business_objective"],
                         p["answer_template"], " ".join(p["analytical_steps"])])
        if not CAUSAL_PATTERNS.search(body):
            continue
        safety = " ".join([p["caveat"], " ".join(p["limitations"]),
                           " ".join(p["evidence_requirements"].get("limitations", []))])
        if not DISCLAIMER_PATTERNS.search(safety):
            offences.append(p["problem_id"])
    assert not offences, (
        "PSs use causal language without a disclaimer: {}".format(offences))


def test_diagnostic_ps_explicitly_disclaim_causation(problems):
    for p in problems:
        if p["analysis_tier"] != "diagnostic":
            continue
        safety = " ".join([p["caveat"], " ".join(p["limitations"]),
                           " ".join(p["evidence_requirements"].get("limitations", []))])
        assert DISCLAIMER_PATTERNS.search(safety), (
            "{} is diagnostic but never disclaims causation".format(p["problem_id"]))


def test_every_ps_declares_its_population(problems):
    for p in problems:
        pop = p["population"]
        assert len(pop.strip()) >= 20, (
            "{}: population is too vague: {!r}".format(p["problem_id"], pop))
        assert p["evidence_requirements"].get("population_definition"), (
            "{}: evidence requirements omit the population".format(p["problem_id"]))


def test_ps_using_persistency_acknowledge_its_restricted_population(problems):
    """Persistency excludes surrendered and death policies; PSs must not ignore that."""
    offenders = []
    for p in problems:
        if "persistency" not in p["relevant_tables"]:
            continue
        text = " ".join([p["population"], p["caveat"],
                         " ".join(p["evidence_requirements"].get("data_quality_notes", []))]).lower()
        if not any(k in text for k in ("dq-006", "persistency", "eligible", "active")):
            offenders.append(p["problem_id"])
    assert not offenders, (
        "PSs use Persistency without noting its restricted population: {}".format(offenders))


# --------------------------------------------------------------------------
# Vagueness, duplication and visualization feasibility
# --------------------------------------------------------------------------

def test_no_ps_is_too_vague(problems):
    for p in problems:
        assert len(p["problem_statement"].split()) >= 4, (
            "{}: statement too vague".format(p["problem_id"]))
        assert len(p["business_objective"].split()) >= 6, (
            "{}: objective too vague".format(p["problem_id"]))
        assert p["problem_statement"].strip().endswith(("?", ".")), (
            "{}: statement is not a question or instruction".format(p["problem_id"]))


def test_no_near_duplicate_problem_statements(problems):
    """Two PSs that ask the same thing of the same data would make routing arbitrary."""
    near = []
    for i, a in enumerate(problems):
        for b in problems[i + 1:]:
            ratio = difflib.SequenceMatcher(
                None, a["problem_statement"].lower(), b["problem_statement"].lower()).ratio()
            if ratio < 0.85:
                continue
            same_scope = (set(a["relevant_tables"]) == set(b["relevant_tables"])
                          and a["intent_type"] == b["intent_type"]
                          and set(a["relevant_columns"]) == set(b["relevant_columns"]))
            if same_scope:
                near.append((a["problem_id"], b["problem_id"], round(ratio, 3)))
    assert not near, "near-duplicate PSs: {}".format(near)


def test_no_two_ps_share_identical_analytical_scope(problems):
    signatures = defaultdict(list)
    for p in problems:
        sig = (tuple(sorted(p["relevant_columns"])), p["intent_type"],
               tuple(sorted(p["required_metrics"])))
        signatures[sig].append(p["problem_id"])
    clashes = {k: v for k, v in signatures.items() if len(v) > 1}
    assert not clashes, "PSs with identical analytical scope: {}".format(list(clashes.values()))


def test_visualization_metadata_is_coherent(problems):
    from scripts.ps_framework import VIZ_TYPES
    for p in problems:
        viz = p["visualization_type"]
        assert viz in VIZ_TYPES, "{}: unknown viz type {}".format(p["problem_id"], viz)
        expected = viz is not None and viz != "table_only"
        assert p["visualization_recommended"] == expected, (
            "{}: visualization_recommended disagrees with type {}".format(p["problem_id"], viz))


def test_requested_visualizations_can_be_generated(problems):
    """A chart type must match the answer shape the PS actually produces."""
    for p in problems:
        viz, answer = p["visualization_type"], p["expected_answer_type"]
        if viz in VIZ_NEEDING_SERIES:
            assert answer in ("timeseries", "table", "distribution"), (
                "{}: {} chart needs a series but the answer is {}".format(
                    p["problem_id"], viz, answer))
        if viz in VIZ_NEEDING_GROUPS:
            assert answer not in ("scalar", "record"), (
                "{}: {} chart cannot be drawn from a {} answer".format(
                    p["problem_id"], viz, answer))
        if answer == "scalar":
            assert viz in (None, "table_only"), (
                "{}: scalar answer should not request a {}".format(p["problem_id"], viz))


def test_evidence_requirements_are_declared(problems):
    from scripts.ps_framework import EVIDENCE_KEYS
    for p in problems:
        ev = p["evidence_requirements"]
        assert ev, "{} declares no evidence requirements".format(p["problem_id"])
        bad = set(ev) - EVIDENCE_KEYS
        assert not bad, "{}: unknown evidence keys {}".format(p["problem_id"], sorted(bad))
        assert ev.get("tables_used") and ev.get("columns_used")


def test_parameters_are_well_formed(problems):
    for p in problems:
        for name, spec in p["parameters"].items():
            assert isinstance(spec, dict), "{}: parameter {} is not a spec".format(
                p["problem_id"], name)
            assert "type" in spec, "{}: parameter {} has no type".format(p["problem_id"], name)
            if spec["type"] == "enum":
                assert spec.get("values"), "{}: enum {} has no values".format(
                    p["problem_id"], name)


# --------------------------------------------------------------------------
# Required capability coverage (specification section 4)
# --------------------------------------------------------------------------

def test_catalogue_covers_every_required_capability(problems):
    measures = {m for p in problems for m in p["intent_type"].split("/")}
    required = {"Count", "Sum", "Average", "Ratio", "Ranking", "TopN", "Comparison",
                "Trend", "Distribution", "Concentration", "Record", "Integrity",
                "Diagnostic", "Driver", "Overview", "Profile", "Anomaly", "Outliers"}
    missing = required - measures
    assert not missing, "catalogue lacks required capabilities: {}".format(sorted(missing))


def test_catalogue_covers_all_five_tables(problems):
    used = {t for p in problems for t in p["relevant_tables"]}
    assert used == {"policy", "owner", "sales", "claims", "persistency"}, used


def test_catalogue_uses_a_broad_share_of_the_schema(problems, contract):
    """The catalogue should exercise most real columns, not cluster on a few."""
    used = {c for p in problems for c in p["relevant_columns"]}
    actual = {"{}.{}".format(k, c)
              for k, t in contract["tables"].items() for c in t["source_columns"]}
    covered = used & actual
    coverage = len(covered) / len(actual)
    assert coverage >= 0.80, (
        "only {:.0%} of real columns are used; unused: {}".format(
            coverage, sorted(actual - used)))


def test_every_analysis_tier_is_represented(problems):
    tiers = Counter(p["analysis_tier"] for p in problems)
    for tier in ("descriptive", "analytical", "diagnostic", "multi_step_business"):
        assert tiers[tier] >= 5, "only {} PSs in tier {}".format(tiers[tier], tier)


def test_why_and_how_questions_are_supported(problems):
    why = [p for p in problems
           if any(q.lower().startswith(("why", "what is driving", "what drives",
                                        "explain", "diagnose", "what makes"))
                  for q in p["example_queries"])]
    assert len(why) >= 5, "only {} PSs answer why/how questions".format(len(why))


def test_multi_table_analysis_is_well_represented(problems):
    multi = [p for p in problems if len(p["relevant_tables"]) > 1]
    assert len(multi) >= 30, "only {} multi-table PSs".format(len(multi))
    assert any(len(p["relevant_tables"]) >= 4 for p in multi), "no PS spans 4+ tables"


# --------------------------------------------------------------------------
# Domain shards
# --------------------------------------------------------------------------

def test_shard_index_exists_and_is_complete(problems):
    index_path = PS_DIR / "_index.json"
    assert index_path.exists(), "data/PS/_index.json missing"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["total_problem_statements"] == len(problems)
    assert set(index["problem_id_to_file"]) == set(ids(problems)), (
        "index does not map every PS id")
    for shard in index["shards"]:
        assert (PS_DIR / shard["file"]).exists(), "missing shard {}".format(shard["file"])
        assert shard["routing_hints"]["tables"], "shard {} has no routing hints".format(
            shard["shard"])


def test_shards_partition_the_catalogue_exactly(problems):
    """Every PS appears in exactly one shard, and shards add up to the whole."""
    seen = {}
    for path in sorted(PS_DIR.glob("ps_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for p in payload["problem_statements"]:
            pid = p["problem_id"]
            assert pid not in seen, "{} appears in both {} and {}".format(
                pid, seen.get(pid), path.name)
            seen[pid] = path.name
    assert set(seen) == set(ids(problems)), (
        "shards do not cover the catalogue exactly; missing {}".format(
            set(ids(problems)) - set(seen)))


def test_shard_contents_match_the_merged_catalogue(problems):
    """Shards and the merged file are generated together and must never diverge."""
    merged = {p["problem_id"]: p for p in problems}
    for path in sorted(PS_DIR.glob("ps_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for p in payload["problem_statements"]:
            assert p == merged[p["problem_id"]], (
                "{} differs between {} and the merged catalogue".format(
                    p["problem_id"], path.name))


def test_ps_directory_holds_only_generated_files():
    """The old reference PS material must not be sitting in data/PS."""
    stray = [f.name for f in PS_DIR.iterdir()
             if f.is_file() and not (f.name.startswith("ps_") or f.name == "_index.json")]
    assert not stray, "unexpected files in data/PS: {}".format(stray)


# --------------------------------------------------------------------------
# Feasibility: every PS can actually be executed against the CSVs
# --------------------------------------------------------------------------

def test_every_ps_population_is_non_empty(problems, raw_frames, derived_columns):
    """Reading a PS's declared source columns must yield rows, not an empty frame."""
    for p in problems:
        for table in p["relevant_tables"]:
            cols = [c.split(".", 1)[1] for c in p["relevant_columns"]
                    if c.startswith(table + ".") and c not in derived_columns]
            if not cols:
                continue
            frame = raw_frames[table][cols]
            assert len(frame) > 0, "{}: {} yields no rows".format(p["problem_id"], table)
            assert not frame.isna().all().any(), (
                "{}: a declared column of {} is entirely null".format(p["problem_id"], table))


def test_grouping_ps_have_a_column_that_actually_varies(problems, raw_frames, derived_columns):
    """A GroupBy or comparison PS needs at least one non-constant source column."""
    for p in problems:
        if "GROUP BY" not in p["required_operations"]:
            continue
        varies = False
        for col in p["relevant_columns"]:
            if col in derived_columns:
                varies = True
                break
            table, name = col.split(".", 1)
            if raw_frames[table][name].nunique(dropna=True) > 1:
                varies = True
                break
        assert varies, (
            "{} groups by columns that are all single-valued".format(p["problem_id"]))


def test_unsupported_concept_exemptions_are_justified(problems):
    """Only a PS that exists to refuse may name unsupported concepts, and it must explain why."""
    exempt = [p for p in problems if p.get("mentions_unsupported_concepts")]
    assert len(exempt) <= 2, "too many PSs exempted from the unsupported-concept check"
    for p in exempt:
        justification = p["mentions_unsupported_concepts"]
        assert len(justification.split()) >= 15, (
            "{}: exemption justification is too thin".format(p["problem_id"]))
        assert "refus" in justification.lower() or "reject" in justification.lower(), (
            "{}: exemption must be for refusal purposes".format(p["problem_id"]))


# Real columns the catalogue deliberately does not use, each with the reason.
INTENTIONALLY_UNUSED_COLUMNS = {
    "owner.Owner.Given.Name":
        "personal data; withheld from analysis and from record output by default",
    "persistency.CHANNEL":
        "single-valued ('Banca') like policy.CHANNEL, so it supports no comparison",
    "persistency.Inforce.Date":
        "duplicates policy.Inforce.Date; the policy table is the single source for it",
    "persistency.PREMIUM.PAYING.TERM":
        "duplicates policy.PREMIUM.PAYING.TERM; the policy table is the single source",
}


def test_unused_columns_are_unused_on_purpose(problems, contract):
    """Every real column is either used by a PS or has a recorded reason for exclusion."""
    used = {c for p in problems for c in p["relevant_columns"]}
    actual = {"{}.{}".format(k, c)
              for k, t in contract["tables"].items() for c in t["source_columns"]}
    unused = actual - used
    undocumented = unused - set(INTENTIONALLY_UNUSED_COLUMNS)
    assert not undocumented, (
        "columns are unused with no recorded reason: {}".format(sorted(undocumented)))
    stale = set(INTENTIONALLY_UNUSED_COLUMNS) - unused
    assert not stale, (
        "these columns are now used, so their exclusion note is stale: {}".format(sorted(stale)))


def test_pii_columns_are_never_used_in_aggregate_analysis(problems):
    """Name and postal code must not leak into any analytical PS."""
    pii = {"owner.Owner.Given.Name", "owner.Owner.Comm.Postal.Code"}
    for p in problems:
        leaked = pii & set(p["relevant_columns"])
        if not leaked:
            continue
        # Only the anomaly-review PS may touch postal code, and only to flag implausible values.
        assert leaked == {"owner.Owner.Comm.Postal.Code"}, (
            "{} uses personal data {}".format(p["problem_id"], sorted(leaked)))
        assert "plausib" in p["business_objective"].lower() + p["caveat"].lower(), (
            "{} uses postal code outside a data-quality check".format(p["problem_id"]))
