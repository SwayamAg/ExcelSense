"""Phase 1 - build the Problem Statement catalogue.

Emits, from one source of truth:
  - config/problem_statements.json        merged canonical catalogue (spec deliverable)
  - data/PS/<domain>.json                 domain shards for the query engine
  - data/PS/_index.json                   manifest: domain -> file, PS id -> file
  - docs/PROBLEM_STATEMENT_CATALOGUE.md   human-readable catalogue

Every table, column, relationship, operation and metric is validated against
config/data_contract.json. Referencing a column the CSVs do not have fails the build.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from ps_framework import PSValidationError, build_ps, load_schema  # noqa: E402
import ps_defs  # noqa: E402
from ps_defs.executions import EXECUTIONS  # noqa: E402
from ps_defs.executions_b import EXECUTIONS_B  # noqa: E402

CATALOGUE_VERSION = "1.0.0"

# Business domain -> shard file. Every domain must appear here or the build fails,
# so a new domain cannot silently land in an unrouted shard.
DOMAIN_SHARDS = OrderedDict([
    ("Portfolio Overview", "ps_portfolio_overview"),
    ("Product Performance", "ps_product_performance"),
    ("Financial / Premium Analysis", "ps_financial_premium"),
    ("Policy Lifecycle", "ps_policy_lifecycle"),
    ("Lapse Analysis", "ps_lapse_surrender"),
    ("Surrender Analysis", "ps_lapse_surrender"),
    ("Persistency / Renewal", "ps_persistency_renewal"),
    ("Claims & Risk", "ps_claims"),
    ("Claims Operations", "ps_claims"),
    ("Agent / Sales Performance", "ps_agent_performance"),
    ("Agent Quality & Conduct Risk", "ps_agent_performance"),
    ("Customer Segmentation", "ps_customer_demographics"),
    ("Geographic Analysis", "ps_geography_branch"),
    ("Branch Analysis", "ps_geography_branch"),
    ("Trends & Cohorts", "ps_trends_cohorts"),
    ("Anomaly / Concentration / Risk", "ps_anomaly_concentration"),
    ("Root-Cause / Diagnostic", "ps_diagnostic_rootcause"),
    ("Integrity & Reconciliation", "ps_integrity_reconciliation"),
    ("Executive / Portfolio Health", "ps_executive_business"),
    ("Prescriptive / Action Planning", "ps_executive_business"),
])


def build_all():
    schema = load_schema()
    definitions = ps_defs.all_definitions()

    plans = dict(EXECUTIONS)
    overlap = set(plans) & set(EXECUTIONS_B)
    if overlap:
        raise SystemExit("execution plans defined twice: {}".format(sorted(overlap)))
    plans.update(EXECUTIONS_B)

    catalogue, errors = [], []
    for spec in definitions:
        spec = dict(spec)
        spec["execution"] = plans.get(spec.get("id"))
        try:
            catalogue.append(build_ps(schema, **spec))
        except PSValidationError as exc:
            errors.append(str(exc))
        except KeyError as exc:
            errors.append("{}: missing required field {}".format(spec.get("id"), exc))

    if errors:
        print("PS validation failed:\n")
        for e in errors:
            print("  - {}".format(e))
        raise SystemExit(1)

    ids = [p["problem_id"] for p in catalogue]
    dupes = [i for i, n in Counter(ids).items() if n > 1]
    if dupes:
        raise SystemExit("duplicate problem_ids: {}".format(sorted(dupes)))

    from src.analysis.patterns import PATTERNS  # noqa: E402
    import src.analysis.advanced_patterns  # noqa: F401,E402

    missing_plan = [p["problem_id"] for p in catalogue if not p.get("execution")]
    if missing_plan:
        raise SystemExit("PSs with no execution plan: {}".format(missing_plan))
    bad_pattern = [(p["problem_id"], p["execution"].get("pattern"))
                   for p in catalogue if p["execution"].get("pattern") not in PATTERNS]
    if bad_pattern:
        raise SystemExit("PSs naming an unknown pattern: {}".format(bad_pattern))

    dangling = validate_plan_references(catalogue)
    if dangling:
        print("execution plans reference columns they never compute:")
        for problem_id, field, target in dangling:
            print("  - {}: {} -> {!r}".format(problem_id, field, target))
        raise SystemExit(1)

    unmapped = {p["business_domain"] for p in catalogue} - set(DOMAIN_SHARDS)
    if unmapped:
        raise SystemExit("domains with no shard mapping: {}".format(sorted(unmapped)))

    catalogue.sort(key=lambda p: p["problem_id"])
    return schema, catalogue


def validate_plan_references(catalogue):
    """Catch plans that rank or compare on a column the plan never produces.

    This fails silently at runtime: the measure is simply absent, so the rate comes
    back null and the ranking quietly puts everything at the bottom. Far better to
    refuse to build.
    """
    GROUPING_PATTERNS = ("group_metrics", "timeseries", "driver_scan", "composite_score")
    dangling = []
    for ps in catalogue:
        plan = ps.get("execution") or {}
        if plan.get("pattern") not in GROUPING_PATTERNS:
            continue

        group_by = plan.get("group_by")
        keys = [group_by] if isinstance(group_by, str) else list(group_by or [])
        computed = set(plan.get("measures", {})) | set(plan.get("rates", {}))
        computed |= set(plan.get("shares", {})) | set(keys)
        computed |= {"group_rows", "rank", "delta_vs_baseline", "pct_delta_vs_baseline",
                     "period", "composite_score"}

        def check(field, target):
            # A '$name' placeholder is resolved from validated parameters at run time.
            if target and not str(target).startswith("$") and target not in computed:
                dangling.append((ps["problem_id"], field, target))

        for field in ("rank_by", "kpi", "trend_measure", "sort_by_group", "tier_measure"):
            check(field, plan.get(field))
        check("baseline.measure", (plan.get("baseline") or {}).get("measure"))
        for alias, spec in plan.get("rates", {}).items():
            check("rates.{}.numerator".format(alias), spec.get("numerator"))
            check("rates.{}.denominator".format(alias), spec.get("denominator"))
        for alias, source in plan.get("shares", {}).items():
            check("shares.{}".format(alias), source)
    return dangling


def shard(catalogue):
    shards = defaultdict(list)
    for ps in catalogue:
        shards[DOMAIN_SHARDS[ps["business_domain"]]].append(ps)
    return OrderedDict(sorted(shards.items()))


def routing_hints(items):
    """Cheap pre-filter signals so the router can shortlist a shard before scoring."""
    tables, concepts, intents, tiers = set(), set(), set(), set()
    for ps in items:
        tables.update(ps["relevant_tables"])
        concepts.update(ps["concepts"])
        intents.add(ps["intent_type"].split("/")[0])
        tiers.add(ps["analysis_tier"])
    return {
        "tables": sorted(tables),
        "concepts": sorted(concepts),
        "intent_structures": sorted(intents),
        "analysis_tiers": sorted(tiers),
    }


def write_outputs(schema, catalogue):
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = {
        "catalogue_version": CATALOGUE_VERSION,
        "generated_at_utc": generated,
        "generated_by": "scripts/build_ps_catalogue.py",
        "data_contract_version": schema.contract["contract_version"],
        "grounded_in": "Talk_to_Data_PoC_Documentation_Metadata_V4",
        "total_problem_statements": len(catalogue),
    }

    # 1. Merged canonical catalogue (spec deliverable).
    cfg = ROOT / "config" / "problem_statements.json"
    cfg.write_text(json.dumps({**meta, "problem_statements": catalogue}, indent=2),
                   encoding="utf-8")

    # 2. Domain shards plus manifest.
    ps_dir = ROOT / "data" / "PS"
    ps_dir.mkdir(parents=True, exist_ok=True)
    for old in ps_dir.glob("*.json"):
        old.unlink()

    shards = shard(catalogue)
    index_domains, id_to_file = [], {}
    for name, items in shards.items():
        domains = sorted({p["business_domain"] for p in items})
        payload = {
            **meta,
            "shard": name,
            "business_domains": domains,
            "problem_count": len(items),
            "problem_ids": [p["problem_id"] for p in items],
            "routing_hints": routing_hints(items),
            "problem_statements": items,
        }
        (ps_dir / (name + ".json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        index_domains.append({
            "shard": name,
            "file": name + ".json",
            "business_domains": domains,
            "problem_count": len(items),
            "problem_ids": [p["problem_id"] for p in items],
            "routing_hints": payload["routing_hints"],
        })
        for p in items:
            id_to_file[p["problem_id"]] = name + ".json"

    index = {
        **meta,
        "description": ("Domain-sharded Problem Statement catalogue. Resolve a PS id through "
                        "problem_id_to_file, or shortlist a shard via routing_hints before "
                        "loading it. config/problem_statements.json holds the same PSs merged."),
        "shard_count": len(shards),
        "shards": index_domains,
        "problem_id_to_file": OrderedDict(sorted(id_to_file.items())),
    }
    (ps_dir / "_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    # 3. Human-readable catalogue.
    (ROOT / "docs" / "PROBLEM_STATEMENT_CATALOGUE.md").write_text(
        render_markdown(catalogue, shards, meta), encoding="utf-8")

    return shards, meta


def render_markdown(catalogue, shards, meta):
    by_domain = defaultdict(list)
    for ps in catalogue:
        by_domain[ps["business_domain"]].append(ps)

    tiers = Counter(p["analysis_tier"] for p in catalogue)
    intents = Counter(p["intent_type"].split("/")[0] for p in catalogue)
    tables = Counter(t for p in catalogue for t in p["relevant_tables"])
    viz = sum(1 for p in catalogue if p["visualization_recommended"])
    multi = sum(1 for p in catalogue if len(p["relevant_tables"]) > 1)

    L = ["# Excel_sense_v2 - Problem Statement Catalogue", "",
         "Generated by `scripts/build_ps_catalogue.py` on {}.".format(meta["generated_at_utc"]),
         "",
         "Every table, column and relationship below is validated against "
         "`config/data_contract.json` at build time. A PS referencing a column the CSVs do not "
         "have fails the build, so nothing in this catalogue is ungrounded.", "",
         "## Summary", "",
         "| Measure | Value |", "|---|---:|",
         "| Problem statements | {} |".format(len(catalogue)),
         "| Business domains | {} |".format(len(by_domain)),
         "| Domain shards | {} |".format(len(shards)),
         "| Multi-table PSs | {} |".format(multi),
         "| PSs recommending a visualization | {} |".format(viz),
         ""]

    L += ["### By analysis tier", "", "| Tier | Count |", "|---|---:|"]
    L += ["| {} | {} |".format(k, v) for k, v in sorted(tiers.items())]

    L += ["", "### By intent structure", "", "| Structure | Count |", "|---|---:|"]
    L += ["| {} | {} |".format(k, v) for k, v in intents.most_common()]

    L += ["", "### Table coverage", "", "| Table | PSs using it |", "|---|---:|"]
    L += ["| `{}` | {} |".format(k, v) for k, v in tables.most_common()]

    L += ["", "### Shards", "",
          "| Shard file | Domains | PSs |", "|---|---|---:|"]
    for name, items in shards.items():
        doms = ", ".join(sorted({p["business_domain"] for p in items}))
        L.append("| `data/PS/{}.json` | {} | {} |".format(name, doms, len(items)))

    L += ["", "---", "", "## Catalogue", ""]
    for domain in sorted(by_domain):
        L += ["### {}".format(domain), ""]
        for ps in by_domain[domain]:
            L += ["#### {} - {}".format(ps["problem_id"], ps["problem_statement"]), "",
                  "- **Intent:** `{}` | **Tier:** `{}` | **Answer:** `{}`".format(
                      ps["intent_type"], ps["analysis_tier"], ps["expected_answer_type"]),
                  "- **Objective:** {}".format(ps["business_objective"]),
                  "- **Tables:** {}".format(", ".join("`{}`".format(t) for t in ps["relevant_tables"])),
                  "- **Columns:** {}".format(", ".join("`{}`".format(c) for c in ps["relevant_columns"])),
                  ]
            if ps["relationships"]:
                L.append("- **Joins:** {}".format(", ".join("`{}`".format(r) for r in ps["relationships"])))
            L += ["- **Operations:** {}".format(", ".join("`{}`".format(o) for o in ps["required_operations"])),
                  "- **Metrics:** {}".format(", ".join("`{}`".format(m) for m in ps["required_metrics"])),
                  "- **Population:** {}".format(ps["population"]),
                  "- **Analytical steps:**"]
            L += ["  {}. {}".format(i, s) for i, s in enumerate(ps["analytical_steps"], 1)]
            L += ["- **Example queries:** {}".format(
                "; ".join('"{}"'.format(q) for q in ps["example_queries"]))]
            if ps["parameters"]:
                L.append("- **Parameters:** `{}`".format(json.dumps(ps["parameters"])))
            L.append("- **Caveat:** {}".format(ps["caveat"]))
            if ps["limitations"]:
                L.append("- **Limitations:** {}".format("; ".join(ps["limitations"])))
            L.append("- **Visualization:** {}".format(ps["visualization_type"] or "none"))
            L.append("")
    return "\n".join(L) + "\n"


def main():
    schema, catalogue = build_all()
    shards, meta = write_outputs(schema, catalogue)
    print("built {} problem statements".format(len(catalogue)))
    print("  config/problem_statements.json")
    print("  docs/PROBLEM_STATEMENT_CATALOGUE.md")
    print("  data/PS/_index.json + {} shards:".format(len(shards)))
    for name, items in shards.items():
        print("    {:<34} {:>3} PSs".format(name + ".json", len(items)))


if __name__ == "__main__":
    main()
