"""
business_eval.py
================
Scores the engine against `business_eval_set.json`.

The brief asks for the *analytical result* to be evaluated separately from the
LLM wording, so nothing here checks generated prose. Six independent scores:

  routing          did the question go to the right engine
                   (business / adhoc / simple_retrieval / insufficient_data)
  intent           did the matched Problem Statement carry the expected
                   business intent
  tier             was the analysis tier at least what the case requires
  operations       did the executed plan include the expected operations
  metrics          did the executed plan compute the expected metrics
  tables           did the plan touch the expected tables
  grounding        did the answer actually name the expected entities
  execution        did every planned step execute without error

    python3 business_eval.py            # full run, writes business_eval_results.json
    python3 business_eval.py --verbose  # also print every failing case
"""


from __future__ import annotations

import os
import sys

# Make src/* importable when run via unittest discovery.
_BASE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_BASE, ".."))
for _sub in ["src", "src/analytics", "src/engine", "src/retrieval", "src/llm", "ui"]:
    _p = os.path.join(_ROOT, _sub)
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import json
import os
import sys
from typing import Any, Dict, List

import business_engine as be
import business_intents as bi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_PATH = os.path.join(BASE_DIR, "business_eval_set.json")
OUT_PATH = os.path.join(BASE_DIR, "business_eval_results.json")

TIER_ORDER = {bi.TIER_RETRIEVAL: 0, bi.TIER_ANALYTICAL: 1, bi.TIER_MULTI_STEP: 2}

# The intent taxonomy overlaps by design: "why are people cashing out?" is
# legitimately either SURRENDER_ANALYSIS or ROOT_CAUSE, and "do savings products
# hold up better than protection?" is either PRODUCT_PERFORMANCE or COMPARATIVE.
# `intent_ok` scores the strict label; `intent_family_ok` accepts a documented
# adjacent intent so the strict number is not mistaken for a hard failure.
ADJACENT_INTENTS = {
    bi.PORTFOLIO_HEALTH: {bi.ANOMALY_CONCENTRATION, bi.PROFITABILITY, bi.COMPARATIVE},
    bi.PRODUCT_PERFORMANCE: {bi.COMPARATIVE, bi.PROFITABILITY, bi.ROOT_CAUSE,
                             bi.LAPSE_ANALYSIS},
    bi.CHANNEL_PERFORMANCE: {bi.COMPARATIVE, bi.ROOT_CAUSE, bi.LAPSE_ANALYSIS},
    bi.CUSTOMER_SEGMENTATION: {bi.COMPARATIVE, bi.LAPSE_ANALYSIS,
                               bi.CUSTOMER_TARGETING, bi.PROFITABILITY},
    bi.LAPSE_ANALYSIS: {bi.ROOT_CAUSE, bi.RETENTION_STRATEGY, bi.COMPARATIVE,
                        bi.CUSTOMER_SEGMENTATION, bi.GEOGRAPHIC,
                        bi.PRODUCT_PERFORMANCE},
    bi.SURRENDER_ANALYSIS: {bi.ROOT_CAUSE, bi.RETENTION_STRATEGY,
                            bi.CUSTOMER_TARGETING, bi.PRODUCT_PERFORMANCE},
    bi.RETENTION_STRATEGY: {bi.LAPSE_ANALYSIS, bi.CUSTOMER_TARGETING,
                            bi.CUSTOMER_SEGMENTATION, bi.COMPARATIVE},
    bi.CLAIMS_RISK: {bi.PROFITABILITY, bi.ANOMALY_CONCENTRATION, bi.ROOT_CAUSE},
    bi.PROFITABILITY: {bi.CLAIMS_RISK, bi.PRODUCT_PERFORMANCE, bi.COMPARATIVE},
    bi.GEOGRAPHIC: {bi.COMPARATIVE, bi.LAPSE_ANALYSIS, bi.PORTFOLIO_HEALTH,
                    bi.CLAIMS_RISK},
    bi.AGENT_PERFORMANCE: {bi.ANOMALY_CONCENTRATION, bi.CUSTOMER_TARGETING,
                           bi.CHANNEL_PERFORMANCE},
    bi.ANOMALY_CONCENTRATION: {bi.PORTFOLIO_HEALTH, bi.AGENT_PERFORMANCE,
                               bi.CLAIMS_RISK},
    bi.COMPARATIVE: {bi.PRODUCT_PERFORMANCE, bi.CHANNEL_PERFORMANCE,
                     bi.GEOGRAPHIC, bi.CUSTOMER_SEGMENTATION, bi.PROFITABILITY,
                     bi.RETENTION_STRATEGY},
    bi.ROOT_CAUSE: {bi.LAPSE_ANALYSIS, bi.SURRENDER_ANALYSIS, bi.PROFITABILITY,
                    bi.PRODUCT_PERFORMANCE, bi.CHANNEL_PERFORMANCE,
                    bi.CLAIMS_RISK, bi.GEOGRAPHIC},
    bi.CUSTOMER_TARGETING: {bi.RETENTION_STRATEGY, bi.CUSTOMER_SEGMENTATION,
                            bi.SURRENDER_ANALYSIS, bi.AGENT_PERFORMANCE},
}


def expected_route(case: Dict[str, Any]) -> List[str]:
    if case["expected_intent"] == "OUT_OF_SCOPE":
        return [be.ROUTE_INSUFFICIENT]
    if case["expected_intent"] == bi.SIMPLE_RETRIEVAL:
        return [be.ROUTE_SIMPLE]
    return [be.ROUTE_BUSINESS, be.ROUTE_ADHOC]


def evaluate(engine: be.BusinessAnalyticsEngine,
             cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    for c in cases:
        resp = engine.ask(c["question"])
        exp_routes = expected_route(c)
        row: Dict[str, Any] = {
            "id": c["id"], "question": c["question"],
            "expected_intent": c["expected_intent"],
            "route": resp.route, "matched_ps": resp.plan.problem_id if resp.plan else None,
            "tier": resp.tier,
        }
        row["routing_ok"] = resp.route in exp_routes

        if resp.route in (be.ROUTE_SIMPLE, be.ROUTE_INSUFFICIENT):
            # nothing analytical to check; routing is the whole test
            for k in ("intent_ok", "intent_family_ok", "tier_ok", "operations_ok",
                      "metrics_ok", "tables_ok", "grounding_ok", "execution_ok"):
                row[k] = None
            row["multi_step_ok"] = None
            rows.append(row)
            continue

        # ---- intent
        row["intent_ok"] = (resp.business_intent == c["expected_intent"])
        row["intent_family_ok"] = row["intent_ok"] or (
            resp.business_intent in ADJACENT_INTENTS.get(c["expected_intent"], set()))

        # ---- tier
        row["tier_ok"] = (TIER_ORDER.get(resp.tier, 0) >=
                          TIER_ORDER.get(c["expected_tier"], 0))

        # ---- operations actually executed
        ops = {r.step.op for r in resp.step_results}
        want_ops = set(c["expected_operations"])
        row["operations_found"] = sorted(ops)
        row["operations_missing"] = sorted(want_ops - ops)
        row["operations_ok"] = want_ops.issubset(ops) if want_ops else None

        # ---- metrics actually computed
        metrics = set()
        for r in resp.step_results:
            p = r.step.params
            if isinstance(p.get("metric"), str):
                metrics.add(p["metric"])
            for m in p.get("metrics") or []:
                metrics.add(m)
            for comp in p.get("components") or []:
                metrics.add(comp["metric"])
            if isinstance(p.get("value_metric"), str):
                metrics.add(p["value_metric"])
            if r.step.op == "portfolio_overview" and r.ok and isinstance(r.result, dict):
                metrics.update(k for k in r.result if not k.startswith("_"))
            if r.step.op == "segment_profile" and r.ok and r.is_frame:
                metrics.update(r.result["metric"].tolist())
        want_m = set(c["expected_metrics"])
        row["metrics_missing"] = sorted(want_m - metrics)
        row["metrics_ok"] = want_m.issubset(metrics) if want_m else None

        # ---- tables
        tables = set(resp.plan.relevant_tables) if resp.plan else set()
        want_t = set(c["expected_tables"])
        row["tables_missing"] = sorted(want_t - tables)
        row["tables_ok"] = want_t.issubset(tables) if want_t else None

        # ---- multi-step requirement
        row["n_steps"] = len(resp.step_results)
        row["multi_step_ok"] = (len(resp.step_results) > 1) if c["requires_multi_step"] else None

        # ---- grounding: the answer must actually name the expected entities
        text = ""
        if resp.answer:
            text = (resp.answer.headline + " " + " ".join(resp.answer.evidence) + " "
                    + " ".join(resp.answer.analysis) + " "
                    + " ".join(resp.answer.implication) + " "
                    + " ".join(resp.answer.recommendation)).lower()
        want_mentions = [m.lower() for m in c["answer_must_mention"]]
        row["mentions_missing"] = [m for m in want_mentions if m not in text]
        row["grounding_ok"] = (not row["mentions_missing"]) if want_mentions else None

        # ---- execution health
        failed = [r for r in resp.step_results if not r.ok]
        row["steps_failed"] = len(failed)
        row["execution_ok"] = (len(failed) == 0)

        # ---- structure: business answers must carry the six sections
        a = resp.answer
        row["structure_ok"] = bool(a and a.headline and a.evidence and a.analysis
                                   and a.implication and a.recommendation)
        rows.append(row)

    # ---- aggregate
    def rate(key: str) -> Dict[str, Any]:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return {"pass": sum(1 for v in vals if v), "total": len(vals),
                "rate": round(sum(1 for v in vals if v) / len(vals), 4) if vals else None}

    summary = {k: rate(k) for k in
               ["routing_ok", "intent_ok", "intent_family_ok", "tier_ok",
                "operations_ok", "metrics_ok", "tables_ok", "grounding_ok",
                "multi_step_ok", "execution_ok", "structure_ok"]}
    analytical = [r for r in rows if r["route"] in (be.ROUTE_BUSINESS, be.ROUTE_ADHOC)]
    summary["_cases"] = len(rows)
    summary["_analytical_cases"] = len(analytical)
    summary["_avg_steps_per_analytical_case"] = round(
        sum(r.get("n_steps", 0) for r in analytical) / max(len(analytical), 1), 2)
    summary["_total_steps_executed"] = sum(r.get("n_steps", 0) for r in rows)
    return {"summary": summary, "rows": rows}


def main(verbose: bool = False):
    with open(EVAL_PATH) as f:
        cases = json.load(f)
    engine = be.get_engine(with_legacy=True)
    res = evaluate(engine, cases)

    s = res["summary"]
    print("=" * 78)
    print(f"BUSINESS EVALUATION - {s['_cases']} held-out questions "
          f"({s['_analytical_cases']} analytical)")
    print("=" * 78)
    for k in ["routing_ok", "intent_ok", "intent_family_ok", "tier_ok",
              "operations_ok", "metrics_ok", "tables_ok", "grounding_ok",
              "multi_step_ok", "execution_ok", "structure_ok"]:
        d = s[k]
        if d["total"] == 0:
            continue
        bar = "#" * int(round((d["rate"] or 0) * 30))
        print(f"  {k:16s} {d['pass']:3d}/{d['total']:3d}  {d['rate']:.3f}  {bar}")
    print(f"\n  average analytical steps per business question: "
          f"{s['_avg_steps_per_analytical_case']}")
    print(f"  total analytical operations executed:          {s['_total_steps_executed']}")

    if verbose:
        print("\nFAILURES")
        for r in res["rows"]:
            bad = [k for k in ("routing_ok", "intent_ok", "tier_ok", "operations_ok",
                               "metrics_ok", "tables_ok", "grounding_ok",
                               "execution_ok", "structure_ok")
                   if r.get(k) is False]
            if bad:
                print(f"  {r['id']} [{','.join(bad)}] route={r['route']} "
                      f"ps={r['matched_ps']} :: {r['question'][:70]}")
                for k2 in ("operations_missing", "metrics_missing", "tables_missing",
                           "mentions_missing"):
                    if r.get(k2):
                        print(f"      {k2}: {r[k2]}")

    with open(OUT_PATH, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n[OK] wrote {OUT_PATH}")
    return res


if __name__ == "__main__":
    main(verbose="--verbose" in sys.argv)
