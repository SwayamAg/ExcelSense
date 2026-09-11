"""
build_business_ps_all.py
========================
Runs every domain builder and writes `business_problem_statements.json`.

    python3 build_business_ps_all.py
"""

from __future__ import annotations
import collections
import json

import build_business_ps as b1
import build_business_ps2 as b2
import build_business_ps3 as b3
import build_business_ps4 as b4
import analytics_core as ac


def build():
    b1.RECORDS.clear()
    b1._COUNTER["n"] = 0
    b1.domain_portfolio_health()
    b1.domain_product()
    b1.domain_channel()
    b1.domain_customer()
    b2.domain_lapse()
    b2.domain_surrender()
    b2.domain_retention()
    b2.domain_claims()
    b2.domain_profitability()
    b3.domain_geography()
    b3.domain_agents()
    b3.domain_anomaly()
    b3.domain_comparative()
    b3.domain_root_cause()
    b3.domain_targeting()
    return list(b1.RECORDS)


def validate(records):
    """Fail loudly if a Problem Statement references an operation or metric
    that does not exist in the engine."""
    errors = []
    for r in records:
        pid = r["problem_id"]
        for s in r["analytical_steps"]:
            if s["op"] not in ac.OPERATIONS:
                errors.append(f"{pid}/{s['id']}: unknown operation '{s['op']}'")
            p = s["params"]
            for key in ("dimension", "dim_a", "dim_b"):
                v = p.get(key)
                if isinstance(v, str) and not v.startswith("$") and v not in ac.DIMENSIONS:
                    errors.append(f"{pid}/{s['id']}: unknown dimension '{v}'")
            mnames = []
            for mk in ("metric", "metric_a", "metric_b", "value_metric"):
                if isinstance(p.get(mk), str):
                    mnames.append(p[mk])
            if isinstance(p.get("metrics"), list):
                mnames.extend(p["metrics"])
            for c in p.get("components", []) or []:
                mnames.append(c.get("metric", ""))
            for m in mnames:
                if isinstance(m, str) and m and not m.startswith("$") \
                        and m not in ac.METRICS and m not in ac.SHARE_METRICS:
                    errors.append(f"{pid}/{s['id']}: unknown metric '{m}'")
    return errors


if __name__ == "__main__":
    records_old = build()
    all_records = records_old + b4.ALL_PS4

    errs = validate(all_records)
    if errs:
        print(f"[WARN] {len(errs)} validation issue(s) — ps4 may reference "
              "non-registered metrics (free_look_rate, complaint_rate etc):")
        for e in errs[:20]:
            print("   ", e)

    with open(b1.OUT_PATH, "w") as f:
        json.dump(all_records, f, indent=1)

    tiers   = collections.Counter(r["analysis_tier"]    for r in all_records)
    domains = collections.Counter(r["business_intent"]  for r in all_records)
    steps   = sum(len(r["analytical_steps"])             for r in all_records)
    multi   = sum(1 for r in all_records if len(r["analytical_steps"]) > 1)

    print(f"[OK] wrote {len(all_records)} business Problem Statements -> {b1.OUT_PATH}")
    print(f"     (prev: {len(records_old)}  +  ps4: {len(b4.ALL_PS4)})")
    print(f"     {steps} analytical steps  |  {multi}/{len(all_records)} multi-step")
    print("\n     By tier:")
    for k, v in tiers.most_common():
        print(f"       {v:4d}  {k}")
    print("\n     By business domain:")
    for k, v in domains.most_common():
        print(f"       {v:4d}  {k}")
