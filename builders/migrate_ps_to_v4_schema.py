"""
migrate_ps_to_v4_schema.py
==========================
Re-point the Problem Statement library at the five production tables.

The library was authored against an earlier dataset. Two kinds of reference no
longer resolve:

  renameable   the concept exists under a different name. `zone` was an alias
               for Owner.Comm.State; `retention_success_rate` was identical to
               `persistency_rate`. These are rewritten in place.

  unsupported  the concept does not exist at all - smoker status, riders,
               medical underwriting, surrender reasons, retention outreach.
               A PS step asking for one cannot be answered, so the PS is
               tagged `unsupported_reason` rather than silently left to fail
               mid-execution. The retriever can then skip it, and the engine
               can explain the gap instead of returning an empty table.

Run:
    python builders/migrate_ps_to_v4_schema.py            # report only
    python builders/migrate_ps_to_v4_schema.py --apply    # write changes
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE, ".."))
for sub in ["src", "src/analytics", "src/engine", "src/retrieval", "src/llm"]:
    p = os.path.join(ROOT_DIR, sub)
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import analytics_core as ac

PS_DIR = os.path.join(ROOT_DIR, "data", "problem_statements")
PS_FILES = ["problem_statements.json", "business_problem_statements.json"]

# Concept survives under another name in the 5-table schema.
DIMENSION_RENAMES: Dict[str, str] = {
    "zone": "state",                    # Owner.Comm.State is the only geography
    "claim_type": "cause_of_death",     # every claim is a death claim
    "retention_status": "renewal_status",
}

METRIC_RENAMES: Dict[str, str] = dict(ac.METRIC_REPLACEMENTS)

# Concept has no counterpart. A step needing one cannot run.
UNSUPPORTED_DIMENSIONS = {
    d: r for d, r in ac.RETIRED_DIMENSIONS.items() if d not in DIMENSION_RENAMES
}

# Fields inside a PS record that may name a dimension or a metric.
DIM_KEYS = ("dimension", "dim", "row_dimension", "col_dimension", "dim_a", "dim_b")
METRIC_KEYS = ("metric", "value_metric")


def _walk_params(params: Any, dim_fix: Counter, met_fix: Counter,
                 blockers: set, apply: bool) -> Any:
    """Rewrite renameable names in one params dict; record what cannot be fixed."""
    if not isinstance(params, dict):
        return params

    for key in DIM_KEYS:
        val = params.get(key)
        if not isinstance(val, str) or val.startswith("$"):
            continue
        if val in DIMENSION_RENAMES:
            dim_fix[f"{val} -> {DIMENSION_RENAMES[val]}"] += 1
            if apply:
                params[key] = DIMENSION_RENAMES[val]
        elif val in UNSUPPORTED_DIMENSIONS:
            blockers.add(f"dimension `{val}`: {UNSUPPORTED_DIMENSIONS[val]}")
        elif val not in ac.DIMENSIONS:
            blockers.add(f"dimension `{val}`: not present in the dimension registry.")

    for key in METRIC_KEYS:
        val = params.get(key)
        if not isinstance(val, str) or val.startswith("$"):
            continue
        if val in METRIC_RENAMES:
            met_fix[f"{val} -> {METRIC_RENAMES[val]}"] += 1
            if apply:
                params[key] = METRIC_RENAMES[val]
        elif val in ac.RETIRED_METRICS:
            blockers.add(f"metric `{val}`: {ac.RETIRED_METRICS[val]}")
        elif val not in ac.METRICS and val not in ac.SHARE_METRICS:
            blockers.add(f"metric `{val}`: not present in the metric registry.")

    # `components` on composite_score carries its own metric names
    for comp in params.get("components") or []:
        if isinstance(comp, dict):
            m = comp.get("metric")
            if isinstance(m, str) and not m.startswith("$"):
                if m in METRIC_RENAMES:
                    met_fix[f"{m} -> {METRIC_RENAMES[m]}"] += 1
                    if apply:
                        comp["metric"] = METRIC_RENAMES[m]
                elif m in ac.RETIRED_METRICS:
                    blockers.add(f"metric `{m}`: {ac.RETIRED_METRICS[m]}")

    # `metrics` lists on multi_metric_table / segment_profile
    metrics_list = params.get("metrics")
    if isinstance(metrics_list, list):
        rebuilt: List[Any] = []
        for m in metrics_list:
            if isinstance(m, str) and m in METRIC_RENAMES:
                met_fix[f"{m} -> {METRIC_RENAMES[m]}"] += 1
                rebuilt.append(METRIC_RENAMES[m] if apply else m)
            else:
                if isinstance(m, str) and m in ac.RETIRED_METRICS:
                    blockers.add(f"metric `{m}`: {ac.RETIRED_METRICS[m]}")
                rebuilt.append(m)
        if apply:
            params["metrics"] = rebuilt

    # candidate dimension lists on driver_scan
    for key in ("candidate_dimensions", "dimensions"):
        dims = params.get(key)
        if isinstance(dims, list):
            rebuilt_d: List[Any] = []
            for d in dims:
                if isinstance(d, str) and d in DIMENSION_RENAMES:
                    dim_fix[f"{d} -> {DIMENSION_RENAMES[d]}"] += 1
                    rebuilt_d.append(DIMENSION_RENAMES[d] if apply else d)
                elif isinstance(d, str) and d in UNSUPPORTED_DIMENSIONS:
                    # a scan can simply drop an unavailable candidate
                    if not apply:
                        rebuilt_d.append(d)
                else:
                    rebuilt_d.append(d)
            if apply:
                params[key] = rebuilt_d

    return params


def migrate(apply: bool = False) -> Tuple[Counter, Counter, Dict[str, List[str]]]:
    dim_fix: Counter = Counter()
    met_fix: Counter = Counter()
    tagged: Dict[str, List[str]] = {}

    for fname in PS_FILES:
        path = os.path.join(PS_DIR, fname)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            records = json.load(f)

        for ps in records:
            pid = ps.get("problem_id", "?")
            blockers: set = set()
            for step in ps.get("analytical_steps") or []:
                _walk_params(step.get("params"), dim_fix, met_fix, blockers, apply)

            # Also clean the declarative column hints so retrieval stops scoring
            # on vocabulary the data does not have.
            if apply and isinstance(ps.get("required_metrics"), list):
                ps["required_metrics"] = [METRIC_RENAMES.get(m, m)
                                          for m in ps["required_metrics"]]

            if blockers:
                tagged[pid] = sorted(blockers)
                if apply:
                    ps["unsupported_reason"] = " ".join(sorted(blockers))
                    ps["grounded_in"] = "Talk_to_Data_PoC_Documentation_Metadata_V4"
            elif apply:
                ps.pop("unsupported_reason", None)
                ps["grounded_in"] = "Talk_to_Data_PoC_Documentation_Metadata_V4"

        if apply:
            shutil.copyfile(path, path + ".pre_v4.bak")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)

    return dim_fix, met_fix, tagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the changes (a .pre_v4.bak copy is kept)")
    args = ap.parse_args()

    dim_fix, met_fix, tagged = migrate(apply=args.apply)

    print("RENAMES" + (" APPLIED" if args.apply else " (dry run)"))
    for label, table in (("dimensions", dim_fix), ("metrics", met_fix)):
        print(f"  {label}:")
        if not table:
            print("      none")
        for name, n in table.most_common():
            print(f"      {name:44s} {n:4d} step(s)")

    print(f"\nPS tagged unsupported: {len(tagged)}")
    reasons: Counter = Counter()
    for pid, blocks in tagged.items():
        for b in blocks:
            reasons[b.split(":")[0]] += 1
    for reason, n in reasons.most_common(12):
        print(f"      {reason:46s} {n:4d} PS")

    if not args.apply:
        print("\n(dry run - re-run with --apply to write)")


if __name__ == "__main__":
    main()
