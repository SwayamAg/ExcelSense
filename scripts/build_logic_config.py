"""Phase 3 - emit the inspectable formula and business-rule configuration.

The implementations are the source of truth; these JSON files are generated from
them so the declared semantics can never drift from the code that executes them.
Also emits the PS -> formula / rule mapping, which is what the Phase 3 gate is
actually about: every formula an active PS references must exist and be tested.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.logic.formula_engine import ENGINE_VERSION as FORMULA_VERSION  # noqa: E402
from src.logic.formula_engine import get_engine  # noqa: E402
from src.logic.rule_engine import ENGINE_VERSION as RULE_VERSION  # noqa: E402
from src.logic.rule_engine import get_rule_engine  # noqa: E402

# Rules that apply to a PS because of what it does, rather than being named by it.
# Keyed by a predicate over the PS record.
RULE_TRIGGERS = [
    ("BR-001", lambda ps: "persistency" in ps["relevant_tables"]),
    ("BR-002", lambda ps: "small_sample_guard" in ps["required_operations"]),
    ("BR-003", lambda ps: "claims" in ps["relevant_tables"]
     and any(m in ps["required_metrics"] for m in
             ("settlement_lag_days", "payout_ratio", "total_amount_paid", "settlement_rate"))),
    ("BR-004", lambda ps: "agent_first_year_persistency" in ps["required_metrics"]),
    ("BR-005", lambda ps: "date_diff" in ps["required_operations"]),
    ("BR-006", lambda ps: "owner" in ps["relevant_tables"]),
    ("BR-007", lambda ps: "outliers" in ps["required_operations"]),
    ("BR-008", lambda ps: "claims" in ps["relevant_tables"] and "ratio" in ps["required_operations"]),
    ("BR-009", lambda ps: any("POLICY.STATUS" in c for c in ps["relevant_columns"])),
    ("BR-010", lambda ps: "days_to_death_from_inforce" in ps["required_metrics"]),
    ("BR-011", lambda ps: bool(ps["formula_ids"])),
    ("BR-012", lambda ps: "agent_quality_composite" in ps["required_metrics"]),
    ("BR-013", lambda ps: len(ps["relevant_tables"]) > 1),
    ("BR-014", lambda ps: ps["analysis_tier"] == "diagnostic"),
    ("BR-015", lambda ps: any(c.endswith(".AppID") for c in ps["relevant_columns"])),
]


def main() -> None:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    formulas = get_engine()
    rules = get_rule_engine()

    catalogue_path = ROOT / "config" / "problem_statements.json"
    catalogue = json.loads(catalogue_path.read_text(encoding="utf-8"))
    problems = catalogue["problem_statements"]

    # --- formulas ---------------------------------------------------------
    usage = Counter(f for ps in problems for f in ps["formula_ids"])
    used_by = defaultdict(list)
    for ps in problems:
        for f in ps["formula_ids"]:
            used_by[f].append(ps["problem_id"])

    formula_payload = {
        "config_version": "1.0.0",
        "engine_version": FORMULA_VERSION,
        "generated_at_utc": generated,
        "generated_by": "scripts/build_logic_config.py",
        "source_of_truth": "src/logic/formula_engine.py",
        "principle": ("The application determines the facts and the formula engine determines "
                      "the calculations. The LLM never computes an authoritative value."),
        "formula_count": len(formulas.formula_ids),
        "formulas": [
            {**f, "used_by_ps_count": usage.get(f["id"], 0),
             "used_by": used_by.get(f["id"], [])}
            for f in formulas.describe()
        ],
    }
    (ROOT / "config" / "formulas.json").write_text(
        json.dumps(formula_payload, indent=2), encoding="utf-8")

    # --- business rules ---------------------------------------------------
    rule_applies_to = defaultdict(list)
    for ps in problems:
        for rule_id, predicate in RULE_TRIGGERS:
            if predicate(ps):
                rule_applies_to[rule_id].append(ps["problem_id"])

    rule_payload = {
        "config_version": "1.0.0",
        "engine_version": RULE_VERSION,
        "generated_at_utc": generated,
        "generated_by": "scripts/build_logic_config.py",
        "source_of_truth": "src/logic/rule_engine.py",
        "principle": ("A business rule decides which rows count and how a result must be "
                      "qualified. Every rule states its rationale so it can be challenged."),
        "rule_count": len(rules.rule_ids),
        "rules": [
            {**r, "applies_to_ps_count": len(rule_applies_to.get(r["id"], [])),
             "applies_to_ps": rule_applies_to.get(r["id"], [])}
            for r in rules.describe()
        ],
    }
    (ROOT / "config" / "business_rules.json").write_text(
        json.dumps(rule_payload, indent=2), encoding="utf-8")

    # --- PS mapping -------------------------------------------------------
    mapping = {
        "config_version": "1.0.0",
        "generated_at_utc": generated,
        "generated_by": "scripts/build_logic_config.py",
        "description": ("Maps every Problem Statement to the formulas and business rules its "
                        "execution must use. The Phase 3 gate requires every referenced "
                        "formula to have a deterministic implementation."),
        "problem_statement_count": len(problems),
        "mappings": [
            {
                "problem_id": ps["problem_id"],
                "formula_ids": ps["formula_ids"],
                "business_rule_ids": [rid for rid, predicate in RULE_TRIGGERS if predicate(ps)],
            }
            for ps in problems
        ],
    }
    (ROOT / "config" / "ps_logic_mapping.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8")

    # --- report -----------------------------------------------------------
    unimplemented = sorted(set(usage) - set(formulas.formula_ids))
    unused = sorted(set(formulas.formula_ids) - set(usage))
    print("formulas registered : {}".format(len(formulas.formula_ids)))
    print("formulas referenced : {}".format(len(usage)))
    print("business rules      : {}".format(len(rules.rule_ids)))
    print("PS mappings         : {}".format(len(problems)))
    if unimplemented:
        print("MISSING IMPLEMENTATIONS: {}".format(unimplemented))
        raise SystemExit(1)
    if unused:
        print("registered but unused by any PS: {}".format(unused))
    failures = formulas.verify_examples()
    if failures:
        print("EXAMPLE VERIFICATION FAILED:")
        for f in failures:
            print("  - {}".format(f))
        raise SystemExit(1)
    print("all declared examples verify")


if __name__ == "__main__":
    main()
