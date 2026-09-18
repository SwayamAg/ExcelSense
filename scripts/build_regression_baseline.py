"""Record what the system currently produces, so a change that alters it is visible.

Specification section 10: detect PS, schema and formula regressions. A regression
test needs something to regress against, and that baseline has to be produced by
running the system rather than written by hand - a hand-written expectation is a
second implementation, and it is the one that will be wrong.

The baseline is committed. Regenerating it is therefore a reviewable act: the diff
shows exactly which figures moved, and an unexplained movement is the finding.

Each entry carries two kinds of record:

  digest    a hash over the whole result, which catches any change at all
  headline  the figures in plain text, so a failure says what moved rather than
            that two hashes differ

Run:  python scripts/build_regression_baseline.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.analysis.advanced_patterns  # noqa: F401,E402  registers extra patterns
from src.data import load_dataset  # noqa: E402
from src.logic.formula_engine import get_engine  # noqa: E402
from src.ps.executor import PSExecutor  # noqa: E402
from src.ps.registry import get_registry  # noqa: E402

FORMULAS = get_engine()

BASELINE_PATH = ROOT / "tests" / "regression" / "baseline.json"
BASELINE_VERSION = "1.0.0"

# Problem statements that need a parameter to run at all. The values are real
# identifiers from the data; a baseline built on a fabricated one would be
# recording a refusal, not a result.
REQUIRED_PARAMETERS = {
    "PS-031": {"app_id": "15011000000"},
    "PS-069": {"agent_id": "977337"},
}

# Figures are compared after rounding. Floating-point arithmetic over a different
# pandas or numpy build can differ in the last bits without anything being wrong;
# six decimal places is far finer than any figure this system reports and far
# coarser than that noise.
PRECISION = 6


def _round(value: Any) -> Any:
    """Round a value for comparison, leaving anything non-numeric alone."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, PRECISION)
    if isinstance(value, dict):
        return {k: _round(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    return value


def _digest(payload: Any) -> str:
    text = json.dumps(_round(payload), sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _scalar_aggregates(aggregates: Dict[str, Any]) -> Dict[str, Any]:
    """The headline figures: the scalars an answer would actually quote."""
    return {k: _round(v) for k, v in sorted(aggregates.items())
            if isinstance(v, (int, float, str)) and not isinstance(v, bool)}


def ps_entry(evidence) -> Dict[str, Any]:
    """The recorded shape of one PS's result."""
    return {
        "sufficient": evidence.is_sufficient,
        "insufficient_reason": (evidence.insufficient.reason
                                if evidence.insufficient else None),
        "population": evidence.population,
        "population_size": evidence.population_size,
        "result_rows": len(evidence.result_table),
        "result_columns": list(evidence.result_columns),
        "record_count": len(evidence.retrieved_records),
        "formulas_applied": sorted(evidence.formulas_applied),
        "business_rules_applied": sorted(
            r.get("id", "") for r in evidence.business_rules_applied),
        "tables_used": sorted(evidence.tables_used),
        "column_count": len(evidence.columns_used),
        "data_quality_notes": sorted(evidence.data_quality_notes),
        "limitation_count": len(evidence.limitations),
        "unsupported_count": len(evidence.unsupported_conclusions),
        "expected_answer_type": evidence.expected_answer_type,
        "headline": _scalar_aggregates(evidence.aggregates),
        "table_digest": _digest(evidence.result_table),
        "aggregate_digest": _digest(evidence.aggregates),
    }


def build_ps_baseline() -> Dict[str, Any]:
    dataset = load_dataset()
    registry = get_registry()
    # No cache: the baseline must record computations, not replays of them.
    executor = PSExecutor(dataset=dataset, registry=registry, use_cache=False)
    entries = {}
    for ps in registry.executable():
        evidence = executor.execute_fresh(ps.id, REQUIRED_PARAMETERS.get(ps.id))
        entries[ps.id] = ps_entry(evidence)
    return {
        "catalogue_version": registry.version,
        "problem_statement_count": len(registry),
        "parameters_used": REQUIRED_PARAMETERS,
        "entries": entries,
    }


def build_schema_baseline() -> Dict[str, Any]:
    """The shape of the data itself, so a changed CSV cannot pass unnoticed."""
    dataset = load_dataset()
    contract = dataset.contract
    tables = {}
    for key in sorted(dataset.table_keys):
        table = dataset.table(key)
        spec = contract.table(key)
        frame = table.frame
        tables[key] = {
            "file": spec.file,
            "rows": len(frame),
            "columns": list(frame.columns),
            "column_count": len(frame.columns),
            "primary_key": spec.primary_key,
            "dtypes": {c: str(frame[c].dtype) for c in frame.columns},
            "null_counts": {c: int(frame[c].isna().sum()) for c in frame.columns},
        }
    return {
        "contract_version": contract.version,
        "hub_table": contract.hub_table,
        "tables": tables,
        "join_count": len(contract.joins),
        "finding_ids": sorted(f["id"] for f in contract.data_quality_findings),
        "unanswerable_domain_count": len(contract.unanswerable_domains),
        "load_warnings": sorted(dataset.warnings),
    }


def build_formula_baseline() -> Dict[str, Any]:
    """Every formula's declared worked examples, with the value they produce now."""
    formulas: Dict[str, Any] = {}
    for formula_id in FORMULAS.formula_ids:
        formula = FORMULAS.get(formula_id)
        examples: List[Dict[str, Any]] = []
        for example in formula.examples:
            inputs = dict(example["inputs"])
            result = FORMULAS.compute(formula_id, **inputs)
            examples.append({
                "inputs": _round(inputs),
                "expected": _round(example["expected"]),
                "produced": _round(result.value),
            })
        formulas[formula_id] = {
            "name": formula.name,
            "expression": formula.expression,
            "null_handling": formula.null_handling,
            "zero_denominator": formula.zero_denominator,
            "rounding": formula.rounding,
            "examples": examples,
        }
    return {"formula_count": len(formulas), "formulas": formulas}


def build_routing_baseline() -> Dict[str, Any]:
    """Where every catalogue example query routes today.

    Accuracy alone is too coarse a regression signal: two questions can swap
    which PS they reach and leave the percentage unchanged. This records the
    decision itself, so a swap shows up as a named diff.
    """
    from src.router import QueryRouter

    dataset = load_dataset()
    registry = get_registry()
    router = QueryRouter(dataset=dataset, registry=registry)

    routes: Dict[str, Any] = {}
    correct = 0
    total = 0
    for ps in registry.all():
        for question in ps.record.get("example_queries", []):
            decision = router.route(question)
            total += 1
            hit = decision.outcome == "routed" and decision.problem_id == ps.id
            correct += 1 if hit else 0
            routes[question] = {
                "expected": ps.id,
                "outcome": decision.outcome,
                "problem_id": decision.problem_id,
                "confidence": round(decision.confidence, 4),
                "correct": hit,
            }
    return {
        "example_query_count": total,
        "correct": correct,
        "accuracy_pct": round(100.0 * correct / total, 2) if total else 0.0,
        "routes": routes,
    }


def build() -> Dict[str, Any]:
    return {
        "baseline_version": BASELINE_VERSION,
        "note": ("Generated by scripts/build_regression_baseline.py. Regenerate "
                 "deliberately and review the diff: a figure that moves without "
                 "an explanation is the regression this file exists to catch."),
        "schema": build_schema_baseline(),
        "formulas": build_formula_baseline(),
        "problem_statements": build_ps_baseline(),
        "routing": build_routing_baseline(),
    }


def main() -> int:
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = build()
    BASELINE_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n",
        encoding="utf-8")
    entries = payload["problem_statements"]["entries"]
    sufficient = sum(1 for e in entries.values() if e["sufficient"])
    print("wrote {} ({:,} bytes)".format(
        BASELINE_PATH.relative_to(ROOT), BASELINE_PATH.stat().st_size))
    print("  {} problem statements, {} sufficient".format(len(entries), sufficient))
    print("  {} tables, {} formulas".format(
        len(payload["schema"]["tables"]), payload["formulas"]["formula_count"]))
    routing = payload["routing"]
    print("  {} example queries, {} routed correctly ({}%)".format(
        routing["example_query_count"], routing["correct"], routing["accuracy_pct"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
