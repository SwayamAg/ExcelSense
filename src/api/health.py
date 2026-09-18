"""Application Health and Startup Validation.

Provides programmatic and CLI validation for all core platform components:
- Dataset integrity (5 CSVs present and readable)
- Schema contract and catalogue readiness (103 PSs)
- Formula and business rule engine initialization (18 formulas, 15 rules)
- Query router and analytical engine readiness
- LLM status (API mode or deterministic composer fallback)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.data.contract import CONTRACT_PATH, DataContract, TABLE_FILES
from src.llm.client import default_client
from src.logic.formula_engine import FormulaEngine
from src.logic.rule_engine import RuleEngine
from src.ps.registry import get_registry
from src.router import QueryRouter


def check_dataset_health(data_dir: Path) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Check dataset files existence and readability."""
    issues = []
    file_stats = {}
    
    if not data_dir.exists() or not data_dir.is_dir():
        issues.append(f"Data directory not found: {data_dir}")
        return False, issues, {}

    all_present = True
    for key, filename in TABLE_FILES.items():
        filepath = data_dir / filename
        if not filepath.exists():
            issues.append(f"Missing required CSV: {filename} ({key})")
            all_present = False
        elif not filepath.is_file() or filepath.stat().st_size == 0:
            issues.append(f"Empty or unreadable CSV: {filename}")
            all_present = False
        else:
            file_stats[key] = {
                "file": filename,
                "size_bytes": filepath.stat().st_size,
                "status": "ready"
            }
            
    return all_present, issues, file_stats


def check_contract_health() -> Tuple[bool, List[str], Dict[str, Any]]:
    """Verify data contract file validity."""
    issues = []
    try:
        contract = DataContract.load()
        tables = contract.tables
        if len(tables) < 5:
            issues.append(f"Contract contains only {len(tables)} tables (expected 5)")
            return False, issues, {}
        return True, issues, {"tables_count": len(tables), "version": contract.version, "status": "valid"}
    except Exception as exc:
        issues.append(f"Failed to load data contract: {exc}")
        return False, issues, {}


def check_catalogue_health() -> Tuple[bool, List[str], Dict[str, Any]]:
    """Verify problem statements catalogue and registries."""
    issues = []
    try:
        registry = get_registry()
        ps_count = len(registry)
        formulas = FormulaEngine().formula_ids
        rules = RuleEngine().rule_ids
        
        if ps_count < 100:
            issues.append(f"Catalogue contains {ps_count} PSs (expected >= 100)")
            return False, issues, {}
            
        return True, issues, {
            "ps_count": ps_count,
            "formulas_count": len(formulas),
            "rules_count": len(rules),
            "status": "ready"
        }
    except Exception as exc:
        issues.append(f"Failed to load PS catalogue / registries: {exc}")
        return False, issues, {}


def check_router_health() -> Tuple[bool, List[str], Dict[str, Any]]:
    """Verify router initialization."""
    issues = []
    try:
        registry = get_registry()
        router = QueryRouter(registry=registry)
        # Test routing a basic sample question
        decision = router.route("What is the overall portfolio lapse rate?")
        return True, issues, {
            "sample_route_outcome": decision.outcome,
            "sample_matched_ps": decision.problem_id,
            "status": "ready"
        }
    except Exception as exc:
        issues.append(f"Failed to initialize or run router: {exc}")
        return False, issues, {}


def check_llm_health() -> Tuple[bool, List[str], Dict[str, Any]]:
    """Verify LLM client status (API or Deterministic fallback)."""
    issues = []
    try:
        client = default_client()
        return True, issues, {
            "provider": client.provider,
            "model": client.model,
            "is_available": client.available,
            "mode": "live_llm" if client.available and client.provider != "deterministic" else "deterministic_fallback"
        }
    except Exception as exc:
        issues.append(f"LLM client check error: {exc}")
        return False, issues, {}


def run_health_check(data_dir: Path | None = None) -> Dict[str, Any]:
    """Execute all health checks and return aggregated status."""
    start_time = time.perf_counter()
    
    if data_dir is None:
        root = Path(__file__).resolve().parents[2]
        data_dir = root / "data" / "csv"

    dataset_ok, dataset_issues, dataset_info = check_dataset_health(data_dir)
    contract_ok, contract_issues, contract_info = check_contract_health()
    catalogue_ok, catalogue_issues, catalogue_info = check_catalogue_health()
    router_ok, router_issues, router_info = check_router_health()
    llm_ok, llm_issues, llm_info = check_llm_health()

    all_issues = dataset_issues + contract_issues + catalogue_issues + router_issues + llm_issues
    healthy = dataset_ok and contract_ok and catalogue_ok and router_ok and llm_ok

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "status": "HEALTHY" if healthy else "UNHEALTHY",
        "healthy": healthy,
        "elapsed_ms": elapsed_ms,
        "components": {
            "dataset": {"healthy": dataset_ok, "info": dataset_info, "issues": dataset_issues},
            "contract": {"healthy": contract_ok, "info": contract_info, "issues": contract_issues},
            "catalogue": {"healthy": catalogue_ok, "info": catalogue_info, "issues": catalogue_issues},
            "router": {"healthy": router_ok, "info": router_info, "issues": router_issues},
            "llm": {"healthy": llm_ok, "info": llm_info, "issues": llm_issues},
        },
        "issues": all_issues
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Excel_sense_v2 System Health Check")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to data/csv directory")
    parser.add_argument("--json", action="store_true", help="Output JSON result format")
    parser.add_argument("--quiet", action="store_true", help="Exit code only with minimal output")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir) if args.data_dir else None
    report = run_health_check(data_dir)

    if args.quiet:
        return 0 if report["healthy"] else 1

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        # Encoding safe indicators
        can_utf8 = False
        try:
            can_utf8 = (sys.stdout.encoding or "").lower().startswith("utf")
        except Exception:
            can_utf8 = False
            
        pass_sym = "✅" if can_utf8 else "[OK]"
        fail_sym = "❌" if can_utf8 else "[FAIL]"
        warn_sym = "⚠️ " if can_utf8 else "[WARN]"

        status_symbol = pass_sym if report["healthy"] else fail_sym
        print(f"\n{status_symbol} System Status: {report['status']} ({report['elapsed_ms']} ms)")
        print("-" * 50)
        for comp, res in report["components"].items():
            sym = pass_sym if res["healthy"] else fail_sym
            info_str = ", ".join(f"{k}={v}" for k, v in res.get("info", {}).items())
            print(f"  {sym} {comp.capitalize():<12} {info_str}")
        
        if report["issues"]:
            print("\nIssues detected:")
            for issue in report["issues"]:
                print(f"  - {warn_sym} {issue}")

    return 0 if report["healthy"] else 1


if __name__ == "__main__":
    sys.exit(main())
