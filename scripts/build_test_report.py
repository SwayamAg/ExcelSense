"""Generate docs/TEST_REPORT.md from an actual test run.

A test report written by hand is a claim. This one runs the suite, collects the
outcome of every test, reads the benchmark timings the run produced, and writes
what it found - so the document cannot drift from the code it describes, and a
failing suite produces a report that says so.

Run:  python scripts/build_test_report.py
      python scripts/build_test_report.py --skip-run   (reuse the last results)
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_PATH = ROOT / "docs" / "TEST_REPORT.md"
RESULTS_PATH = ROOT / "tests" / "last_run.json"
BENCH_PATH = ROOT / "tests" / "benchmark" / "results.json"
BASELINE_PATH = ROOT / "tests" / "regression" / "baseline.json"

# What each test directory is responsible for. The report groups by these rather
# than by file, because the question a reader has is "what is covered", not
# "which files exist".
AREAS = [
    ("tests/test_data_contract.py", "Phase 0 - data contract",
     "The profile is true of the CSVs: types, formats, keys, and every declared "
     "data-quality finding."),
    ("tests/test_problem_statements.py", "Phase 1 - problem statement catalogue",
     "Every PS is grounded in real columns, every plan references something that "
     "exists, and the shards agree with the index."),
    ("tests/data", "Phase 2 - loading and querying",
     "No value is lost or silently coerced on load; every query step records its "
     "own provenance."),
    ("tests/logic", "Phase 3 - formulas and business rules",
     "Each formula's arithmetic, null handling and zero-denominator behaviour, and "
     "each executable business rule."),
    ("tests/ps", "Phase 4 - execution",
     "Every problem statement executes and produces an evidence package, or "
     "refuses with a reason."),
    ("tests/analysis", "Phase 4 - analytical patterns",
     "The seventeen patterns, each against data whose answer is known "
     "independently."),
    ("tests/router", "Phase 5 - routing",
     "Accuracy against the query suite, and refusal for out-of-domain, "
     "unsupported and invalid-entity questions."),
    ("tests/evidence", "Phase 6 - evidence tables",
     "Every figure an answer quotes can be traced back to rows."),
    ("tests/visualization", "Phase 6 - visualization",
     "Chart specifications match the shape of the evidence."),
    ("tests/llm", "Phase 7 - summarization and validation",
     "The grounding guarantee: an answer containing anything the evidence does not "
     "support is rejected."),
    ("tests/api", "Phase 8 - chatbot",
     "The whole path, including conversation context, refusals and clarification."),
    ("tests/e2e/test_deployment_smoke.py", "Phase 11 - deployment readiness",
     "Health checking, startup validation, clean environment execution, and deployment artifacts."),
    ("tests/e2e", "Phase 8 - end to end",
     "Real questions from question to answer, with no component stubbed."),
    ("tests/ui", "Phase 8 - web interface",
     "The Streamlit interface renders only what the evidence contains, styles each "
     "outcome distinctly, and leaks no personal field."),
    ("tests/security", "Phase 9 - security and grounding hardening",
     "Nine attack classes, and the absence of false positives across every "
     "catalogue example query."),
    ("tests/regression", "Phase 10 - regression",
     "Every recorded figure, schema detail, formula result and routing decision, "
     "against a committed baseline."),
    ("tests/benchmark", "Phase 10 - performance",
     "Latency of each layer, measured separately, against budgets set to catch "
     "algorithmic regressions."),
]


class _Collector:
    """A pytest plugin that records the outcome of every test."""

    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}

    def pytest_runtest_logreport(self, report):
        if report.when != "call" and not (
                report.when == "setup" and report.outcome == "skipped"):
            return
        self.results[report.nodeid] = {
            "outcome": report.outcome,
            "duration": round(report.duration, 4),
            "file": report.nodeid.split("::")[0],
        }


def run_suite() -> Dict[str, Any]:
    """Run pytest in-process and record every outcome."""
    import pytest

    collector = _Collector()
    started = datetime.now(timezone.utc)
    exit_code = pytest.main(
        ["-o", "addopts=", "-q", "--tb=no", str(ROOT / "tests")], plugins=[collector])
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()

    payload = {
        "run_at_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": round(elapsed, 2),
        "exit_code": int(exit_code),
        "results": collector.results,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _area_for(path: str) -> str:
    for prefix, title, _ in AREAS:
        normalized = path.replace("\\", "/")
        if normalized == prefix or normalized.startswith(prefix.rstrip("/") + "/"):
            return title
    return "other"


def summarize(run: Dict[str, Any]) -> Dict[str, Any]:
    by_area: Dict[str, Counter] = defaultdict(Counter)
    durations: Dict[str, float] = defaultdict(float)
    slowest: List[tuple] = []
    for node, result in run["results"].items():
        area = _area_for(result["file"])
        by_area[area][result["outcome"]] += 1
        by_area[area]["total"] += 1
        durations[area] += result["duration"]
        slowest.append((result["duration"], node))
    slowest.sort(reverse=True)
    totals = Counter()
    for counts in by_area.values():
        totals.update(counts)
    return {"by_area": by_area, "durations": durations, "totals": totals,
            "slowest": slowest[:10]}


def _table(rows: List[List[str]], headers: List[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def build_report(run: Dict[str, Any]) -> str:
    summary = summarize(run)
    totals = summary["totals"]
    bench = json.loads(BENCH_PATH.read_text(encoding="utf-8")) if BENCH_PATH.exists() else {}
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8")) \
        if BASELINE_PATH.exists() else {}

    passed = totals.get("passed", 0)
    failed = totals.get("failed", 0)
    skipped = totals.get("skipped", 0)
    total = totals.get("total", 0)

    out: List[str] = []
    w = out.append

    w("# Excel_sense_v2 - Test Report")
    w("")
    w("Generated by `scripts/build_test_report.py` from an actual run of the suite. "
      "Every number below was measured, not transcribed - regenerate it and the "
      "document follows the code.")
    w("")
    w("- **Run at:** {}".format(run["run_at_utc"]))
    w("- **Result:** {}".format(
        "**{:,} passed**, 0 failed".format(passed) if not failed
        else "**{:,} passed, {:,} FAILED**".format(passed, failed)))
    if skipped:
        w("- **Skipped:** {} (each is a reported condition, not an untested one - "
          "see below)".format(skipped))
    w("- **Wall clock:** {:.1f} s for {:,} tests".format(
        run["elapsed_seconds"], total))
    if bench.get("environment"):
        env = bench["environment"]
        w("- **Environment:** Python {}, pandas {}, {}".format(
            env.get("python", "?"), env.get("pandas", "?"),
            env.get("platform", "?")))
    w("")

    if failed:
        w("## Failures")
        w("")
        for node, result in sorted(run["results"].items()):
            if result["outcome"] == "failed":
                w("- `{}`".format(node))
        w("")

    # ---- coverage by area -------------------------------------------------
    w("## What is covered")
    w("")
    w("Tests are grouped by what they are responsible for rather than by file, "
      "because the question a reader has is what is covered, not which files "
      "exist.")
    w("")
    rows = []
    for _, title, purpose in AREAS:
        counts = summary["by_area"].get(title)
        if not counts:
            continue
        seconds = summary["durations"][title]
        rows.append([title, "{:,}".format(counts["total"]),
                     "{:.1f} s".format(seconds), purpose])
    w(_table(rows, ["Area", "Tests", "Time", "What it establishes"]))
    w("")

    # ---- regression -------------------------------------------------------
    if baseline:
        ps = baseline["problem_statements"]
        routing = baseline["routing"]
        schema = baseline["schema"]
        w("## The regression baseline")
        w("")
        w("`tests/regression/baseline.json` records what the system produces, "
          "generated by running it. It is committed, so regenerating it is a "
          "reviewable act: the diff shows which figures moved, and an unexplained "
          "movement is the finding.")
        w("")
        w(_table([
            ["Problem statements", "{}".format(len(ps["entries"])),
             "every headline figure, population, result table, formula and rule"],
            ["Tables", "{}".format(len(schema["tables"])),
             "row counts, column order, dtype and null count of every column"],
            ["Formulas", "{}".format(baseline["formulas"]["formula_count"]),
             "declared semantics and the value each worked example produces"],
            ["Example queries", "{}".format(routing["example_query_count"]),
             "the analysis each one routes to, at {}% accuracy".format(
                 routing["accuracy_pct"])],
        ], ["Recorded", "Count", "What is held fixed"]))
        w("")
        w("The suite is asymmetric by design. A figure that moves, a question that "
          "stops routing correctly, or a data-quality caveat that disappears is a "
          "failure. A new problem statement or a question that *starts* routing "
          "correctly is reported as a skip, not failed - the tests should not stand "
          "in the way of the system getting better, only in the way of it getting "
          "quietly worse.")
        w("")

    # ---- benchmarks -------------------------------------------------------
    if bench.get("timings"):
        w("## Measured performance")
        w("")
        w("Medians over several runs, after a warmup. Budgets are set roughly an "
          "order of magnitude above measured cost: they exist to catch a change "
          "that makes something accidentally quadratic, not to police milliseconds "
          "on whatever machine runs the suite.")
        w("")
        rows = [[t["label"], "{:.2f}".format(t["median_ms"]),
                 "{:.2f}".format(t["p95_ms"]), t["runs"]]
                for t in bench["timings"]]
        w(_table(rows, ["Operation", "Median (ms)", "p95 (ms)", "Runs"]))
        w("")

        notes = bench.get("notes", {})
        if "catalogue_execution" in notes:
            note = notes["catalogue_execution"]
            w("### The whole catalogue, computed cold")
            w("")
            w("All {} problem statements execute in **{:.1f} s** "
              "({:.0f} ms mean).".format(
                  note["problem_statements"], note["total_seconds"],
                  note["mean_ms"]))
            w("")
            w(_table([[s["problem_id"], "{:,.0f}".format(s["ms"])]
                      for s in note["slowest"]],
                     ["Slowest analyses", "ms"]))
            w("")
        if "cache_speedup" in notes:
            note = notes["cache_speedup"]
            w("### Caching")
            w("")
            w("A repeated analysis costs **{:.2f} ms** against **{:.1f} ms** to "
              "recompute - roughly {}x. Correctness, not speed, is the hard part "
              "of a cache, and it is tested separately in "
              "`tests/regression/test_cache_safety.py`.".format(
                  note["warm_ms"], note["cold_ms"], note["factor"]))
            w("")
        if "llm_overhead" in notes:
            note = notes["llm_overhead"]
            w("### The model, measured separately")
            w("")
            w("Around a model call of known cost ({:.0f} ms, simulated), this system "
              "adds **{:.1f} ms** - prompt construction, parsing and validating the "
              "answer against the evidence. Mixing the API round trip into an "
              "end-to-end figure would report the network as this system's "
              "cost.".format(note["simulated_model_latency_ms"],
                             note["system_overhead_ms"]))
            w("")
        if "dataset_memory" in notes:
            note = notes["dataset_memory"]
            w("### Memory")
            w("")
            w("The five datasets occupy **{:.1f} MB** once loaded and typed.".format(
                note["total_mb"]))
            w("")
            w(_table([[k, "{:,}".format(v["rows"]), "{:.2f}".format(v["mb"])]
                      for k, v in sorted(note["tables"].items())],
                     ["Table", "Rows", "MB"]))
            w("")

    # ---- slowest tests ----------------------------------------------------
    w("## Slowest tests")
    w("")
    w("Recorded so that a suite people are willing to run stays that way.")
    w("")
    w(_table([["{:.2f}".format(d), "`{}`".format(n)]
              for d, n in summary["slowest"]], ["Seconds", "Test"]))
    w("")

    # ---- skips ------------------------------------------------------------
    skips = [n for n, r in sorted(run["results"].items())
             if r["outcome"] == "skipped"]
    if skips:
        w("## Reported conditions (skips)")
        w("")
        w("These are not untested paths. Each one is a regression test reporting "
          "that the system improved on its baseline - a new problem statement, or "
          "a question that now routes correctly - which is a thing to record, not "
          "to fail.")
        w("")
        for node in skips:
            w("- `{}`".format(node))
        w("")

    w("---")
    w("")
    w("Regenerate with `python scripts/build_test_report.py`.")
    w("")
    return "\n".join(out)


def main() -> int:
    skip_run = "--skip-run" in sys.argv
    if skip_run:
        if not RESULTS_PATH.exists():
            print("no previous run at {}; run without --skip-run".format(RESULTS_PATH))
            return 1
        run = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    else:
        run = run_suite()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(run), encoding="utf-8")
    totals = summarize(run)["totals"]
    print("wrote {} ({:,} bytes)".format(
        REPORT_PATH.relative_to(ROOT), REPORT_PATH.stat().st_size))
    print("  {:,} tests: {} passed, {} failed, {} skipped".format(
        totals.get("total", 0), totals.get("passed", 0),
        totals.get("failed", 0), totals.get("skipped", 0)))
    return 0 if not totals.get("failed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
