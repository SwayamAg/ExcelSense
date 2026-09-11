"""
validate_grounding.py
=====================
Checks that everything the system can say is traceable to the five production
CSVs described in `Talk_to_Data_PoC_Documentation_Metadata_V4.docx`.

Run directly for a report:
    python tests/validate_grounding.py

Run as a test to gate regressions:
    python -m unittest tests.validate_grounding

Four checks:

  1. schema        every raw CSV carries the columns the metadata document
                   declares, and the grain matches (Persistency_Details is one
                   row per AppID, which the README previously got wrong).
  2. metrics       every registered metric computes against a real column, with
                   no magic constants and no tautological denominators.
  3. dimensions    every registered dimension resolves to a column that exists
                   and has more than one distinct value.
  4. ps library    every Problem Statement references metrics and dimensions
                   that still exist.

The PS check reports rather than fails, because the library is large and the
count is the number worth watching over time.
"""

from __future__ import annotations

import json
import os
import re
import sys
import unittest

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE, ".."))
for sub in ["src", "src/analytics", "src/engine", "src/retrieval", "src/llm", "ui"]:
    p = os.path.join(ROOT_DIR, sub)
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd

import analytics_core as ac

CSV_DIR = os.path.join(ROOT_DIR, "data", "csv")

# A placeholder AppID carried by one Owner_Details row and one
# Persistency_Details row, present in no policy. The metadata document flags
# owner-file AppID coverage as a known quality caveat.
PHANTOM_APPID = 15000000000.0

# Columns each file must carry, per the metadata document (section 3.1 - 3.5).
DECLARED_SCHEMA = {
    "Policy_Details.csv": (
        25000,
        ["AppID", "SALES_ID", "Branch_No", "Prod_No", "POLICY.STATUS", "CHANNEL",
         "Login.Date", "Inforce.Date", "Inforce.month", "Trad.ULIP", "Plan.Name",
         "Mode", "PREMIUM.PAYING.TERM", "Policy.Term", "Sum.Assured", "APE"]),
    "Owner_Details.csv": (
        25000,
        ["AppID", "ClientID", "Owner.Given.Name", "Owner.Gender", "Owner.Age",
         "Owner.Education", "Owner.Occupation", "Owner.Earned.Income",
         "Owner.Comm.City.Name", "Owner.Comm.State", "Owner.Comm.Country",
         "Owner.Comm.Postal.Code"]),
    "Sales_Details.csv": (
        1956,
        ["SALES_ID", "Surrender Rate", "Early Claim Rate",
         "PIVC Number Mismatch Rate", "PIVC Concern Raised Rate",
         "RCU Rejections Rate", "First Year Persistency Rate", "Type"]),
    "Claims_Details.csv": (
        134,
        ["AppID", "Claim_No", "Date.of.Death", "Cause.of.Death", "Claim.Amount",
         "Date.of.Intimation", "Status.of.Claim", "Amount.Paid", "Settlement.Date"]),
    "Persistency_Details.csv": (
        24542,
        ["AppID", "SALES_ID", "POLICY.STATUS", "CHANNEL", "Inforce.Date",
         "PREMIUM.PAYING.TERM", "Renewal_Due_Date", "Renewal_Paid_Date",
         "Renewal_Status", "Renewal_Premium_Amount", "Persistency_Bucket",
         "Payment_Mode"]),
}

# Concepts that appear nowhere in the five files. A metric or dimension whose
# name implies one of these is making a claim the data cannot support.
ABSENT_CONCEPTS = [
    "smoker", "tobacco", "rider", "medical_underwriting", "underwriting",
    "nominee", "marital", "surrender_value", "surrender_charge", "surrender_reason",
    "incentive", "commission", "retention_contact", "retention_incentive",
    "retention_success", "campaign", "offer", "complaint_reason", "qualification",
]


def _load(fname: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(CSV_DIR, fname), low_memory=False)


def _metric_source(name: str) -> str:
    """The registered source text of one metric, for constant/tautology checks."""
    src = open(os.path.join(ROOT_DIR, "src", "analytics", "analytics_core.py"),
               encoding="utf-8").read()
    m = re.search(r'register\(MetricSpec\(\s*\n\s*"' + re.escape(name) + r'".*?\n\n',
                  src, re.S)
    return m.group(0) if m else ""


class SchemaGrounding(unittest.TestCase):
    """The five files are present and shaped as the metadata document says."""

    def test_files_present(self):
        for fname in DECLARED_SCHEMA:
            self.assertTrue(os.path.exists(os.path.join(CSV_DIR, fname)),
                            f"{fname} missing from data/csv/")

    def test_columns_and_rowcounts(self):
        for fname, (rows, cols) in DECLARED_SCHEMA.items():
            df = _load(fname)
            self.assertEqual(len(df), rows, f"{fname}: expected {rows} rows, got {len(df)}")
            missing = [c for c in cols if c not in df.columns]
            self.assertEqual(missing, [], f"{fname}: declared columns missing: {missing}")

    def test_persistency_grain_is_one_row_per_appid(self):
        """The metadata document declares '1 row per policy/application AppID'.

        The README and MULTI_TABLE_RELATIONS.md previously declared the grain as
        AppID + Persistency_Bucket, which would make the three buckets a survival
        curve. They are disjoint cohorts; this asserts which reading is right.
        """
        df = _load("Persistency_Details.csv")
        self.assertEqual(df["AppID"].nunique(), len(df),
                         "Persistency_Details is not one row per AppID")

    def test_referential_integrity_matches_known_state(self):
        """Claims and Persistency join cleanly; Owner_Details has one known break.

        One Owner_Details row carries AppID 15000000000, which matches no policy,
        and one policy correspondingly has no owner row. The metadata document
        flags owner-file AppID coverage as a quality caveat. Asserting the exact
        counts means a future extract that changes them fails here rather than
        quietly shifting every owner-joined denominator.
        """
        pol = set(_load("Policy_Details.csv")["AppID"])

        orphans = set(_load("Claims_Details.csv")["AppID"]) - pol
        self.assertEqual(len(orphans), 0,
                         f"Claims_Details: {len(orphans)} AppIDs not in Policy_Details")

        # One placeholder AppID appears in both Owner_Details and
        # Persistency_Details and matches no policy.
        for fname in ("Owner_Details.csv", "Persistency_Details.csv"):
            orphans = set(_load(fname)["AppID"]) - pol
            self.assertEqual(sorted(orphans), [PHANTOM_APPID],
                             f"{fname}: orphan AppIDs changed from the known placeholder")

        owners = set(_load("Owner_Details.csv")["AppID"])
        self.assertEqual(len(pol - owners), 1,
                         "Count of policies without an owner row changed from the known 1")

    def test_persistency_covers_most_but_not_all_policies(self):
        """459 policies have no renewal record, so persistency denominators are
        24,542 - not 25,000. Any metric based on the persistency frame is
        measured over records due, not over the whole book."""
        pol = set(_load("Policy_Details.csv")["AppID"])
        pers = set(_load("Persistency_Details.csv")["AppID"])
        self.assertEqual(len(pol - pers), 459,
                         "Policies missing a persistency record changed from the known 459")


class MetricGrounding(unittest.TestCase):
    """Every metric computes from a real column, with no invented constants."""

    @classmethod
    def setUpClass(cls):
        cls.ctx = ac.DataContext.get()

    def test_all_metrics_compute(self):
        broken = []
        for name, spec in ac.METRICS.items():
            try:
                spec.compute(self.ctx.frame(spec.base))
            except Exception as e:
                broken.append(f"{name}: {type(e).__name__}: {e}")
        self.assertEqual(broken, [], f"metrics failed to compute: {broken}")

    def test_no_magic_constants(self):
        """A numerator multiplied by a hard-coded rate is an assumption, not a measurement."""
        offenders = []
        for name in ac.METRICS:
            src = _metric_source(name)
            if re.search(r"(numerator|denominator)=lambda d:[^\n]*\*\s*0\.\d+", src):
                offenders.append(name)
        self.assertEqual(offenders, [],
                         f"metrics multiply by an invented constant: {offenders}")

    def test_no_tautological_rates(self):
        """len(d)/len(d) always returns 100% and measures nothing."""
        offenders = []
        for name in ac.METRICS:
            src = _metric_source(name)
            if re.search(r"numerator=lambda d: len\(d\),\s*\n\s*denominator=lambda d: len\(d\)", src):
                offenders.append(name)
        self.assertEqual(offenders, [],
                         f"metrics are tautologies returning 100%: {offenders}")

    def test_no_metric_names_absent_concepts(self):
        offenders = [n for n in ac.METRICS
                     for c in ABSENT_CONCEPTS if c in n]
        self.assertEqual(offenders, [],
                         f"metrics named for concepts absent from the data: {offenders}")

    def test_retired_metrics_explain_themselves(self):
        for name in ac.RETIRED_METRICS:
            self.assertNotIn(name, ac.METRICS, f"{name} is both retired and registered")
            with self.assertRaises(KeyError) as cm:
                _ = ac.METRICS[name]
            self.assertIn("not available", str(cm.exception),
                          f"{name} raised an unhelpful error")


class DimensionGrounding(unittest.TestCase):
    """Every dimension maps to a real column with something to compare."""

    @classmethod
    def setUpClass(cls):
        cls.ctx = ac.DataContext.get()
        cls.frames = {b: cls.ctx.frame(b)
                      for b in ("policy", "claim", "persistency", "customer", "agent")}

    def test_columns_exist(self):
        missing = []
        for name, dim in ac.DIMENSIONS.items():
            frame = self.frames.get(dim.base)
            if frame is None or dim.column not in frame.columns:
                missing.append(f"{name} -> {dim.base}.{dim.column}")
        self.assertEqual(missing, [], f"dimensions with no backing column: {missing}")

    def test_degenerate_dimensions_are_declared(self):
        """A single-valued column may stay registered only if it is guarded.

        CHANNEL and Mode are real fields the metadata document describes, so
        removing them would misrepresent the schema. What must not happen is a
        breakdown on them returning one row that reads as a comparison, so each
        one has to be listed in DEGENERATE_DIMENSIONS and refused at use.
        """
        undeclared = []
        for name, dim in ac.DIMENSIONS.items():
            frame = self.frames.get(dim.base)
            if frame is None or dim.column not in frame.columns:
                continue
            if frame[dim.column].nunique(dropna=True) < 2 \
                    and dim.column not in ac.DEGENERATE_DIMENSIONS:
                undeclared.append(f"{name} -> {dim.column}")
        self.assertEqual(undeclared, [],
                         f"single-valued dimensions that are not guarded: {undeclared}")

    def test_no_dimension_names_absent_concepts(self):
        offenders = [n for n in ac.DIMENSIONS
                     for c in ABSENT_CONCEPTS if c in n]
        self.assertEqual(offenders, [],
                         f"dimensions named for absent concepts: {offenders}")

    def test_degenerate_dimensions_refuse(self):
        """CHANNEL is Banca throughout; a breakdown must say so, not return one row."""
        for dim in ("channel", "sales_channel"):
            with self.assertRaises(ac.InsufficientDataError):
                ac.check_dimension(dim)

    def test_retired_dimensions_refuse(self):
        for dim in ("smoker", "marital_status", "surrender_reason"):
            with self.assertRaises(KeyError):
                ac.check_dimension(dim)


class ProblemStatementGrounding(unittest.TestCase):
    """Report how much of the PS library still resolves against the registries."""

    @classmethod
    def setUpClass(cls):
        ps_dir = os.path.join(ROOT_DIR, "data", "problem_statements")
        cls.records = []
        for fname in ("problem_statements.json", "business_problem_statements.json"):
            path = os.path.join(ps_dir, fname)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    cls.records.extend(json.load(f))

    def test_report_unresolvable(self):
        bad_metric, bad_dim, bad_op = {}, {}, {}
        for ps in self.records:
            pid = ps.get("problem_id", "?")
            for step in ps.get("analytical_steps") or []:
                params = step.get("params") or {}
                for key in ("metric", "value_metric"):
                    m = params.get(key)
                    if isinstance(m, str) and not m.startswith("$") \
                            and m not in ac.METRICS and m not in ac.SHARE_METRICS:
                        bad_metric.setdefault(m, []).append(pid)
                for key in ("dimension", "dim", "row_dimension", "col_dimension"):
                    d = params.get(key)
                    if isinstance(d, str) and not d.startswith("$") and d not in ac.DIMENSIONS:
                        bad_dim.setdefault(d, []).append(pid)
                op = step.get("op")
                if op and op not in ac.OPERATIONS:
                    bad_op.setdefault(op, []).append(pid)

        print(f"\n  PS records checked: {len(self.records)}")
        for label, table in (("metrics", bad_metric), ("dimensions", bad_dim),
                             ("operations", bad_op)):
            if table:
                print(f"  unresolvable {label}: {len(table)}")
                for name, pids in sorted(table.items(), key=lambda kv: -len(kv[1]))[:12]:
                    swap = ac.METRIC_REPLACEMENTS.get(name) or ac.RETIRED_DIMENSIONS.get(name, "")
                    hint = f"  -> {swap}" if swap else ""
                    print(f"      {name:34s} {len(pids):4d} PS{hint}")
            else:
                print(f"  unresolvable {label}: none")


if __name__ == "__main__":
    unittest.main(verbosity=2)
