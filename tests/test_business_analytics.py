"""
test_business_analytics.py
==========================
Deterministic regression tests for the business-analytics layer operating
exclusively on the 5 production CSV tables:
  1. Policy_Details.csv
  2. Owner_Details.csv
  3. Persistency_Details.csv
  4. Sales_Details.csv
  5. Claims_Details.csv
"""

from __future__ import annotations

import json
import math
import os
import sys
import unittest
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE, ".."))
for sub in ["src", "src/analytics", "src/engine", "src/retrieval", "src/llm", "ui"]:
    p = os.path.join(ROOT_DIR, sub)
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import analytics_core as ac
import business_engine as be
import business_intents as bi
import business_retrieval as brt
from analysis_planner import BusinessPlanner, ParameterBinder

DATA = os.path.join(ROOT_DIR, "data")



class TestDataContext(unittest.TestCase):
    """The 5 fact tables must match the raw CSV row counts and integrity."""

    @classmethod
    def setUpClass(cls):
        cls.ctx = ac.DataContext.get(DATA)
        cls.policies = cls.ctx.raw_policies
        cls.claims = cls.ctx.raw_claims
        cls.persistency = cls.ctx.raw_persistency
        cls.owners = cls.ctx.raw_owners
        cls.sales = cls.ctx.raw_sales

    def test_policy_fact_has_exactly_one_row_per_policy(self):
        self.assertEqual(len(self.ctx.policy_fact), len(self.policies))
        self.assertEqual(self.ctx.policy_fact["app_id"].nunique(), len(self.policies))
        self.assertEqual(len(self.ctx.policy_fact), 25000)

    def test_claim_fact_row_count_preserved(self):
        self.assertEqual(len(self.ctx.claim_fact), len(self.claims))
        self.assertEqual(len(self.ctx.claim_fact), 134)

    def test_persistency_fact_row_count_preserved(self):
        self.assertEqual(len(self.ctx.persistency_fact), len(self.persistency))
        self.assertEqual(len(self.ctx.persistency_fact), 24542)

    def test_customer_fact_row_count_preserved(self):
        self.assertEqual(len(self.ctx.customer_fact), self.owners["ClientID"].nunique())

    def test_agent_fact_row_count_preserved(self):
        self.assertEqual(len(self.ctx.agent_fact), len(self.sales))
        self.assertEqual(len(self.ctx.agent_fact), 1956)

    def test_claim_totals_match_raw(self):
        self.assertAlmostEqual(
            float(self.ctx.claim_fact["claim_amount"].sum()),
            float(self.claims["Claim.Amount"].astype(str).str.replace(",", "").astype(float).sum()),
            places=1
        )

    def test_every_fact_frame_carries_the_shared_dimensions(self):
        for base in ("policy", "claim", "persistency"):
            frame = self.ctx.frame(base)
            for col in ("plan_name", "channel", "state", "trad_ulip"):
                self.assertIn(col, frame.columns, f"{col} missing from {base} fact")


class TestMetricDenominators(unittest.TestCase):
    """Every rate must use the production denominator declared in metadata."""

    @classmethod
    def setUpClass(cls):
        cls.ctx = ac.DataContext.get(DATA)

    def test_inforce_rate_computation(self):
        res = ac.overall_metric(self.ctx, "inforce_rate")
        self.assertEqual(int(res["denominator"]), 25000)
        self.assertAlmostEqual(res["value"], 87.22, places=2)

    def test_lapse_rate_computation(self):
        res = ac.overall_metric(self.ctx, "lapse_rate")
        self.assertGreater(res["value"], 0.0)
        self.assertLess(res["value"], 20.0)

    def test_persistency_13m_rate_uses_13m_cohort(self):
        res = ac.overall_metric(self.ctx, "persistency_13m_rate")
        self.assertEqual(int(res["denominator"]), 6943)
        self.assertAlmostEqual(res["value"], 53.36, places=1)

    def test_persistency_25m_rate_uses_25m_cohort(self):
        res = ac.overall_metric(self.ctx, "persistency_25m_rate")
        self.assertEqual(int(res["denominator"]), 8189)
        self.assertAlmostEqual(res["value"], 65.45, places=1)

    def test_claim_settlement_ratio_matches_paid_claims(self):
        res = ac.overall_metric(self.ctx, "claim_settlement_ratio")
        self.assertEqual(int(res["denominator"]), 134)
        self.assertAlmostEqual(res["value"], 99.25, places=1)

    def test_every_registry_metric_declares_a_definition_and_unit(self):
        for name, spec in ac.METRICS.items():
            self.assertTrue(spec.definition.strip(), f"{name} has no definition")
            self.assertIn(spec.unit, {"%", "x", "INR", "count", "ratio", "days", "₹", "years"}, name)


class TestRankingAndScoring(unittest.TestCase):
    def setUp(self):
        self.ctx = ac.DataContext.get(DATA)

    def test_rank_desc_is_actually_descending(self):
        df = ac.rank(self.ctx, "plan", "policy_count", order="desc")
        val_col = "value" if "value" in df.columns else "policy_count"
        vals = df[val_col].dropna().tolist()
        self.assertEqual(vals, sorted(vals, reverse=True))

    def test_composite_score_stays_bounded(self):
        df = ac.composite_score(
            self.ctx, "plan",
            [{"metric": "lapse_rate", "weight": 0.5, "direction": "high"},
             {"metric": "total_ape", "weight": 0.5, "direction": "high"}]
        )
        self.assertTrue((df["priority_score"] >= 0).all())
        self.assertTrue((df["priority_score"] <= 1.0001).all())


class TestAnalyticalOperations(unittest.TestCase):
    def setUp(self):
        self.ctx = ac.DataContext.get(DATA)

    def test_portfolio_overview_reports_core_kpis(self):
        ov = ac.portfolio_overview(self.ctx)
        for key in ("policy_count", "total_ape", "inforce_rate", "persistency_13m_rate", "claim_settlement_ratio"):
            self.assertIn(key, ov)
            self.assertIn("value", ov[key])
            self.assertIn("label", ov[key])


class TestEngineRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eng = be.get_engine(with_legacy=True)

    def test_business_question_routes_to_multi_step_analysis(self):
        r = self.eng.ask("What are the biggest risks in our insurance portfolio?")
        self.assertEqual(r.route, be.ROUTE_BUSINESS)
        self.assertGreater(len(r.step_results), 1)

    def test_instance_lookup_resolves_sales_id(self):
        r = self.eng.ask("Show me details of agent 270084")
        self.assertIn(r.route, (be.ROUTE_SIMPLE, be.ROUTE_BUSINESS))
        self.assertIn("270084", str(r.resolved_entities.get("sales_id", "")) + str(getattr(r, "legacy_plan", {})))


if __name__ == "__main__":
    unittest.main(verbosity=2)
