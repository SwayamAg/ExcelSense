"""
test_grounding_regression.py
============================
Automated regression tests verifying:
1. Exact grounding and count consistency (e.g. biggest risks)
2. Multi-intent diagnostic & prescriptive query (lowest persistency area + why + how)
3. Cross-domain correlation without metric substitution (claims vs persistency rate)
4. Multi-metric balanced growth vs persistency calculations
5. Strict non-causal language and outer quote stripping in LLM responses
"""
import os
import sys

# Make src/* importable: these suites previously inserted only their own
# directory, so every `import analytics_core` failed before a test could run.
_BASE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_BASE, ".."))
for _sub in ["src", "src/analytics", "src/engine", "src/retrieval", "src/llm", "ui"]:
    _p = os.path.join(_ROOT, _sub)
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


import unittest
import analytics_core as ac
import business_engine as be
import ollama_client as _oc

class TestBusinessEngineGrounding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = be.get_engine(with_legacy=True)

    def test_query_1_biggest_risks_exact_count(self):
        q = "What are the biggest risks in our insurance portfolio?"
        resp = self.engine.ask(q, llm=False)
        self.assertEqual(resp.route, "business_analysis")
        self.assertIsNotNone(resp.answer)
        self.assertIn("The 4 largest exposures", resp.answer.headline)
        self.assertEqual(len(resp.answer.evidence), 4)
        for e in resp.answer.evidence:
            self.assertTrue(any(k in e for k in ("Canara HSBC", "Outlier", "Single-policy", "exposure", "lapse rate", "loss ratio")))

    def test_query_2_area_lowest_persistency_why_how(self):
        q = "Which area is having lowest persistency rate and why. how can I improve persistency rate."
        resp = self.engine.ask(q, llm=False)
        self.assertEqual(resp.route, "business_analysis")
        self.assertIsNotNone(resp.answer)
        # Direct answer upfront with sample threshold caution
        self.assertIn("Sikkim", resp.answer.headline)
        self.assertIn("Chandigarh", resp.answer.headline)
        # Why & How: evidence includes direct finding and actionable levers
        self.assertTrue(any("Direct Finding" in e for e in resp.answer.evidence))
        self.assertTrue(any("Lever" in e for e in resp.answer.evidence))
        self.assertTrue(len(resp.answer.recommendation) >= 3)

    def test_query_3_claims_vs_persistency_no_metric_substitution(self):
        q = "What is the relation between claims and persistency rate across products?"
        resp = self.engine.ask(q, llm=False)
        self.assertEqual(resp.route, "business_analysis")
        self.assertIsNotNone(resp.answer)
        # Verified metrics used
        self.assertIn("overall persistency rate", resp.answer.headline)
        self.assertIn("claim settlement ratio", resp.answer.headline)
        self.assertIn("Pearson r = -0.04", resp.answer.headline)
        # Correlation step executed
        corr_steps = [s for s in resp.step_results if s.step.op == "correlation_analysis"]
        self.assertTrue(len(corr_steps) > 0)
        self.assertEqual(corr_steps[0].step.params.get("metric_a"), "persistency_rate")
        self.assertEqual(corr_steps[0].step.params.get("metric_b"), "claim_settlement_ratio")

    def test_query_4_channel_comparison_is_refused(self):
        """A channel comparison must be refused, not answered with one row.

        CHANNEL is 'Banca' for all 25,000 policies, so ranking channels against
        each other produces a single-row table that reads as a leaderboard. This
        test previously asserted that leaderboard was produced; it now asserts
        the refusal, and that the reason names the degenerate column.
        """
        q = "Which sales channel gives us the best balance of growth and persistency?"
        resp = self.engine.ask(q, llm=False)
        self.assertIsNotNone(resp.answer)
        self.assertIn("could not be completed", resp.answer.headline)

        reasons = " ".join(resp.answer.unavailable or [])
        self.assertIn("cannot be compared", reasons)
        self.assertIn("Banca", reasons)

    def test_degenerate_dimension_refusal_is_specific(self):
        """The refusal must say which column is degenerate and why."""
        with self.assertRaises(ac.InsufficientDataError) as cm:
            ac.check_dimension("channel")
        self.assertIn("25,000", str(cm.exception))

    def test_llm_response_no_outer_quotes(self):
        from answer_composer import polish
        q = "What are the biggest risks in our insurance portfolio?"
        resp = self.engine.ask(q, llm=False)
        if _oc.is_ollama_available():
            narrative = polish(resp.answer)
            # ollama may return None on connection drop – only validate when we got a response
            if narrative is None:
                self.skipTest("Ollama returned None (connection error); skipping quote assertion.")
            self.assertFalse(narrative.startswith('"'))
            self.assertFalse(narrative.endswith('"'))
            self.assertFalse(narrative.startswith('"""'))
            self.assertFalse(narrative.endswith('"""'))

if __name__ == "__main__":
    unittest.main()
