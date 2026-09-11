"""
build_business_eval.py
======================
Writes `business_eval_set.json`: 64 held-out business questions.

Every question here is phrased differently from any `example_queries` entry in
the Problem Statement library, so this is an out-of-sample test of the
retriever, the planner and the executor - not a memorisation check.

Each case declares:
  question                    the user's words
  expected_intent             business intent taxonomy label
  expected_tier               simple_retrieval | analytical | multi_step_business
  expected_tables             tables that must be touched
  expected_operations         analytical operations the plan must include
  expected_metrics            metrics the plan must compute
  requires_multi_step         True when one query cannot answer it
  answer_must_mention         substrings the deterministic answer must contain
                              (entity names / metric labels only - never
                              generated wording)
"""

from __future__ import annotations

import json
import os

import business_intents as bi

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "business_eval_set.json")

E = []


def case(question, intent, tier, tables, ops, metrics, multi,
         mentions=None, note=None):
    E.append({
        "id": f"BEV-{len(E) + 1:03d}",
        "question": question,
        "expected_intent": intent,
        "expected_tier": tier,
        "expected_tables": tables,
        "expected_operations": ops,
        "expected_metrics": metrics,
        "requires_multi_step": multi,
        "answer_must_mention": mentions or [],
        "note": note,
    })


M, A, S = bi.TIER_MULTI_STEP, bi.TIER_ANALYTICAL, bi.TIER_RETRIEVAL

# --- A. portfolio health -------------------------------------------------
case("Give me a board-level read on the state of the book right now.",
     bi.PORTFOLIO_HEALTH, M, ["policies", "claims"],
     ["portfolio_overview", "multi_metric_table"],
     ["lapse_rate", "persistency_rate"], True, ["Persistency rate", "Lapse rate"])
case("If I had to name the three things most likely to hurt this portfolio, what are they?",
     bi.PORTFOLIO_HEALTH, M, ["policies", "claims"],
     ["portfolio_overview", "compare_to_baseline", "concentration"],
     ["lapse_rate", "loss_ratio"], True)
case("Where would you point the executive committee first?",
     bi.PORTFOLIO_HEALTH, M, ["policies"], ["composite_score"],
     ["lapse_rate"], True)
case("What is working and what is not in this book?",
     bi.PORTFOLIO_HEALTH, M, ["policies"], ["composite_score", "multi_metric_table"],
     ["persistency_rate", "loss_ratio"], True)
case("Break the book down by what we sell, how we sell it and where.",
     bi.PORTFOLIO_HEALTH, A, ["policies"], ["multi_metric_table"],
     ["policy_share"], False)

# --- B. product ----------------------------------------------------------
case("Rank the product line on a scorecard, not on one number.",
     bi.PRODUCT_PERFORMANCE, M, ["policies", "claims"],
     ["multi_metric_table", "composite_score"],
     ["persistency_rate", "loss_ratio"], True)
case("Which of our products is quietly bleeding policyholders?",
     bi.PRODUCT_PERFORMANCE, A, ["policies"], ["multi_metric_table"],
     ["lapse_rate"], True, ["Annuity"])
case("Which products sell in volume but do not stay on the books?",
     bi.PRODUCT_PERFORMANCE, A, ["policies"], ["multi_metric_table"],
     ["policy_share", "lapse_rate"], True)
case("Which product lines look economically unattractive to keep pushing?",
     bi.PRODUCT_PERFORMANCE, M, ["policies", "claims"],
     ["multi_metric_table"], ["loss_ratio"], True, ["Term Life"])
case("Give me a full read on the Whole Life book.",
     bi.PRODUCT_PERFORMANCE, M, ["policies"], ["segment_profile", "multi_metric_table"],
     ["lapse_rate"], True, ["Whole Life"])
case("Do savings products hold up better than protection products?",
     bi.PRODUCT_PERFORMANCE, A, ["policies"], ["multi_metric_table"],
     ["lapse_rate"], False, ["Savings"])
case("Does it matter whether a plan is regular pay or limited pay?",
     bi.PRODUCT_PERFORMANCE, A, ["policies"], ["multi_metric_table"],
     ["lapse_rate"], True, ["Limited"])

# --- C. channel ----------------------------------------------------------
case("Which route to market actually earns its keep once persistency is counted?",
     bi.CHANNEL_PERFORMANCE, M, ["policies"],
     ["multi_metric_table", "composite_score"],
     ["persistency_rate", "lapse_rate"], True)
case("Where do our large cases come from?",
     bi.CHANNEL_PERFORMANCE, A, ["policies"], ["multi_metric_table"],
     ["avg_sum_assured"], False)
case("Is any distribution partner writing business that does not stick?",
     bi.CHANNEL_PERFORMANCE, M, ["policies"], ["multi_metric_table"],
     ["lapse_rate"], True, ["Online"])
case("Should the sales head be worried about any particular channel?",
     bi.CHANNEL_PERFORMANCE, M, ["policies"], ["composite_score"],
     ["lapse_rate"], True)
case("Does the same channel work equally well for every product?",
     bi.CHANNEL_PERFORMANCE, M, ["policies"], ["crosstab_metric"],
     ["lapse_rate"], True)
case("How does the bank channel stack up against the rest of the book?",
     bi.CHANNEL_PERFORMANCE, M, ["policies"], ["segment_profile"],
     ["lapse_rate"], True, ["Bancassurance"])

# --- D. customer segmentation -------------------------------------------
case("Who actually pays the bills around here?",
     bi.CUSTOMER_SEGMENTATION, M, ["policies", "customers"],
     ["multi_metric_table", "entity_shortlist"],
     ["premium_share"], True)
case("Which slices of the customer base are most likely to walk away?",
     bi.CUSTOMER_SEGMENTATION, M, ["policies", "customers"],
     ["compare_to_baseline", "driver_scan"], ["lapse_rate"], True)
case("Paint me a picture of our most valuable policyholder.",
     bi.CUSTOMER_SEGMENTATION, M, ["policies", "customers"],
     ["segment_profile", "multi_metric_table"], ["avg_annual_premium"], True)
case("What does a customer who stops paying tend to look like?",
     bi.CUSTOMER_SEGMENTATION, M, ["policies", "customers"],
     ["segment_profile"], ["lapse_rate"], True)
case("Does earning more change how a policyholder behaves?",
     bi.CUSTOMER_SEGMENTATION, M, ["policies", "customers"],
     ["multi_metric_table", "compare_to_baseline"],
     ["lapse_rate", "avg_sum_assured"], True)
case("Do tobacco users cost us more than non-users?",
     bi.CUSTOMER_SEGMENTATION, M, ["policies", "customers", "claims"],
     ["multi_metric_table"], ["risk_claim_rate", "risk_loss_ratio"], True,
     ["Smoker"])
case("Is anyone paying more premium than they can realistically afford?",
     bi.CUSTOMER_SEGMENTATION, M, ["policies", "customers"],
     ["multi_metric_table", "entity_shortlist"],
     ["avg_premium_to_income"], True)

# --- E. lapse ------------------------------------------------------------
case("What sits behind the policies we lose to non-payment?",
     bi.LAPSE_ANALYSIS, M, ["policies"], ["driver_scan", "portfolio_overview"],
     ["lapse_rate"], True)
case("Does how often someone is billed change whether they keep paying?",
     bi.LAPSE_ANALYSIS, M, ["policies"], ["compare_to_baseline", "multi_metric_table"],
     ["lapse_rate"], True, ["Payment mode"])
case("Are the policies we lose bunched into particular corners of the book?",
     bi.LAPSE_ANALYSIS, M, ["policies"], ["crosstab_metric"], ["lapse_rate"], True)
case("Give me a persistency improvement agenda I can take to the exec.",
     bi.LAPSE_ANALYSIS, M, ["policies", "retention"],
     ["portfolio_overview", "driver_scan", "composite_score"],
     ["lapse_rate"], True)
case("Which part of the country loses the most policies?",
     bi.LAPSE_ANALYSIS, A, ["policies"], ["compare_to_baseline"],
     ["lapse_rate"], True, ["zone"])
case("Do policies fail early or late in their life?",
     bi.LAPSE_ANALYSIS, A, ["policies"], ["compare_to_baseline"], ["lapse_rate"], True)
case("Do riders or a medical check make any difference to whether a policy survives?",
     bi.LAPSE_ANALYSIS, A, ["policies"], ["compare_to_baseline"], ["lapse_rate"], True)

# --- F. surrender --------------------------------------------------------
case("Why are people cashing out of their policies?",
     bi.SURRENDER_ANALYSIS, M, ["policies", "surrenders"],
     ["multi_metric_table", "driver_scan"], ["surrender_rate"], True)
case("Which products lose the most policies to early exit?",
     bi.SURRENDER_ANALYSIS, A, ["policies", "surrenders"],
     ["compare_to_baseline"], ["surrender_rate"], True, ["Money Back"])
case("Are we losing our biggest policies to cash-outs?",
     bi.SURRENDER_ANALYSIS, M, ["policies", "surrenders"],
     ["segment_profile", "compare_to_baseline"], ["surrender_rate"], True)
case("How much money has gone out of the door through cash-outs?",
     bi.SURRENDER_ANALYSIS, A, ["policies", "surrenders"],
     ["multi_metric_table"], ["total_surrender_value_paid"], True)
case("Which live policies should we protect from being cashed out?",
     bi.SURRENDER_ANALYSIS, M, ["policies", "surrenders"],
     ["entity_shortlist", "segment_profile"], ["surrender_rate"], True)

# --- G. retention --------------------------------------------------------
case("What is the best way to reach someone before they drop out?",
     bi.RETENTION_STRATEGY, M, ["retention"],
     ["multi_metric_table", "compare_to_baseline"],
     ["retention_success_rate"], True, ["SMS"])
case("Are some policyholders simply easier to talk round than others?",
     bi.RETENTION_STRATEGY, M, ["retention", "policies"],
     ["multi_metric_table"], ["retention_success_rate"], True)
case("Give the save team a call list for Monday morning.",
     bi.RETENTION_STRATEGY, M, ["policies", "retention"],
     ["entity_shortlist", "multi_metric_table"], ["lapse_rate"], True)
case("Are we spending sensibly on saving policies?",
     bi.RETENTION_STRATEGY, A, ["retention"], ["multi_metric_table"],
     ["incentive_per_retained_policy", "retention_success_rate"], True)
case("Where should the save budget go next year?",
     bi.RETENTION_STRATEGY, M, ["retention", "policies"],
     ["portfolio_overview", "multi_metric_table", "composite_score"],
     ["retention_success_rate", "lapse_rate"], True)
case("Which reinstatement offer actually persuades people?",
     bi.RETENTION_STRATEGY, A, ["retention"], ["multi_metric_table"],
     ["retention_success_rate"], True)

# --- H. claims -----------------------------------------------------------
case("What sits behind the money we pay out on claims?",
     bi.CLAIMS_RISK, M, ["policies", "claims"],
     ["portfolio_overview", "multi_metric_table", "compare_to_baseline"],
     ["risk_claim_rate", "risk_loss_ratio"], True)
case("Which products see more claim activity than they should?",
     bi.CLAIMS_RISK, A, ["policies", "claims"], ["multi_metric_table"],
     ["claim_rate", "risk_claim_rate"], True)
case("What kind of claims come off each product?",
     bi.CLAIMS_RISK, A, ["policies", "claims"], ["crosstab_metric"],
     ["claim_count"], True)
case("Do we see claims coming in suspiciously soon after a policy is written?",
     bi.CLAIMS_RISK, M, ["policies", "claims"], ["multi_metric_table"],
     ["early_claim_rate"], True)
case("Is anyone selecting against us on the underwriting side?",
     bi.CLAIMS_RISK, M, ["policies", "claims"],
     ["multi_metric_table", "portfolio_overview"],
     ["risk_claim_rate", "early_claim_rate"], True)
case("Where should the claims function be spending its review time?",
     bi.CLAIMS_RISK, M, ["policies", "claims"],
     ["portfolio_overview", "composite_score"],
     ["risk_loss_ratio"], True)

# --- I. economics --------------------------------------------------------
case("Which products cost us the most in claims per rupee of premium?",
     bi.PROFITABILITY, A, ["policies", "claims"], ["multi_metric_table"],
     ["loss_ratio", "risk_loss_ratio"], True, ["Term Life"])
case("Which parts of the book look worth writing and which do not?",
     bi.PROFITABILITY, M, ["policies", "claims"],
     ["composite_score", "multi_metric_table"], ["loss_ratio"], True)
case("Where does claims spend run ahead of what we take in?",
     bi.PROFITABILITY, M, ["policies", "claims"],
     ["compare_to_baseline", "driver_scan"], ["loss_ratio"], True)
case("Does checking someone medically actually earn its cost?",
     bi.PROFITABILITY, A, ["policies", "claims"], ["multi_metric_table"],
     ["risk_loss_ratio"], True)

# --- J. geography --------------------------------------------------------
case("Which region runs the tightest ship?",
     bi.GEOGRAPHIC, M, ["policies"], ["multi_metric_table", "composite_score"],
     ["persistency_rate"], True)
case("Is anywhere in the country carrying more claims than it should?",
     bi.GEOGRAPHIC, A, ["policies", "claims"], ["multi_metric_table"],
     ["loss_ratio"], True)
case("Do different regions sell different things?",
     bi.GEOGRAPHIC, M, ["policies"], ["crosstab_metric"], ["policy_count"], True)
case("Which offices should the regional director visit first?",
     bi.GEOGRAPHIC, M, ["policies"], ["composite_score", "multi_metric_table"],
     ["lapse_rate"], True)

# --- K. agents -----------------------------------------------------------
case("Who are the advisors worth keeping, once book quality is counted?",
     bi.AGENT_PERFORMANCE, M, ["sales", "policies"],
     ["entity_shortlist", "multi_metric_table"], ["persistency_rate"], True)
case("Is anyone writing a lot of business that falls over afterwards?",
     bi.AGENT_PERFORMANCE, M, ["sales", "policies"],
     ["entity_shortlist", "outliers"], ["lapse_rate"], True)
case("Does anyone in the sales force look statistically odd?",
     bi.AGENT_PERFORMANCE, M, ["sales", "policies"], ["outliers"],
     ["lapse_rate"], True)
case("Is our incentive scheme buying volume at the cost of quality?",
     bi.AGENT_PERFORMANCE, M, ["sales", "policies"],
     ["entity_shortlist", "multi_metric_table"], ["lapse_rate"], True)

# --- L. anomaly / concentration -----------------------------------------
case("Is there anything in this book that just looks wrong?",
     bi.ANOMALY_CONCENTRATION, M, ["policies", "claims"],
     ["outliers", "concentration"], ["lapse_rate"], True)
case("How much of the book rests on very few things?",
     bi.ANOMALY_CONCENTRATION, M, ["policies"],
     ["concentration", "top_entity_concentration"],
     ["total_sum_assured"], True)
case("Do a handful of claims account for most of the payout?",
     bi.ANOMALY_CONCENTRATION, M, ["claims", "policies"],
     ["top_entity_concentration", "concentration"],
     ["total_claim_amount"], True)

# --- M. comparative ------------------------------------------------------
case("Put every distribution channel side by side on everything that matters.",
     bi.COMPARATIVE, M, ["policies", "retention"],
     ["multi_metric_table"], ["lapse_rate", "retention_success_rate"], True)
case("Set Whole Life and Term Life against each other.",
     bi.COMPARATIVE, A, ["policies", "claims"], ["multi_metric_table"],
     ["lapse_rate", "loss_ratio"], True, ["Whole Life", "Term Life"])
case("How do monthly payers compare with annual payers?",
     bi.COMPARATIVE, A, ["policies"], ["multi_metric_table"],
     ["lapse_rate"], True, ["Monthly", "Annual"])

# --- N. root cause -------------------------------------------------------
case("Talk me through why Whole Life is a problem.",
     bi.ROOT_CAUSE, M, ["policies", "claims"],
     ["segment_profile", "compare_to_baseline", "driver_scan"],
     ["lapse_rate", "loss_ratio"], True, ["Whole Life"])
case("Why does the Online channel keep losing policies?",
     bi.ROOT_CAUSE, M, ["policies", "retention"],
     ["segment_profile", "compare_to_baseline", "driver_scan"],
     ["lapse_rate"], True, ["Online"])
case("Unpack our headline lapse number for me.",
     bi.ROOT_CAUSE, M, ["policies"],
     ["portfolio_overview", "driver_scan", "multi_metric_table"],
     ["lapse_rate", "policy_share"], True)
case("What separates the parts of the book that work from the parts that do not?",
     bi.ROOT_CAUSE, M, ["policies", "claims"],
     ["composite_score", "multi_metric_table", "driver_scan"],
     ["persistency_rate"], True)

# --- O. targeting --------------------------------------------------------
case("Give me the live policies we can least afford to lose.",
     bi.CUSTOMER_TARGETING, M, ["policies", "customers"],
     ["entity_shortlist"], ["avg_annual_premium"], True)
case("Which customers have already started drifting away from us?",
     bi.CUSTOMER_TARGETING, M, ["policies", "customers"],
     ["entity_shortlist"], ["retention_success_rate"], True)
case("Who has room to buy more cover from us?",
     bi.CUSTOMER_TARGETING, M, ["policies", "customers"],
     ["entity_shortlist", "multi_metric_table"], ["avg_sum_assured"], True)

# --- P. prescriptive (how to improve) -----------------------------------
case("How can I improve persistency rate across our portfolio?",
     bi.PRESCRIPTIVE, M, ["policies"],
     ["portfolio_overview", "compare_to_baseline", "driver_scan"],
     ["persistency_rate"], True)
case("What actions would reduce lapse rate for middle-aged policyholders?",
     bi.PRESCRIPTIVE, M, ["policies"],
     ["portfolio_overview", "compare_to_baseline", "driver_scan"],
     ["lapse_rate"], True)
case("How can we improve 13th month persistency across sales channels?",
     bi.PRESCRIPTIVE, M, ["policies"],
     ["portfolio_overview", "compare_to_baseline", "driver_scan"],
     ["persistency_13m_rate"], True)

# --- Q. cross-domain correlation -----------------------------------------
case("What is the relation between claims and persistency rate?",
     bi.CROSS_DOMAIN, M, ["policies", "claims"],
     ["correlation_analysis", "composite_score"],
     ["lapse_rate", "claim_settlement_ratio"], True)
case("Is there a relationship between PIVC mismatch and lapse rate?",
     bi.CROSS_DOMAIN, M, ["policies"],
     ["correlation_analysis"],
     ["pivc_mismatch_rate", "lapse_rate"], True)
case("Does higher agent attrition correlate with lower persistency?",
     bi.CROSS_DOMAIN, M, ["policies", "agents"],
     ["correlation_analysis"],
     ["attrition_rate", "persistency_rate"], True)

# --- R. geographic diagnostics (where and why) ---------------------------
case("Which area is having lowest persistency rate and why?",
     bi.GEOGRAPHIC, M, ["policies"],
     ["leaderboard", "driver_scan", "multi_metric_table"],
     ["persistency_rate"], True)
case("Which states have the worst lapse rate and what is driving it?",
     bi.GEOGRAPHIC, M, ["policies"],
     ["leaderboard", "driver_scan", "multi_metric_table"],
     ["lapse_rate"], True)

# --- S. simple retrieval (must still route to the legacy engine) ---------
case("How many customers are smokers?", bi.SIMPLE_RETRIEVAL, S,
     ["customers"], [], [], False)
case("How many policies are currently In-Force?", bi.SIMPLE_RETRIEVAL, S,
     ["policies"], [], [], False)
case("What is the total surrender value paid out?", bi.SIMPLE_RETRIEVAL, S,
     ["surrenders"], [], [], False)
case("How many surrender records are there?", bi.SIMPLE_RETRIEVAL, S,
     ["surrenders"], [], [], False)

# --- T. out of scope (must be refused honestly) --------------------------
case("How do we compare with our competitors on market share?",
     "OUT_OF_SCOPE", S, [], [], [], False)
case("What is our solvency ratio?", "OUT_OF_SCOPE", S, [], [], [], False)
case("What will our lapse rate be next year?", "OUT_OF_SCOPE", S, [], [], [], False)
case("What is our expense ratio by product?", "OUT_OF_SCOPE", S, [], [], [], False)
case("What do customers say in their complaints?", "OUT_OF_SCOPE", S, [], [], [], False)


if __name__ == "__main__":
    with open(OUT, "w") as f:
        json.dump(E, f, indent=1)
    import collections
    print(f"[OK] wrote {len(E)} held-out business evaluation cases -> {OUT}")
    c = collections.Counter(e["expected_intent"] for e in E)
    for k, v in c.most_common():
        print(f"   {v:3d}  {k}")

