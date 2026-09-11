"""build_business_ps4.py
Generates 75 new problem statements covering:
  A. Geographic + Diagnostic (25 PSs)  - 'which area is worst and why?'
  B. Prescriptive / Action Planning (30 PSs) - 'how to improve X?'
  C. Cross-domain Correlation (20 PSs) - 'relation between X and Y'
"""
import business_intents as bi

def _ps(pid, stmt, intent, tier, questions, steps, template, metrics, tables, caveat=None):
    domain_label = bi.BUSINESS_INTENTS.get(intent, {}).get("label", intent) if hasattr(bi, "BUSINESS_INTENTS") else intent
    return {
        "problem_id": pid,
        "problem_statement": stmt,
        "business_intent": intent,
        "business_domain": domain_label,
        "intent_type": f"Business/{intent}",
        "analysis_tier": tier,
        "answer_template": template,
        "required_metrics": metrics,
        "relevant_tables": tables,
        "caveat": caveat or "",
        "analytical_steps": steps,
        "example_queries": list(questions),
        "sample_questions": questions,
        "parameters": {},
    }

def _step(sid, title, op, params, purpose):
    return {"id": sid, "title": title, "op": op, "params": params, "purpose": purpose}

M = bi.TIER_MULTI_STEP

# ==========================================================================
# BUCKET A — Geographic + Diagnostic (25 PSs)
# ==========================================================================
GEO_DIAGNOSTIC = []

GEO_DIAGNOSTIC.append(_ps(
    "geo-d-001",
    "Which state has the lowest persistency rate and what factors explain it?",
    bi.GEOGRAPHIC, M,
    ["Which area has the lowest persistency rate and why?",
     "Which state has the worst persistency?",
     "Why does one state have lower persistency than others?",
     "Show me the bottom states by persistency and diagnose the cause."],
    [_step("s1","Portfolio baseline","portfolio_overview",{},"Reference KPIs."),
     _step("s2","Persistency by state","compare_to_baseline",
           {"metric":"persistency_rate","dimension":"state"},"Rank states by persistency gap."),
     _step("s3","Driver scan inside worst state","driver_scan",
           {"metric":"persistency_rate","candidate_dimensions":["plan_name","payment_mode","income_band","age_band","trad_ulip"]},
           "Find sub-dimensions driving the gap."),
     _step("s4","Multi-metric profile by state","multi_metric_table",
           {"dimension":"state","metrics":["policy_count","persistency_rate","lapse_rate","avg_ape"],"sort_by":"persistency_rate","ascending":True},
           "Volume and value context.")],
    "diagnostic",["persistency_rate","lapse_rate","policy_count","avg_ape"],["policies","persistency"],
))

GEO_DIAGNOSTIC.append(_ps(
    "geo-d-002",
    "Which zone performs worst on persistency and what is driving the underperformance?",
    bi.GEOGRAPHIC, M,
    ["Which zone has the worst persistency rate?",
     "Compare persistency across zones and tell me why the bottom zone is underperforming.",
     "Geographic persistency analysis worst zone root cause."],
    [_step("s1","Portfolio baseline","portfolio_overview",{},"Reference KPIs."),
     _step("s2","Persistency by zone","compare_to_baseline",{"metric":"persistency_rate","dimension":"zone"},"Rank zones."),
     _step("s3","Lapse by zone","compare_to_baseline",{"metric":"lapse_rate","dimension":"zone"},"Cross-check lapse."),
     _step("s4","Driver scan inside worst zone","driver_scan",
           {"metric":"lapse_rate","candidate_dimensions":["plan_name","payment_mode","income_band","occupation"]},
           "Sub-dimension drivers."),
     _step("s5","Zone summary","multi_metric_table",
           {"dimension":"zone","metrics":["policy_count","persistency_rate","lapse_rate","avg_ape","pivc_mismatch_rate"]},"Full zone table.")],
    "diagnostic",["persistency_rate","lapse_rate","pivc_mismatch_rate"],["policies","persistency"],
))

GEO_DIAGNOSTIC.append(_ps(
    "geo-d-003",
    "Which states have the lowest 13th month persistency and why?",
    bi.GEOGRAPHIC, M,
    ["13th month persistency by state which states are worst?",
     "Where is 13m persistency lowest geographically?",
     "Which area has the lowest 13th month persistency rate?"],
    [_step("s1","Portfolio baseline","portfolio_overview",{},"Reference KPIs."),
     _step("s2","13m persistency by state","compare_to_baseline",{"metric":"persistency_13m_rate","dimension":"state"},"State-level 13m."),
     _step("s3","Driver scan","driver_scan",{"metric":"persistency_13m_rate","candidate_dimensions":["plan_name","payment_mode","income_band","occupation"]},"Sub-dimension drivers."),
     _step("s4","State summary","multi_metric_table",{"dimension":"state","metrics":["policy_count","persistency_13m_rate","persistency_25m_rate","lapse_rate"],"sort_by":"persistency_13m_rate","ascending":True},"State table.")],
    "diagnostic",["persistency_13m_rate","persistency_25m_rate","lapse_rate"],["policies","persistency"],
))

GEO_DIAGNOSTIC.append(_ps(
    "geo-d-004",
    "Which states have the highest lapse rate and what factors explain the concentration?",
    bi.GEOGRAPHIC, M,
    ["Highest lapse rate by state and why?",
     "Which states have the most policy lapses?",
     "Lapse rate geographic breakdown with root cause.",
     "State-level lapse analysis."],
    [_step("s1","Portfolio baseline","portfolio_overview",{},"Reference KPIs."),
     _step("s2","Lapse by state","compare_to_baseline",{"metric":"lapse_rate","dimension":"state"},"State lapse vs baseline."),
     _step("s3","Driver scan","driver_scan",{"metric":"lapse_rate","candidate_dimensions":["plan_name","payment_mode","trad_ulip","income_band","occupation"]},"Find sub-dimension drivers."),
     _step("s4","State multi-metric","multi_metric_table",{"dimension":"state","metrics":["policy_count","lapse_rate","persistency_rate","avg_ape","avg_sum_assured"],"sort_by":"lapse_rate","ascending":False},"State summary.")],
    "diagnostic",["lapse_rate","persistency_rate","avg_ape"],["policies","persistency"],
))

GEO_DIAGNOSTIC.append(_ps(
    "geo-d-005",
    "Which states have the highest PIVC mismatch rates and why?",
    bi.GEOGRAPHIC, M,
    ["PIVC mismatch by state which areas are worst?",
     "Where is sales quality lowest geographically?",
     "State-level PIVC analysis."],
    [_step("s1","Portfolio baseline","portfolio_overview",{},"Reference KPIs."),
     _step("s2","PIVC by state","compare_to_baseline",{"metric":"pivc_mismatch_rate","dimension":"state"},"State-level PIVC."),
     _step("s3","Driver scan","driver_scan",{"metric":"pivc_mismatch_rate","candidate_dimensions":["plan_name","sales_type","trad_ulip"]},"Sub-dimension PIVC drivers."),
     _step("s4","State summary","multi_metric_table",{"dimension":"state","metrics":["policy_count","pivc_mismatch_rate","rcu_rejection_rate","lapse_rate"],"sort_by":"pivc_mismatch_rate","ascending":False},"Sales quality by state.")],
    "diagnostic",["pivc_mismatch_rate","rcu_rejection_rate","lapse_rate"],["policies","sales_reps"],
))

for pid, stmt, metric, dim, qs, extra_metrics in [
    ("geo-d-006","Which branch has the highest lapse rate and what is driving it?",
     "lapse_rate","branch",["Which branch has the most lapses?","Branch-level lapse analysis."],
     ["policy_count","persistency_rate","avg_ape"]),
    ("geo-d-007","Compare persistency rate across zones and identify the weakest region.",
     "persistency_rate","zone",["Zone persistency comparison.","Which zone has the lowest persistency?"],
     ["policy_count","lapse_rate","avg_ape","total_ape"]),
    ("geo-d-008","Which zone has the highest RCU rejection rate and what explains it?",
     "rcu_rejection_rate","zone",["RCU rejection rate by zone.","Which zone has the most RCU rejections?"],
     ["rcu_rejection_rate","pivc_mismatch_rate","policy_count"]),
    ("geo-d-009","Which state has the most early claims?",
     "claim_settlement_ratio","state",["Claims by state.","Which states have the most claims?"],
     ["claim_settlement_ratio","avg_settlement_tat","policy_count"]),
    ("geo-d-010","Analyse 25th month persistency across states and identify geographic risk hotspots.",
     "persistency_25m_rate","state",["25m persistency by state.","Which states have poor 25th month persistency?"],
     ["persistency_25m_rate","persistency_13m_rate","lapse_rate"]),
]:
    GEO_DIAGNOSTIC.append(_ps(
        pid, stmt, bi.GEOGRAPHIC, M, qs,
        [_step("s1","Portfolio baseline","portfolio_overview",{},"Reference KPIs."),
         _step("s2",f"Metric by {dim}","compare_to_baseline",{"metric":metric,"dimension":dim},f"Rank {dim}s."),
         _step("s3","Driver scan","driver_scan",{"metric":metric,"candidate_dimensions":["plan_name","payment_mode","income_band","trad_ulip"]},"Sub-dimension drivers."),
         _step("s4","Summary","multi_metric_table",{"dimension":dim,"metrics":extra_metrics,"sort_by":metric,"ascending":True},"Summary table.")],
        "diagnostic", extra_metrics, ["policies","persistency"],
    ))

for pid, stmt, metric1, metric2, dim, qs in [
    ("geo-d-011","Which zone has the worst combined lapse and PIVC mismatch?",
     "lapse_rate","pivc_mismatch_rate","zone",["Zone with worst lapse and PIVC.","Geographic sales quality and persistency risk."]),
    ("geo-d-012","Analyse state-level surrender rates and explain the geographic pattern.",
     "surrender_rate","lapse_rate","state",["Surrender rate by state.","Which states surrender the most?"]),
    ("geo-d-013","Which area has the most policies with RCU rejections?",
     "rcu_rejection_rate","pivc_mismatch_rate","state",["RCU by state.","Geographic RCU rejection pattern."]),
    ("geo-d-014","Zone-level analysis of persistency and claims concentration.",
     "persistency_rate","claim_settlement_ratio","zone",["Zone persistency vs claims.","Which zone has poor persistency and high claims?"]),
    ("geo-d-015","Which state has the highest premium to income ratio and associated lapse risk?",
     "lapse_rate","avg_ape","state",["Geographic affordability analysis.","Which area has the highest premium burden?"]),
]:
    GEO_DIAGNOSTIC.append(_ps(
        pid, stmt, bi.GEOGRAPHIC, M, qs,
        [_step("s1","Portfolio baseline","portfolio_overview",{},"Reference KPIs."),
         _step("s2",f"{metric1} by {dim}","compare_to_baseline",{"metric":metric1,"dimension":dim},f"{metric1} by {dim}."),
         _step("s3",f"{metric2} by {dim}","compare_to_baseline",{"metric":metric2,"dimension":dim},f"{metric2} by {dim}."),
         _step("s4","Summary","multi_metric_table",{"dimension":dim,"metrics":["policy_count",metric1,metric2,"avg_ape"]},"Multi-metric summary.")],
        "comparison", [metric1,metric2,"policy_count","avg_ape"], ["policies","persistency"],
    ))

for pid, stmt, qs, metric in [
    ("geo-d-016","Identify which state has the highest concentration of high-sum-assured policies with low persistency.",
     ["High value low persistency states."],"avg_sum_assured"),
    ("geo-d-017","Which zone should management prioritise for a persistency improvement campaign?",
     ["Zone persistency priority ranking.","Where to focus retention geographically?"],"persistency_rate"),
    ("geo-d-018","Analyse persistency rate across states and flag the three worst performing.",
     ["State persistency ranking.","Bottom 3 states by persistency."],"persistency_rate"),
    ("geo-d-019","Which branch clusters show the highest lapse rate?",
     ["Branch lapse analysis.","Branch product mix and lapse."],"lapse_rate"),
    ("geo-d-020","Compare North vs South zone persistency and explain the gap.",
     ["North vs South persistency.","Zone comparison persistency."],"persistency_rate"),
    ("geo-d-021","Which state has the highest free-look cancellation rate?",
     ["Free-look by state.","Geographic free-look analysis."],"free_look_rate"),
    ("geo-d-022","Analyse the geographic distribution of sales complaints.",
     ["Complaint rate by state.","Geographic complaint analysis."],"complaint_rate"),
    ("geo-d-023","Identify states where both lapse and surrender rates exceed the portfolio average.",
     ["States with high lapse and surrender.","Dual-risk states."],"lapse_rate"),
    ("geo-d-024","Which zone has the worst 37th month persistency?",
     ["37m persistency by zone.","Zone-level 37m persistency."],"persistency_rate"),
    ("geo-d-025","Which area shows the highest agent PIVC mismatch concentration?",
     ["Geographic PIVC concentration.","Agent quality by area."],"pivc_mismatch_rate"),
]:
    GEO_DIAGNOSTIC.append(_ps(
        pid, stmt, bi.GEOGRAPHIC, M, qs,
        [_step("s1","Portfolio baseline","portfolio_overview",{},"Reference KPIs."),
         _step("s2","Metric by state","compare_to_baseline",{"metric":metric,"dimension":"state"},"State-level ranking."),
         _step("s3","Driver scan","driver_scan",{"metric":metric,"candidate_dimensions":["plan_name","payment_mode","income_band","trad_ulip"]},"Sub-dimension drivers."),
         _step("s4","State summary","multi_metric_table",{"dimension":"state","metrics":["policy_count",metric,"lapse_rate","avg_ape"]},"State summary.")],
        "diagnostic", [metric,"lapse_rate","policy_count"], ["policies","persistency"],
    ))

# ==========================================================================
# BUCKET B — Prescriptive / Action Planning (30 PSs)
# ==========================================================================
PRESCRIPTIVE = []

PRESCRIPTIVE.append(_ps(
    "pres-001",
    "How can management improve the overall persistency rate? What levers are available?",
    bi.PRESCRIPTIVE, M,
    ["How can I improve persistency rate?",
     "How to improve persistency?",
     "What can we do to improve persistency?",
     "What actions would improve persistency rate?",
     "What should management do to improve policy persistency?"],
    [_step("s1","Portfolio baseline","portfolio_overview",{},"Reference KPIs."),
     _step("s2","Lapse by payment mode","compare_to_baseline",{"metric":"lapse_rate","dimension":"payment_mode"},"Payment mode lever."),
     _step("s3","Lapse by product","compare_to_baseline",{"metric":"lapse_rate","dimension":"plan_name"},"Product-level lapse."),
     _step("s4","Lapse by income band","compare_to_baseline",{"metric":"lapse_rate","dimension":"income_band"},"Affordability lens."),
     _step("s5","Driver scan all dimensions","driver_scan",{"metric":"lapse_rate","candidate_dimensions":["plan_name","payment_mode","state","occupation","income_band","age_band","trad_ulip"]},"Full driver scan.")],
    "prescriptive",["lapse_rate","persistency_rate","policy_count"],["policies","persistency"],
))

PRESCRIPTIVE.append(_ps(
    "pres-002",
    "How can we reduce lapse in the Cheque payment mode segment?",
    bi.PRESCRIPTIVE, M,
    ["How to reduce lapse in Cheque payers?",
     "What can we do for Cheque segment lapse?",
     "Lapse reduction for Cheque payment mode."],
    [_step("s1","Portfolio baseline","portfolio_overview",{},"Reference KPIs."),
     _step("s2","Lapse by payment mode","compare_to_baseline",{"metric":"lapse_rate","dimension":"payment_mode"},"Confirm Cheque lapse."),
     _step("s3","Driver scan inside Cheque","driver_scan",{"metric":"lapse_rate","focus_filters":{"payment_mode":"Cheque"},"candidate_dimensions":["plan_name","income_band","age_band","state","occupation"]},"Sub-segment drivers within Cheque."),
     _step("s4","Multi-metric by payment mode","multi_metric_table",{"dimension":"payment_mode","metrics":["policy_count","lapse_rate","persistency_rate","avg_ape"]},"Full payment mode comparison.")],
    "prescriptive",["lapse_rate","persistency_rate","policy_count"],["policies","persistency"],
))

PRESCRIPTIVE.append(_ps(
    "pres-003",
    "What specific actions would improve 13th month persistency?",
    bi.PRESCRIPTIVE, M,
    ["How to improve 13th month persistency?",
     "What actions would improve 13m persistency?",
     "Improve 13m renewal rate what levers exist?",
     "How do I boost first renewal persistency?"],
    [_step("s1","Portfolio baseline","portfolio_overview",{},"Reference KPIs."),
     _step("s2","13m persistency by product","compare_to_baseline",{"metric":"persistency_13m_rate","dimension":"plan_name"},"Product-level 13m gap."),
     _step("s3","13m persistency by payment mode","compare_to_baseline",{"metric":"persistency_13m_rate","dimension":"payment_mode"},"Payment mode lever."),
     _step("s4","Driver scan","driver_scan",{"metric":"persistency_13m_rate","candidate_dimensions":["plan_name","payment_mode","income_band","age_band","occupation","state","trad_ulip"]},"Full driver scan."),
     _step("s5","Multi-metric by product","multi_metric_table",{"dimension":"plan_name","metrics":["policy_count","persistency_13m_rate","persistency_25m_rate","lapse_rate"],"sort_by":"persistency_13m_rate","ascending":True},"Product persistency ladder.")],
    "prescriptive",["persistency_13m_rate","persistency_25m_rate","lapse_rate"],["policies","persistency"],
))

PRESCRIPTIVE.append(_ps(
    "pres-004",
    "What steps should management take to reduce surrender rates?",
    bi.PRESCRIPTIVE, M,
    ["How to reduce surrender rate?",
     "What actions would reduce policy surrenders?",
     "How can we fix high surrender rates?"],
    [_step("s1","Portfolio baseline","portfolio_overview",{},"Reference KPIs."),
     _step("s2","Surrender by product","compare_to_baseline",{"metric":"surrender_rate","dimension":"plan_name"},"Product-level surrender gap."),
     _step("s3","Surrender by income band","compare_to_baseline",{"metric":"surrender_rate","dimension":"income_band"},"Income lens."),
     _step("s4","Driver scan","driver_scan",{"metric":"surrender_rate","candidate_dimensions":["plan_name","payment_mode","income_band","age_band","trad_ulip","state"]},"Full driver scan.")],
    "prescriptive",["surrender_rate","lapse_rate","policy_count"],["policies"],
))

PRESCRIPTIVE.append(_ps(
    "pres-005",
    "What interventions would improve agent quality and portfolio persistency?",
    bi.PRESCRIPTIVE, M,
    ["How to improve agent quality?",
     "What steps improve sales rep quality?",
     "How to fix agent PIVC mismatch issues?",
     "How can we improve sales quality?"],
    [_step("s1","Portfolio baseline","portfolio_overview",{},"Reference KPIs."),
     _step("s2","PIVC by sales type","compare_to_baseline",{"metric":"pivc_mismatch_rate","dimension":"sales_type"},"PIVC mismatch by agent type."),
     _step("s3","RCU by sales type","compare_to_baseline",{"metric":"rcu_rejection_rate","dimension":"sales_type"},"RCU rejection by agent type."),
     _step("s4","Driver scan","driver_scan",{"metric":"pivc_mismatch_rate","candidate_dimensions":["sales_type","plan_name","state","trad_ulip"]},"PIVC sub-dimension drivers."),
     _step("s5","Agent summary","multi_metric_table",{"dimension":"sales_type","metrics":["policy_count","pivc_mismatch_rate","rcu_rejection_rate","lapse_rate","persistency_rate"]},"Agent quality scorecard.")],
    "prescriptive",["pivc_mismatch_rate","rcu_rejection_rate","lapse_rate"],["policies","sales_reps"],
))

PRESCRIPTIVE.append(_ps(
    "pres-006",
    "How should management prioritise retention effort to achieve the highest save rate?",
    bi.PRESCRIPTIVE, M,
    ["How to improve retention strategy?",
     "How can we improve retention effectiveness?",
     "What should the retention team focus on?",
     "How to prioritise retention effort?",
     "Improve retention where to focus?"],
    [_step("s1","Portfolio baseline","portfolio_overview",{},"Reference KPIs."),
     _step("s2","Lapse by product","compare_to_baseline",{"metric":"lapse_rate","dimension":"plan_name"},"Products with highest lapse."),
     _step("s3","Lapse by payment mode","compare_to_baseline",{"metric":"lapse_rate","dimension":"payment_mode"},"Payment mode as lever."),
     _step("s4","Driver scan","driver_scan",{"metric":"lapse_rate","candidate_dimensions":["plan_name","payment_mode","income_band","age_band","state"]},"Rank all levers.")],
    "prescriptive",["lapse_rate","persistency_rate","avg_ape"],["policies","persistency"],
))

for pid, stmt, metric, dims, qs in [
    ("pres-007","How can we improve persistency among ULIP product holders?","lapse_rate",["plan_name","payment_mode","income_band","age_band"],["How to reduce ULIP lapse?","Improve ULIP persistency."]),
    ("pres-008","How can we reduce lapse rate in the low-income segment?","lapse_rate",["plan_name","payment_mode","occupation","age_band"],["Reduce lapse in low income segment."]),
    ("pres-009","What product-level changes would improve overall persistency?","persistency_rate",["plan_name","trad_ulip","payment_mode","income_band"],["Product strategy to improve persistency."]),
    ("pres-010","How to improve 25th month persistency across the portfolio?","persistency_25m_rate",["plan_name","payment_mode","income_band","age_band"],["Improve 25m persistency.","How to boost 25th month renewal rates?"]),
    ("pres-011","What interventions would reduce the PIVC mismatch rate?","pivc_mismatch_rate",["sales_type","plan_name","state","trad_ulip"],["Reduce PIVC mismatch.","How to fix PIVC issues?"]),
    ("pres-012","How can management reduce RCU rejection rates?","rcu_rejection_rate",["sales_type","plan_name","state"],["Reduce RCU rejections.","How to improve RCU acceptance rate?"]),
    ("pres-013","How to improve persistency in the Bancassurance channel?","lapse_rate",["plan_name","income_band","age_band","state"],["Improve banca persistency."]),
    ("pres-014","How can we improve persistency among young policyholders?","lapse_rate",["plan_name","payment_mode","income_band","occupation"],["Young customer persistency improvement."]),
    ("pres-015","What steps would improve persistency among traditional product holders?","lapse_rate",["plan_name","payment_mode","income_band","state"],["Traditional product persistency improvement."]),
    ("pres-016","How to improve first year persistency across the portfolio?","persistency_13m_rate",["plan_name","payment_mode","sales_type","state"],["Improve first year persistency."]),
    ("pres-017","What product mix changes would improve portfolio persistency?","persistency_rate",["plan_name","trad_ulip","ppt_band","term_band"],["Product mix improvement for persistency."]),
    ("pres-018","How to improve free-look cancellation rates?","free_look_rate",["plan_name","sales_type","state","trad_ulip"],["Reduce free-look cancellations."]),
    ("pres-019","How can we improve collection efficiency for renewal premiums?","renewal_premium_collection_rate",["payment_mode","plan_name","state","income_band"],["Improve collection efficiency."]),
    ("pres-020","What actions would improve persistency for high-sum-assured policies?","persistency_rate",["plan_name","payment_mode","age_band","occupation"],["Improve high sum assured persistency."]),
    ("pres-021","How can we reduce lapse in the 36-50 age band?","lapse_rate",["plan_name","payment_mode","income_band","occupation"],["Reduce lapse in middle age group."]),
    ("pres-022","What channel strategy changes would improve persistency?","persistency_rate",["plan_name","trad_ulip","payment_mode","income_band"],["Channel strategy for persistency."]),
    ("pres-023","How to reduce complaint rates across sales channels?","complaint_rate",["sales_type","plan_name","state"],["Reduce sales complaints."]),
    ("pres-024","What interventions would reduce policy surrender in the first 3 years?","surrender_rate",["plan_name","payment_mode","income_band","age_band"],["Reduce early surrender."]),
    ("pres-025","How can the claims team improve settlement ratios?","claim_settlement_ratio",["cause_of_death","state"],["Improve claim settlement ratio."]),
    ("pres-026","What product design changes would reduce lapse risk?","lapse_rate",["plan_name","trad_ulip","ppt_band","term_band"],["Product design for lower lapse."]),
    ("pres-027","How to improve persistency for salaried segment policyholders?","lapse_rate",["plan_name","payment_mode","income_band","age_band"],["Salaried customer persistency."]),
    ("pres-028","What steps would reduce lapse in the North zone?","lapse_rate",["plan_name","payment_mode","income_band","state"],["North zone lapse reduction."]),
    ("pres-029","How can we improve premium collection rates for ECS mode?","renewal_premium_collection_rate",["payment_mode","plan_name","income_band"],["ECS collection improvement."]),
    ("pres-030","How can we improve portfolio quality metrics across all dimensions?","persistency_rate",["plan_name","payment_mode","state","sales_type","income_band"],["Overall portfolio quality improvement."]),
]:
    PRESCRIPTIVE.append(_ps(
        pid, stmt, bi.PRESCRIPTIVE, M, qs,
        [_step("s1","Portfolio baseline","portfolio_overview",{},"Reference KPIs."),
         _step("s2","Metric by product","compare_to_baseline",{"metric":metric,"dimension":"plan_name"},"Product-level metric gap."),
         _step("s3","Metric by payment mode","compare_to_baseline",{"metric":metric,"dimension":"payment_mode"},"Payment mode as lever."),
         _step("s4","Driver scan all dimensions","driver_scan",{"metric":metric,"candidate_dimensions":dims},"Rank all levers by gap x volume.")],
        "prescriptive", [metric,"policy_count","persistency_rate"], ["policies","persistency"],
    ))

# ==========================================================================
# BUCKET C — Cross-domain Correlation (20 PSs)
# ==========================================================================
CROSS_DOMAIN = []

CROSS_DOMAIN.append(_ps(
    "xd-001",
    "What is the relationship between claims experience and persistency rate across products?",
    bi.CROSS_DOMAIN, M,
    ["What is the relation between claims and persistency rate?",
     "Is there a link between claims and persistency?",
     "How are claims related to persistency?",
     "Do high claim products also have low persistency?",
     "Claims and persistency relationship."],
    [_step("s1","Portfolio baseline","portfolio_overview",{},"Reference KPIs."),
     _step("s2","Claim settlement by product","compare_to_baseline",{"metric":"claim_settlement_ratio","dimension":"plan_name"},"Claims per product."),
     _step("s3","Persistency by product","compare_to_baseline",{"metric":"persistency_rate","dimension":"plan_name"},"Persistency per product."),
     _step("s4","Claims vs persistency correlation","correlation_analysis",{"metric_a":"persistency_rate","metric_b":"claim_settlement_ratio","dimension":"plan_name"},"Pearson r and combined-risk segments."),
     _step("s5","Multi-metric product table","multi_metric_table",{"dimension":"plan_name","metrics":["policy_count","persistency_rate","lapse_rate","claim_settlement_ratio","avg_ape"]},"Side-by-side product view.")],
    "correlation",["persistency_rate","claim_settlement_ratio"],["policies","claims","persistency"],
))

CROSS_DOMAIN.append(_ps(
    "xd-002",
    "Is PIVC mismatch linked to higher lapse rates? Quantify the relationship.",
    bi.CROSS_DOMAIN, M,
    ["Is PIVC mismatch linked to higher lapse?",
     "Does PIVC mismatch affect persistency?",
     "Relation between PIVC and lapse rate."],
    [_step("s1","Portfolio baseline","portfolio_overview",{},"Reference KPIs."),
     _step("s2","PIVC by product","compare_to_baseline",{"metric":"pivc_mismatch_rate","dimension":"plan_name"},"PIVC per product."),
     _step("s3","Lapse by product","compare_to_baseline",{"metric":"lapse_rate","dimension":"plan_name"},"Lapse per product."),
     _step("s4","PIVC vs lapse correlation","correlation_analysis",{"metric_a":"pivc_mismatch_rate","metric_b":"lapse_rate","dimension":"plan_name"},"Correlation PIVC vs lapse."),
     _step("s5","Product quality table","multi_metric_table",{"dimension":"plan_name","metrics":["policy_count","pivc_mismatch_rate","rcu_rejection_rate","lapse_rate"]},"Sales quality and persistency.")],
    "correlation",["pivc_mismatch_rate","lapse_rate","rcu_rejection_rate"],["policies","sales_reps"],
))

CROSS_DOMAIN.append(_ps(
    "xd-003",
    "How does payment mode affect both lapse rate and persistency simultaneously?",
    bi.CROSS_DOMAIN, M,
    ["Does payment mode affect lapse and persistency together?",
     "Payment mode effect on both lapse and persistency.",
     "Relation between payment mode and persistency."],
    [_step("s1","Portfolio baseline","portfolio_overview",{},"Reference KPIs."),
     _step("s2","Lapse by payment mode","compare_to_baseline",{"metric":"lapse_rate","dimension":"payment_mode"},"Lapse per payment mode."),
     _step("s3","Persistency by payment mode","compare_to_baseline",{"metric":"persistency_rate","dimension":"payment_mode"},"Persistency per payment mode."),
     _step("s4","Lapse vs persistency correlation","correlation_analysis",{"metric_a":"lapse_rate","metric_b":"persistency_13m_rate","dimension":"payment_mode"},"Correlation at payment mode level."),
     _step("s5","Payment mode summary","multi_metric_table",{"dimension":"payment_mode","metrics":["policy_count","lapse_rate","persistency_rate","persistency_13m_rate","avg_ape"]},"Multi-metric table.")],
    "correlation",["lapse_rate","persistency_rate","persistency_13m_rate"],["policies","persistency"],
))

for pid, stmt, ma, mb, dim, qs in [
    ("xd-004","Do agents with high sales volume also have good persistency quality?","policy_count","persistency_rate","sales_type",["Agent volume vs persistency.","Does high sales volume hurt persistency?"]),
    ("xd-005","Is there a link between income level and lapse rate?","lapse_rate","avg_ape","income_band",["Income and lapse correlation.","Does income level affect lapse?"]),
    ("xd-006","What is the relationship between sum assured and persistency rate?","avg_sum_assured","persistency_rate","plan_name",["Sum assured and persistency link.","Does sum assured affect persistency?"]),
    ("xd-007","Do customer segments with high lapse rates also have high claims rates?","lapse_rate","claim_settlement_ratio","income_band",["Lapse and claims by income.","Dual risk lapse and claims."]),
    ("xd-008","Is RCU rejection rate linked to subsequent lapse rates?","rcu_rejection_rate","lapse_rate","plan_name",["RCU rejection and lapse link.","Does RCU rejection predict lapse?"]),
    ("xd-009","What is the relationship between premium size and persistency?","avg_ape","persistency_rate","plan_name",["Premium size and persistency.","Does premium size affect renewal rates?"]),
    ("xd-010","Is there a correlation between age band and lapse rate?","lapse_rate","avg_ape","age_band",["Age and lapse correlation.","Does age affect lapse rate?"]),
    ("xd-011","Do products with high PIVC mismatch also have high surrender rates?","pivc_mismatch_rate","surrender_rate","plan_name",["PIVC mismatch and surrender.","Sales quality and surrender."]),
    ("xd-012","What is the relationship between persistency bucket and claim frequency?","persistency_13m_rate","claim_settlement_ratio","plan_name",["Persistency and claim frequency."]),
    ("xd-013","Is product term length related to lapse rate?","lapse_rate","avg_sum_assured","term_band",["Policy term and lapse.","Does term length affect lapse?"]),
    ("xd-014","Do states with high PIVC mismatch also show high lapse rates?","pivc_mismatch_rate","lapse_rate","state",["State PIVC and lapse.","Geographic sales quality and persistency."]),
    ("xd-015","Is there a relationship between premium paying term and persistency?","persistency_rate","avg_ape","ppt_band",["PPT and persistency.","Premium paying term and renewal rates."]),
    ("xd-016","Do high-claim products also show high surrender rates?","claim_settlement_ratio","surrender_rate","plan_name",["Claims and surrender.","High claim high surrender products."]),
    ("xd-017","What is the correlation between agent PIVC rate and portfolio lapse rate?","pivc_mismatch_rate","lapse_rate","sales_type",["Agent quality and portfolio lapse."]),
    ("xd-018","Is there a link between occupation type and policy lapse?","lapse_rate","avg_ape","occupation",["Occupation and lapse link.","Does job type affect lapse?"]),
    ("xd-019","What is the relationship between income band and claim rates?","avg_ape","claim_settlement_ratio","income_band",["Income and claims relationship."]),
    ("xd-020","Do products with higher APE also show better persistency?","avg_ape","persistency_rate","plan_name",["APE and persistency.","Does ticket size affect persistency?"]),
]:
    CROSS_DOMAIN.append(_ps(
        pid, stmt, bi.CROSS_DOMAIN, M, qs,
        [_step("s1","Portfolio baseline","portfolio_overview",{},"Reference KPIs."),
         _step("s2",f"{ma} by {dim}","compare_to_baseline",{"metric":ma,"dimension":dim},f"{ma} per {dim}."),
         _step("s3",f"{mb} by {dim}","compare_to_baseline",{"metric":mb,"dimension":dim},f"{mb} per {dim}."),
         _step("s4",f"Correlation {ma} vs {mb}","correlation_analysis",{"metric_a":ma,"metric_b":mb,"dimension":dim},f"Pearson r and combined-risk segments."),
         _step("s5","Multi-metric summary","multi_metric_table",{"dimension":dim,"metrics":["policy_count",ma,mb]},"Side-by-side table.")],
        "correlation", [ma,mb,"policy_count"], ["policies","persistency","claims"],
    ))

# ==========================================================================
# EXPORT
# ==========================================================================
ALL_PS4 = GEO_DIAGNOSTIC + PRESCRIPTIVE + CROSS_DOMAIN

if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "business_problem_statements_ps4.json")
    with open(out, "w") as f:
        json.dump(ALL_PS4, f, indent=2)
    intents = {}
    for x in ALL_PS4:
        intents[x["business_intent"]] = intents.get(x["business_intent"], 0) + 1
    print(f"Generated {len(ALL_PS4)} PSs:")
    for k, v in sorted(intents.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print(f"Saved to {out}")
