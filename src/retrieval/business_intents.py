"""
business_intents.py
===================
The business-intent taxonomy and the domain concept ontology that lets the
retriever bridge management vocabulary ("where should we focus?") to the
analytical concepts actually present in the dataset (lapse, surrender,
retention, value, concentration).

Nothing here executes anything - it is pure vocabulary + classification
metadata used by `business_retrieval.py` and `analysis_planner.py`.
"""

from __future__ import annotations

from typing import Dict, List

# =====================================================================
# 1. BUSINESS INTENT TAXONOMY
# =====================================================================
# Each intent maps to the section letters of the business-question brief.

PORTFOLIO_HEALTH = "PORTFOLIO_HEALTH"            # A
PRODUCT_PERFORMANCE = "PRODUCT_PERFORMANCE"      # B
CHANNEL_PERFORMANCE = "CHANNEL_PERFORMANCE"      # C
CUSTOMER_SEGMENTATION = "CUSTOMER_SEGMENTATION"  # D
LAPSE_ANALYSIS = "LAPSE_ANALYSIS"                # E
SURRENDER_ANALYSIS = "SURRENDER_ANALYSIS"        # F
RETENTION_STRATEGY = "RETENTION_STRATEGY"        # G
CLAIMS_RISK = "CLAIMS_RISK"                      # H
PROFITABILITY = "PROFITABILITY"                  # I
GEOGRAPHIC = "GEOGRAPHIC"                        # J
AGENT_PERFORMANCE = "AGENT_PERFORMANCE"          # K
ANOMALY_CONCENTRATION = "ANOMALY_CONCENTRATION"  # L
COMPARATIVE = "COMPARATIVE"                      # M
ROOT_CAUSE = "ROOT_CAUSE"                        # N
CUSTOMER_TARGETING = "CUSTOMER_TARGETING"        # actionable shortlists
PRESCRIPTIVE = "PRESCRIPTIVE"                    # P - how to improve / action planning
CROSS_DOMAIN = "CROSS_DOMAIN"                    # Q - relation between two metric families
SIMPLE_RETRIEVAL = "SIMPLE_RETRIEVAL"            # falls through to the legacy engine

BUSINESS_INTENTS: Dict[str, Dict[str, str]] = {
    PORTFOLIO_HEALTH: {
        "label": "Executive / Portfolio Health",
        "description": "Overall state of the book, biggest risks, what needs management attention.",
    },
    PRODUCT_PERFORMANCE: {
        "label": "Product Performance & Strategy",
        "description": "Which products win or lose across volume, value, persistency and claims.",
    },
    CHANNEL_PERFORMANCE: {
        "label": "Sales Channel Performance",
        "description": "Growth vs persistency trade-off across distribution channels.",
    },
    CUSTOMER_SEGMENTATION: {
        "label": "Customer Segmentation & Value",
        "description": "Which customer segments are valuable, which are risky, and how they differ.",
    },
    LAPSE_ANALYSIS: {
        "label": "Lapse / Persistency Analysis",
        "description": "What is associated with policy lapse and where lapse risk concentrates.",
    },
    SURRENDER_ANALYSIS: {
        "label": "Surrender Analysis",
        "description": "What distinguishes surrendered policies and where surrender risk sits.",
    },
    RETENTION_STRATEGY: {
        "label": "Retention Strategy",
        "description": "Effectiveness of retention effort and where to spend it next.",
    },
    CLAIMS_RISK: {
        "label": "Claims & Risk",
        "description": "Claim frequency, mix, early claims and where claims risk concentrates.",
    },
    PROFITABILITY: {
        "label": "Profitability / Claims Economics",
        "description": "Loss ratios and claims-to-premium economics by product, channel and segment.",
    },
    GEOGRAPHIC: {
        "label": "Geographic / Zone Analysis",
        "description": "Zone, state and branch level performance and risk.",
    },
    AGENT_PERFORMANCE: {
        "label": "Agent / Sales Performance",
        "description": "Agent quality beyond raw volume - persistency, value and portfolio quality.",
    },
    ANOMALY_CONCENTRATION: {
        "label": "Anomaly / Concentration / Risk Detection",
        "description": "Outliers, concentration risk and unusual patterns worth investigating.",
    },
    COMPARATIVE: {
        "label": "Comparative / Benchmarking",
        "description": "Multi-metric side-by-side comparison of named groups.",
    },
    ROOT_CAUSE: {
        "label": "Root-Cause / Diagnostic",
        "description": "Why a specific group under-performs; multi-hop driver decomposition.",
    },
    CUSTOMER_TARGETING: {
        "label": "Actionable Targeting",
        "description": "Ranked lists of individual customers, policies or agents to act on.",
    },
    PRESCRIPTIVE: {
        "label": "Prescriptive / Action Planning",
        "description": "How to improve a metric — identifies the highest-ROI levers with evidence-backed rationale.",
    },
    CROSS_DOMAIN: {
        "label": "Cross-Domain Correlation",
        "description": "Relationship between two distinct metric families (e.g. claims vs persistency) across a shared dimension.",
    },
    SIMPLE_RETRIEVAL: {
        "label": "Simple Retrieval",
        "description": "A single lookup or aggregate answered by the classic query engine.",
    },
}

# Analysis tiers - visible in the architecture, per the brief.
TIER_RETRIEVAL = "simple_retrieval"      # one query is enough
TIER_ANALYTICAL = "analytical"           # aggregation + derived metric + comparison
TIER_MULTI_STEP = "multi_step_business"  # several operations chained into a plan


# =====================================================================
# 2. CONCEPT ONTOLOGY
# =====================================================================
# concept -> surface forms.  Used to (a) expand a user query before TF-IDF and
# (b) score concept overlap against each Problem Statement's declared concepts.

CONCEPT_ONTOLOGY: Dict[str, List[str]] = {
    "portfolio": [
        "portfolio", "book", "book of business", "overall", "aggregate", "company wide",
        "company-wide", "whole business", "entire portfolio", "management level",
        "executive", "senior management", "board", "state of the business", "health check",
    ],
    "health": [
        "health", "healthy", "how are we doing", "performance", "state", "assessment",
        "shape", "wellbeing", "status", "scorecard", "kpi", "key indicators",
    ],
    "risk": [
        "risk", "risky", "exposure", "danger", "threat", "vulnerability", "downside",
        "biggest risks", "concern", "concerning", "red flag", "warning", "worry",
    ],
    "attention": [
        "attention", "focus", "prioritise", "prioritize", "priority", "where should",
        "act on", "action", "intervene", "intervention", "investigate", "look into",
        "needs work", "management attention", "escalate", "next steps",
        "where do we start", "start with", "first priority", "point me",
        "worried about", "should we be worried", "visit first", "point the",
        "agenda", "worth the effort", "earns its keep", "tightest ship",
    ],
    "lapse": [
        "lapse", "lapsed", "lapsing", "lapsation", "stopped paying", "premium default",
        "non payment", "non-payment", "dropped out", "discontinued", "persistency",
        "persistent", "renewal", "stickiness", "churn", "attrition", "drop off",
        "stick", "sticks", "does not stick", "hold up", "holds up", "survive",
        "survives", "keep paying", "keeps paying", "still paying", "walk away",
        "leave us", "stay with us", "drift away", "drop out", "bleeding",
        "losing policies", "lose policies", "lost policies", "fall over",
        "falls over", "quietly bleeding",
    ],
    "surrender": [
        "surrender", "surrendered", "surrendering", "cash out", "cashed out",
        "early exit", "exited", "withdrew", "withdrawal", "encashment",
        "cash-out", "cash-outs", "cashing out", "cash in", "pull their money",
        "take the money out",
    ],
    "retention": [
        "retention", "retain", "retained", "save the policy", "win back", "winback",
        "reinstatement", "grace period", "outreach", "save rate", "recover",
        "retention team", "retention campaign", "retention effort",
    ],
    "claims": [
        "claim", "claims", "claimed", "payout", "payouts", "settlement", "settled",
        "repudiated", "repudiation", "rejected claim", "death claim", "health claim",
        "critical illness", "maturity claim", "early claim", "claims burden",
    ],
    "economics": [
        "loss ratio", "economics", "profitability", "profitable", "unprofitable",
        "margin", "claims to premium", "claims-to-premium", "cost", "attractive",
        "unattractive", "value destruction", "bleeding", "unfavourable", "unfavorable",
        "favourable", "favorable", "economic",
    ],
    "product": [
        "product", "products", "plan", "plans", "term life", "endowment", "ulip",
        "money back", "whole life", "annuity", "child plan", "health rider",
        "protection", "savings", "investment", "retirement", "product mix",
        "product line", "portfolio of products",
    ],
    "channel": [
        "channel", "channels", "distribution", "sales channel", "agency",
        "bancassurance", "banca", "broker", "online", "direct", "digital",
        "intermediary", "route to market", "sourcing",
    ],
    "customer": [
        "customer", "customers", "policyholder", "policyholders", "client", "clients",
        "segment", "segments", "segmentation", "demographic", "persona", "cohort",
        "smoker", "non-smoker", "income", "age group", "gender", "occupation",
        "married", "marital",
    ],
    "targeting": [
        "list", "call list", "shortlist", "target list", "who should we",
        "which customers should", "which policies should", "name them",
        "give me the", "top 10 customers", "contact list", "candidates",
        "prioritise for", "prioritize for", "least afford to lose",
        "can we win back", "win back", "cross-sell", "cross sell",
        "room to buy", "underinsured", "under-insured",
    ],
    "value": [
        "value", "valuable", "high value", "high-value", "premium size", "ticket size",
        "sum assured", "coverage", "worth", "revenue", "contribution", "big ticket",
        "top customers", "clv", "lifetime value",
    ],
    "agent": [
        "agent", "agents", "advisor", "advisors", "sales person", "salesperson",
        "sales force", "salesforce", "producer", "producers", "seller", "rep",
        "target achievement", "nbp", "quota",
    ],
    "zone": [
        "zone", "zones", "region", "regional", "geography", "geographic", "location",
        "city", "state", "branch", "branches", "territory", "north", "south",
        "east", "west", "central", "part of the country", "parts of the country",
        "office", "offices", "regional director", "geographically",
    ],
    "concentration": [
        "concentration", "concentrated", "diversified", "diversification",
        "spread", "hhi", "herfindahl", "top 10", "dominated", "reliance",
        "over-reliant", "single point", "clustering", "handful", "a few",
        "account for most", "bulk of", "majority of", "rests on", "depend on",
        "dependent on", "share of the total", "largest few", "biggest few",
        "over-exposed", "exposure to one",
    ],
    "anomaly": [
        "anomaly", "anomalies", "unusual", "outlier", "outliers", "strange",
        "abnormal", "irregular", "suspicious", "stands out", "odd", "surprising",
        "unexpected", "deviation",
    ],
    "comparison": [
        "compare", "comparison", "versus", "vs", "against", "side by side",
        "benchmark", "benchmarking", "relative", "difference between", "how does",
        "better than", "worse than", "rank", "ranking", "league table",
    ],
    "diagnostic": [
        "why", "reason", "cause", "causes", "driver", "drivers", "driving",
        "explain", "explanation", "root cause", "what is behind", "what accounts for",
        "underlying", "factors", "contributing", "diagnose",
    ],
    "recommendation": [
        "recommend", "recommendation", "should we", "should management", "what should",
        "advise", "advice", "strategy", "plan of action", "propose", "suggest",
        "best course", "next best action",
    ],
    "growth": [
        "growth", "grow", "volume", "new business", "acquisition", "sales volume",
        "policy count", "how many policies", "scale", "market", "expansion",
    ],
    "persistency": [
        "persistency", "13th month", "13m", "25th month", "25m", "37th month", "37m",
        "renewal", "renewal status", "renewal paid", "renewal pending", "renewal premium",
        "persistency bucket", "payment mode", "ecs", "auto debit", "cheque", "online",
        "upi", "credit card", "cash", "persistency rate", "renewal efficiency",
    ],
    "sales_quality": [
        "sales quality", "pivc", "pivc mismatch", "pivc number mismatch", "pivc concern",
        "rcu", "rcu rejection", "rcu rejections", "free look", "free-look", "cancellation",
        "sales complaint", "complaint rate", "first year persistency", "active", "inactive",
        "mis-selling", "sales risk",
    ],
    "cause_of_death": [
        "cause of death", "cancer", "heart attack", "altered sensorium", "natural death",
        "accident", "claim settlement", "tat", "turnaround time", "intimation", "wip",
    ],
    "product_category": [
        "trad", "traditional", "ulip", "unit linked", "flexi edge", "guaranteed assured income",
        "guaranteed fortune", "short term", "long term",
    ],
    "prescriptive": [
        "how to improve", "how can i improve", "how can we improve", "how do i improve",
        "how do we improve", "how to increase", "how to reduce", "how to boost",
        "how to fix", "how to address", "what actions", "what steps", "what can be done",
        "what should management do", "what interventions", "lever", "levers",
        "what can we do", "what would help", "improve the", "reduce the",
        "increase the", "boost the", "fix the", "action plan", "action items",
        "roadmap", "what is the remedy", "recommendations to improve",
    ],
    "correlation": [
        "relation between", "relationship between", "linked to", "link between",
        "associated with", "does affect", "connection between", "impact of",
        "influence of", "correlation between", "correlated", "relate to",
        "does claims affect", "does lapse affect", "do claims affect",
        "relationship of", "how are they related", "is there a link", "does x affect",
        "what is the effect of", "how does one affect", "jointly", "together",
    ],
}

# reverse index: surface term -> concepts
_TERM_TO_CONCEPTS: Dict[str, List[str]] = {}
for _concept, _terms in CONCEPT_ONTOLOGY.items():
    for _t in _terms:
        _TERM_TO_CONCEPTS.setdefault(_t, []).append(_concept)


def concepts_in(text: str) -> List[str]:
    """Concepts whose surface forms appear in `text` (longest match wins)."""
    t = f" {text.lower()} "
    found: List[str] = []
    for term, concepts in _TERM_TO_CONCEPTS.items():
        needle = f" {term} " if " " in term else None
        hit = (needle in t) if needle else (f" {term} " in t or f" {term}s " in t
                                           or f" {term}," in t or f" {term}." in t
                                           or f" {term}?" in t)
        if hit:
            for c in concepts:
                if c not in found:
                    found.append(c)
    return found


def expand_query(text: str, max_terms: int = 40) -> str:
    """Append the canonical concept names (and a few sibling terms) to a query.

    This is the cheap, offline substitute for a dense retriever: a question
    phrased as "where should management focus retention effort" gains the
    tokens `retention lapse surrender attention priority ...` so that it can
    reach Problem Statements that never use the word "focus".
    """
    found = concepts_in(text)
    extra: List[str] = []
    for c in found:
        extra.append(c)
        extra.extend(CONCEPT_ONTOLOGY[c][:4])
    seen, out = set(), []
    for w in extra:
        if w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) >= max_terms:
            break
    return text.lower() + " " + " ".join(out)


# =====================================================================
# 3. INTENT TRIGGERS
# =====================================================================
# Weighted phrases used to classify a free-text question into a business
# intent when Problem-Statement retrieval alone is ambiguous.

INTENT_TRIGGERS: Dict[str, Dict[str, float]] = {
    PORTFOLIO_HEALTH: {
        "portfolio health": 3.0, "overall portfolio": 2.5, "how healthy": 3.0,
        "biggest risks": 3.0, "management-level": 2.5, "management level": 2.5,
        "state of the portfolio": 3.0, "portfolio assessment": 3.0,
        "executive summary": 2.5, "key indicators": 2.0, "overall health": 3.0,
        "management attention": 2.0, "strongest and weakest": 2.5,
        "assessment of the current portfolio": 3.0, "how is the portfolio": 3.0,
    },
    PRODUCT_PERFORMANCE: {
        "which product": 2.5, "products perform": 3.0, "product performance": 3.0,
        "product strategy": 3.0, "products need": 2.5, "product mix": 2.0,
        "products should": 2.5, "strong-performing product": 3.0,
        "product portfolio": 2.0, "by product": 1.5,
    },
    CHANNEL_PERFORMANCE: {
        "sales channel": 3.0, "which channel": 2.5, "channel performance": 3.0,
        "distribution channel": 3.0, "channels generate": 2.5, "by channel": 1.5,
        "agency vs": 2.5, "bancassurance": 1.5, "growth and persistency": 3.0,
    },
    CUSTOMER_SEGMENTATION: {
        "customer segment": 3.0, "segments are most": 3.0, "which customers": 2.0,
        "customer value": 2.5, "high-value customer": 3.0, "high value customer": 3.0,
        "characteristics distinguish": 3.0, "smoker": 1.5, "income": 1.0,
        "demographic": 2.0, "customer characteristics": 2.5,
    },
    LAPSE_ANALYSIS: {
        "lapse": 2.0, "lapsed": 1.5, "lapsing": 2.5, "persistency": 2.0,
        "lapse rate": 3.0, "lapse risk": 3.0, "improve persistency": 3.0,
        "associated with policy lapse": 3.0, "highest lapse": 2.5,
    },
    SURRENDER_ANALYSIS: {
        "surrender": 2.0, "surrendered": 2.0, "surrender rate": 3.0,
        "surrender risk": 3.0, "driving policy surrender": 3.0,
        "likely to surrender": 3.0, "surrender reason": 2.5,
    },
    RETENTION_STRATEGY: {
        "retention": 2.0, "retention channel": 3.0, "retention effort": 3.0,
        "retention team": 3.0, "retention strategy": 3.0, "retention intervention": 3.0,
        "most effective": 1.5, "win back": 2.0, "retention effectiveness": 3.0,
        "candidates for retention": 3.0,
    },
    CLAIMS_RISK: {
        "claims risk": 3.0, "claim activity": 3.0, "claim type": 2.5,
        "early claim": 3.0, "adverse selection": 3.0, "claims team": 2.5,
        "claims burden": 3.0, "driving claims": 3.0, "repudiat": 2.5,
    },
    PROFITABILITY: {
        "loss ratio": 3.0, "claims economics": 3.0, "economically attractive": 3.0,
        "unfavorable claims": 3.0, "unfavourable claims": 3.0, "profitab": 2.5,
        "claims relative to premium": 3.0, "claims-to-premium": 3.0,
        "economics": 2.0,
    },
    GEOGRAPHIC: {
        "zone": 2.0, "which zones": 3.0, "region": 2.0, "geographic": 2.5,
        "branch": 2.0, "state": 1.0, "zones perform": 3.0, "by zone": 2.0,
    },
    AGENT_PERFORMANCE: {
        "agent": 2.0, "agents": 2.0, "which agents": 3.0, "agent performance": 3.0,
        "sales performance": 2.5, "outperform targets": 3.0, "advisor": 2.0,
        "portfolio quality": 2.0,
    },
    ANOMALY_CONCENTRATION: {
        "unusual": 2.5, "anomal": 3.0, "outlier": 3.0, "concentration": 3.0,
        "concentrated": 2.5, "unusually": 2.5, "patterns management": 3.0,
        "statistical": 2.0, "stands out": 2.5,
    },
    COMPARATIVE: {
        "compare": 3.0, "comparison": 3.0, " vs ": 3.0, "versus": 3.0,
        "side by side": 3.0, "benchmark": 2.5, "difference between": 2.5,
        "across volume": 2.5,
    },
    ROOT_CAUSE: {
        "why is": 3.0, "why does": 3.0, "why are": 3.0, "what is driving": 3.0,
        "what factors": 3.0, "root cause": 3.0, "what explains": 3.0,
        "reasons behind": 3.0, "what appears to distinguish": 3.0,
        "performing poorly": 2.5,
    },
    CUSTOMER_TARGETING: {
        "which customers should": 3.0, "prioritize for": 2.5, "prioritise for": 2.5,
        "target list": 3.0, "who should we": 2.5, "shortlist": 3.0,
        "customers to contact": 3.0, "best candidates": 3.0,
    },
    PRESCRIPTIVE: {
        "how to improve": 4.0, "how can i improve": 4.0, "how can we improve": 4.0,
        "how do i improve": 4.0, "how do we improve": 4.0, "how to increase": 3.5,
        "how to reduce": 3.5, "how to boost": 3.5, "how to fix": 3.5,
        "what actions": 3.0, "what steps": 3.0, "what can be done": 3.0,
        "what interventions": 3.5, "action plan": 3.0, "action items": 3.0,
        "what should management do": 3.5, "what can we do": 3.0,
        "recommendations to improve": 3.5, "what would help": 2.5,
        "improve persistency": 4.0, "reduce lapse": 4.0, "improve retention": 3.5,
    },
    CROSS_DOMAIN: {
        "relation between": 4.0, "relationship between": 4.0, "linked to": 3.0,
        "link between": 4.0, "does affect": 3.5, "connection between": 4.0,
        "impact of": 3.0, "influence of": 3.0, "correlation between": 4.0,
        "is there a link": 4.0, "how are they related": 4.0, "how does one affect": 3.5,
        "claims and persistency": 4.0, "lapse and claims": 4.0,
        "pivc and lapse": 4.0, "claims affect persistency": 4.0,
    },
}


def score_intents(query: str) -> Dict[str, float]:
    """Trigger-phrase score per business intent (0 when nothing fires)."""
    q = f" {query.lower()} "
    scores: Dict[str, float] = {}
    for intent, triggers in INTENT_TRIGGERS.items():
        s = 0.0
        for phrase, w in triggers.items():
            if phrase in q:
                s += w
        if s:
            scores[intent] = round(s, 2)
    return dict(sorted(scores.items(), key=lambda kv: -kv[1]))


def classify_intent(query: str) -> str:
    scores = score_intents(query)
    return next(iter(scores), SIMPLE_RETRIEVAL)


__all__ = [
    "BUSINESS_INTENTS", "CONCEPT_ONTOLOGY", "INTENT_TRIGGERS",
    "concepts_in", "expand_query", "score_intents", "classify_intent",
    "TIER_RETRIEVAL", "TIER_ANALYTICAL", "TIER_MULTI_STEP",
    "PORTFOLIO_HEALTH", "PRODUCT_PERFORMANCE", "CHANNEL_PERFORMANCE",
    "CUSTOMER_SEGMENTATION", "LAPSE_ANALYSIS", "SURRENDER_ANALYSIS",
    "RETENTION_STRATEGY", "CLAIMS_RISK", "PROFITABILITY", "GEOGRAPHIC",
    "AGENT_PERFORMANCE", "ANOMALY_CONCENTRATION", "COMPARATIVE", "ROOT_CAUSE",
    "CUSTOMER_TARGETING", "PRESCRIPTIVE", "CROSS_DOMAIN", "SIMPLE_RETRIEVAL",
]
