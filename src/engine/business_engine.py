"""
business_engine.py
==================
Top-level orchestrator: natural-language business question in, structured,
evidence-backed business answer out.

    BusinessAnalyticsEngine.ask(question)
        -> EngineResponse(route, plan, step_results, answer, retrieval, timings)

Routing is explicit and visible, which is the architectural point:

    business_analysis   a business Problem Statement matched -> multi-step
                        AnalyticalPlan executed on analytics_core
    adhoc_analysis      no PS matched well, but the question is analytical ->
                        a baseline-anchored comparison is planned on the fly
    simple_retrieval    the question is a lookup -> answered directly from the
                        fact frames by `engine.lookup` (pandas)
    insufficient_data   the question asks for something the dataset does not
                        contain -> the system says so instead of guessing
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
for sub in ["src", "src/analytics", "src/engine", "src/retrieval", "src/llm", "ui"]:
    p = os.path.join(ROOT_DIR, sub)
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd

import analytics_core as ac
import business_intents as bi
import business_retrieval as brt
from analysis_planner import AnalyticalPlan, BusinessPlanner, StepResult
from answer_composer import AnswerComposer, BusinessAnswer, polish

ROUTE_BUSINESS = "business_analysis"
ROUTE_ADHOC = "adhoc_analysis"
ROUTE_SIMPLE = "simple_retrieval"
ROUTE_INSUFFICIENT = "insufficient_data"
ROUTE_GREETING = "greeting_identity"
ROUTE_OUT_OF_DOMAIN = "out_of_domain"

# Concepts the dataset genuinely cannot speak to. Asking about these gets an
# honest refusal rather than a confident-looking table.
OUT_OF_SCOPE = {
    "competitor": "competitor or market-share data",
    "market share": "competitor or market-share data",
    "industry benchmark": "external industry benchmarks",
    "regulator": "regulatory filings or supervisory data",
    "irda": "regulatory filings or supervisory data",
    "solvency": "solvency, capital or reserving data",
    "capital requirement": "solvency, capital or reserving data",
    "reserve": "reserving data",
    "embedded value": "embedded-value or actuarial-valuation data",
    "expense ratio": "expense data",
    "commission": "commission data",
    "operating cost": "expense data",
    "acquisition cost": "acquisition-cost data",
    "investment return": "investment-return data",
    "forecast": "forward-looking or forecast data",
    "will our lapse": "forward-looking or forecast data",
    "will be next year": "forward-looking or forecast data",
    "predict next": "forward-looking or forecast data",
    "projection for next": "forward-looking or forecast data",
    "nps": "customer-satisfaction or NPS data",
    "satisfaction score": "customer-satisfaction or NPS data",
    "credit score": "credit-bureau data",
    "kyc": "KYC or onboarding-document data",
}

GREETING_PATTERNS = {
    "tell me about u", "tell me about you", "tell me about yourself",
    "who are you", "who r u", "what are you", "what r u", "what is this",
    "what is excelsense", "what can you do", "what do you do",
    "help", "help me", "how does this work", "how to use",
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "greetings", "introduce yourself", "what is your purpose", "what is your job",
    "start", "menu"
}

OUT_OF_DOMAIN_PATTERNS = [
    # General trivia / science / tech / coding / creative
    "capital of", "president of", "prime minister", "weather", "forecast for tomorrow",
    "recipe", "how to cook", "write code", "python script", "javascript", "write a poem",
    "tell a joke", "tell me a joke", "movie", "song", "lyrics", "cricket", "football",
    "who won", "world cup", "olympics", "photosynthesis", "quantum", "gravity", "elon musk",
    "translate to", "write an essay", "how to make",
    # Non-life insurance / other financial products
    "car insurance", "auto insurance", "motor insurance", "bike insurance", "vehicle insurance",
    "health insurance", "mediclaim", "dental coverage", "home insurance", "property insurance",
    "travel insurance", "crypto", "cryptocurrency", "bitcoin", "ethereum", "stock market",
    "stock price", "credit card", "personal loan", "home loan", "gold loan", "fixed deposit",
    "mutual fund", "sip return", "demat"
]

CORE_INSURANCE_TERMS = (
    "policy", "policies", "premium", "premiums", "persistency", "lapse", "lapsed", "lapsing",
    "surrender", "surrendered", "claim", "claims", "payout", "payouts", "sum assured", "term",
    "rider", "agent", "agents", "advisor", "advisors", "sales", "salesperson", "branch",
    "channel", "banca", "customer", "customers", "client", "clients", "demographic", "income",
    "marital", "settlement", "repudiation", "death", "turnaround", "tat", "pivc", "rcu",
    "tenure", "zone", "product", "products", "ulip", "endowment", "pension", "fortune",
    "wealth", "smart guaranteed", "assured income", "saving", "savings", "hhi", "loss ratio",
    "ticket size", "conversion", "breakdown", "distribution", "portfolio", "book", "retention",
    "cohort", "kpi", "13m", "25m", "37m", "p13m", "p25m", "p37m"
)


def is_greeting_or_identity(question: str) -> bool:
    """Detect introductory, identity, or greeting questions, handling quotes and combined phrases."""
    import re
    # Normalize: strip quotes, punctuation, extra whitespace
    q = re.sub(r'[\'"`“”‘’]', ' ', str(question or '')).lower().strip()
    q = re.sub(r'\s+', ' ', q).rstrip('?.!,')

    # Core phrases that mean identity/intro/help
    greeting_phrases = (
        "tell me about you", "tell me about u", "tell me about yourself",
        "who are you", "who r u", "what are you", "what r u", "what is this",
        "what is excelsense", "what can you do", "what do you do",
        "introduce yourself", "what is your purpose", "what is your job",
        "help me", "how does this work", "how to use this", "how to use",
        "what are your capabilities", "explain yourself"
    )
    if any(phrase in q for phrase in greeting_phrases):
        return True

    # Standalone single words/short greetings
    single_greetings = {"hi", "hello", "hey", "start", "menu", "help", "greetings", "good morning", "good afternoon", "good evening"}
    words = [w for w in q.split() if w not in {"or", "and", "the", "a", "an"}]
    if q in single_greetings or (words and words[0] in single_greetings and len(words) <= 3):
        return True

    return False


def check_domain(question: str) -> Tuple[bool, Optional[str]]:
    """Determine if a question falls outside the life insurance analytics domain.
    Returns (is_out_of_domain, reason_or_category)."""
    if is_greeting_or_identity(question):
        return False, None

    import re
    q = re.sub(r'[\'"`“”‘’]', ' ', str(question or '')).lower().strip()
    q = re.sub(r'\s+', ' ', q)

    # 1. Explicit out of domain patterns
    for p in OUT_OF_DOMAIN_PATTERNS:
        if p in q:
            return True, f"queries regarding {p}"

    # 2. Entity ID patterns (e.g. POL0001, CLI0001, AGT0001, CLM0001)
    if re.search(r"\b(pol|cli|agt|clm|app|sal)\d+\b", q, re.IGNORECASE):
        return False, None

    # 3. Ontology concepts
    concepts = bi.concepts_in(question)
    if concepts:
        return False, None

    # 4. Core domain vocabulary
    if any(re.search(rf"\b{re.escape(term)}\b", q) for term in CORE_INSURANCE_TERMS):
        return False, None

    # 5. Conversational follow-ups
    followup_cues = ("why", "how come", "what about", "and for", "drill down", "show more",
                     "explain", "compare", "breakdown", "rank", "list", "filter", "which one")
    if any(q == fc or q.startswith(fc + " ") for fc in followup_cues):
        return False, None

    return True, "general non-life-insurance queries"


@dataclass
class EngineResponse:
    question: str
    route: str
    answer: Optional[BusinessAnswer] = None
    plan: Optional[AnalyticalPlan] = None
    step_results: List[StepResult] = field(default_factory=list)
    retrieval: List[Dict[str, Any]] = field(default_factory=list)
    resolved_entities: Dict[str, Any] = field(default_factory=dict)
    business_intent: Optional[str] = None
    legacy_output: Optional[str] = None
    legacy_plan: Optional[Dict[str, Any]] = None
    legacy_df: Optional[pd.DataFrame] = None
    message: Optional[str] = None
    timings_ms: Dict[str, float] = field(default_factory=dict)

    @property
    def answer_text(self) -> str:
        return self.to_markdown()

    def to_markdown(self) -> str:
        if self.answer:
            return self.answer.to_markdown()
        if self.legacy_output:
            return self.legacy_output
        return self.message or "No answer produced."

    @property
    def tier(self) -> str:
        if self.plan:
            return self.plan.tier
        return bi.TIER_RETRIEVAL


class BusinessAnalyticsEngine:
    """The business-analytics front door.

    `legacy_system` is retained only so older call sites keep working; lookups
    are answered by `engine.lookup` regardless of what is passed.
    """
    _singleton: Optional["BusinessAnalyticsEngine"] = None

    @classmethod
    def get(cls, ctx: Optional[ac.DataContext] = None,
            records: Optional[Sequence[Dict[str, Any]]] = None,
            legacy_system: Any = None,
            use_llm: bool = False) -> "BusinessAnalyticsEngine":
        if cls._singleton is None or ctx is not None:
            cls._singleton = cls(ctx, records, legacy_system, use_llm)
        return cls._singleton

    def __init__(self,
                 ctx: Optional[ac.DataContext] = None,
                 records: Optional[Sequence[Dict[str, Any]]] = None,
                 legacy_system: Any = None,
                 use_llm: bool = False):
        self.ctx = ctx or ac.DataContext.get()
        self.records = list(records or brt.load_ps_library())
        self.retriever = brt.BusinessRetriever(self.records)
        self.planner = BusinessPlanner(self.ctx, self.records)
        self.composer = AnswerComposer()
        self.legacy_system = legacy_system
        import ollama_client as _oc
        self.use_llm = use_llm and bool(
            _oc.is_ollama_available() or os.environ.get("GROQ_API_KEY") or os.environ.get("XAI_API_KEY"))

    def answer(self, question: str, llm: Optional[bool] = None) -> EngineResponse:
        return self.ask(question, llm)

    # ---- scope guard --------------------------------------------------
    @staticmethod
    def check_scope(question: str) -> Optional[str]:
        q = question.lower()
        for phrase, what in OUT_OF_SCOPE.items():
            if phrase in q:
                return what
        return None

    # ---- routing ------------------------------------------------------
    def route(self, question: str) -> Dict[str, Any]:
        if is_greeting_or_identity(question):
            return {"route": ROUTE_GREETING, "hit": None, "top5": []}

        missing = self.check_scope(question)
        if missing:
            return {"route": ROUTE_INSUFFICIENT, "missing": missing,
                    "hit": None, "top5": []}

        is_ood, ood_reason = check_domain(question)
        if is_ood:
            return {"route": ROUTE_OUT_OF_DOMAIN, "reason": ood_reason,
                    "hit": None, "top5": []}

        best, top5 = self.retriever.best_business(question)
        style = self.retriever.question_style(question)
        top1_is_business = bool(top5) and brt.is_business_ps(top5[0].record)

        # A question naming a specific record is an instance lookup, answered
        # directly from the fact frames rather than planned.
        entities = self.planner.resolver.resolve(question)
        names_an_id = any(k in entities for k in
                          ("app_id", "client_id", "sales_id", "claim_no"))

        if style == "lookup" or (names_an_id and style != "business"):
            return {"route": ROUTE_SIMPLE, "hit": None, "top5": top5, "style": style}

        if best is not None:
            return {"route": ROUTE_BUSINESS, "hit": best, "top5": top5, "style": style}

        if not top1_is_business:
            return {"route": ROUTE_SIMPLE, "hit": None, "top5": top5, "style": style}

        return {"route": ROUTE_ADHOC, "hit": best, "top5": top5, "style": style}

    # ---- main entry point ---------------------------------------------
    def ask(self, question: str, llm: Optional[bool] = None) -> EngineResponse:
        t0 = time.perf_counter()
        decision = self.route(question)
        t_route = (time.perf_counter() - t0) * 1000

        resp = EngineResponse(question=question, route=decision["route"])
        resp.retrieval = [h.as_dict() for h in decision.get("top5", [])]
        resp.timings_ms["routing"] = round(t_route, 1)

        if decision["route"] == ROUTE_GREETING:
            resp.message = (
                "I am **ExcelSense**, an AI analytics engine specialized exclusively for **Life Insurance Portfolio Data**.\n\n"
                "I compute verified metrics across 5 core production domains:\n"
                "- 📋 **Policies**: Product performance, premium volumes, term distributions, sum assured.\n"
                "- ⏳ **Persistency & Lapse**: 13M, 25M, and 37M persistency rates, early lapse warnings, renewal behavior.\n"
                "- 🏥 **Claims**: Claim frequency, payout amounts, settlement ratios, repudiation patterns.\n"
                "- 👥 **Customers & Demographics**: Policyholder age bands, income tiers, marital status, risk profiles.\n"
                "- 🧑‍💼 **Agents & Sales**: Salesperson persistency rankings, ticket sizes, tenure bands, concentration risk."
            )
            resp.answer = BusinessAnswer(
                question=question,
                headline="👋 Welcome to ExcelSense — Enterprise Life Insurance Analytics",
                evidence=[
                    "Specialized domain: Life Insurance portfolio, persistency, claims, customer profiles, and agent sales performance.",
                    "5 production tables: Policy_Details, Persistency_Details, Claims_Details, Owner_Details, and Sales_Details.",
                    "Deterministic computations: Every metric is calculated directly from verified data records.",
                ],
                analysis=[
                    "Ask any analytical question about your life insurance book (e.g. 'Which product has the highest lapse rate?', 'Rank agents by persistency', or 'What is driving claims in the North zone?')."
                ],
                recommendation=[
                    "Try asking: 'Which products have the highest 13th-month lapse rate?'",
                    "Try asking: 'Show top 10 agents by persistency risk'",
                    "Try asking: 'What is the relationship between customer income and claims payout?'"
                ]
            )
            return resp

        if decision["route"] == ROUTE_OUT_OF_DOMAIN:
            resp.message = (
                "I am **ExcelSense**, an AI assistant specialized strictly for **Life Insurance portfolio analytics**.\n\n"
                "I cannot answer general knowledge, coding, creative, or non-life-insurance questions "
                "(such as health/auto insurance, general trivia, weather, or creative writing).\n\n"
                "Please ask a question related to life insurance policies, claims, persistency, customer demographics, or agent performance."
            )
            resp.answer = BusinessAnswer(
                question=question,
                headline="⛔ Out of Domain Request",
                evidence=[
                    "ExcelSense is strictly restricted to Life Insurance portfolio data and policy analytics.",
                    "General trivia, coding, other insurance types (health/motor), and unrelated topics are outside the domain boundary."
                ],
                analysis=[
                    "Domain guardrail intercepted the query before analytical execution to prevent inaccurate or out-of-domain answers."
                ],
                recommendation=[
                    "Please rephrase your query to focus on Life Insurance policies, claims, persistency, customers, or sales performance."
                ]
            )
            return resp

        if decision["route"] == ROUTE_INSUFFICIENT:
            resp.message = (
                f"This question needs {decision['missing']}, which is not present in "
                f"this dataset. The available tables cover policies, customers, "
                f"products, plans, claims, surrenders, retention interactions and "
                f"agents - nothing about {decision['missing']}. I can answer a related "
                f"question from what is here, but I will not estimate the part that is "
                f"missing.")
            resp.answer = BusinessAnswer(
                question=question,
                headline="The dataset does not support this question.",
                evidence=[resp.message],
                analysis=[
                    "Scope check runs before any analysis, so the system reports the "
                    "gap rather than substituting a proxy metric."],
                caveats=["Answering this would require data the system does not have."])
            return resp

        # ---- check for salesperson-specific deep-dive
        agent_match = self._find_agent_entity(question)
        if agent_match:
            agent_resp = self._profile_salesperson(agent_match, question, resp, llm)
            if agent_resp is not None:
                return agent_resp

        if decision["route"] == ROUTE_SIMPLE:
            return self._run_legacy(question, resp)

        # ---- build the plan
        t1 = time.perf_counter()
        if decision["route"] == ROUTE_BUSINESS:
            hit = decision["hit"]
            plan = self.planner.plan_from_ps(question, hit.record, resp.retrieval)
            ps_record = hit.record
        else:
            intent = bi.classify_intent(question)
            plan = self.planner.plan_adhoc(question, intent)
            plan.retrieval_scores = resp.retrieval
            ps_record = None
        resp.timings_ms["planning"] = round((time.perf_counter() - t1) * 1000, 1)

        # ---- execute
        t2 = time.perf_counter()
        results = self.planner.execute(plan, ps_record)
        resp.timings_ms["execution"] = round((time.perf_counter() - t2) * 1000, 1)

        # ---- compose
        t3 = time.perf_counter()
        answer = self.composer.compose(plan, results)
        resp.timings_ms["composition"] = round((time.perf_counter() - t3) * 1000, 1)

        # every step failed -> be honest about it
        if results and not any(r.ok for r in results):
            answer.headline = ("The analysis could not be completed on this dataset.")
            answer.evidence = []
            answer.caveats.append(
                "Every planned step returned no usable data - see the list of "
                "unavailable steps above.")

        resp.plan = plan
        resp.step_results = results
        resp.answer = answer
        resp.business_intent = plan.business_intent
        resp.resolved_entities = plan.resolved_entities

        if (llm if llm is not None else self.use_llm):
            t4 = time.perf_counter()
            answer.llm_narrative = polish(answer)
            resp.timings_ms["llm"] = round((time.perf_counter() - t4) * 1000, 1)

        resp.timings_ms["total"] = round(sum(resp.timings_ms.values()), 1)
        return resp

    # ---- salesperson & agent intelligence -----------------------------
    def _find_agent_entity(self, question: str) -> Optional[Dict[str, Any]]:
        """Resolve a SALES_ID named in the question.

        Sales_Details identifies agents by a numeric SALES_ID only - there is no
        name, and no AGT-style code. The previous implementation also matched an
        `AGT\\w+` pattern and looked up agent names in `Sales_Info.csv`, both of
        which belong to an earlier dataset that this project no longer carries.
        """
        import re
        q = question.strip()

        # "salesperson 800812", "agent 270084", "sales_id 977337"
        sales_id_match = re.search(
            r'\b(?:salesperson|sales person|agent|advisor|rep|sales_id|sales id)\s*#?\s*(\d{4,7})\b',
            q, re.IGNORECASE)
        if sales_id_match:
            return {"type": "sales_id", "id": int(sales_id_match.group(1))}

        # A bare id alongside a word that implies an agent is being discussed.
        if re.search(r'\b(agent|salesperson|sales person|advisor|persistency|performance)\b',
                     q, re.IGNORECASE):
            num_match = re.search(r'\b(\d{6})\b', q)
            if num_match:
                return {"type": "sales_id", "id": int(num_match.group(1))}
        return None

    def _profile_salesperson(self, agent_entity: Dict[str, Any], question: str,
                             resp: EngineResponse, llm: Optional[bool] = None) -> Optional[EngineResponse]:
        target_id = agent_entity["id"]
        entity_type = agent_entity["type"]

        # Sales_Details is the only agent source in this dataset: a numeric
        # SALES_ID with stated rates. There is no agent name or AGT-style code.

        # 2. Try agent_fact / Sales_Details
        af = self.ctx.agent_fact
        sales_id_num = None
        try:
            sales_id_num = int(target_id)
        except Exception:
            pass

        if sales_id_num is not None and not af.empty:
            match = af[af["sales_id"] == sales_id_num]
            if not match.empty:
                r = match.iloc[0]
                sales_id = int(r["sales_id"])
                f_pers = float(r.get("sales_first_year_persistency", 75.0))
                obs_pers = float(r.get("observed_persistency_rate", 0.0)) * 100.0
                obs_lapse = float(r.get("observed_lapse_rate", 0.0)) * 100.0
                obs_surrender = float(r.get("observed_surrender_rate", 0.0)) * 100.0
                policies = int(r.get("policies_sold", 0))
                sum_assured = float(r.get("total_sum_assured", 0.0))
                total_ape = float(r.get("total_ape", 0.0))
                pivc_conc = float(r.get("pivc_concern_rate", 0.0))
                rcu_rej = float(r.get("rcu_rejection_rate", 0.0))
                early_claim = float(r.get("sales_early_claim_rate", 0.0))
                main_plan = str(r.get("main_plan", "General Portfolio"))
                branch = str(r.get("main_branch", "General"))
                st_type = str(r.get("sales_type", "Active"))

                peer_f_pers = float(af["sales_first_year_persistency"].dropna().mean())
                peer_lapse = float(af["observed_lapse_rate"].dropna().mean()) * 100.0
                peer_surrender = float(af["observed_surrender_rate"].dropna().mean()) * 100.0

                diff_pers = f_pers - peer_f_pers
                diff_lapse = obs_lapse - peer_lapse

                headline = (
                    f"Salesperson Deep-Dive: **Salesperson #{sales_id}** ({st_type}) — "
                    f"First Year Persistency: **{f_pers:.1f}%** ({abs(diff_pers):.1f}% {'above' if diff_pers>=0 else 'below'} peer benchmark), "
                    f"Policies Sold: **{policies}**, APE Book: **Rs {total_ape/1e5:.2f} L**."
                )

                evidence = [
                    f"**Salesperson ID & Status**: Salesperson `#{sales_id}` ({st_type}), Servicing Branch `{branch}`.",
                    f"**Persistency & Quality**: First Year Persistency is **{f_pers:.1f}%** (Peer Baseline: {peer_f_pers:.1f}%), Observed Persistency: **{obs_pers:.1f}%**, Observed Lapse: **{obs_lapse:.1f}%**.",
                    f"**Volume & Book Value**: Delivered **{policies} policies** representing **Rs {sum_assured/1e7:.2f} cr** in Sum Assured and **Rs {total_ape/1e5:.2f} L** Annualized Premium (APE).",
                    f"**Quality & Verification Flags**: PIVC Concern Rate: **{pivc_conc:.2f}%**, RCU Rejection Rate: **{rcu_rej:.2f}%**, Early Claim Rate: **{early_claim:.2f}%**.",
                    f"**Primary Product Focus**: Top selling plan is `{main_plan}`.",
                ]

                analysis = [
                    f"**Book Quality Assessment**: {'High quality persistency with minimal early attrition.' if f_pers >= 75.0 else 'Persistency is below the 75% target threshold, indicating elevated cancellation propensity.'}",
                    f"**Underwriting Integrity**: RCU rejection rate ({rcu_rej:.2f}%) and PIVC concern rate ({pivc_conc:.2f}%) reflect {'clean customer onboarding.' if rcu_rej < 1.0 else 'potential quality friction during sales verification.'}",
                ]

                implication = [
                    f"Policyholder book generated **Rs {total_ape/1e5:.2f} L** in recurring premium value across {policies} policies.",
                    f"Sustaining retention requires focus on top product `{main_plan}` renewal cycle.",
                ]

                recommendation = [
                    f"Engage policyholders 30 days prior to premium due date to secure regular mode renewal.",
                    f"Monitor PIVC pre-issuance verification closely to avoid post-login cancellations.",
                ]

                scorecard_df = pd.DataFrame([
                    {"Metric": "First Year Persistency", "Salesperson Value": f"{f_pers:.2f}%", "Peer Average": f"{peer_f_pers:.2f}%", "Variance": f"{diff_pers:+.2f}%"},
                    {"Metric": "Observed Lapse Rate", "Salesperson Value": f"{obs_lapse:.2f}%", "Peer Average": f"{peer_lapse:.2f}%", "Variance": f"{diff_lapse:+.2f}%"},
                    {"Metric": "Observed Surrender Rate", "Salesperson Value": f"{obs_surrender:.2f}%", "Peer Average": f"{peer_surrender:.2f}%", "Variance": f"{obs_surrender-peer_surrender:+.2f}%"},
                    {"Metric": "Policies Sold", "Salesperson Value": f"{policies}", "Peer Average": f"{af['policies_sold'].mean():.1f}", "Variance": "-"},
                    {"Metric": "Total Sum Assured", "Salesperson Value": f"Rs {sum_assured/1e7:.2f} cr", "Peer Average": f"Rs {af['total_sum_assured'].mean()/1e7:.2f} cr", "Variance": "-"},
                    {"Metric": "Total Annual Premium (APE)", "Salesperson Value": f"Rs {total_ape/1e5:.2f} L", "Peer Average": f"Rs {af['total_ape'].mean()/1e5:.2f} L", "Variance": "-"},
                    {"Metric": "PIVC Concern Rate", "Salesperson Value": f"{pivc_conc:.2f}%", "Peer Average": f"{af['pivc_concern_rate'].mean():.2f}%", "Variance": "-"},
                ])

                ans = BusinessAnswer(
                    question=question,
                    headline=headline,
                    evidence=evidence,
                    analysis=analysis,
                    implication=implication,
                    recommendation=recommendation,
                    caveats=[f"Metrics aggregated from Sales_Details and Policy_Details fact tables for salesperson #{sales_id}."],
                    tables=[
                        (f"Salesperson Scorecard — #{sales_id}", scorecard_df)
                    ],
                )

                resp.route = ROUTE_BUSINESS
                resp.answer = ans
                resp.business_intent = bi.AGENT_PERFORMANCE
                resp.resolved_entities = {"sales_id": sales_id}

                if (llm if llm is not None else self.use_llm):
                    ans.llm_narrative = polish(ans)

                return resp

        return None

    # ---- direct lookup -------------------------------------------------
    def _run_legacy(self, question: str, resp: EngineResponse) -> EngineResponse:
        """Answer a lookup straight from the fact frames.

        This replaced a NetworkX graph plus a hand-written Cypher interpreter.
        The graph was built from these same pandas frames and every table is one
        hop from Policy_Details, so it offered no reachability a filter does not
        already have. The route name is kept for compatibility.
        """
        import lookup as lk
        t0 = time.perf_counter()

        result = lk.answer(self.ctx, question)
        resp.resolved_entities = {result.detail.get("key"): result.detail.get("id")}             if result.detail.get("key") else {}
        resp.legacy_df = result.frame
        resp.timings_ms["lookup"] = round((time.perf_counter() - t0) * 1000, 1)
        resp.timings_ms["total"] = round(sum(resp.timings_ms.values()), 1)

        if not result.ok:
            # Not a lookup after all - fall through to an ad-hoc analytical plan
            # rather than reporting nothing.
            resp.route = ROUTE_ADHOC
            plan = self.planner.plan_adhoc(question, bi.classify_intent(question))
            plan.retrieval_scores = resp.retrieval
            results = self.planner.execute(plan, None)
            resp.plan, resp.step_results = plan, results
            resp.answer = self.composer.compose(plan, results)
            return resp

        resp.legacy_output = result.to_markdown()
        return resp

    # ---- convenience ---------------------------------------------------
    def explain_plan(self, question: str) -> List[Dict[str, str]]:
        """Expose the analytical plan without executing it."""
        decision = self.route(question)
        if decision["route"] == ROUTE_BUSINESS:
            return self.planner.plan_from_ps(question, decision["hit"].record).describe()
        if decision["route"] == ROUTE_ADHOC:
            return self.planner.plan_adhoc(question, bi.classify_intent(question)).describe()
        return []

    def capabilities(self) -> Dict[str, Any]:
        biz = [r for r in self.records if brt.is_business_ps(r)]
        return {
            "problem_statements_total": len(self.records),
            "business_problem_statements": len(biz),
            "legacy_problem_statements": len(self.records) - len(biz),
            "business_domains": sorted({r.get("business_domain") or r.get("business_intent") or "GENERAL" for r in biz}),
            "metrics": {name: {"label": s.label, "unit": s.unit,
                               "definition": s.definition}
                        for name, s in ac.METRICS.items()},
            "share_metrics": list(ac.SHARE_METRICS),
            "dimensions": sorted({d.name for d in ac.DIMENSIONS.values()}),
            "operations": sorted(ac.OPERATIONS),
            "analytical_steps": sum(len(r["analytical_steps"]) for r in biz),
        }


_ENGINE: Optional[BusinessAnalyticsEngine] = None


def get_engine(with_legacy: bool = True) -> BusinessAnalyticsEngine:
    """Cached engine.

    `with_legacy` is retained for call-site compatibility; lookups are now
    answered from the fact frames by `engine.lookup`, so there is no graph to
    build and nothing to wire in.
    """
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = BusinessAnalyticsEngine(legacy_system=None)
    return _ENGINE


if __name__ == "__main__":
    import sys
    eng = get_engine()
    qs = sys.argv[1:] or [
        "What are the biggest risks in our insurance portfolio?",
        "Which products should management prioritize for retention?",
        "Why is Whole Life performing poorly?",
    ]
    for q in qs:
        r = eng.ask(q)
        print("=" * 100)
        print(f"Q: {q}\nROUTE: {r.route}  TIER: {r.tier}  PS: "
              f"{r.plan.problem_id if r.plan else '-'}  {r.timings_ms.get('total')}ms\n")
        print(r.to_markdown())
