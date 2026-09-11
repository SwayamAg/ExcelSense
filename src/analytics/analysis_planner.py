"""
analysis_planner.py
===================
Turns a natural-language business question into an *executable analytical plan*
and runs it.

This is the module that breaks the old one-question-one-query assumption:

    question
      -> entity resolution against the dataset's real vocabulary
      -> business-intent classification
      -> Problem Statement retrieval (business_retrieval)
      -> AnalyticalPlan  (N ordered steps, each an analytics_core operation)
      -> parameter binding ($product, $zone, $top_product from step s1, ...)
      -> deterministic execution, step by step, with per-step failure isolation
      -> StepResult list handed to answer_composer

Nothing here computes a number itself; every figure comes from analytics_core.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

import analytics_core as ac
import business_intents as bi


# =====================================================================
# PLAN OBJECTS
# =====================================================================

@dataclass
class AnalysisStep:
    id: str
    title: str
    op: str
    params: Dict[str, Any]
    purpose: str


@dataclass
class StepResult:
    step: AnalysisStep
    ok: bool
    result: Any = None
    error: Optional[str] = None

    @property
    def is_frame(self) -> bool:
        return isinstance(self.result, pd.DataFrame)

    def preview(self, n: int = 8) -> Any:
        if self.is_frame:
            return self.result.head(n)
        return self.result


@dataclass
class AnalyticalPlan:
    question: str
    tier: str
    business_intent: str
    problem_id: Optional[str]
    problem_statement: Optional[str]
    answer_template: str
    steps: List[AnalysisStep] = field(default_factory=list)
    resolved_entities: Dict[str, Any] = field(default_factory=dict)
    bindings: Dict[str, Any] = field(default_factory=dict)
    required_metrics: List[str] = field(default_factory=list)
    relevant_tables: List[str] = field(default_factory=list)
    caveat: Optional[str] = None
    retrieval_scores: List[Dict[str, Any]] = field(default_factory=list)

    def describe(self) -> List[Dict[str, str]]:
        """Human-readable plan, exposed *before* execution as the brief requires."""
        return [{"step": s.id, "title": s.title, "operation": s.op,
                 "purpose": s.purpose,
                 "parameters": _params_summary(s.params)} for s in self.steps]


def _params_summary(p: Dict[str, Any]) -> str:
    bits = []
    for k in ("dimension", "dim_a", "dim_b", "metric", "value_metric", "base", "column"):
        if k in p:
            bits.append(f"{k}={p[k]}")
    if "metrics" in p:
        bits.append(f"metrics={len(p['metrics'])}")
    if p.get("filters"):
        bits.append(f"filters={p['filters']}")
    if p.get("focus_filters"):
        bits.append(f"focus={p['focus_filters']}")
    if p.get("components"):
        bits.append("components=" + "+".join(
            f"{c['metric']}({c.get('weight', 1)},{c.get('direction', 'high')})"
            for c in p["components"]))
    return ", ".join(bits) or "-"


# =====================================================================
# ENTITY RESOLUTION
# =====================================================================

class EntityResolver:
    """Resolves the dataset's actual categorical vocabulary out of free text.

    Unlike the legacy regex resolver (which only knew IDs, zones and eight
    product names), this builds its vocabulary from the loaded data, so any
    channel, state, branch, payment mode, claim type, surrender reason,
    retention channel or offer in the CSVs can be named in a question.
    """

    # dimension -> also accept these aliases for its values
    # Aliases map user phrasing onto values that actually occur in the CSVs.
    # The previous table carried Agency/Online/Broker/Direct channels, smoker
    # status and In-Force/Matured/Paid-up statuses, none of which exist here:
    # CHANNEL is 'Banca' throughout and POLICY.STATUS has exactly four values.
    VALUE_ALIASES = {
        "channel": {"banca": "Banca", "bancassurance": "Banca",
                    "bank channel": "Banca", "bank": "Banca"},
        "policy_status": {
            "premium paying": "Premium paying (regular)",
            "paying": "Premium paying (regular)",
            "regular": "Premium paying (regular)",
            "in force": "Premium paying (regular)",
            "in-force": "Premium paying (regular)",
            "inforce": "Premium paying (regular)",
            "active": "Premium paying (regular)",
            "lapsed": "Lapsed", "lapse": "Lapsed",
            "surrendered": "Surrendered", "surrender": "Surrendered",
            "death": "Death", "deceased": "Death",
        },
        "trad_ulip": {"traditional": "TRAD", "trad": "TRAD",
                      "ulip": "ULIP", "unit linked": "ULIP",
                      "unit-linked": "ULIP"},
        "renewal_status": {"paid": "Paid", "collected": "Paid",
                           "pending": "Pending", "outstanding": "Pending",
                           "unpaid": "Pending"},
    }

    # dimensions whose values are safe to auto-detect from free text
    SCANNABLE = ["product", "plan_name", "product_category", "trad_ulip", "channel", "state",
                 "payment_mode", "persistency_bucket", "policy_status", "cause_of_death", "gender",
                 "occupation", "education", "sales_type", "income_band", "age_band", "ppt_band",
                 "term_band", "renewal_status"]

    def __init__(self, ctx: ac.DataContext):
        self.ctx = ctx
        self.vocab: Dict[str, Dict[str, Any]] = {}
        for dname in self.SCANNABLE:
            dim = ac.DIMENSIONS.get(dname)
            if dim is None:
                continue
            for base in ("policy", "claim", "retention"):
                frame = ctx.frame(base)
                if dim.column in frame.columns:
                    vals = [v for v in frame[dim.column].dropna().unique()]
                    self.vocab[dname] = {str(v).lower(): v for v in vals}
                    break
        for dname, aliases in self.VALUE_ALIASES.items():
            if dname in self.vocab:
                for a, v in aliases.items():
                    self.vocab[dname].setdefault(a, v)

    def resolve(self, query: str) -> Dict[str, Any]:
        q = f" {query.lower()} "
        out: Dict[str, Any] = {}

        # --- identifiers, in the shapes this dataset actually uses.
        # The previous patterns (CUST/POL/AGT/CLM-/SRD/RET) came from an earlier
        # dataset and matched nothing here, so `names_an_id` in the router never
        # fired for a real AppID, ClientID or SALES_ID.
        for pattern, key in [
                (r"\b(?:appid|app id|policy(?:\s*(?:no|number|id))?)\s*#?\s*(\d{8,12})\b", "app_id"),
                (r"\b(?:clientid|client id|client|customer|owner)\s*#?\s*(\d{8,12})\b", "client_id"),
                (r"\b(?:sales_id|sales id|salesperson|sales person|agent|advisor|rep)\s*#?\s*(\d{4,7})\b", "sales_id"),
                (r"\b(?:claim(?:\s*(?:no|number|id))?)\s*#?\s*(\d{1,8})\b", "claim_no")]:
            m = re.findall(pattern, query, re.IGNORECASE)
            if m:
                out[key] = str(m[0])

        # --- categorical values, longest surface form first
        for dname in self.SCANNABLE:
            table = self.vocab.get(dname) or {}
            best = None
            for surface, value in table.items():
                if len(surface) < 3:
                    continue
                if f" {surface} " in q or f" {surface}s " in q or f" {surface}," in q \
                        or f" {surface}." in q or f" {surface}?" in q:
                    if best is None or len(surface) > len(best[0]):
                        best = (surface, value)
            if best:
                out[dname] = best[1]

        # a bare "surrender"/"lapse" mention must not be read as a status filter
        if out.get("policy_status") and not re.search(
                r"\b(lapsed|surrendered|matured|paid[- ]up|in[- ]?force|active)\b",
                query, re.IGNORECASE):
            out.pop("policy_status")

        # "Term Life" also matches the "Protection" category; product wins
        if "product" in out and "product_category" in out:
            out.pop("product_category")

        return out


# =====================================================================
# PARAMETER BINDING
# =====================================================================

_PLACEHOLDER = re.compile(r"^\$(\w+)$")


class ParameterBinder:
    """Resolves `$name` placeholders inside step parameters.

    Three sources, in priority order:
      1. an entity the user actually named in the question
      2. a value produced by an earlier step (declared as
         {"source": "step:s1", "field": "product_name"})
      3. the Problem Statement's declared default
    """

    def __init__(self, plan_params: Dict[str, Any], resolved: Dict[str, Any]):
        self.spec = plan_params or {}
        self.resolved = resolved
        self.values: Dict[str, Any] = {}
        self.provenance: Dict[str, str] = {}

    def seed_from_entities(self):
        for name, spec in self.spec.items():
            if spec.get("free_filters"):
                filters = self._free_filters()
                if filters:
                    self.values[name] = filters
                    self.provenance[name] = "named in question"
                else:
                    self.values[name] = dict(spec.get("default") or {})
                    self.provenance[name] = "problem-statement default"
                continue
            dim = spec.get("dimension")
            if dim and dim in self.resolved:
                self.values[name] = self.resolved[dim]
                self.provenance[name] = "named in question"
            elif "default" in spec:
                self.values[name] = spec["default"]
                self.provenance[name] = "problem-statement default"

    def _free_filters(self) -> Dict[str, Any]:
        keep = ["product", "product_category", "channel", "zone", "state",
                "payment_mode", "smoker", "gender", "marital_status",
                "policy_status", "premium_payment_type"]
        return {k: v for k, v in self.resolved.items() if k in keep}

    def absorb_step_result(self, step_id: str, result: Any):
        """Fill any placeholder declared as coming from this step."""
        for name, spec in self.spec.items():
            if self.values.get(name) is not None and self.provenance.get(name) == "named in question":
                continue
            src = spec.get("source")
            if src != f"step:{step_id}":
                continue
            field_name = spec.get("field")
            take = int(spec.get("take", 1))
            vals = self._extract(result, field_name, take)
            if vals:
                self.values[name] = vals[0] if take == 1 else vals
                self.provenance[name] = f"derived from step {step_id}"

    @staticmethod
    def _extract(result: Any, field_name: Optional[str], take: int) -> List[Any]:
        if not isinstance(result, pd.DataFrame) or result.empty or not field_name:
            return []
        if field_name not in result.columns:
            return []
        return [v for v in result[field_name].head(take).tolist() if pd.notna(v)]

    def bind(self, params: Any) -> Any:
        if isinstance(params, dict):
            return {k: self.bind(v) for k, v in params.items()}
        if isinstance(params, list):
            return [self.bind(v) for v in params]
        if isinstance(params, str):
            m = _PLACEHOLDER.match(params)
            if m:
                return self.values.get(m.group(1))
            return params
        return params


# =====================================================================
# PLANNER
# =====================================================================

# =====================================================================
# SEMANTIC QUERY EXTRACTION & ADAPTATION
# =====================================================================

def extract_query_semantics(question: str) -> Dict[str, Any]:
    """Extract explicit metrics, dimensions, and operational intents from natural query."""
    q = f" {question.lower()} "
    
    # 1. Metrics
    metrics = []
    if "surrender" in q:
        metrics.append("surrender_rate")
    if "13th" in q or "13m" in q:
        metrics.append("persistency_13m_rate")
    elif "25th" in q or "25m" in q:
        metrics.append("persistency_25m_rate")
    elif "persistency" in q or "renewal" in q:
        metrics.append("persistency_rate")
    
    if "lapse" in q:
        metrics.append("lapse_rate")
    if "pivc" in q:
        metrics.append("pivc_mismatch_rate")
    if "rcu" in q:
        metrics.append("rcu_rejection_rate")
    if "claim" in q or "settlement ratio" in q:
        if "tat" in q or "turnaround" in q or "speed" in q:
            metrics.append("avg_settlement_tat")
        elif "repudiat" in q or "reject" in q:
            metrics.append("claim_repudiation_rate")
        else:
            metrics.append("claim_settlement_ratio")
    if any(w in q for w in ("ticket", "ape", "premium amount", "avg premium")):
        metrics.append("avg_ape")
    if "sum assured" in q or "coverage" in q:
        metrics.append("avg_sum_assured")
    if "loss ratio" in q:
        metrics.append("loss_ratio")

    # 2. Dimensions
    dims = []
    if any(w in q for w in ("area", "state", "region", "geograph", "location", "territor")):
        dims.append("state")
    if "zone" in q:
        dims.append("zone")
    if any(w in q for w in ("product", "plan", "policy type", "which policy")):
        dims.append("plan_name")
    if any(w in q for w in ("channel", "banca", "agency", "broker", "direct")):
        dims.append("channel")
    if any(w in q for w in ("payment mode", "mode of payment", "cheque", "auto-debit", "ecs", "monthly", "annual payment")):
        dims.append("payment_mode")
    if "occupation" in q:
        dims.append("occupation")
    if "income" in q:
        dims.append("income_band")
    if "age" in q or "age group" in q:
        dims.append("age_band")
    if "agent" in q or "advisor" in q or "sales rep" in q:
        dims.append("agent")
    if "gender" in q:
        dims.append("gender")
    if "education" in q:
        dims.append("education")

    # 3. Intents / modifiers
    has_why = bool(re.search(r"\b(why|reason|reasons|driver|drivers|cause|causes|factor|factors)\b", q))
    has_how = bool(re.search(r"\b(how|improve|improving|increase|increasing|reduce|reducing|intervene|action|plan|recommendation|levers)\b", q))
    has_lowest = bool(re.search(r"\b(lowest|worst|bottom|least|poor|poorest|underperforming|lagging|minimum|min)\b", q))
    has_highest = bool(re.search(r"\b(highest|best|top|maximum|max|leading|most)\b", q))
    has_correlation = bool(re.search(r"\b(relation|relationship|link|linked|correlat|association|btw|between)\b", q))

    return {
        "metrics": metrics,
        "dimensions": dims,
        "has_why": has_why,
        "has_how": has_how,
        "has_lowest": has_lowest,
        "has_highest": has_highest,
        "has_correlation": has_correlation,
    }


# =====================================================================
# PLANNER
# =====================================================================

class BusinessPlanner:
    """Builds and executes AnalyticalPlans."""

    def __init__(self, ctx: Optional[ac.DataContext] = None,
                 ps_records: Optional[Sequence[Dict[str, Any]]] = None):
        self.ctx = ctx or ac.DataContext.get()
        self.resolver = EntityResolver(self.ctx)
        self.ps_records = list(ps_records or [])
        self.ps_by_id = {r["problem_id"]: r for r in self.ps_records}

    # ---- plan construction -------------------------------------------
    def plan_from_ps(self, question: str, ps_record: Dict[str, Any],
                     retrieval_scores: Optional[List[Dict[str, Any]]] = None
                     ) -> AnalyticalPlan:
        resolved = self.resolver.resolve(question)
        sem = extract_query_semantics(question)
        steps = [AnalysisStep(s["id"], s["title"], s["op"], dict(s["params"]), s["purpose"])
                 for s in ps_record.get("analytical_steps", [])]

        # Adapt steps if user explicitly specified dimensions / metrics
        if sem["dimensions"] or sem["metrics"] or sem["has_why"] or sem["has_how"] or sem["has_correlation"]:
            target_metric = sem["metrics"][0] if sem["metrics"] else None
            target_dim = sem["dimensions"][0] if sem["dimensions"] else None

            # 1. Correlation adaptation: ensure exact metrics requested
            if sem["has_correlation"] and len(sem["metrics"]) >= 2:
                for s in steps:
                    if s.op == "correlation_analysis":
                        s.params["metric_a"] = sem["metrics"][0]
                        s.params["metric_b"] = sem["metrics"][1]
                        if target_dim:
                            s.params["dimension"] = target_dim

            # 2. Dimensional comparison adaptation: if user asks for specific dimension (e.g. area/state)
            if target_dim:
                has_dim_step = any(s.params.get("dimension") == target_dim for s in steps)
                if not has_dim_step:
                    metric_for_step = target_metric or (steps[1].params.get("metric") if len(steps) > 1 and "metric" in steps[1].params else "persistency_rate")
                    dim_label = ac.DIMENSIONS[target_dim].label if target_dim in ac.DIMENSIONS else target_dim.title()
                    m_label = ac.METRICS[metric_for_step].label if metric_for_step in ac.METRICS else metric_for_step
                    new_step = AnalysisStep(
                        id=f"s_dim_{target_dim}",
                        title=f"{m_label} by {dim_label}",
                        op="compare_to_baseline",
                        params={"dimension": target_dim, "metric": metric_for_step},
                        purpose=f"Evaluate user requested dimension ({dim_label})."
                    )
                    # Insert right after baseline overview (position 1)
                    insert_pos = 1 if (steps and steps[0].op == "portfolio_overview") else 0
                    steps.insert(insert_pos, new_step)

            # 3. Diagnostic "why" adaptation: ensure driver_scan step exists
            if sem["has_why"] and not any(s.op in ("driver_scan", "segment_profile") for s in steps):
                m_scan = target_metric or "persistency_rate"
                steps.append(AnalysisStep(
                    id="s_driver_scan",
                    title=f"Driver scan for {m_scan}",
                    op="driver_scan",
                    params={"metric": m_scan},
                    purpose="Decompose drivers across dimensions to answer why."
                ))

            # 4. Prescriptive "how to improve" adaptation: ensure driver_scan or compare steps for levers
            if sem["has_how"] and not any(s.op in ("driver_scan", "compare_to_baseline") for s in steps):
                m_presc = target_metric or "persistency_rate"
                steps.append(AnalysisStep(
                    id="s_levers_scan",
                    title=f"Intervention levers for {m_presc}",
                    op="driver_scan",
                    params={"metric": m_presc},
                    purpose="Scan high-impact levers to answer how to improve."
                ))

        # Choose appropriate template
        declared_template = ps_record.get("answer_template", "leaderboard")
        ans_template = declared_template
        if declared_template == "leaderboard":
            if sem["has_how"] and any(s.op in ("driver_scan", "compare_to_baseline") for s in steps):
                ans_template = "prescriptive"
            elif sem["has_why"] and any(s.op in ("driver_scan", "segment_profile") for s in steps):
                ans_template = "diagnostic"
            elif sem["has_correlation"] and any(s.op == "correlation_analysis" for s in steps):
                ans_template = "correlation"

        return AnalyticalPlan(
            question=question,
            tier=ps_record.get("analysis_tier", bi.TIER_ANALYTICAL),
            business_intent=ps_record.get("business_intent", bi.SIMPLE_RETRIEVAL),
            problem_id=ps_record.get("problem_id"),
            problem_statement=ps_record.get("problem_statement"),
            answer_template=ans_template,
            steps=steps,
            resolved_entities=resolved,
            required_metrics=list(ps_record.get("required_metrics", [])),
            relevant_tables=list(ps_record.get("relevant_tables", [])),
            caveat=ps_record.get("caveat"),
            retrieval_scores=retrieval_scores or [],
        )

    def plan_adhoc(self, question: str, intent: str) -> AnalyticalPlan:
        """Fallback plan when no Problem Statement matches well enough.

        Still analytical rather than a raw dump: it picks the metric family the
        question is about and the dimension it mentions, and produces a
        baseline-anchored comparison.
        """
        resolved = self.resolver.resolve(question)
        q = question.lower()

        metric = "lapse_rate"
        if "surrender" in q:
            metric = "surrender_rate"
        elif "13th" in q or "13m" in q:
            metric = "persistency_13m_rate"
        elif "25th" in q or "25m" in q:
            metric = "persistency_25m_rate"
        elif "persistency" in q or "renewal" in q:
            metric = "persistency_rate"
        elif "pivc" in q:
            metric = "pivc_mismatch_rate"
        elif "rcu" in q:
            metric = "rcu_rejection_rate"
        elif "tat" in q or "settlement speed" in q:
            metric = "avg_settlement_tat"
        elif "claim" in q:
            metric = "claim_settlement_ratio"
        elif any(w in q for w in ("value", "premium", "ape", "ticket")):
            metric = "avg_ape"
        elif "sum assured" in q or "coverage" in q:
            metric = "avg_sum_assured"

        dimension = None
        for dname in ("product", "plan_name", "trad_ulip", "product_category", "state", "zone",
                      "payment_mode", "persistency_bucket", "occupation", "income_band",
                      "age_band", "ppt_band", "term_band", "branch", "agent", "sales_type",
                      "gender", "education", "policy_status", "cause_of_death"):
            if dname in ac.DIMENSIONS:
                dim = ac.DIMENSIONS[dname]
                if dim.label.lower() in q or dname.replace("_", " ") in q:
                    dimension = dname
                    break
        if dimension is None:
            dimension = "product"

        secondary = ["policy_count", metric]
        for extra in ("persistency_rate", "avg_sum_assured", "avg_ape"):
            if extra in ac.METRICS and extra not in secondary:
                secondary.append(extra)

        steps = [
            AnalysisStep("s1", "Portfolio baseline", "portfolio_overview", {},
                         "Establish the reference numbers."),
            AnalysisStep("s2", f"{ac.METRICS[metric].label} by {ac.DIMENSIONS[dimension].label}",
                         "compare_to_baseline",
                         {"dimension": dimension, "metric": metric},
                         "Rate, gap to baseline and sample size for each group."),
            AnalysisStep("s3", "Supporting metrics", "multi_metric_table",
                         {"dimension": dimension, "metrics": secondary,
                          "sort_by": metric},
                         "Volume and value context for the ranking."),
        ]
        return AnalyticalPlan(
            question=question, tier=bi.TIER_ANALYTICAL, business_intent=intent,
            problem_id=None,
            problem_statement=f"Ad-hoc analysis: {ac.METRICS[metric].label} by "
                              f"{ac.DIMENSIONS[dimension].label}.",
            answer_template="leaderboard", steps=steps, resolved_entities=resolved,
            required_metrics=secondary,
            relevant_tables=["policies", "customers", "products"],
            caveat=None,
        )

    # ---- execution ----------------------------------------------------
    def execute(self, plan: AnalyticalPlan,
                ps_record: Optional[Dict[str, Any]] = None) -> List[StepResult]:
        ps_record = ps_record or (self.ps_by_id.get(plan.problem_id) if plan.problem_id else None)
        binder = ParameterBinder((ps_record or {}).get("parameters", {}),
                                 plan.resolved_entities)
        binder.seed_from_entities()

        results: List[StepResult] = []
        for step in plan.steps:
            params = binder.bind(step.params)
            params = _drop_unbound(params)
            try:
                res = ac.run_operation(self.ctx, step.op, params)
                ok = not (isinstance(res, pd.DataFrame) and res.empty)
                bound_step = AnalysisStep(step.id, step.title, step.op, params, step.purpose)
                results.append(StepResult(bound_step, ok, res,
                                          None if ok else "No rows matched this step."))
                binder.absorb_step_result(step.id, res)
            except ac.InsufficientDataError as e:
                results.append(StepResult(step, False, None, str(e)))
            except Exception as e:  # keep one bad step from killing the plan
                results.append(StepResult(step, False, None,
                                          f"{type(e).__name__}: {e}"))

        plan.bindings = {k: v for k, v in binder.values.items() if v is not None}
        plan.binding_provenance = binder.provenance  # type: ignore[attr-defined]
        return results


def _drop_unbound(params: Any) -> Any:
    """Remove parameters whose placeholder could not be resolved.

    A `filters=None` must not reach the operation as a literal None filter -
    dropping the key means the step simply runs unfiltered, which is the
    correct degradation.
    """
    if isinstance(params, dict):
        out = {}
        for k, v in params.items():
            v = _drop_unbound(v)
            if v is None and k in ("filters", "focus_filters", "dimension",
                                   "dim_a", "dim_b", "metric"):
                if k in ("filters", "focus_filters"):
                    out[k] = {}
                continue
            out[k] = v
        return out
    if isinstance(params, list):
        return [_drop_unbound(v) for v in params]
    return params


__all__ = ["AnalysisStep", "StepResult", "AnalyticalPlan", "EntityResolver",
           "ParameterBinder", "BusinessPlanner"]
