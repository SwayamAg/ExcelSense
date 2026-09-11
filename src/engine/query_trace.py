"""
query_trace.py
==============
Builds a per-question derivation trace and renders it as Graphviz DOT.

Every question produces a *different* diagram, because the diagram is built
from what actually happened to that question: which entities were resolved,
which Problem Statement won and by what signal mix, which analytical steps were
planned, which of them read which raw tables, which step fed a parameter into
which later step, how many rows each returned, and which answer template
composed the result.

Four views:

  pipeline_dot()    end-to-end flow: question -> scope -> entities -> retrieval
                    -> plan -> each executed step -> composition -> answer.
                    Step-to-step parameter dependencies are drawn as edges.

  lineage_dot()     data lineage: raw CSVs -> fact frames -> metrics/dimensions
                    -> the steps that used them. Only the tables the question
                    genuinely touched appear.

  entity_path_dot() the relationship path through the entity model. For a
                    lookup answered by the legacy engine this is the *actual*
                    Cypher MATCH pattern executed against the NetworkX graph,
                    with live node counts. For an analytical answer it is the
                    join path those pandas merges correspond to - labelled as
                    such, because no graph traversal took place.

  step_table()      the same trace as a flat auditable table.

Nothing here recomputes a number; it reads the EngineResponse produced by
`business_engine.ask()`.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

import analytics_core as ac

# ---------------------------------------------------------------------
# palette (matches the dashboard's dark theme)
# ---------------------------------------------------------------------
C = {
    "bg": "#0d1117",
    "node": "#161b22",
    "border": "#30363d",
    "text": "#e6edf3",
    "muted": "#8b949e",
    "blue": "#58a6ff",
    "green": "#3fb950",
    "amber": "#d29922",
    "red": "#f85149",
    "purple": "#bc8cff",
    "cyan": "#39c5cf",
}

KIND_COLOR = {
    "question": C["purple"],
    "scope": C["green"],
    "entity": C["cyan"],
    "retrieval": C["blue"],
    "plan": C["blue"],
    "step": C["green"],
    "step_failed": C["red"],
    "compose": C["amber"],
    "answer": C["purple"],
    "table": C["muted"],
    "frame": C["cyan"],
    "metric": C["amber"],
    "dimension": C["blue"],
    "graph": C["green"],
    "blocked": C["red"],
}

# The property-graph edge set, used to render the relationship path.
GRAPH_EDGES: List[Tuple[str, str, str]] = [
    ("Customer", "Policy", "HOLDS"),
    ("Policy", "Product", "OF_PRODUCT"),
    ("Policy", "Plan", "UNDER_PLAN"),
    ("Plan", "Product", "VARIANT_OF"),
    ("Agent", "Policy", "SOLD"),
    ("Policy", "Branch", "BOOKED_AT"),
    ("Agent", "Branch", "WORKS_AT"),
    ("Claim", "Policy", "AGAINST"),
    ("Claim", "Customer", "FILED_BY"),
    ("Policy", "Surrender", "HAS_SURRENDER"),
    ("Surrender", "Customer", "FILED_BY"),
    ("Policy", "Retention", "HAS_RETENTION_EFFORT"),
    ("Retention", "Customer", "CONTACTED_CUSTOMER"),
]

TABLE_TO_NODE = {
    "policies": "Policy", "customers": "Customer", "products": "Product",
    "plans": "Plan", "sales": "Agent", "claims": "Claim",
    "surrenders": "Surrender", "retention": "Retention",
}

FRAME_OF_BASE = {
    "policy": "policy_fact", "claim": "claim_fact", "retention": "retention_fact",
    "customer": "customer_fact", "agent": "agent_fact",
}


# ---------------------------------------------------------------------
# trace objects
# ---------------------------------------------------------------------

@dataclass
class TraceNode:
    id: str
    kind: str
    title: str
    lines: List[str] = field(default_factory=list)
    status: str = "ok"          # ok | warn | fail
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceEdge:
    src: str
    dst: str
    label: str = ""
    style: str = "solid"        # solid | dashed | bold


@dataclass
class QueryTrace:
    question: str
    route: str
    nodes: List[TraceNode] = field(default_factory=list)
    edges: List[TraceEdge] = field(default_factory=list)
    lineage_nodes: List[TraceNode] = field(default_factory=list)
    lineage_edges: List[TraceEdge] = field(default_factory=list)
    path_nodes: List[TraceNode] = field(default_factory=list)
    path_edges: List[TraceEdge] = field(default_factory=list)
    path_caption: str = ""
    path_is_real_traversal: bool = False
    summary: Dict[str, Any] = field(default_factory=dict)
    step_rows: List[Dict[str, Any]] = field(default_factory=list)

    def node(self, nid: str) -> Optional[TraceNode]:
        return next((n for n in self.nodes if n.id == nid), None)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _short(v: Any, n: int = 46) -> str:
    t = str(v)
    return t if len(t) <= n else t[: n - 1] + "…"


def _fmt_filters(f: Any) -> str:
    if not f or not isinstance(f, dict):
        return ""
    bits = []
    for k, v in f.items():
        if isinstance(v, dict):
            v = "/".join(f"{a} {b}" for a, b in v.items())
        elif isinstance(v, (list, tuple)):
            v = "/".join(str(x) for x in v)
        bits.append(f"{k}={v}")
    return ", ".join(bits)


def _rows_of(result: Any) -> Optional[int]:
    if isinstance(result, pd.DataFrame):
        return len(result)
    return None


def _scalar_summary(op: str, result: Any) -> Optional[str]:
    """A one-line figure for steps that return a dict rather than a frame, so
    the trace shows what the step produced instead of just 'computed'."""
    if not isinstance(result, dict):
        return None
    if op == "concentration":
        return (f"HHI {result.get('hhi')} ({result.get('interpretation')}), "
                f"top {result.get('top_n')} = {result.get('top_n_share_pct')}%")
    if op == "top_entity_concentration":
        return (f"top {result.get('top_n')} of {result.get('n_records'):,} records "
                f"= {result.get('top_n_share_pct')}% of "
                f"{str(result.get('column', '')).replace('_', ' ')}")
    if op == "overall_metric":
        return f"{result.get('label', '')}: {result.get('formatted', '-')}"
    if op == "portfolio_overview":
        kpis = [k for k in result if not str(k).startswith("_")]
        return f"{len(kpis)} KPIs computed, each with its denominator"
    return None


def _step_metrics(params: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    if isinstance(params.get("metric"), str):
        out.append(params["metric"])
    for m in params.get("metrics") or []:
        if isinstance(m, str):
            out.append(m)
    for c in params.get("components") or []:
        if isinstance(c, dict) and isinstance(c.get("metric"), str):
            out.append(c["metric"])
    if isinstance(params.get("value_metric"), str):
        out.append(params["value_metric"])
    seen, uniq = set(), []
    for m in out:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq


def _step_dimensions(params: Dict[str, Any]) -> List[str]:
    out = []
    for k in ("dimension", "dim_a", "dim_b"):
        v = params.get(k)
        if isinstance(v, str):
            out.append(v)
    return out


def _step_sources(op: str, params: Dict[str, Any]) -> List[str]:
    """Raw tables a single executed step genuinely reads."""
    tables: List[str] = []

    def add(ts):
        for t in ts:
            if t not in tables:
                tables.append(t)

    if op == "portfolio_overview":
        add(["policies", "customers", "products", "claims", "surrenders", "retention"])
        return tables
    if op == "top_entity_concentration":
        add(ac.BASE_SOURCES.get(params.get("base", "policy"), ("policies",)))
        if params.get("column") == "claim_amount":
            add(["claims"])
        return tables
    if op == "entity_shortlist":
        add(ac.BASE_SOURCES.get(params.get("base", "policy"), ("policies",)))
        for c in params.get("score_components") or []:
            col = c.get("column", "")
            if "claim" in col:
                add(["claims"])
            if "income" in col or "premium_to_income" in col or "sa_to_income" in col:
                add(["customers"])
        return tables

    for m in _step_metrics(params):
        add(ac.metric_sources(m))
    for d in _step_dimensions(params):
        add(ac.dimension_sources(d))
    for f in (params.get("filters"), params.get("focus_filters")):
        if isinstance(f, dict):
            for k in f:
                add(ac.dimension_sources(k))
    if op == "segment_profile":
        add(["policies", "customers", "claims"])
    if not tables:
        add(["policies"])
    return tables


def _step_base(op: str, params: Dict[str, Any]) -> str:
    if op in ("entity_shortlist", "top_entity_concentration"):
        return params.get("base", "policy")
    metrics = _step_metrics(params)
    for m in metrics:
        spec = ac.METRICS.get(m)
        if spec:
            return spec.base
        if m in ac.SHARE_METRICS:
            src = ac.SHARE_METRICS[m][0]
            if src in ac.METRICS:
                return ac.METRICS[src].base
    return "policy"


# ---------------------------------------------------------------------
# trace construction
# ---------------------------------------------------------------------

def build_trace(resp: Any, graph: Any = None) -> QueryTrace:
    """Build the derivation trace for one EngineResponse.

    `graph` is the optional NetworkX graph; when supplied and the question was
    answered by the legacy engine, the entity-path view shows real node counts.
    """
    tr = QueryTrace(question=resp.question, route=resp.route)

    # ---- 1. question
    tr.nodes.append(TraceNode(
        "q", "question", "User question", [_short(resp.question, 62)]))

    # ---- 2. scope guard
    if resp.route == "insufficient_data":
        tr.nodes.append(TraceNode(
            "scope", "blocked", "Scope check: BLOCKED",
            [_short(resp.message or "required data is absent", 72)], status="fail"))
        tr.edges.append(TraceEdge("q", "scope"))
        tr.nodes.append(TraceNode(
            "ans", "answer", "Answer",
            ["States what data is missing.", "No metric was computed."], status="warn"))
        tr.edges.append(TraceEdge("scope", "ans", "refuse"))
        tr.summary = {"steps": 0, "operations": 0, "tables": 0,
                      "route": resp.route, "total_ms": resp.timings_ms.get("total", 0)}
        return tr

    tr.nodes.append(TraceNode(
        "scope", "scope", "Scope check: passed",
        ["Question maps onto tables the dataset actually has."]))
    tr.edges.append(TraceEdge("q", "scope"))

    # ---- 3. entity resolution
    ents = resp.resolved_entities or {}
    if ents:
        lines = [f"{k} = {v}" for k, v in list(ents.items())[:5]]
    else:
        lines = ["none found - analysis runs portfolio-wide"]
    tr.nodes.append(TraceNode(
        "ent", "entity", "Entity resolution", lines, status="ok",
        meta={"entities": ents}))
    tr.edges.append(TraceEdge("scope", "ent"))

    # ---- 4. retrieval
    top = (resp.retrieval or [{}])[0]
    if resp.route == "simple_retrieval":
        tr.nodes.append(TraceNode(
            "ret", "retrieval", "Routing: single-fact lookup",
            ["No multi-step analysis needed.",
             "Delegated to the graph query engine."]))
    else:
        sig = top.get("signals", {})
        lines = [f"matched {top.get('problem_id', '-')}  score {top.get('score', 0):.3f}"]
        if sig:
            lines.append("signals  " + "  ".join(
                f"{k[:3]} {v:.2f}" for k, v in sig.items()))
        if len(resp.retrieval or []) > 1:
            runner = resp.retrieval[1]
            lines.append(f"runner-up {runner['problem_id']} ({runner['score']:.3f})")
        tr.nodes.append(TraceNode(
            "ret", "retrieval", "Problem Statement retrieval", lines,
            meta={"top5": resp.retrieval}))
    tr.edges.append(TraceEdge("ent", "ret"))

    # ---- 5a. legacy lookup route
    if resp.route == "simple_retrieval":
        lp = resp.legacy_plan or {}
        rows = 0 if resp.legacy_df is None else len(resp.legacy_df)
        tr.nodes.append(TraceNode(
            "cy", "step", f"Cypher plan: {lp.get('type', '-')}",
            [_short(lp.get("cypher") or "no query generated", 70)],
            status="ok" if lp.get("cypher") else "warn"))
        tr.edges.append(TraceEdge("ret", "cy"))
        tr.nodes.append(TraceNode(
            "exec", "graph", "Graph traversal",
            [f"{rows} row(s) returned from the NetworkX property graph"],
            status="ok" if rows else "warn"))
        tr.edges.append(TraceEdge("cy", "exec"))
        tr.nodes.append(TraceNode("ans", "answer", "Answer", ["Direct result table."]))
        tr.edges.append(TraceEdge("exec", "ans"))
        _build_cypher_path(tr, lp.get("cypher"), graph)
        tr.summary = {"steps": 1, "operations": 1, "tables": 0,
                      "route": resp.route, "total_ms": resp.timings_ms.get("total", 0)}
        tr.step_rows = [{
            "step": "cypher", "operation": lp.get("type", "-"),
            "parameters": _short(lp.get("cypher") or "-", 90),
            "reads": "NetworkX property graph", "rows": rows,
            "status": "ok" if rows else "empty"}]
        return tr

    # ---- 5b. analytical route
    plan = resp.plan
    plines = []
    if plan is not None:
        if plan.problem_id:
            plines.append(f"{plan.problem_id} - {_short(plan.problem_statement, 58)}")
        else:
            plines.append("ad-hoc plan (no Problem Statement matched)")
        plines.append(f"tier {plan.tier}  -  {len(plan.steps)} step(s)")
        if plan.bindings:
            plines.append("bound: " + _short(
                ", ".join(f"{k}={v}" for k, v in plan.bindings.items()), 58))
    tr.nodes.append(TraceNode("plan", "plan", "Analytical plan", plines))
    tr.edges.append(TraceEdge("ret", "plan"))

    # parameter provenance: which step produced which binding
    provenance: Dict[str, str] = getattr(plan, "binding_provenance", {}) or {}
    derived_from: Dict[str, str] = {}
    for name, prov in provenance.items():
        m = re.match(r"derived from step (\w+)", str(prov))
        if m:
            derived_from[name] = m.group(1)

    all_tables: List[str] = []
    prev = "plan"
    for i, sr in enumerate(resp.step_results):
        sid = f"s_{sr.step.id}"
        rows = _rows_of(sr.result)
        params = sr.step.params or {}
        detail = []
        dims = _step_dimensions(params)
        if dims:
            detail.append("by " + " x ".join(
                ac.DIMENSIONS[d].label if d in ac.DIMENSIONS else str(d) for d in dims))
        mets = _step_metrics(params)
        if mets:
            detail.append(_short(", ".join(
                ac.METRICS[m].label if m in ac.METRICS else m for m in mets), 52))
        filt = _fmt_filters(params.get("filters") or params.get("focus_filters"))
        if filt:
            detail.append(f"filtered to {_short(filt, 44)}")
        if sr.ok:
            scalar = _scalar_summary(sr.step.op, sr.result)
            if rows is not None:
                detail.append(f"{rows} row(s)")
            elif scalar:
                detail.append(_short(scalar, 58))
            else:
                detail.append("computed")
        else:
            detail.append(f"unavailable: {_short(sr.error, 48)}")

        srcs = _step_sources(sr.step.op, params)
        for t in srcs:
            if t not in all_tables:
                all_tables.append(t)

        tr.nodes.append(TraceNode(
            sid, "step" if sr.ok else "step_failed",
            f"{sr.step.id.upper()}  {sr.step.op}",
            [_short(sr.step.title, 56)] + detail,
            status="ok" if sr.ok else "fail",
            meta={"sources": srcs, "metrics": mets, "dimensions": dims,
                  "base": _step_base(sr.step.op, params), "rows": rows,
                  "purpose": sr.step.purpose}))
        tr.edges.append(TraceEdge(prev, sid, "" if i else "execute"))
        prev = sid

        tr.step_rows.append({
            "step": sr.step.id, "operation": sr.step.op, "title": sr.step.title,
            "dimensions": ", ".join(dims) or "-",
            "metrics": ", ".join(mets) or "-",
            "filters": filt or "-",
            "reads": ", ".join(ac.RAW_TABLE_FILES.get(t, t) for t in srcs),
            "rows": rows if rows is not None else "-",
            "result": _short(_scalar_summary(sr.step.op, sr.result) or
                             (f"{rows} row table" if rows is not None else "-"), 60),
            "status": "ok" if sr.ok else "unavailable",
            "purpose": sr.step.purpose,
        })

    # parameter-dependency edges (step -> step), drawn dashed
    step_ids = {sr.step.id for sr in resp.step_results}
    for sr in resp.step_results:
        params = sr.step.params or {}
        used = set()
        for name, src_step in derived_from.items():
            if src_step not in step_ids or src_step == sr.step.id:
                continue
            val = (plan.bindings or {}).get(name)
            if val is None:
                continue
            if _params_mention(params, val):
                used.add(src_step)
        for src_step in used:
            tr.edges.append(TraceEdge(
                f"s_{src_step}", f"s_{sr.step.id}",
                f"${_binding_name(derived_from, src_step)}", "dashed"))

    # ---- 6. composition + answer
    ans = resp.answer
    tmpl = plan.answer_template if plan is not None else "-"
    clines = [f"template: {tmpl}"]
    if ans is not None:
        clines.append(f"{len(ans.evidence)} evidence - {len(ans.analysis)} analysis "
                      f"- {len(ans.recommendation)} recommendation")
        if ans.caveats:
            clines.append(f"{len(ans.caveats)} caveat(s) attached")
    tr.nodes.append(TraceNode("comp", "compose", "Answer composition", clines))
    tr.edges.append(TraceEdge(prev, "comp", "compose"))
    tr.nodes.append(TraceNode(
        "ans", "answer", "Business answer",
        [_short(ans.headline, 66)] if ans else ["-"]))
    tr.edges.append(TraceEdge("comp", "ans"))

    _build_lineage(tr, resp, all_tables)
    _build_join_path(tr, all_tables)

    tr.summary = {
        "steps": len(resp.step_results),
        "operations": len({sr.step.op for sr in resp.step_results}),
        "tables": len(all_tables),
        "metrics": len({m for sr in resp.step_results
                        for m in _step_metrics(sr.step.params or {})}),
        "route": resp.route,
        "total_ms": resp.timings_ms.get("total", 0),
        "rows_examined": sum(r["rows"] for r in tr.step_rows
                             if isinstance(r["rows"], int)),
    }
    return tr


def _binding_name(derived_from: Dict[str, str], src_step: str) -> str:
    for name, s in derived_from.items():
        if s == src_step:
            return name
    return "param"


def _params_mention(params: Any, value: Any) -> bool:
    if isinstance(params, dict):
        return any(_params_mention(v, value) for v in params.values())
    if isinstance(params, (list, tuple)):
        return any(_params_mention(v, value) for v in params)
    if isinstance(value, (list, tuple)):
        return any(params == v for v in value) if not isinstance(params, (list, tuple)) else params == list(value)
    return params == value


def _build_lineage(tr: QueryTrace, resp: Any, tables: Sequence[str]):
    """raw CSV -> fact frame -> metric/dimension -> step."""
    ctx = ac.DataContext.get()
    counts = {
        "policies": len(ctx.policies), "customers": len(ctx.customers),
        "products": len(ctx.products), "plans": len(ctx.plans),
        "sales": len(ctx.sales), "claims": len(ctx.claims),
        "surrenders": len(ctx.surrenders), "retention": len(ctx.retention),
    }
    for t in tables:
        tr.lineage_nodes.append(TraceNode(
            f"t_{t}", "table", ac.RAW_TABLE_FILES.get(t, t),
            [f"{counts.get(t, 0):,} rows"]))

    frames_used: Dict[str, List[str]] = {}
    for sr in resp.step_results:
        base = _step_base(sr.step.op, sr.step.params or {})
        frames_used.setdefault(base, []).append(sr.step.id)

    frame_rows = {
        "policy": len(ctx.policy_fact), "claim": len(ctx.claim_fact),
        "retention": len(ctx.retention_fact), "customer": len(ctx.customer_fact),
        "agent": len(ctx.agent_fact),
    }
    for base in frames_used:
        fid = f"f_{base}"
        tr.lineage_nodes.append(TraceNode(
            fid, "frame", FRAME_OF_BASE.get(base, base),
            [f"{frame_rows.get(base, 0):,} rows - one per "
             f"{'policy' if base == 'policy' else base}"]))
        for t in ac.BASE_SOURCES.get(base, ()):
            if t in tables:
                tr.lineage_edges.append(TraceEdge(f"t_{t}", fid, "join"))

    # extra tables feeding the policy frame for specific metrics
    for sr in resp.step_results:
        base = _step_base(sr.step.op, sr.step.params or {})
        fid = f"f_{base}"
        for t in _step_sources(sr.step.op, sr.step.params or {}):
            if t in tables and not any(
                    e.src == f"t_{t}" and e.dst == fid for e in tr.lineage_edges):
                tr.lineage_edges.append(TraceEdge(f"t_{t}", fid, "join"))

    seen_metric = set()
    for sr in resp.step_results:
        params = sr.step.params or {}
        base = _step_base(sr.step.op, params)
        sid = f"ls_{sr.step.id}"
        tr.lineage_nodes.append(TraceNode(
            sid, "step" if sr.ok else "step_failed",
            f"{sr.step.id.upper()} {sr.step.op}",
            [_short(sr.step.title, 40)], status="ok" if sr.ok else "fail"))

        mets = _step_metrics(params)[:4]
        dims = _step_dimensions(params)[:2]
        if not mets and not dims:
            tr.lineage_edges.append(TraceEdge(f"f_{base}", sid))
        for m in mets:
            mid = f"m_{m}"
            if mid not in seen_metric:
                seen_metric.add(mid)
                spec = ac.METRICS.get(m)
                lines = [_short(spec.definition, 62)] if spec else ["portfolio share"]
                tr.lineage_nodes.append(TraceNode(
                    mid, "metric",
                    spec.label if spec else ac.SHARE_METRICS.get(m, (m, m))[1], lines))
                tr.lineage_edges.append(TraceEdge(f"f_{base}", mid, "compute"))
            tr.lineage_edges.append(TraceEdge(mid, sid))
        for d in dims:
            did = f"d_{d}"
            if did not in seen_metric:
                seen_metric.add(did)
                dim = ac.DIMENSIONS.get(d)
                tr.lineage_nodes.append(TraceNode(
                    did, "dimension", f"group by {dim.label if dim else d}",
                    [f"column {dim.column}" if dim else ""]))
                tr.lineage_edges.append(TraceEdge(f"f_{base}", did, "group"))
            tr.lineage_edges.append(TraceEdge(did, sid))


def _build_join_path(tr: QueryTrace, tables: Sequence[str]):
    """The property-graph relationship path the pandas joins correspond to."""
    node_names = [TABLE_TO_NODE[t] for t in tables if t in TABLE_TO_NODE]
    if not node_names:
        return
    if len(node_names) > 1 and "Policy" not in node_names:
        node_names.append("Policy")
    node_set = set(node_names)
    ctx = ac.DataContext.get()
    counts = {
        "Policy": len(ctx.policies), "Customer": len(ctx.customers),
        "Product": len(ctx.products), "Plan": len(ctx.plans),
        "Agent": len(ctx.sales), "Claim": len(ctx.claims),
        "Surrender": len(ctx.surrenders), "Retention": len(ctx.retention),
    }
    for n in sorted(node_set):
        tr.path_nodes.append(TraceNode(
            f"g_{n}", "graph", n, [f"{counts.get(n, 0):,} nodes"]))
    for a, b, rel in GRAPH_EDGES:
        if a in node_set and b in node_set:
            tr.path_edges.append(TraceEdge(f"g_{a}", f"g_{b}", rel))
    tr.path_is_real_traversal = False
    tr.path_caption = (
        "This analysis ran on pandas fact tables, not on the graph. The path "
        "below is the relationship path in the property graph that those joins "
        "correspond to - it is shown to make the entity model explicit, not to "
        "claim a traversal took place.")


def _build_cypher_path(tr: QueryTrace, cypher: Optional[str], graph: Any):
    """The Cypher MATCH pattern that was actually executed on the graph.

    Retained for trace records written while the graph route existed. Lookups
    are now answered directly from the fact frames by `engine.lookup`, so no
    Cypher is generated and this path is not reached for new questions.
    """
    if not cypher:
        tr.path_caption = ("Lookups are answered directly from the fact tables; "
                           "no graph query is generated.")
        return
    try:
        import graph_engine as ge  # removed with the graph route
    except ImportError:
        tr.path_caption = ("This trace predates the removal of the graph route, "
                           "so its Cypher pattern can no longer be rendered.")
        return
    m = re.search(r"MATCH\s+(.*?)(?:\s+WHERE|\s+RETURN|$)", cypher,
                  re.IGNORECASE | re.DOTALL)
    if not m:
        tr.path_caption = "The executed query had no MATCH pattern to display."
        return
    try:
        node_patterns, edge_patterns = ge.parse_match(m.group(1))
    except Exception:
        tr.path_caption = "The MATCH pattern could not be parsed for display."
        return

    live = {}
    if graph is not None:
        for _, attrs in graph.nodes(data=True):
            nt = attrs.get("node_type")
            if nt:
                live[nt] = live.get(nt, 0) + 1

    for var, label in node_patterns.items():
        lines = [f"variable {var}"]
        if label and label in live:
            lines.append(f"{live[label]:,} nodes of this type in the graph")
        tr.path_nodes.append(TraceNode(
            f"g_{var}", "graph", label or var, lines))
    for src, dst, rel in edge_patterns:
        tr.path_edges.append(TraceEdge(f"g_{src}", f"g_{dst}", rel))

    tr.path_is_real_traversal = True
    tr.path_caption = (
        "This is the pattern that was actually traversed on the NetworkX "
        "property graph to answer the question.")


# ---------------------------------------------------------------------
# DOT rendering
# ---------------------------------------------------------------------

def _esc(t: str) -> str:
    return html.escape(str(t), quote=False)


def _node_dot(n: TraceNode) -> str:
    color = KIND_COLOR.get("step_failed" if n.status == "fail" else n.kind, C["blue"])
    if n.status == "warn":
        color = C["amber"]
    body = "".join(
        f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9" COLOR="{C["muted"]}">'
        f'{_esc(l)}</FONT></TD></TR>'
        for l in n.lines if l)
    label = (
        f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2">'
        f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="11" COLOR="{color}">'
        f'<B>{_esc(n.title)}</B></FONT></TD></TR>{body}</TABLE>>')
    return (f'  "{n.id}" [label={label}, shape=box, style="rounded,filled", '
            f'fillcolor="{C["node"]}", color="{color}", penwidth=1.4];')


def _edge_dot(e: TraceEdge) -> str:
    style = {"dashed": "dashed", "bold": "bold"}.get(e.style, "solid")
    color = C["amber"] if e.style == "dashed" else C["border"]
    lbl = (f', label="{_esc(e.label)}", fontcolor="{C["muted"]}", fontsize=9'
           if e.label else "")
    return (f'  "{e.src}" -> "{e.dst}" [color="{color}", style={style}, '
            f'penwidth=1.2, arrowsize=0.7{lbl}];')


def _wrap(body: str, rankdir: str = "TB") -> str:
    return "\n".join([
        "digraph trace {",
        f'  bgcolor="{C["bg"]}";',
        f"  rankdir={rankdir};",
        "  nodesep=0.35; ranksep=0.42; splines=true; pad=0.2;",
        f'  node [fontname="Helvetica", fontcolor="{C["text"]}"];',
        f'  edge [fontname="Helvetica"];',
        body,
        "}",
    ])


def pipeline_dot(tr: QueryTrace) -> str:
    body = "\n".join([_node_dot(n) for n in tr.nodes] +
                     [_edge_dot(e) for e in tr.edges])
    return _wrap(body, "TB")


def lineage_dot(tr: QueryTrace) -> str:
    if not tr.lineage_nodes:
        return ""
    ranks = []
    for kind in ("table", "frame"):
        ids = [n.id for n in tr.lineage_nodes if n.kind == kind]
        if ids:
            ranks.append("  { rank=same; " + " ".join(f'"{i}"' for i in ids) + " }")
    body = "\n".join([_node_dot(n) for n in tr.lineage_nodes] +
                     [_edge_dot(e) for e in tr.lineage_edges] + ranks)
    return _wrap(body, "LR")


def entity_path_dot(tr: QueryTrace) -> str:
    if not tr.path_nodes:
        return ""
    body = "\n".join([_node_dot(n) for n in tr.path_nodes] +
                     [_edge_dot(e) for e in tr.path_edges])
    return _wrap(body, "LR")


def step_table(tr: QueryTrace) -> pd.DataFrame:
    if not tr.step_rows:
        return pd.DataFrame()
    df = pd.DataFrame(tr.step_rows)
    # keep the column Arrow-serialisable: a step that returned a scalar rather
    # than a frame reports "-" alongside integer row counts.
    if "rows" in df.columns:
        df["rows"] = df["rows"].map(
            lambda v: f"{v:,}" if isinstance(v, (int, float)) and v == v else "-")
    return df


__all__ = ["QueryTrace", "TraceNode", "TraceEdge", "build_trace",
           "pipeline_dot", "lineage_dot", "entity_path_dot", "step_table",
           "GRAPH_EDGES", "TABLE_TO_NODE"]
