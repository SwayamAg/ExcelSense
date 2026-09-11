"""
business_dashboard.py
=====================
Streamlit UI for the business-analytics engine.

Rendered by `dashboard.py` when the sidebar mode is "Business Analyst". The
classic pipeline-trace UI is untouched and still available from the same app.

The screen deliberately shows the machinery, not just the answer:
    routing decision -> retrieval -> analytical plan (before results)
    -> per-step execution -> structured business answer -> supporting tables
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

CSS = """
<style>
  .ba-hero {
    background: linear-gradient(135deg, #10233d 0%, #0d1117 60%);
    border: 1px solid #1f6feb44; border-radius: 16px;
    padding: 22px 28px; margin-bottom: 18px;
  }
  .ba-hero h2 { color:#e6edf3 !important; margin:0 0 4px; font-size:1.35rem; }
  .ba-hero p  { color:#8b949e !important; margin:0; font-size:.9rem; }

  .ba-route {
    display:inline-block; padding:3px 12px; border-radius:20px;
    font-size:11px; font-weight:700; letter-spacing:.6px; text-transform:uppercase;
  }
  .ba-answer {
    background:#0f1b2d; border:1px solid #1f6feb55; border-left:4px solid #58a6ff;
    border-radius:12px; padding:18px 22px; margin:6px 0 16px;
  }
  .ba-answer .h { color:#e6edf3; font-size:1.05rem; font-weight:600; line-height:1.55; }

  .ba-sec { border:1px solid #30363d; border-radius:12px;
            background:#161b22; padding:14px 18px; margin-bottom:12px; height:100%; }
  .ba-sec .t { font-size:11px; font-weight:700; letter-spacing:1px;
               text-transform:uppercase; color:#8b949e; margin-bottom:8px; }
  .ba-sec ul { margin:0; padding-left:18px; }
  .ba-sec li { color:#c9d1d9 !important; font-size:13px; margin-bottom:6px; line-height:1.5; }
  .ba-sec.caveat { border-color:#d2992255; background:#1d1804; }
  .ba-sec.caveat li { color:#e3b341 !important; }
  .ba-sec.gap { border-color:#f8514955; background:#1d0d0d; }
  .ba-sec.gap li { color:#ff7b72 !important; }

  .ba-step { border:1px solid #30363d; border-left:3px solid #3fb950;
             background:#12181f; border-radius:8px; padding:10px 14px; margin-bottom:8px; }
  .ba-step.bad { border-left-color:#f85149; }
  .ba-step .n { font-size:11px; color:#6e7681; letter-spacing:.5px; }
  .ba-step .ti { color:#e6edf3; font-size:13px; font-weight:600; }
  .ba-step .op { font-family:'JetBrains Mono',monospace; font-size:11px; color:#79c0ff; }
  .ba-step .pu { color:#8b949e; font-size:12px; margin-top:3px; }

  .ba-kpi { background:#161b22; border:1px solid #30363d; border-radius:10px;
            padding:12px 14px; text-align:center; }
  .ba-kpi .v { font-size:19px; font-weight:700; color:#58a6ff; }
  .ba-kpi .l { font-size:10.5px; color:#8b949e; margin-top:3px; line-height:1.3; }
  .ba-kpi .d { font-size:9.5px; color:#484f58; margin-top:2px; }

  .ba-chip { display:inline-block; background:#161b22; border:1px solid #30363d;
             border-radius:8px; padding:6px 12px; margin:0 6px 6px 0; }
  .ba-chip b { color:#58a6ff; font-size:14px; }
  .ba-chip span { color:#8b949e; font-size:11px; margin-left:5px; }
  .ba-note { color:#8b949e; font-size:12px; border-left:2px solid #30363d;
             padding-left:10px; margin:6px 0 12px; line-height:1.5; }
  .ba-note.warn { border-left-color:#d29922; color:#e3b341; }
</style>
"""

ROUTE_STYLE = {
    "business_analysis": ("#1f3a5f", "#58a6ff", "Multi-step business analysis"),
    "adhoc_analysis": ("#3a2e00", "#d29922", "Ad-hoc analytical plan"),
    "simple_retrieval": ("#1a3a2a", "#3fb950", "Simple retrieval (legacy engine)"),
    "insufficient_data": ("#3a1a1a", "#f85149", "Insufficient data"),
}

EXAMPLES: Dict[str, List[str]] = {
    "Executive & Portfolio Strategy": [
        "What are the biggest risks in our insurance portfolio?",
        "Give me a management-level assessment of the current portfolio.",
        "Which areas of the portfolio require management attention?",
    ],
    "Salesperson & Advisor Deep-Dive": [
        "What is the persistency rate and risk profile of salesperson 270084?",
        "Show performance metrics, persistency rate, and risk flags for salesperson 977337.",
        "Evaluate salesperson 800722 compared to peers across persistency, surrender rate, and PIVC mismatch.",
        "Which salespeople have the lowest persistency rate and highest surrender rate?",
        "Compare top salespeople by first year persistency rate and sales related complaints.",
        "Show advisor health diagnostics for salesperson 800812 and recommend interventions.",
    ],
    "Plan & Retention Decisions": [
        "Which plans should management prioritize for retention?",
        "Which plans have unfavorable claims economics or high mortality payout?",
        "Which plans have high volume but poor persistency?",
        "Analyze plan retention priorities across annual premium and lapse exposure.",
    ],
    "Channel & Distribution Analytics": [
        "Which sales channel gives us the best balance of volume, APE, and persistency?",
        "Compare Banca channel performance across volume, lapse, surrender and persistency buckets.",
        "Are there sales advisors with unusual compliance or RCU rejection patterns?",
    ],
    "Cross-Domain & Multi-Factor Complex Queries": [
        "Analyze the trade-off between plan claim payout and persistency bucket across all policies.",
        "Identify high-risk combinations of customer age, payment mode, and policy term driving lapse.",
        "What factors distinguish high persistency advisors from low persistency advisors?",
        "Which customer segments are most at risk of lapsing across top states (UP, Karnataka, WB)?",
        "Why does ECS/Auto-Debit payment mode compare favorably against Cash renewals?",
        "What demographic factors are associated with policy lapse?",
    ],
    "Risk, Anomaly & Concentration Scans": [
        "Are there unusual concentrations or patterns management should investigate?",
        "Are policies or sum assured unusually concentrated in a few top accounts or states?",
        "What is driving mortality claims risk and early claim frequency?",
    ],
    "Direct Lookup & Instance Queries": [
        "Show me details for policy AppID 1000000015",
        "Show profile and demographics for customer ClientID 7140735859",
        "Show performance and scorecard for salesperson 270084",
        "Show claim settlement details for Claim 490",
    ],
    "Honest Scope Refusal": [
        "How do we compare with our competitors on market share?",
        "What is our investment expense ratio by fund manager?",
    ],
}


def _resolve_history_file() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, "data", "history", "query_history.json"),
        os.path.join(base, "..", "data", "history", "query_history.json"),
        os.path.join(base, "..", "..", "data", "history", "query_history.json"),
        os.path.join(base, "query_history.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    target = os.path.join(base, "data", "history", "query_history.json") if not base.endswith("ui") else os.path.join(base, "..", "data", "history", "query_history.json")
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    return os.path.abspath(target)


HISTORY_FILE = _resolve_history_file()


def _load_history_from_disk() -> List[Dict[str, Any]]:
    import json
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            print(f"[History Load Error] {e}")
    return []


def _save_history_to_disk(history_list: List[Dict[str, Any]]) -> None:
    import json
    try:
        clean_list = []
        for item in history_list[-50:]:
            entry = {
                "query": item.get("query", ""),
                "timestamp": item.get("timestamp", ""),
                "route": item.get("route", "business_analysis"),
                "headline": item.get("headline", ""),
                "evidence": item.get("evidence", []),
                "analysis": item.get("analysis", []),
                "implication": item.get("implication", []),
                "recommendation": item.get("recommendation", []),
                "caveats": item.get("caveats", []),
                "unavailable": item.get("unavailable", []),
                "llm_narrative": item.get("llm_narrative"),
                "legacy_output": item.get("legacy_output"),
                "tables": item.get("tables", []),
            }
            clean_list.append(entry)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(clean_list, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[History Save Error] {e}")


def _reconstruct_engine_response(h_item: Dict[str, Any]) -> Any:
    from answer_composer import BusinessAnswer
    from business_engine import EngineResponse
    tables_list = []
    for t in h_item.get("tables", []):
        if isinstance(t, (list, tuple)) and len(t) == 2:
            title, rows = t[0], t[1]
            if isinstance(rows, list):
                tables_list.append((title, pd.DataFrame(rows)))
            elif isinstance(rows, pd.DataFrame):
                tables_list.append((title, rows))

    ans = BusinessAnswer(
        question=h_item.get("query", ""),
        headline=h_item.get("headline", ""),
        evidence=h_item.get("evidence", []),
        analysis=h_item.get("analysis", []),
        implication=h_item.get("implication", []),
        recommendation=h_item.get("recommendation", []),
        caveats=h_item.get("caveats", []),
        unavailable=h_item.get("unavailable", []),
        llm_narrative=h_item.get("llm_narrative"),
        tables=tables_list,
    ) if h_item.get("headline") else None

    resp = EngineResponse(
        question=h_item.get("query", ""),
        route=h_item.get("route", "business_analysis"),
        answer=ans,
        legacy_output=h_item.get("legacy_output"),
    )
    return resp


@st.cache_resource(show_spinner="Building fact tables, metric registry and PS library...")
def load_engine():
    import business_engine as be
    return be.get_engine(with_legacy=True)


def _render_html(content: str) -> None:
    """Render HTML content safely without triggering markdown code-block escapes."""
    if not content or not str(content).strip():
        return
    if hasattr(st, "html"):
        st.html(content)
    else:
        st.markdown(content, unsafe_allow_html=True)


def _bullets(title: str, items: List[str], cls: str = "") -> str:
    if not items:
        return ""
    lis = "".join(f"<li>{_md_bold(i)}</li>" for i in items)
    return f'<div class="ba-sec {cls}"><div class="t">{title}</div><ul>{lis}</ul></div>'


def _md_bold(text: str) -> str:
    """Minimal markdown -> HTML for the bold and code spans the composer emits."""
    import re
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", str(text))
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    return text


def render():
    _render_html(CSS)
    engine = load_engine()
    caps = engine.capabilities()

    hero_html = f"""<div class="ba-hero">
  <h2>Insurance Business Analytics Engine</h2>
  <p>Business question &rarr; intent &rarr; analytical plan &rarr; deterministic
     multi-step execution &rarr; evidence-backed answer.
     {caps['business_problem_statements']} business problem statements &middot;
     {caps['analytical_steps']} analytical steps &middot;
     {len(caps['metrics'])} declared metrics &middot;
     {len(caps['dimensions'])} dimensions &middot;
     {len(caps['operations'])} analytical operations.</p>
</div>"""
    _render_html(hero_html)

    import ollama_client as _oc
    ollama_online = _oc.is_ollama_available()
    active_model = _oc.get_default_model() if ollama_online else "local"

    # Initialize query history from local disk if not already loaded in session
    if "ba_history" not in st.session_state:
        st.session_state["ba_history"] = _load_history_from_disk()

    # ---- Sidebar with Examples & History
    with st.sidebar:
        st.markdown("### 🔍 Business Questions")
        for group, qs in EXAMPLES.items():
            with st.expander(group, expanded=(group == "Salesperson & Advisor Deep-Dive")):
                for q in qs:
                    if st.button(q, key=f"ba_{hash(q)}", width="stretch"):
                        st.session_state["ba_query"] = q
        st.divider()

        # History Section in Sidebar
        st.markdown("### 🕒 Query History (Local)")
        if st.session_state["ba_history"]:
            c_hist, c_clr = st.columns([3, 1])
            with c_hist:
                st.caption(f"{len(st.session_state['ba_history'])} saved queries (JSON)")
            with c_clr:
                if st.button("Clear", key="clr_hist"):
                    st.session_state["ba_history"] = []
                    _save_history_to_disk([])
                    st.rerun()

            for idx, h_item in enumerate(reversed(st.session_state["ba_history"][-15:])):
                q_text = h_item["query"]
                q_label = (q_text[:36] + "…") if len(q_text) > 36 else q_text
                ts = h_item.get("timestamp", "")
                if st.button(f"📌 {q_label}", key=f"hist_btn_{idx}", help=f"[{ts}] {q_text}", width="stretch"):
                    st.session_state["ba_query"] = q_text
                    st.session_state["ba_last"] = _reconstruct_engine_response(h_item)
                    st.rerun()
        else:
            st.caption("No queries recorded yet.")

        st.divider()
        llm_label = "🤖 LLM Executive Answer" + (f" (Ollama: {active_model})" if ollama_online else "")
        # Off by default: the deterministic answer is the product, and a prose
        # layer that rephrases on every run undercuts the verification panels.
        use_llm = st.checkbox(llm_label, value=False,
                              help=f"Optional narrative pass over the already-computed "
                                   f"figures using local Ollama model {active_model}. "
                                   f"The deterministic answer is always shown regardless.")

    default_q = st.session_state.get(
        "ba_query", "What are the biggest risks in our insurance portfolio?")
    query = st.text_area("Business question", value=default_q, height=78,
                         label_visibility="collapsed")
    run = st.button("Run analysis")

    if not run and "ba_last" not in st.session_state:
        st.info("Ask a business question above, or pick one from the sidebar.")
        with st.expander("What this engine can and cannot answer"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Business domains covered**")
                for d in caps["business_domains"]:
                    st.markdown(f"- {d}")
            with c2:
                st.markdown("**Not in the dataset**")
                st.markdown(
                    "- expenses, commission, reserves\n"
                    "- competitor or market-share data\n"
                    "- regulatory or solvency data\n"
                    "- complaints, NPS, satisfaction\n"
                    "- any forward-looking projection")
        with st.expander("Metric registry (every denominator declared)"):
            mdf = pd.DataFrame([
                {"metric": k, "label": v["label"], "unit": v["unit"],
                 "definition": v["definition"]}
                for k, v in caps["metrics"].items()])
            st.dataframe(mdf, hide_index=True, height=420)
        return

    if run and query.strip():
        import datetime
        st.session_state["ba_query"] = query
        spinner_msg = f"Planning, executing, and synthesizing with Ollama ({active_model})..." if use_llm else "Planning and executing analytical plan..."
        with st.spinner(spinner_msg):
            resp_obj = engine.ask(query, llm=use_llm)
            st.session_state["ba_last"] = resp_obj

            # Record and save complete response to local disk
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            tables_serializable = []
            if resp_obj.answer and resp_obj.answer.tables:
                for t_title, t_df in resp_obj.answer.tables:
                    if isinstance(t_df, pd.DataFrame):
                        tables_serializable.append([t_title, t_df.to_dict(orient="records")])

            entry = {
                "query": query,
                "timestamp": now_str,
                "route": resp_obj.route,
                "headline": resp_obj.answer.headline if resp_obj.answer else "",
                "evidence": resp_obj.answer.evidence if resp_obj.answer else [],
                "analysis": resp_obj.answer.analysis if resp_obj.answer else [],
                "implication": resp_obj.answer.implication if resp_obj.answer else [],
                "recommendation": resp_obj.answer.recommendation if resp_obj.answer else [],
                "caveats": resp_obj.answer.caveats if resp_obj.answer else [],
                "unavailable": resp_obj.answer.unavailable if resp_obj.answer else [],
                "llm_narrative": resp_obj.answer.llm_narrative if resp_obj.answer else None,
                "legacy_output": resp_obj.legacy_output,
                "tables": tables_serializable,
            }
            # Remove any prior entry for the same question so latest result is stored
            st.session_state["ba_history"] = [h for h in st.session_state["ba_history"] if h.get("query") != query]
            st.session_state["ba_history"].append(entry)
            _save_history_to_disk(st.session_state["ba_history"])

    resp = st.session_state.get("ba_last")
    if resp is None:
        return

    ans = resp.answer

    # ---- legacy route: show the classic output
    if resp.route == "simple_retrieval":
        st.markdown("This is a single-fact lookup, so it was answered by the "
                    "original graph query engine rather than the analytics planner.")
        st.markdown(resp.legacy_output or "_no output_")
        return

    if ans is None:
        st.warning(resp.message or "No answer produced.")
        return

    # ── ANSWER ────────────────────────────────────────────────────────────────
    _render_html(f'<div class="ba-answer"><div class="h">{_md_bold(ans.headline)}</div></div>')

    # ── LLM GENERATED ANSWER TILE (Only rendered when narrative exists) ──
    if ans.llm_narrative:
        import re
        clean_narrative = re.sub(r"\*\*([^*]+?)\*\*", r"\1", str(ans.llm_narrative))
        clean_narrative = re.sub(r"\*([^*]+?)\*", r"\1", clean_narrative).replace("**", "")
        llm_tile_html = f"""<div class="ba-sec" style="border: 1px solid #7c3aed; border-left: 5px solid #a855f7; background: linear-gradient(135deg, #18122b 0%, #161b22 100%); padding: 18px 22px; margin-bottom: 16px; box-shadow: 0 4px 20px rgba(124,58,237,0.18);">
  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
    <div style="font-size: 11.5px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; color: #c084fc;">
      🤖 LLM Generated Executive Answer
    </div>
    <div style="font-size: 11px; background: #3b0764; color: #e9d5ff; padding: 2px 10px; border-radius: 12px; border: 1px solid #9333ea;">
      Ollama: {active_model}
    </div>
  </div>
  <div style="color: #f3f4f6; font-size: 14px; line-height: 1.65; white-space: pre-wrap;">
{clean_narrative}
  </div>
</div>"""
        _render_html(llm_tile_html)

    # ── EVIDENCE · BUSINESS IMPLICATION (left) | ANALYSIS · RECOMMENDATION (right)
    left, right = st.columns([1, 1], gap="medium")
    with left:
        _render_html(_bullets("Evidence", ans.evidence))
        _render_html(_bullets("Business implication", ans.implication))
    with right:
        _render_html(_bullets("Analysis", ans.analysis))
        _render_html(_bullets("Recommendation", ans.recommendation))

    if ans.caveats:
        _render_html(_bullets("Analytical Caveats & Statistical Boundaries", ans.caveats, cls="caveat"))

    if ans.unavailable:
        _render_html(_bullets("Data Gaps / Out of Scope Elements", ans.unavailable, cls="gap"))

    # ── Supporting Charts & Data Tables ──
    if ans.tables:
        st.divider()
        st.markdown("### Supporting Evidence & Data Tables")
        first_title, first_df = ans.tables[0]
        if isinstance(first_df, pd.DataFrame) and not first_df.empty:
            _chart(first_df)

        for title, df in ans.tables:
            with st.expander(f"📊 {title} ({len(df)} rows)", expanded=False):
                st.dataframe(df)


def _chart(df: pd.DataFrame):
    """A single honest bar chart when the frame has one obvious category+metric."""
    import plotly.graph_objects as go
    import analytics_core as ac

    metric_cols = [c for c in df.columns
                   if (c in ac.METRICS or c in ac.SHARE_METRICS)
                   and pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns
                if not pd.api.types.is_numeric_dtype(df[c])
                and not c.endswith("_reliable") and c not in ("method", "direction")]
    if not metric_cols or not cat_cols or len(df) < 2 or len(df) > 30:
        return
    x, y = cat_cols[0], metric_cols[0]
    d = df.dropna(subset=[y])
    if d.empty:
        return
    rel_col = f"{y}_reliable" if f"{y}_reliable" in d.columns else (
        "reliable" if "reliable" in d.columns else None)
    colors = ["#58a6ff"] * len(d)
    if rel_col is not None:
        colors = ["#58a6ff" if bool(v) else "#6e7681" for v in d[rel_col]]
    fig = go.Figure(go.Bar(x=d[x].astype(str), y=d[y], marker_color=colors))
    fig.update_layout(margin=dict(l=8, r=8, t=34, b=8), height=290,
                      paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                      font_color="#8b949e",
                      xaxis=dict(gridcolor="#21262d"),
                      yaxis=dict(gridcolor="#21262d",
                                 title=ac.METRICS[y].label if y in ac.METRICS else y),
                      title=dict(text=f"{y} by {x}"
                                      + ("  (grey = below sample threshold)"
                                         if rel_col is not None else ""),
                                 font=dict(size=12, color="#c9d1d9")))
    st.plotly_chart(fig)


if __name__ == "__main__":
    try:
        st.set_page_config(
            page_title="Insurance Business Analytics Engine",
            page_icon="💼",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        pass
    render()

