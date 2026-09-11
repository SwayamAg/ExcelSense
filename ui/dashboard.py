"""
dashboard.py
============
Insurance Business Analytics Engine
A unified AI-powered business analytics & query engine for insurance management.

Handles all query types seamlessly:
  - Multi-step business problem statements (prescriptive, diagnostic, portfolio health, etc.)
  - Ad-hoc analytics (custom metric-dimension comparisons, rankings, driver scans)
  - Direct database / graph record lookups
  - Out-of-scope honest boundary detection
"""

from __future__ import annotations

import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analytics_core as ac
import business_engine as be
import ollama_client as _oc

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Business Analytics Engine",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS & Design System ────────────────────────────────────────────────
CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: #0d1117; }
  section[data-testid="stSidebar"] { background: #161b22 !important; border-right: 1px solid #30363d; }

  .ba-hero {
    background: linear-gradient(135deg, #10233d 0%, #0d1117 60%);
    border: 1px solid #1f6feb44; border-radius: 16px;
    padding: 22px 28px; margin-bottom: 18px;
    box-shadow: 0 4px 24px rgba(31,111,235,0.12);
  }
  .ba-hero h2 { color:#e6edf3 !important; margin:0 0 6px; font-size:1.45rem; font-weight:700; }
  .ba-hero p  { color:#8b949e !important; margin:0; font-size:.92rem; line-height:1.5; }

  .ba-answer {
    background:#0f1b2d; border:1px solid #1f6feb55; border-left:4px solid #58a6ff;
    border-radius:12px; padding:18px 22px; margin:6px 0 16px;
    box-shadow: 0 2px 12px rgba(88,166,255,0.08);
  }
  .ba-answer .h { color:#e6edf3; font-size:1.08rem; font-weight:600; line-height:1.55; }

  .ba-sec {
    border:1px solid #30363d; border-radius:12px;
    background:#161b22; padding:16px 20px; margin-bottom:14px; height:100%;
  }
  .ba-sec .t {
    font-size:11.5px; font-weight:700; letter-spacing:1px;
    text-transform:uppercase; color:#8b949e; margin-bottom:10px;
  }
  .ba-sec ul { margin:0; padding-left:18px; }
  .ba-sec li { color:#c9d1d9 !important; font-size:13.5px; margin-bottom:8px; line-height:1.55; }
  .ba-sec.caveat { border-color:#d2992255; background:#1d1804; }
  .ba-sec.caveat li { color:#e3b341 !important; }
  .ba-sec.gap { border-color:#f8514955; background:#1d0d0d; }
  .ba-sec.gap li { color:#ff7b72 !important; }

  .ba-chip {
    display:inline-block; background:#161b22; border:1px solid #30363d;
    border-radius:8px; padding:6px 12px; margin:0 6px 6px 0;
  }
  .ba-chip b { color:#58a6ff; font-size:14px; }
  .ba-chip span { color:#8b949e; font-size:11px; margin-left:5px; }

  /* Query box */
  .stTextArea textarea {
    background: #161b22 !important; color: #e6edf3 !important;
    border: 1px solid #30363d !important; border-radius: 10px !important;
    font-size: 15px !important; line-height: 1.5 !important;
  }
  .stTextArea textarea:focus {
    border-color: #58a6ff !important; box-shadow: 0 0 10px rgba(88,166,255,0.2) !important;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #1f6feb, #238636) !important;
    color: #fff !important; border: none !important;
    font-weight: 600 !important; border-radius: 8px !important;
    padding: 8px 24px !important;
  }
</style>
"""

EXAMPLES: Dict[str, List[str]] = {
    "Executive & Risks": [
        "What are the biggest risks in our insurance portfolio?",
        "Give me a management-level assessment of the current portfolio.",
        "Which areas of the portfolio require management attention?",
    ],
    "Area & Persistency Analysis": [
        "Which area is having lowest persistency rate and why. how can I improve persistency rate.",
        "What is the relation between claims and persistency rate across products?",
        "Why is persistency declining in the lowest performing region?",
    ],
    "Product Performance": [
        "Which products should management prioritize for retention?",
        "Which products have unfavorable claims economics?",
        "Which products have high volume but poor persistency?",
    ],
    "Channel & Agents": [
        "Which sales channel gives us the best balance of growth and persistency?",
        "Compare the major sales channels across volume, policy value, lapse, surrender and retention.",
        "Are there agents with unusual performance patterns?",
    ],
    "Customer & Retention": [
        "Which customer segments are most at risk of lapsing?",
        "Which customers should the retention team prioritize?",
        "Which retention channels are most effective?",
    ],
    "Direct Record Lookup": [
        "How many customers are smokers?",
        "Show me the details of agent AGT0101",
        "How many policies were surrendered in the West zone?",
    ],
    "Honest Scope Boundaries": [
        "How do we compare with our competitors on market share?",
        "What is our expense ratio by product?",
    ],
}


@st.cache_resource(show_spinner="Initializing Analytics Core, Metric Registry & Problem Catalog...")
def load_engine():
    return be.get_engine(with_legacy=True)


def _render_html(content: str) -> None:
    """Render HTML content safely without triggering markdown code-block escapes."""
    if not content or not str(content).strip():
        return
    if hasattr(st, "html"):
        st.html(content)
    else:
        st.markdown(content, unsafe_allow_html=True)


def _md_bold(text: str) -> str:
    """Markdown bolding converter for HTML rendering."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", str(text))
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    return text


def _bullets(title: str, items: List[str], cls: str = "") -> str:
    if not items:
        return ""
    lis = "".join(f"<li>{_md_bold(i)}</li>" for i in items)
    return f'<div class="ba-sec {cls}"><div class="t">{title}</div><ul>{lis}</ul></div>'


def _chart(df: pd.DataFrame):
    """Honest bar chart when a metric-dimension combination is present."""
    metric_cols = [c for c in df.columns
                   if (c in ac.METRICS or c in ac.SHARE_METRICS)
                   and pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns
                if not pd.api.types.is_numeric_dtype(df[c])
                and not c.endswith("_reliable") and c not in ("method", "direction")]
    if not metric_cols or not cat_cols or len(df) < 2 or len(df) > 35:
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
                      title=dict(text=f"{y.replace('_', ' ').title()} by {x.replace('_', ' ').title()}"
                                      + ("  (grey = below sample threshold)"
                                         if rel_col is not None else ""),
                                 font=dict(size=12, color="#c9d1d9")))
    st.plotly_chart(fig)


def main():
    _render_html(CSS)
    engine = load_engine()
    caps = engine.capabilities()

    hero_html = f"""<div class="ba-hero">
  <h2>Insurance Business Analytics Engine</h2>
  <p>Ask any question &rarr; Semantic intent &rarr; Deterministic analytical plan &rarr; Multi-step execution &rarr; Grounded evidence & answer.
     Covers <b>{caps['business_problem_statements']}</b> problem statements &middot; 
     <b>{len(caps['metrics'])}</b> declared metrics &middot; 
     <b>{len(caps['dimensions'])}</b> dimensions &middot; 
     <b>{len(caps['operations'])}</b> analytical operations.</p>
</div>"""
    _render_html(hero_html)

    ollama_online = _oc.is_ollama_available()
    active_model = _oc.get_default_model() if ollama_online else "local"

    # ---- Sidebar with query examples & model status
    with st.sidebar:
        st.markdown("### Example Questions")
        for group, qs in EXAMPLES.items():
            with st.expander(group, expanded=(group == "Executive & Risks")):
                for q in qs:
                    if st.button(q, key=f"ba_{hash(q)}", width="stretch"):
                        st.session_state["ba_query"] = q
        st.divider()
        llm_label = "🤖 LLM Narrative Synthesis" + (f" (Ollama: {active_model})" if ollama_online else "")
        use_llm = st.checkbox(llm_label, value=ollama_online,
                              help=f"Synthesizes fluent executive answer using local Ollama model {active_model}.")

    default_q = st.session_state.get(
        "ba_query", "What are the biggest risks in our insurance portfolio?")
    query = st.text_area("Question", value=default_q, height=78,
                         label_visibility="collapsed", placeholder="Ask any insurance analytics or portfolio question...")
    
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        run = st.button("Run analysis")

    if not run and "ba_last" not in st.session_state:
        st.info("Ask any business or portfolio question above, or select an example from the sidebar.")
        with st.expander("Scope & Metric Capabilities"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Business Domains Covered**")
                for d in caps["business_domains"]:
                    st.markdown(f"- {d}")
            with c2:
                st.markdown("**Honest Scope Boundaries (Not in Dataset)**")
                st.markdown(
                    "- expenses, commission, reserves\n"
                    "- competitor or market-share data\n"
                    "- regulatory or solvency capital data\n"
                    "- complaints, NPS, satisfaction scores\n"
                    "- speculative forward-looking forecasts")
        with st.expander("Metric Registry (Declared Denominators)"):
            mdf = pd.DataFrame([
                {"Metric": k, "Label": v["label"], "Unit": v["unit"], "Definition": v["definition"]}
                for k, v in caps["metrics"].items()])
            st.dataframe(mdf, hide_index=True, height=400)
        return

    if run and query.strip():
        st.session_state["ba_query"] = query
        with st.spinner("Analyzing portfolio data and synthesizing answer..."):
            st.session_state["ba_last"] = engine.ask(query, llm=use_llm)

    resp = st.session_state.get("ba_last")
    if resp is None:
        return

    ans = resp.answer

    # ── CASE 1: Direct Lookup (Legacy Graph Query) ──
    if resp.route == "simple_retrieval":
        direct_html = """<div class="ba-answer">
  <div style="font-size: 11px; text-transform: uppercase; color: #3fb950; font-weight: 700; margin-bottom: 6px;">
    📍 Direct Database / Graph Record Lookup
  </div>
  <div class="h">Direct match retrieved from dataset entities.</div>
</div>"""
        _render_html(direct_html)
        if resp.legacy_df is not None and not resp.legacy_df.empty:
            st.dataframe(resp.legacy_df)
        elif resp.legacy_output:
            st.markdown(resp.legacy_output)
        return

    if ans is None:
        st.warning(resp.message or "No answer produced.")
        return

    # ── CASE 2: Executive Answer Headline ──
    _render_html(f'<div class="ba-answer"><div class="h">{_md_bold(ans.headline)}</div></div>')

    # ── CASE 3: LLM Executive Answer Tile ──
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

    # ── CASE 4: Structured Cards (Evidence, Implication, Analysis, Recommendation) ──
    left, right = st.columns([1, 1], gap="medium")
    with left:
        _render_html(_bullets("Evidence", ans.evidence))
        _render_html(_bullets("Business implication", ans.implication))
    with right:
        _render_html(_bullets("Analysis", ans.analysis))
        _render_html(_bullets("Recommendation", ans.recommendation))

    if ans.caveats:
        _render_html(_bullets("Analytical Caveats & Statistical Boundaries", ans.caveats, cls="caveat"))

    # ── CASE 5: Supporting Charts & Data Tables ──
    if ans.tables:
        st.divider()
        st.markdown("### Supporting Evidence & Data Tables")
        first_title, first_df = ans.tables[0]
        if isinstance(first_df, pd.DataFrame) and not first_df.empty:
            _chart(first_df)
        
        for title, df in ans.tables:
            with st.expander(f"📊 {title} ({len(df)} rows)", expanded=False):
                st.dataframe(df)


if __name__ == "__main__":
    main()
