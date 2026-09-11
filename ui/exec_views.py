"""
exec_views.py
=============
The executive-facing renderings of an analytical result.

Three displays, in the order an executive reads them:

  Method 1  a combined-risk ranking - one bar chart, one name at the top, with
            the score decomposed on demand so the ranking is auditable (V3).
  Method 2  a quadrant scatter - two metrics against each other, bubble area by
            book size, so a compound question ("worst on X *and* Y") shows the
            trade-off instead of asserting a single answer.
  V4        drill to the source records behind any point, with a CSV export.

Cohort gating happens before scoring, not after. `composite_score` min-max
normalises across whatever cohort it is handed, so leaving 1,485 one- and
two-policy agents in the population would rescale every score around noise.
The excluded counts are rendered next to the chart rather than dropped, because
a ranking that silently covers a quarter of the book is worse than no ranking.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
for _sub in ["src", "src/analytics", "src/engine", "src/retrieval", "src/llm"]:
    _p = os.path.join(ROOT_DIR, _sub)
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import streamlit as st

import analytics_core as ac

# Book size below which an agent-level rate is treated as noise rather than
# signal. Matches analytics_core.DEFAULT_MIN_N.
MIN_BOOK = 20

# Palette, matching business_dashboard's dark theme.
C_BLUE = "#58a6ff"
C_RED = "#f85149"
C_AMBER = "#d29922"
C_GREY = "#6e7681"
C_TEXT = "#c9d1d9"
C_MUTED = "#8b949e"
C_PANEL = "#161b22"
C_BG = "#0d1117"
C_GRID = "#21262d"

# Rates that are stated at agent level in Sales_Details and joined onto each of
# an agent's policy rows. Their `n` is a policy count, but the underlying
# evidence is a single declared figure - drilling to rows shows it repeated.
STATED_AGENT_RATES = {
    "sales_persistency_benchmark": "First Year Persistency Rate",
    "early_claim_rate": "Early Claim Rate",
    "surrender_rate": "Surrender Rate",
    "rcu_rejection_rate": "RCU Rejections Rate",
    "pivc_mismatch_rate": "PIVC Number Mismatch Rate",
}


# =====================================================================
# COHORT
# =====================================================================

@dataclass
class AgentCohort:
    """Which agents are eligible to be ranked, and who was left out."""
    ids: List[Any] = field(default_factory=list)
    min_book: int = MIN_BOOK
    n_eligible: int = 0
    n_total: int = 0
    n_small_book: int = 0
    n_missing_rate: int = 0

    @property
    def exclusion_note(self) -> str:
        return (f"ranked over {self.n_eligible:,} agents · "
                f"{self.n_small_book:,} excluded (book < {self.min_book}) · "
                f"{self.n_missing_rate:,} excluded (no rate on file)")


def build_agent_cohort(ctx: ac.DataContext, min_book: int = MIN_BOOK,
                       require: Sequence[str] = ("sales_first_year_persistency",
                                                 "sales_early_claim_rate")) -> AgentCohort:
    """Agents with a book large enough to rank and the rates needed to rank on."""
    af = ctx.frame("agent")
    present = [c for c in require if c in af.columns]

    small = af["policies_sold"] < min_book if "policies_sold" in af.columns else False
    missing = None
    for c in present:
        col_missing = af[c].isna()
        missing = col_missing if missing is None else (missing | col_missing)
    if missing is None:
        missing = af.index.to_series().map(lambda _: False)

    eligible = af[~small & ~missing]
    return AgentCohort(
        ids=eligible["sales_id"].tolist(),
        min_book=min_book,
        n_eligible=len(eligible),
        n_total=len(af),
        n_small_book=int(small.sum()) if hasattr(small, "sum") else 0,
        n_missing_rate=int(missing.sum()),
    )


# =====================================================================
# METHOD 1 - combined risk ranking
# =====================================================================

def agent_risk_ranking(ctx: ac.DataContext, cohort: AgentCohort,
                       components: Optional[List[Dict[str, Any]]] = None,
                       top_k: int = 6) -> pd.DataFrame:
    """Rank the eligible cohort on a weighted blend of two or more metrics."""
    components = components or [
        {"metric": "sales_persistency_benchmark", "weight": 0.5, "direction": "low"},
        {"metric": "early_claim_rate", "weight": 0.5, "direction": "high"},
    ]
    if not cohort.ids:
        return pd.DataFrame()
    return ac.composite_score(
        ctx, dimension="sales_id", components=components,
        filters={"sales_id": cohort.ids}, top_k=top_k, min_n=1,
    )


def render_method1(df: pd.DataFrame, cohort: AgentCohort,
                   components: Optional[List[Dict[str, Any]]] = None,
                   highlight: Optional[str] = None,
                   key: str = "m1") -> Optional[str]:
    """Ranked horizontal bars + the score decomposition (V3).

    Returns the top-ranked id, so the caller can carry it into turn state.
    """
    if df is None or df.empty:
        st.info("No agent met the ranking threshold for this question.")
        return None

    import plotly.graph_objects as go

    components = components or [
        {"metric": "sales_persistency_benchmark", "weight": 0.5, "direction": "low"},
        {"metric": "early_claim_rate", "weight": 0.5, "direction": "high"},
    ]
    d = df.copy()
    d["label"] = d["sales_id"].astype(str)
    top_id = str(d.iloc[0]["sales_id"])
    focus = str(highlight) if highlight else top_id

    colors = [C_RED if lbl == focus else C_BLUE for lbl in d["label"]]
    order = d.iloc[::-1]  # plotly draws bottom-up

    fig = go.Figure(go.Bar(
        x=order["composite_score"], y=order["label"], orientation="h",
        marker_color=list(reversed(colors)),
        text=[f"{v:.2f}" for v in order["composite_score"]],
        textposition="outside",
        hovertemplate="Agent %{y}<br>combined risk %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        height=max(220, 44 * len(d)),
        margin=dict(l=8, r=48, t=34, b=8),
        paper_bgcolor=C_PANEL, plot_bgcolor=C_BG, font_color=C_MUTED,
        xaxis=dict(gridcolor=C_GRID, title="combined risk score", range=[0, 1.05]),
        yaxis=dict(gridcolor=C_GRID, title="", type="category"),
        title=dict(text="Combined risk ranking", font=dict(size=13, color=C_TEXT)),
    )
    st.plotly_chart(fig, width="stretch", key=f"{key}_chart")
    st.caption(cohort.exclusion_note)

    _render_score_decomposition(d, components, focus, key=key)
    return top_id


def _render_score_decomposition(d: pd.DataFrame, components: List[Dict[str, Any]],
                                focus: str, key: str = "m1") -> None:
    """V3 - show every term that produced the score, and the scale it sits on."""
    row = d[d["sales_id"].astype(str) == str(focus)]
    if row.empty:
        return
    r = row.iloc[0]

    with st.expander(f"How this score is built — Agent {focus}", expanded=False):
        lines: List[Dict[str, Any]] = []
        for comp in components:
            m = comp["metric"]
            if m not in d.columns:
                continue
            w = float(comp.get("weight", 1.0))
            contribution = float(r.get(f"w_{m}", float("nan")))
            norm = contribution / w if w else float("nan")
            spec = ac.METRICS.get(m)
            n = r.get(f"{m}_n")
            reliable = bool(r.get(f"{m}_reliable", False))
            lines.append({
                "component": spec.label if spec else m,
                "value": spec.format(r[m]) if spec else f"{r[m]}",
                "normalised": f"{norm:.2f}",
                "weight": f"{w:g}",
                "contribution": f"{contribution:.2f}",
                "n": f"{int(n):,}" if pd.notna(n) else "-",
                "reliable": "yes" if reliable else "no",
            })
        if lines:
            st.dataframe(pd.DataFrame(lines), hide_index=True, width="stretch")
        st.markdown(
            f"**Total combined risk {float(r['composite_score']):.3f}** — the sum of "
            f"the contribution column.")
        st.caption(
            "Normalised values are min-max scaled across the ranked cohort, so the "
            "scale is relative to those agents and shifts if the cohort changes. "
            "Raw values do not.")

        stated = [c["metric"] for c in components if c["metric"] in STATED_AGENT_RATES]
        if stated:
            names = ", ".join(f"`{STATED_AGENT_RATES[m]}`" for m in stated)
            st.warning(
                f"The n column counts policies, but {names} are declared once per "
                f"agent in Sales_Details and copied onto each of their policy rows. "
                f"The underlying evidence is a single stated figure, not n independent "
                f"observations.")


# =====================================================================
# METHOD 2 - quadrant scatter
# =====================================================================

def agent_quadrant_frame(ctx: ac.DataContext, cohort: AgentCohort,
                         x_col: str = "sales_first_year_persistency",
                         y_col: str = "sales_early_claim_rate") -> pd.DataFrame:
    """Per-agent points for the quadrant: two rates plus book size."""
    af = ctx.frame("agent")
    keep = [c for c in ["sales_id", x_col, y_col, "policies_sold", "sales_type"]
            if c in af.columns]
    d = af[keep].copy()
    if cohort.ids:
        d = d[d["sales_id"].isin(cohort.ids)]
    return d.dropna(subset=[c for c in (x_col, y_col) if c in d.columns])


def render_method2(d: pd.DataFrame, cohort: AgentCohort,
                   x_col: str = "sales_first_year_persistency",
                   y_col: str = "sales_early_claim_rate",
                   highlight: Optional[str] = None,
                   key: str = "m2") -> None:
    """Persistency against early-claim rate, bubble area by book size.

    The danger quadrant is shaded rather than labelled per point, and bubble
    area carries book size, so small-book noise reads as small without needing
    a caveat sentence.
    """
    if d is None or d.empty:
        st.info("Not enough agents with both rates to plot.")
        return

    import plotly.graph_objects as go

    x_med = float(d[x_col].median())
    y_med = float(d[y_col].median())
    sizes = d["policies_sold"] if "policies_sold" in d.columns else pd.Series([10] * len(d))
    sizeref = 2.0 * float(sizes.max()) / (38.0 ** 2) if float(sizes.max()) > 0 else 1

    focus = str(highlight) if highlight else None
    is_focus = d["sales_id"].astype(str) == focus if focus else pd.Series([False] * len(d), index=d.index)
    colors = [C_RED if f else C_BLUE for f in is_focus]

    fig = go.Figure()
    # danger quadrant: low persistency, high claim rate
    fig.add_shape(type="rect", x0=d[x_col].min(), x1=x_med, y0=y_med, y1=d[y_col].max(),
                  fillcolor="rgba(248,81,73,0.07)", line=dict(width=0), layer="below")
    fig.add_hline(y=y_med, line=dict(color=C_GRID, width=1, dash="dot"))
    fig.add_vline(x=x_med, line=dict(color=C_GRID, width=1, dash="dot"))

    customdata = d[["sales_id", "policies_sold"]].values if "policies_sold" in d.columns else d[["sales_id"]].values
    fig.add_trace(go.Scatter(
        x=d[x_col], y=d[y_col], mode="markers",
        marker=dict(size=sizes, sizemode="area", sizeref=sizeref, sizemin=4,
                    color=colors, opacity=0.75,
                    line=dict(width=[2 if f else 0 for f in is_focus], color="#ffffff")),
        customdata=customdata,
        hovertemplate=("Agent %{customdata[0]}<br>persistency %{x:.1f}%"
                       "<br>early claim %{y:.2f}%"
                       + ("<br>book %{customdata[1]:,} policies" if "policies_sold" in d.columns else "")
                       + "<extra></extra>"),
    ))
    fig.update_layout(
        height=420, margin=dict(l=8, r=8, t=38, b=8),
        paper_bgcolor=C_PANEL, plot_bgcolor=C_BG, font_color=C_MUTED,
        xaxis=dict(gridcolor=C_GRID, title="first year persistency (%)"),
        yaxis=dict(gridcolor=C_GRID, title="early claim rate (%)"),
        showlegend=False,
        title=dict(text="Persistency vs early-claim rate · bubble = book size",
                   font=dict(size=13, color=C_TEXT)),
    )
    st.plotly_chart(fig, width="stretch", key=f"{key}_chart")
    st.caption(
        f"Shaded corner = below-median persistency and above-median early-claim rate. "
        f"Medians: {x_med:.1f}% / {y_med:.2f}%. {cohort.exclusion_note}.")


# =====================================================================
# AGENT IMPROVEMENT - what separates high from low persistency
# =====================================================================

def agent_book_contrast(ctx: ac.DataContext, cohort: AgentCohort,
                        focus_id: Optional[str] = None,
                        dimensions: Sequence[str] = ("premium_paying_term", "trad_ulip",
                                                     "income_band", "plan_name",
                                                     "age_band", "state"),
                        top_n: int = 3) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compare what high-persistency agents sell against everyone else.

    This is the only defensible answer to "how do sales people improve
    persistency". Nothing in the five tables records an intervention - no
    training, activity, coaching or contact history - so no causal claim is
    available. What *is* available is the composition of the books: which
    products, terms and customer segments the top quartile writes, against the
    rest. That is an association and a place to look, not a prescription.
    """
    af = ctx.frame("agent")
    pf = ctx.frame("policy")
    elig = af[af["sales_id"].isin(cohort.ids)] if cohort.ids else af
    if elig.empty or "sales_first_year_persistency" not in elig.columns:
        return pd.DataFrame(), {}

    cutoff = float(elig["sales_first_year_persistency"].quantile(0.75))
    high_ids = elig[elig["sales_first_year_persistency"] >= cutoff]["sales_id"]
    low_ids = elig[elig["sales_first_year_persistency"] < cutoff]["sales_id"]

    if focus_id is not None:
        subject = pf[pf["sales_id"].astype(str) == str(focus_id)]
        subject_label = f"Agent {focus_id}"
    else:
        subject = pf[pf["sales_id"].isin(low_ids)]
        subject_label = "Lower-persistency agents"
    peers = pf[pf["sales_id"].isin(high_ids)]

    if subject.empty or peers.empty:
        return pd.DataFrame(), {}

    rows: List[Dict[str, Any]] = []
    for dname in dimensions:
        dim = ac.DIMENSIONS.get(dname)
        col = dim.column if dim else dname
        if col not in pf.columns:
            continue
        a = subject[col].value_counts(normalize=True).mul(100)
        b = peers[col].value_counts(normalize=True).mul(100)
        merged = pd.DataFrame({"subject_pct": a, "peer_pct": b}).fillna(0.0)
        merged["gap_pts"] = merged["subject_pct"] - merged["peer_pct"]
        merged = merged.reindex(merged["gap_pts"].abs().sort_values(ascending=False).index)
        for value, r in merged.head(1).iterrows():
            rows.append({
                "What": dim.label if dim else dname,
                "Where they differ most": str(value)[:46],
                "Them": f"{r['subject_pct']:.0f}%",
                "Top quartile": f"{r['peer_pct']:.0f}%",
                "Gap": f"{r['gap_pts']:+.0f} pts",
                "_abs": abs(r["gap_pts"]),
            })

    if not rows:
        return pd.DataFrame(), {}

    out = (pd.DataFrame(rows).sort_values("_abs", ascending=False)
           .head(top_n).drop(columns=["_abs"]).reset_index(drop=True))
    meta = {
        "subject_label": subject_label,
        "subject_policies": len(subject),
        "peer_policies": len(peers),
        "peer_agents": int(high_ids.nunique()),
        "cutoff": cutoff,
    }
    return out, meta


# =====================================================================
# V4 - drill to source records
# =====================================================================

def agent_source_rows(ctx: ac.DataContext, sales_id: Any,
                      columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """The policy records behind one agent."""
    pf = ctx.frame("policy")
    if "sales_id" not in pf.columns:
        return pd.DataFrame()
    sub = pf[pf["sales_id"].astype(str) == str(sales_id)]
    cols = [c for c in (columns or ["app_id", "plan_name", "policy_status", "trad_ulip",
                                    "ape", "sum_assured", "branch_no", "state",
                                    "inforce_date", "premium_paying_term"])
            if c in sub.columns]
    return sub[cols].reset_index(drop=True) if cols else sub.reset_index(drop=True)


def render_v4_drill(ctx: ac.DataContext, sales_id: Any, key: str = "v4",
                    metrics_shown: Sequence[str] = ()) -> None:
    """The records behind a number, plus an export. This is the verification."""
    rows = agent_source_rows(ctx, sales_id)
    if rows.empty:
        st.info(f"No policy records found for agent {sales_id}.")
        return

    st.markdown(f"**Agent {sales_id} · {len(rows):,} policies**")

    status_col = "policy_status" if "policy_status" in rows.columns else None
    if status_col:
        counts = rows[status_col].value_counts()
        chips = " · ".join(f"{k}: {v}" for k, v in counts.items())
        st.caption(chips)

    st.dataframe(rows, hide_index=True, width="stretch",
                 column_config=_column_config(rows))
    st.download_button(
        "Download these rows (CSV)",
        data=rows.to_csv(index=False).encode("utf-8"),
        file_name=f"agent_{sales_id}_policies.csv",
        mime="text/csv",
        key=f"{key}_dl",
    )

    stated = [m for m in metrics_shown if m in STATED_AGENT_RATES]
    if stated:
        names = ", ".join(f"`{STATED_AGENT_RATES[m]}`" for m in stated)
        st.caption(
            f"These rows are the agent's book. {names} are not computed from them — "
            f"they are declared per agent in Sales_Details. Use these rows to verify "
            f"book size, product mix and status, not the rate itself.")


def _column_config(df: pd.DataFrame) -> Dict[str, Any]:
    """Render rupee and rate columns as money and percentages, not raw floats."""
    cfg: Dict[str, Any] = {}
    for col in df.columns:
        spec = ac.METRICS.get(col)
        unit = spec.unit if spec else None
        if col in ("ape", "sum_assured", "total_ape", "total_sum_assured",
                   "claim_amount", "amount_paid", "renewal_premium_amount") or unit == "₹":
            cfg[col] = st.column_config.NumberColumn(col, format="₹%,.0f")
        elif unit == "%" or col.endswith("_rate") or col.endswith("_pct"):
            cfg[col] = st.column_config.NumberColumn(col, format="%.2f%%")
        elif col.endswith("_reliable"):
            cfg[col] = st.column_config.CheckboxColumn(col)
    return cfg


# =====================================================================
# PROOF PANEL
# =====================================================================

def render_proof(resolved: Any, key: str = "proof") -> None:
    """'How I read your question' - the interpretation layer, never numbers."""
    if resolved is None:
        return
    with st.expander("How I read your question", expanded=bool(getattr(resolved, "ambiguity", None))):
        rows = resolved.proof_rows() if hasattr(resolved, "proof_rows") else []
        for label, value in rows:
            st.markdown(f"- **{label}** → {value}")
        if getattr(resolved, "interpretation", ""):
            st.caption(resolved.interpretation)
        if getattr(resolved, "source", "") == "llm":
            st.caption("Resolved by the local model, then checked against the metric "
                       "and dimension registries. The model names what to compute; "
                       "every figure comes from the analytics engine.")
        if getattr(resolved, "degraded", False):
            st.warning(resolved.degraded_reason or "Follow-up resolution was degraded.")


__all__ = [
    "MIN_BOOK", "AgentCohort", "build_agent_cohort",
    "agent_risk_ranking", "render_method1",
    "agent_quadrant_frame", "render_method2",
    "agent_source_rows", "render_v4_drill", "render_proof",
]
