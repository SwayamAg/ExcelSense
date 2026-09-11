"""
chat_dashboard.py
=================
The conversational surface over the analytics engine.

A turn is: question -> resolution -> (clarify | execute) -> render -> persist.

Resolution is the part the stateless engine cannot do on its own. Turn 1 is
resolved by rules because there is no prior state and a model call would only
add latency. Turn 2 onward goes to the local model, whose answer is validated
against the metric and dimension registries before it is trusted, and whose
self-reported confidence is damped by deterministic checks (see
`conversation._calibrate_confidence`) because small local models report 1.0
almost unconditionally.

When resolution lands below the confidence threshold, or a data-backed check
finds the question presumes something the data does not support, the turn
*asks* instead of answering. That is the behaviour worth having in a chat
surface: a follow-up is cheap to ask again, and a confidently wrong answer is
expensive.

Conversational turns are persisted to `data/history/chat_history.json` in the
same record shape the classic dashboard writes to `query_history.json` - same
field names, same table encoding - with the transcript fields (`turn_id`,
`parent_turn`, `turn_state`, `render_kind`) added on top. The two files are kept
apart because they have different lifecycles, not different formats: a
conversation is an ordered thread, a query log is a set of independent asks.
"""

from __future__ import annotations

import datetime
import html
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
for _sub in ["src", "src/analytics", "src/engine", "src/retrieval", "src/llm", "ui"]:
    _p = os.path.join(ROOT_DIR, _sub)
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import streamlit as st

import analytics_core as ac
import business_engine as be
import conversation as cv
import exec_views as ev

HISTORY_FILE = os.path.join(ROOT_DIR, "data", "history", "chat_history.json")
ARCHIVE_FILE = os.path.join(ROOT_DIR, "data", "history", "chat_archive.json")

# Map a rendered turn onto the classic dashboard's `route` vocabulary, so a
# record written here means the same thing as one written there.
ROUTE_BY_KIND = {
    "agent_risk": "business_analysis",
    "why": "business_analysis",
    "agent_improvement": "business_analysis",
    "agent_composition": "business_analysis",
    "entity_comparison": "business_analysis",
    "engine": "business_analysis",
    "clarification": "needs_clarification",
    "refusal": "insufficient_data",
    "greeting": "greeting_identity",
    "out_of_domain": "out_of_domain",
}

CSS = """
<style>
  .ct-turn-q {
    background:#0f1b2d; border:1px solid #1f6feb55; border-left:4px solid #58a6ff;
    border-radius:10px; padding:12px 18px; margin:14px 0 10px;
    color:#e6edf3; font-size:.98rem; font-weight:600;
  }
  .ct-headline {
    background:#161b22; border:1px solid #30363d; border-radius:10px;
    padding:16px 20px; margin-bottom:12px;
    color:#e6edf3; font-size:1rem; line-height:1.6;
  }
  .ct-clarify {
    background:#1d1804; border:1px solid #d2992255; border-left:4px solid #d29922;
    border-radius:10px; padding:16px 20px; margin-bottom:12px;
    color:#e3b341; font-size:.97rem; line-height:1.6;
  }
  .ct-refuse {
    background:#1d0d0d; border:1px solid #f8514955; border-left:4px solid #f85149;
    border-radius:10px; padding:16px 20px; margin-bottom:12px;
    color:#ff7b72; font-size:.97rem; line-height:1.6;
  }
  .ct-metric { color:#c9d1d9; font-size:.92rem; margin:2px 0; }
  .ct-metric b { color:#58a6ff; }
</style>
"""

# Questions this surface answers directly with Methods 1 and 2 rather than
# handing to the PS planner. Compound agent-risk superlatives are exactly the
# shape the planner handles worst: it picks one metric and ranks on it.
_AGENT_RISK_CUES = (
    "persistency", "claim rate", "early claim", "surrender rate",
    "salesperson", "sales person", "advisor", "agent",
)
_SUPERLATIVE_CUES = ("lowest", "highest", "worst", "best", "poorest", "top", "bottom")


# =====================================================================
# PERSISTENCE
# =====================================================================

def _load_history() -> List[Dict[str, Any]]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            print(f"[chat history load] {e}")
    return []


def _save_history(turns: List[Dict[str, Any]]) -> None:
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(turns[-200:], f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"[chat history save] {e}")


def _load_archives() -> List[Dict[str, Any]]:
    """Past conversations, each a list of turns in the shared record shape."""
    if os.path.exists(ARCHIVE_FILE):
        try:
            with open(ARCHIVE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            print(f"[archive load] {e}")
    return []


def _archive_current(turns: List[Dict[str, Any]]) -> None:
    """Set the open thread aside so it can be reopened later.

    Without this, starting a new conversation discarded the old one and there
    was nothing to click back into.
    """
    if not turns:
        return
    archives = _load_archives()
    first = turns[0].get("query") or turns[0].get("question", "")
    # don't stack duplicates of the same thread
    if archives and (archives[-1].get("turns") or [{}])[0].get("query") == first \
            and len(archives[-1].get("turns") or []) == len(turns):
        return
    archives.append({
        "saved_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "title": first[:120],
        "turns": turns,
    })
    try:
        os.makedirs(os.path.dirname(ARCHIVE_FILE), exist_ok=True)
    _save_archives(archives)


def _save_archives(archives: List[Dict[str, Any]]) -> None:
    """Persist conversation archive list to disk."""
    try:
        os.makedirs(os.path.dirname(ARCHIVE_FILE), exist_ok=True)
        with open(ARCHIVE_FILE, "w", encoding="utf-8") as f:
            json.dump(archives[-30:], f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"[archive save] {e}")


def _clear_all_chat_history() -> None:
    """Wipe active chat history and all archived conversations."""
    _save_history([])
    _save_archives([])
    for fpath in (HISTORY_FILE, ARCHIVE_FILE):
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except Exception:
                pass


# =====================================================================
# LOADERS
# =====================================================================

@st.cache_resource(show_spinner="Building fact tables and metric registry...")
def _load_ctx() -> ac.DataContext:
    return ac.DataContext.get()


@st.cache_resource(show_spinner="Loading the analytics engine...")
def _load_engine():
    try:
        import business_engine as be
        return be.get_engine(with_legacy=True)
    except Exception as e:
        print(f"[engine load] {e}")
        return None


# =====================================================================
# CLASSIFICATION
# =====================================================================

_AGENT_SUBJECT_CUES = ("sales person", "salesperson", "sales persons", "salespersons",
                       "sales people", "advisor", "advisors", "agent", "agents", "rep ",
                       "reps")
_IMPROVE_CUES = ("improve", "increase", "raise", "lift", "better", "fix", "reduce",
                 "boost", "uplift", "what distinguishes", "what separates")


def _is_agent_risk_question(question: str) -> bool:
    q = (question or "").lower()
    return (any(c in q for c in _AGENT_RISK_CUES)
            and any(c in q for c in _SUPERLATIVE_CUES))


def _is_agent_improvement_question(question: str) -> bool:
    """Asking how sales people improve is a question about people, not products.

    Without this, "how to improve persistency of other sales persons" had no
    superlative, fell through to the generic planner, and came back with a
    ranking of *plans* by lapse rate - an answer to a different question.
    """
    q = (question or "").lower()
    return (any(c in q for c in _AGENT_SUBJECT_CUES)
            and any(c in q for c in _IMPROVE_CUES))


# Wording that actually asks about the future. A refusal is a strong response,
# so it needs evidence in the question itself.
_FORECAST_CUES = ("will ", "next year", "next quarter", "next month", "predict",
                  "forecast", "projection", "going to", "expected to", "likely to",
                  "future", "in a year", "by next")

# "How do we improve X" is prescriptive, not predictive. The local model
# frequently labels it `forecast`, which would refuse an answerable question.
_PRESCRIPTIVE_CUES = ("how to improve", "how can we improve", "how do we improve",
                      "how should", "what should", "how to increase", "how to raise",
                      "how to fix", "how to reduce", "ways to", "what can be done")


def _is_forecast(resolved: cv.ResolvedFollowUp, question: str) -> bool:
    """True only when the question really asks what happens next.

    The model's `intent` alone is not enough. A 0.5B model labelled "how to
    improve persistency score of other sales persons" as forecast, which turned
    an answerable prescriptive question into a refusal. Model intent now needs
    corroboration from the wording, and explicit prescriptive phrasing wins.
    """
    q = (question or "").lower()
    if any(p in q for p in _PRESCRIPTIVE_CUES):
        return False
    lexical = any(p in q for p in _FORECAST_CUES)
    if lexical:
        return True
    # Model says forecast but nothing in the question does: don't refuse.
    return False


# =====================================================================
# TURN RENDERING
# =====================================================================

def _render_agent_risk_turn(ctx: ac.DataContext, question: str,
                            resolved: cv.ResolvedFollowUp,
                            turn_no: int) -> cv.TurnState:
    """Methods 1 and 2, plus V4 on the selected agent."""
    cohort = ev.build_agent_cohort(ctx)
    ranking = ev.agent_risk_ranking(ctx, cohort, top_k=6)

    if ranking.empty:
        st.warning("No agent met the ranking threshold.")
        return (cv.TurnState(question=question, cohort={"eligible": cohort.n_eligible}),
                "No agent met the ranking threshold.")

    top = ranking.iloc[0]
    top_id = str(top["sales_id"])
    pers = float(top["sales_persistency_benchmark"])
    claim = float(top["early_claim_rate"])
    book = ev.agent_source_rows(ctx, top_id)
    peer_pers = float(ev.agent_quadrant_frame(ctx, cohort)["sales_first_year_persistency"].median())

    headline = (
        f"No single agent is both the lowest on persistency and the highest on "
        f"claim rate. Ranked on the two together, <b>Agent {top_id}</b> is the "
        f"clearest case: {len(book):,} policies, persistency {pers:.1f}% against a "
        f"peer median of {peer_pers:.1f}%, early-claim rate {claim:.2f}%.")
    st.markdown(f'<div class="ct-headline">{headline}</div>', unsafe_allow_html=True)

    ev.render_method1(ranking, cohort, highlight=top_id, key=f"t{turn_no}m1")

    # The quadrant is supporting reasoning, not the answer, so it sits behind a
    # toggle like the score decomposition and the source rows.
    with st.expander("Persistency vs early-claim rate — all ranked agents", expanded=False):
        ev.render_method2(ev.agent_quadrant_frame(ctx, cohort), cohort,
                          highlight=top_id, key=f"t{turn_no}m2")

    with st.expander(f"Rows behind Agent {top_id}", expanded=False):
        ev.render_v4_drill(ctx, top_id, key=f"t{turn_no}v4",
                           metrics_shown=["sales_persistency_benchmark", "early_claim_rate"])

    return cv.TurnState(
        question=question,
        focus_entity=cv.FocusEntity("agent", top_id),
        focus_metrics=["sales_persistency_benchmark", "early_claim_rate"],
        active_filters={"sales_id": top_id},
        cohort={"ranked": cohort.n_eligible,
                "excluded_small_book": cohort.n_small_book,
                "excluded_no_rate": cohort.n_missing_rate},
        ops_run=["composite_score", "multi_metric_table"],
        result_ref=f"turn_{turn_no}_ranking",
    ), headline


def _render_clarification(ctx: ac.DataContext, question: str,
                          resolved: cv.ResolvedFollowUp, turn_no: int) -> cv.TurnState:
    """Ask rather than guess. Offer the defensible alternatives."""
    entity = resolved.entity
    if resolved.ambiguity:
        body = (f'"{question.strip()}" presumes something the records do not support. '
                f'{resolved.ambiguity}. A breakdown on that would rest on the largest '
                f'slice alone, below the {ev.MIN_BOOK} needed to be reliable.')
    else:
        body = (f'I am not confident enough about what "{question.strip()}" is asking '
                f'to answer it without checking. Pick the reading you meant.')
    st.markdown(f'<div class="ct-clarify">{body}</div>', unsafe_allow_html=True)

    if entity is not None and resolved.dimension:
        dim = ac.DIMENSIONS.get(resolved.dimension)
        rows = ev.agent_source_rows(ctx, entity.id)
        col = dim.column if dim else None
        if col and col in rows.columns:
            spread = rows[col].value_counts().head(8).rename_axis(
                dim.label if dim else col).reset_index(name="policies")
            st.markdown(f"**Their spread across {dim.label.lower() if dim else col}** — counts, not rates")
            st.dataframe(spread, hide_index=True, width="stretch")

    st.markdown("**What I can compare instead:**")
    c1, c2 = st.columns(2)
    with c1:
        st.button(f"vs all ranked agents", key=f"t{turn_no}_alt1", width="stretch")
    with c2:
        st.button("vs agents with a similar book size", key=f"t{turn_no}_alt2", width="stretch")

    return cv.TurnState(
        question=question,
        focus_entity=entity,
        focus_metrics=resolved.metric and [resolved.metric] or [],
        active_filters=resolved.filters,
        ops_run=[],
        result_ref=None,
    ), body


def _render_refusal(question: str, resolved: cv.ResolvedFollowUp,
                    entity: Optional[cv.FocusEntity], ctx: ac.DataContext,
                    turn_no: int) -> cv.TurnState:
    """Forward-looking questions get the scope answer, not a guess."""
    who = entity.label if entity else "these records"
    n = len(ev.agent_source_rows(ctx, entity.id)) if entity else 0
    extra = f" I can show which of their {n:,} policies have already lapsed, and how their renewal dates fall — but not a prediction." if n else ""
    body = (f"That needs forward-looking data this dataset does not hold. Nothing "
            f"here records what {who} will do next — only what has already "
            f"happened.{extra}")
    st.markdown(f'<div class="ct-refuse">{body}</div>', unsafe_allow_html=True)
    return cv.TurnState(question=question, focus_entity=entity,
                        active_filters=resolved.filters, ops_run=[]), body


def _render_entity_comparison(ctx: ac.DataContext, question: str,
                              resolved: cv.ResolvedFollowUp,
                              turn_no: int) -> cv.TurnState:
    """One agent against the ranked peer cohort, with denominators shown (V1)."""
    entity = resolved.entity
    cohort = ev.build_agent_cohort(ctx)
    q = ev.agent_quadrant_frame(ctx, cohort)
    rows = ev.agent_source_rows(ctx, entity.id)

    af = ctx.frame("agent")
    me = af[af["sales_id"].astype(str) == str(entity.id)]
    if me.empty or q.empty:
        st.info(f"No comparable record for {entity.label}.")
        return (cv.TurnState(question=question, focus_entity=entity),
                f"No comparable record for {entity.label}.")
    me = me.iloc[0]

    # Every agent-level column Sales_Details actually carries. A question asking
    # for "risk flags" was previously answered with persistency, early claim and
    # book size only - the five compliance columns were never shown.
    # `higher_is_worse` decides which direction gets flagged, not just coloured.
    peers = af[af["sales_id"].isin(cohort.ids)] if cohort.ids else af
    pairs = [
        ("Persistency",        "sales_first_year_persistency", "%",     False),
        ("Early claim rate",   "sales_early_claim_rate",       "%",     True),
        ("Surrender rate",     "sales_surrender_rate",         "%",     True),
        ("RCU rejections",     "rcu_rejection_rate",           "%",     True),
        ("PIVC mismatch",      "pivc_mismatch_rate",           "%",     True),
        ("PIVC concern",       "pivc_concern_rate",            "%",     True),
        ("Free-look cancels",  "free_look_rate",               "%",     True),
        ("Sales complaints",   "complaint_rate",               "%",     True),
        ("Book size",          "policies_sold",                "count", False),
    ]

    lines, flags = [], []
    for label, col, unit, worse_is_high in pairs:
        if col not in af.columns or pd.isna(me.get(col)):
            continue
        mine = float(me[col])
        med = float(peers[col].median())
        diff = mine - med
        adverse = (diff > 0) if worse_is_high else (diff < 0)
        arrow = "▲" if diff > 0 else ("▼" if diff < 0 else "=")
        if unit == "count":
            shown, peer_shown, gap = f"{mine:,.0f}", f"{med:,.0f}", f"{arrow} {abs(diff):,.0f}"
        else:
            shown, peer_shown, gap = f"{mine:.2f}%", f"{med:.2f}%", f"{arrow} {abs(diff):.2f} pts"
        if col == "sales_first_year_persistency":
            shown += f" ({len(rows):,} policies)"
        lines.append({"Metric": label, "This agent": shown,
                      "Peer median": peer_shown, "Gap": gap,
                      "Flag": "⚠" if adverse and abs(diff) > 0.01 else ""})
        if adverse and abs(diff) > 0.01:
            flags.append(f"{label} {shown} vs {peer_shown} peer median")

    status = str(me.get("sales_type", "")) or "unknown"
    if flags:
        headline = (f"<b>{entity.label}</b> ({status}, {len(rows):,} policies) is adverse "
                    f"to the peer median on <b>{len(flags)} of {len(lines)}</b> measures — "
                    f"worst: {flags[0]}.")
    else:
        headline = (f"<b>{entity.label}</b> ({status}, {len(rows):,} policies) is at or "
                    f"better than the peer median on every measured indicator.")
    st.markdown(f'<div class="ct-headline">{headline}</div>', unsafe_allow_html=True)

    table = pd.DataFrame(lines)
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(cohort.exclusion_note + ". ⚠ marks a reading worse than the peer median.")
    st.download_button("Download this scorecard (CSV)",
                       data=table.to_csv(index=False).encode("utf-8"),
                       file_name=f"agent_{entity.id}_scorecard.csv", mime="text/csv",
                       key=f"t{turn_no}_sc_dl")

    with st.expander("Persistency vs early-claim rate — all ranked agents", expanded=False):
        ev.render_method2(q, cohort, highlight=str(entity.id), key=f"t{turn_no}m2")

    with st.expander(f"Rows behind {entity.label}", expanded=False):
        ev.render_v4_drill(ctx, entity.id, key=f"t{turn_no}v4",
                           metrics_shown=["sales_persistency_benchmark", "early_claim_rate"])

    return cv.TurnState(
        question=question, focus_entity=entity,
        focus_metrics=["sales_persistency_benchmark", "early_claim_rate"],
        active_filters=resolved.filters,
        cohort={"ranked": cohort.n_eligible},
        ops_run=["compare_to_baseline"],
        result_ref=f"turn_{turn_no}_comparison",
    ), headline


# Words that mean the question already says what to measure.
_OWN_MEASURE_CUES = ("how many", "how much", "most", "least", "count", "number of",
                     "volume", "share", "mix", "breakdown", "split", "distribution",
                     "total", "average", "biggest", "largest", "smallest")

# Asking what an agent's own book looks like.
_COMPOSITION_CUES = ("sell", "sells", "selling", "sold", "write", "writes", "book",
                     "portfolio", "mix", "breakdown", "split", "distribution",
                     "which plans", "what plans", "which products", "what products",
                     "which states", "what states")


def _is_agent_composition_question(question: str, resolved: cv.ResolvedFollowUp) -> bool:
    """"Which plans do they sell most" is about one agent's own book.

    The generic planner has no way to scope to a single SALES_ID, so it answered
    portfolio-wide. When an agent is in focus and the question asks what they
    sell, the answer comes from their own policy rows.
    """
    if resolved.entity is None or resolved.entity.type != "agent":
        return False
    return any(c in (question or "").lower() for c in _COMPOSITION_CUES)


def _render_agent_composition_turn(ctx: ac.DataContext, question: str,
                                   resolved: cv.ResolvedFollowUp, turn_no: int):
    """Break one agent's own book down by the dimension the question names."""
    entity = resolved.entity
    rows = ev.agent_source_rows(ctx, entity.id)
    if rows.empty:
        msg = f"No policy records for {entity.label}."
        st.markdown(f'<div class="ct-refuse">{msg}</div>', unsafe_allow_html=True)
        return cv.TurnState(question=question, focus_entity=entity), msg, [], []

    q = (question or "").lower()
    if "state" in q or "region" in q or "geograph" in q:
        col, label = "state", "State"
    elif "status" in q:
        col, label = "policy_status", "Policy status"
    elif "trad" in q or "ulip" in q or "categor" in q:
        col, label = "trad_ulip", "Trad vs ULIP"
    else:
        col, label = "plan_name", "Plan"
    if col not in rows.columns:
        col, label = "plan_name", "Plan"

    counts = rows[col].value_counts()
    ape = rows.groupby(col)["ape"].sum() if "ape" in rows.columns else None
    table = pd.DataFrame({
        label: counts.index,
        "Policies": counts.values,
        "Share": [f"{v / len(rows) * 100:.0f}%" for v in counts.values],
    })
    if ape is not None:
        table["APE"] = [f"Rs {ape.get(k, 0):,.0f}" for k in counts.index]

    top_name, top_n = str(counts.index[0]), int(counts.iloc[0])
    headline = (f"<b>{entity.label}</b> sells mostly <b>{top_name}</b> — "
                f"{top_n} of {len(rows):,} policies ({top_n / len(rows) * 100:.0f}%), "
                f"across {counts.size} {label.lower()} value(s).")
    st.markdown(f'<div class="ct-headline">{headline}</div>', unsafe_allow_html=True)
    st.dataframe(table.head(10), hide_index=True, width="stretch")
    st.download_button("Download (CSV)", data=table.to_csv(index=False).encode("utf-8"),
                       file_name=f"agent_{entity.id}_{col}.csv", mime="text/csv",
                       key=f"t{turn_no}_comp_dl")

    with st.expander(f"Rows behind {entity.label}", expanded=False):
        ev.render_v4_drill(ctx, entity.id, key=f"t{turn_no}v4")

    evidence = [f"{r[label]}: {r['Policies']} policies ({r['Share']})"
                for _, r in table.head(3).iterrows()]
    state = cv.TurnState(
        question=question, focus_entity=entity,
        focus_metrics=list(resolved.metric and [resolved.metric] or []),
        active_filters=resolved.filters, ops_run=["metric_by_dimension"],
        result_ref=f"turn_{turn_no}_composition")
    return state, headline, evidence, [[f"{label} mix", table.to_dict(orient="records")]]


# A "why is A higher than B" question names a metric and a segmenting dimension.
# Matching them explicitly beats letting the retriever guess, which produced
# "Diagnostic decomposition of the focus group" led by a 23-policy outlier.
_WHY_CUES = ("why", "what is driving", "what drives", "what explains",
             "reason for", "root cause", "how come", "what causes")

_METRIC_WORDS = [
    ("lapse", "lapse_rate"), ("lapsing", "lapse_rate"),
    ("persistency", "persistency_rate"), ("persisting", "persistency_rate"),
    ("surrender", "surrender_rate"), ("surrendering", "surrender_rate"),
    ("claim settlement", "claim_settlement_ratio"),
    ("early claim", "early_claim_rate"),
    ("renewal", "renewal_pending_rate"),
]

_DIMENSION_WORDS = [
    ("younger", "age_band"), ("older", "age_band"), ("age", "age_band"),
    ("income", "income_band"), ("earning", "income_band"),
    ("occupation", "occupation"), ("education", "education"),
    ("state", "state"), ("region", "state"), ("city", "city"),
    ("plan", "plan_name"), ("product", "plan_name"),
    ("ulip", "trad_ulip"), ("traditional", "trad_ulip"),
    ("payment mode", "payment_mode"), ("paying term", "ppt_band"),
    ("gender", "gender"), ("male", "gender"), ("female", "gender"),
    ("bucket", "persistency_bucket"),
]


def _match_why_question(question: str,
                        resolved: cv.ResolvedFollowUp,
                        prev: Optional[cv.TurnState]) -> Optional[Tuple[str, str]]:
    """(metric, dimension) for a why-question, or None."""
    q = (question or "").lower()
    # A follow-up to a why-turn continues it: "what about by income?" carries
    # the question word implicitly, so requiring "why" again would drop it to
    # the generic planner and lose the gap framing.
    continues_why = bool(prev is not None and "explain_gap" in (prev.ops_run or []))
    if not any(c in q for c in _WHY_CUES) and not continues_why:
        return None
    if continues_why and not any(c in q for c in _WHY_CUES):
        # only continue when the follow-up actually names a new cut
        if not any(w in q for w, _ in _DIMENSION_WORDS):
            return None

    metric = next((m for w, m in _METRIC_WORDS if w in q), None)
    if metric is None and resolved.metric in ac.METRICS:
        metric = resolved.metric
    if metric is None and prev is not None:
        metric = next((m for m in (prev.focus_metrics or []) if m in ac.METRICS), None)
    if metric is None:
        return None

    dimension = next((d for w, d in _DIMENSION_WORDS if w in q), None)
    if dimension is None and resolved.dimension in ac.DIMENSIONS:
        dimension = resolved.dimension
    if dimension is None:
        return None
    return metric, dimension


def _render_why_turn(ctx: ac.DataContext, question: str, metric: str, dimension: str,
                     resolved: cv.ResolvedFollowUp, turn_no: int):
    """State the gap, name the confounder, say whether it survives."""
    try:
        r = ac.explain_gap(ctx, metric, dimension)
    except Exception as e:
        msg = str(e)
        st.markdown(f'<div class="ct-refuse">{msg}</div>', unsafe_allow_html=True)
        return cv.TurnState(question=question), msg, [], []

    unit = "%" if r["unit"] == "%" else ""
    # Band labels such as "<30" and "50L+" are rendered inside a styled div, so
    # an unescaped "<" is parsed as the start of a tag and the label vanishes.
    hi_g, lo_g = html.escape(r["high_group"]), html.escape(r["low_group"])
    mult = f" ({r['ratio']:.1f}×)" if r.get("ratio") else ""
    lead = "Yes" if r.get("material") else "Barely"
    tail = ("." if r.get("material") else
            " — too small a spread to act on, and within normal variation.")
    headline = (f"{lead} — <b>{hi_g}</b> at {r['high_value']:.1f}{unit} against "
                f"<b>{lo_g}</b> at {r['low_value']:.1f}{unit}: a "
                f"<b>{r['gap_pts']:.1f} point</b> gap{mult}{tail}")
    st.markdown(f'<div class="ct-headline">{headline}</div>', unsafe_allow_html=True)

    evidence: List[str] = []
    conf = r["confounders"][0] if r["confounders"] else None
    if conf:
        evidence.append(
            f"**{conf['label']}** differs most between them: *{html.escape(str(conf['level']))}* "
            f"is {conf['high_group_pct']:.0f}% of {hi_g} against "
            f"{conf['low_group_pct']:.0f}% of {lo_g}.")
    if r["verdict"]:
        evidence.append(html.escape(r["verdict"]))
    evidence.append(
        f"Measured over {r['high_n']:,} and {r['low_n']:,} records; groups holding "
        f"fewer than {r.get('anchor_floor', ac.DEFAULT_MIN_N):,} are not used as "
        f"the comparison ends.")
    for line in evidence:
        st.markdown(f'<div class="ct-metric">• {_md_inline(line)}</div>',
                    unsafe_allow_html=True)

    groups = pd.DataFrame(r["groups"]).rename(
        columns={dimension: r["dimension_label"], "value": r["metric_label"],
                 "sample_size": "Records"})
    groups[r["metric_label"]] = groups[r["metric_label"]].map(lambda v: f"{v:.1f}{unit}")

    with st.expander(f"{r['metric_label']} by {r['dimension_label'].lower()}", expanded=False):
        st.dataframe(groups, hide_index=True, width="stretch")
        st.download_button("Download (CSV)", data=groups.to_csv(index=False).encode("utf-8"),
                           file_name=f"{metric}_by_{dimension}.csv", mime="text/csv",
                           key=f"t{turn_no}_why_dl")

    if r["stratified"]:
        meta = r["stratified_meta"]
        with st.expander(f"Gap within each {meta.get('held_dimension','group').lower()}",
                         expanded=False):
            st.dataframe(pd.DataFrame(r["stratified"]), hide_index=True, width="stretch")
            st.caption("Each row recomputes the gap inside one group. A gap that "
                       "survives here is not explained by composition alone.")

    st.caption("Association, not cause — the dataset records no interventions.")

    state = cv.TurnState(
        question=question, focus_metrics=[metric],
        active_filters=resolved.filters, ops_run=["explain_gap"],
        result_ref=f"turn_{turn_no}_why")
    return state, headline, evidence, [
        [f"{r['metric_label']} by {r['dimension_label']}", groups.to_dict(orient="records")]]


def _render_agent_improvement_turn(ctx: ac.DataContext, question: str,
                                   resolved: cv.ResolvedFollowUp, turn_no: int):
    """What separates high-persistency agents from the rest."""
    cohort = ev.build_agent_cohort(ctx)
    focus = resolved.entity.id if (resolved.entity and resolved.entity.type == "agent") else None
    table, meta = ev.agent_book_contrast(ctx, cohort, focus_id=focus)

    if table.empty:
        msg = "Not enough comparable agents to contrast."
        st.markdown(f'<div class="ct-refuse">{msg}</div>', unsafe_allow_html=True)
        return cv.TurnState(question=question), msg, [], []

    top = table.iloc[0]
    headline = (f"{meta['subject_label']} differ from the top quartile most on "
                f"<b>{top['What']}</b>: {top['Where they differ most']} is "
                f"{top['Them']} of their book vs {top['Top quartile']} "
                f"({top['Gap']}).")
    st.markdown(f'<div class="ct-headline">{headline}</div>', unsafe_allow_html=True)

    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(
        f"Top quartile = {meta['peer_agents']:,} agents above "
        f"{meta['cutoff']:.0f}% first-year persistency "
        f"({meta['peer_policies']:,} policies) vs {meta['subject_policies']:,} policies. "
        f"Association only — nothing in these tables records training, activity or "
        f"coaching, so this is where to look, not proof of cause.")

    evidence = [f"{r['What']}: {r['Where they differ most']} — "
                f"{r['Them']} vs {r['Top quartile']} ({r['Gap']})"
                for _, r in table.iterrows()]

    state = cv.TurnState(
        question=question,
        focus_entity=resolved.entity,
        focus_metrics=["sales_persistency_benchmark"],
        active_filters=resolved.filters,
        cohort={"ranked": cohort.n_eligible},
        ops_run=["segment_profile", "compare_to_baseline"],
        result_ref=f"turn_{turn_no}_contrast",
    )
    return state, headline, evidence, [["Book composition gap", table.to_dict(orient="records")]]


def _contextualise(question: str, prev: Optional[cv.TurnState],
                   resolved: cv.ResolvedFollowUp) -> str:
    """Expand a bare follow-up into a question the stateless engine can answer.

    `engine.ask()` takes one string and has no memory, so handing it "why is
    that happening?" makes it retrieve against a question with no subject - and
    it will happily answer about a different metric than the turn before. The
    previous turn's metrics and focus are spliced back in, and the rewritten
    question is shown in the proof panel so the user can see what was asked.
    """
    q = (question or "").strip()
    if prev is None:
        return q

    # Only rewrite when the question cannot stand on its own: short, and either
    # pointing back with a pronoun or opening with a bare verb.
    standalone = len(q.split()) > 9 and not cv.reads_as_continuation(q)
    if standalone:
        return q

    bits: List[str] = []

    # A question that names its own measure ("which plans do they sell most")
    # must not have the previous turn's metric spliced in - that is how a
    # volume question came back answered with persistency rate.
    carries_own_measure = any(w in q.lower() for w in _OWN_MEASURE_CUES)
    if not carries_own_measure:
        metrics = resolved.metric and [resolved.metric] or list(prev.focus_metrics or [])
        labels = []
        for m in metrics[:2]:
            spec = ac.METRICS.get(m)
            if spec:
                labels.append(spec.label.lower())
        if labels:
            bits.append(" and ".join(labels))

    if resolved.entity is not None:
        bits.append(f"for {resolved.entity.label}")
    elif prev.active_filters:
        for key, val in list(prev.active_filters.items())[:1]:
            if isinstance(val, str) and not val.isdigit():
                bits.append(f"for {val}")

    if not bits:
        return q
    return f"{q.rstrip('?').strip()} — {' '.join(bits)}"


def _render_engine_turn(question: str, turn_no: int, asked: Optional[str] = None):
    """Anything outside the agent-risk path goes to the existing planner.

    Renders the same blocks the classic dashboard does - evidence, analysis,
    business implication, recommendation, caveats, data gaps - so a question
    answered here is not a thinner answer than the same question asked there.
    """
    engine = _load_engine()
    if engine is None:
        msg = "The analytics engine is unavailable for this question."
        st.warning(msg)
        return cv.TurnState(question=question), msg, [], []

    asked = asked or question
    if asked != question:
        st.caption(f"Read as: *{asked}*")

    with st.spinner("Planning and executing..."):
        resp = engine.ask(asked, llm=False)

    ans = resp.answer
    if ans is None:
        msg = resp.message or "No answer produced."
        st.markdown(f'<div class="ct-refuse">{msg}</div>', unsafe_allow_html=True)
        return cv.TurnState(question=question), msg, [], []

    st.markdown(f'<div class="ct-headline">{_md_inline(ans.headline)}</div>',
                unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="medium")
    with left:
        _bullet_block("Evidence", ans.evidence)
        _bullet_block("Business implication", ans.implication)
    with right:
        _bullet_block("Analysis", ans.analysis)
        _bullet_block("Recommendation", ans.recommendation)

    if ans.caveats:
        with st.expander("Analytical caveats", expanded=False):
            for c in ans.caveats:
                st.markdown(f"- {c}")
    if ans.unavailable:
        with st.expander("Not answerable from this dataset", expanded=False):
            for u in ans.unavailable:
                st.markdown(f"- {u}")

    tables_payload = []
    for title, df in (ans.tables or [])[:3]:
        with st.expander(f"{title} ({len(df)} rows)", expanded=False):
            st.dataframe(df, hide_index=True, width="stretch")
        if isinstance(df, pd.DataFrame):
            tables_payload.append([title, df.head(50).to_dict(orient="records")])

    metrics_used = []
    for sr in (resp.step_results or []):
        if not getattr(sr, "ok", False):
            continue
        params = getattr(sr.step, "params", {}) or {}
        for key in ("metric", "metric_a", "metric_b", "value_metric"):
            m = params.get(key)
            if isinstance(m, str) and m in ac.METRICS and m not in metrics_used:
                metrics_used.append(m)
        for comp in params.get("components") or []:
            m = comp.get("metric") if isinstance(comp, dict) else None
            if isinstance(m, str) and m in ac.METRICS and m not in metrics_used:
                metrics_used.append(m)

    state = cv.TurnState(
        question=question,
        focus_metrics=metrics_used[:4],
        active_filters=dict(getattr(resp, "resolved_entities", {}) or {}),
        ops_run=[s.step.op for s in (resp.step_results or []) if getattr(s, "ok", False)],
        result_ref=f"turn_{turn_no}_engine",
    )
    return state, ans.headline, list(ans.evidence or []), tables_payload


def _md_inline(text: str) -> str:
    """Convert the composer's markdown to HTML.

    The bullets are emitted inside a styled <div>, where Streamlit does not run
    the markdown parser - so `**bold**` was reaching the screen as literal
    asterisks.
    """
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\g<1></b>", str(text))
    out = re.sub(r"`(.+?)`", r"<code>\g<1></code>", out)
    # Single-asterisk emphasis runs after the double form, so `**bold**` is
    # already consumed and cannot be split into two stray italics.
    out = re.sub(r"(?<!\*)\*(?!\s)([^*]+?)(?<!\s)\*(?!\*)", r"<i>\g<1></i>", out)
    return out


def _bullet_block(title: str, items: Optional[List[str]], limit: int = 3) -> None:
    if not items:
        return
    st.markdown(f"**{title}**")
    for item in items[:limit]:
        st.markdown(f'<div class="ct-metric">• {_md_inline(item)}</div>',
                    unsafe_allow_html=True)
    if len(items) > limit:
        with st.expander(f"{len(items) - limit} more"):
            for item in items[limit:]:
                st.markdown(f'<div class="ct-metric">• {_md_inline(item)}</div>',
                            unsafe_allow_html=True)


# =====================================================================
# REPLAY
# =====================================================================

def _replay_turn(ctx: ac.DataContext, entry: Dict[str, Any]) -> None:
    """Re-render one recorded turn.

    Charts and tables are recomputed from the stored focus rather than stored as
    data: the fact frames are cached, the computation is deterministic, and it
    keeps the history file small. Narrative answers from the planner are replayed
    from the text that was recorded, since re-asking the engine would be slow and
    could drift.
    """
    turn_no = entry.get("turn_id", 0)
    question = entry.get("query") or entry.get("question", "")
    st.markdown(f'<div class="ct-turn-q">{question}</div>', unsafe_allow_html=True)

    kind = entry.get("render_kind")
    params = entry.get("render_params") or {}
    key = f"replay{turn_no}"

    if entry.get("headline"):
        css = {"clarification": "ct-clarify", "refusal": "ct-refuse"}.get(kind, "ct-headline")
        st.markdown(f'<div class="{css}">{_md_inline(entry["headline"])}</div>',
                    unsafe_allow_html=True)

    for line in (entry.get("evidence") or [])[:3]:
        st.markdown(f'<div class="ct-metric">• {_md_inline(line)}</div>',
                    unsafe_allow_html=True)

    focus_id = params.get("focus_id")

    if kind == "agent_risk":
        cohort = ev.build_agent_cohort(ctx)
        ranking = ev.agent_risk_ranking(ctx, cohort, top_k=6)
        if not ranking.empty:
            ev.render_method1(ranking, cohort, highlight=focus_id, key=f"{key}m1")
            with st.expander("Persistency vs early-claim rate — all ranked agents",
                             expanded=False):
                ev.render_method2(ev.agent_quadrant_frame(ctx, cohort), cohort,
                                  highlight=focus_id, key=f"{key}m2")
        if focus_id:
            with st.expander(f"Rows behind Agent {focus_id}", expanded=False):
                ev.render_v4_drill(ctx, focus_id, key=f"{key}v4",
                                   metrics_shown=["sales_persistency_benchmark",
                                                  "early_claim_rate"])

    elif kind == "entity_comparison" and focus_id:
        cohort = ev.build_agent_cohort(ctx)
        with st.expander("Persistency vs early-claim rate — all ranked agents",
                         expanded=False):
            ev.render_method2(ev.agent_quadrant_frame(ctx, cohort), cohort,
                              highlight=focus_id, key=f"{key}m2")
        with st.expander(f"Rows behind Agent {focus_id}", expanded=False):
            ev.render_v4_drill(ctx, focus_id, key=f"{key}v4",
                               metrics_shown=["sales_persistency_benchmark",
                                              "early_claim_rate"])

    for title, rows in (entry.get("tables") or [])[:2]:
        if rows:
            with st.expander(f"{title} ({len(rows)} rows)", expanded=False):
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.divider()


# =====================================================================
# MAIN
# =====================================================================

def render() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    ctx = _load_ctx()

    if "chat_conv" not in st.session_state:
        st.session_state["chat_conv"] = cv.Conversation.from_list(
            [h.get("turn_state", {}) for h in _load_history() if h.get("turn_state")])
    if "chat_log" not in st.session_state:
        st.session_state["chat_log"] = _load_history()

    conv: cv.Conversation = st.session_state["chat_conv"]

    # ---- sidebar
    with st.sidebar:
        st.markdown("### Conversation")
        try:
            import ollama_client as oc
            online = oc.is_ollama_available()
            model = oc.get_default_model() if online else None
        except Exception:
            online, model = False, None

        use_llm_resolver = st.checkbox(
            "Use local model to resolve follow-ups", value=online, disabled=not online,
            help="The model only decides WHAT to compute. It never sees raw data and "
                 "never produces a number.")
        if online:
            st.caption(f"Local model: `{model}`")
        else:
            st.caption("Ollama offline — follow-ups resolve by carry-forward rules.")

        st.divider()
        st.caption(f"{len(conv)} turns in this conversation")
        c_new, c_clr = st.columns([1, 1])
        with c_new:
            if st.button("➕ New Chat", width="stretch", help="Archive current chat and start a fresh session"):
                _archive_current(st.session_state["chat_log"])
                conv.reset()
                st.session_state["chat_log"] = []
                _save_history([])
                st.session_state.pop("chat_last_processed", None)
                st.rerun()
        with c_clr:
            if st.button("🗑️ Clear Chat", width="stretch", help="Delete current conversation without saving"):
                conv.reset()
                st.session_state["chat_log"] = []
                _save_history([])
                st.session_state.pop("chat_last_processed", None)
                st.rerun()

        # ---- past conversations, resumable
        #
        # History survived a restart before this, but only as one ever-growing
        # thread. Archiving on "new conversation" and listing the archives makes
        # a past thread something you can reopen and keep asking from: loading
        # one restores both the transcript and the turn state a follow-up needs.
        archives = _load_archives()
        if archives:
            st.divider()
            c_arc_title, c_arc_del = st.columns([2, 1])
            with c_arc_title:
                st.markdown("### Past conversations")
            with c_arc_del:
                if st.button("Clear All", key="clr_all_arc", help="Delete all archived conversations"):
                    _save_archives([])
                    st.rerun()

            for i, arc in enumerate(reversed(archives[-10:])):
                turns = arc.get("turns") or []
                if not turns:
                    continue
                first = turns[0].get("query") or turns[0].get("question", "")
                label = (first[:32] + "…") if len(first) > 32 else first
                if st.button(f"💬 {label}", key=f"arc_{i}", width="stretch",
                             help=f"{arc.get('saved_at','')} · {len(turns)} turns — click to reopen"):
                    _archive_current(st.session_state["chat_log"])
                    st.session_state["chat_log"] = turns
                    st.session_state["chat_conv"] = cv.Conversation.from_list(
                        [t.get("turn_state", {}) for t in turns if t.get("turn_state")])
                    st.session_state.pop("chat_last_processed", None)
                    _save_history(turns)
                    st.rerun()

        if st.session_state.get("chat_log") or archives:
            st.divider()
            if st.button("⚠️ Wipe All Chat History", key="wipe_all_history",
                         help="Delete active chat log and all archives permanently", width="stretch"):
                conv.reset()
                st.session_state["chat_log"] = []
                st.session_state.pop("chat_last_processed", None)
                _clear_all_chat_history()
                st.rerun()

        st.divider()
    st.markdown("## Insurance Analytics — Conversation")
    st.caption("Every figure is computed by the analytics engine. Ask a follow-up and "
               "the system will tell you how it read your question.")

    # ---- replay every prior turn in full
    #
    # Streamlit re-runs the whole script on each interaction, so anything not
    # re-emitted disappears. This loop previously printed a placeholder caption,
    # which made the conversation look compressed. Each turn now records what it
    # rendered and is replayed from that, so the transcript reads as
    # question -> answer -> follow-up -> answer.
    for entry in st.session_state["chat_log"]:
        _replay_turn(ctx, entry)

    # ---- input
    pending = st.session_state.pop("chat_pending", None)
    question = st.chat_input("Ask a question, or follow up on the answer above")
    question = pending or question
    if not question:
        if not conv.turns:
            st.info("Start with a question — try one from the sidebar.")
        return

    # A rerun (a button press, a widget change, a test harness replay) can
    # re-deliver the last question. Answering it twice appends a duplicate turn
    # and shifts what the next follow-up inherits, so mark the question as
    # claimed before any work happens rather than inferring it from the tail.
    claimed = st.session_state.get("chat_last_processed")
    if claimed is not None and claimed.strip() == question.strip():
        st.caption("(already answered above — ask something new to continue)")
        return
    st.session_state["chat_last_processed"] = question

    st.markdown(f'<div class="ct-turn-q">{question}</div>', unsafe_allow_html=True)

    # ---- resolve
    turn_no = len(conv) + 1
    resolver = cv.FollowUpResolver(ctx=ctx, use_llm=use_llm_resolver, timeout=45)
    with st.spinner("Reading your question..."):
        resolved = resolver.resolve(question, conv)

    if turn_no > 1:
        ev.render_proof(resolved, key=f"t{turn_no}proof")

    # ---- route
    headline, evidence, tables = "", [], []
    if be.is_greeting_or_identity(question):
        kind = "greeting"
        state, headline, evidence, tables = _render_engine_turn(question, turn_no, question)
    elif be.check_domain(question)[0] and not (turn_no > 1 and (resolved.entity is not None or resolved.inherited_from is not None)):
        kind = "out_of_domain"
        state, headline, evidence, tables = _render_engine_turn(question, turn_no, question)
    elif _is_forecast(resolved, question):
        kind = "refusal"
        state, headline = _render_refusal(question, resolved, resolved.entity, ctx, turn_no)
    elif turn_no > 1 and resolved.needs_clarification and resolved.entity is not None:
        kind = "clarification"
        state, headline = _render_clarification(ctx, question, resolved, turn_no)
    elif _match_why_question(question, resolved, conv.last()):
        kind = "why"
        metric, dimension = _match_why_question(question, resolved, conv.last())
        state, headline, evidence, tables = _render_why_turn(
            ctx, question, metric, dimension, resolved, turn_no)
    elif _is_agent_improvement_question(question):
        kind = "agent_improvement"
        state, headline, evidence, tables = _render_agent_improvement_turn(
            ctx, question, resolved, turn_no)
    elif _is_agent_risk_question(question):
        kind = "agent_risk"
        state, headline = _render_agent_risk_turn(ctx, question, resolved, turn_no)
    elif _is_agent_composition_question(question, resolved):
        kind = "agent_composition"
        state, headline, evidence, tables = _render_agent_composition_turn(
            ctx, question, resolved, turn_no)
    elif resolved.entity is not None and resolved.entity.type == "agent":
        kind = "entity_comparison"
        state, headline = _render_entity_comparison(ctx, question, resolved, turn_no)
    else:
        kind = "engine"
        asked = _contextualise(question, conv.last(), resolved)
        state, headline, evidence, tables = _render_engine_turn(question, turn_no, asked)

    # ---- persist
    #
    # The entry is a superset of the classic dashboard's history record: the
    # same field names and shapes (`query`, `route`, `headline`, `evidence`,
    # `analysis`, `implication`, `recommendation`, `caveats`, `unavailable`,
    # `llm_narrative`, `legacy_output`, `tables`), plus the conversational
    # fields a transcript needs. Either reader can open either file.
    conv.append(state)
    st.session_state["chat_log"].append({
        # --- classic dashboard schema
        "query": question,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "route": ROUTE_BY_KIND.get(kind, "business_analysis"),
        "headline": headline,
        "evidence": evidence,
        "analysis": [],
        "implication": [],
        "recommendation": [],
        "caveats": [],
        "unavailable": [],
        "llm_narrative": None,
        "legacy_output": None,
        "tables": tables,
        # --- conversational extensions
        "question": question,          # kept as an alias for readability
        "turn_id": state.turn_id,
        "parent_turn": state.turn_id - 1 if state.turn_id > 1 else None,
        "resolved_from": (f"inherited: {resolved.entity.label} from {resolved.inherited_from}"
                          if resolved.inherited_from else
                          (f"named: {resolved.entity.label}" if resolved.entity else "no entity")),
        "resolution_source": resolved.source,
        "confidence": resolved.confidence,
        "ambiguity": resolved.ambiguity,
        "degraded": resolved.degraded,
        "ops_run": state.ops_run,
        "turn_state": state.to_dict(),
        "render_kind": kind,
        "render_params": {"focus_id": state.focus_entity.id if state.focus_entity else None},
    })
    _save_history(st.session_state["chat_log"])
    st.divider()


if __name__ == "__main__":
    try:
        st.set_page_config(page_title="Insurance Analytics — Conversation",
                           page_icon="💬", layout="wide",
                           initial_sidebar_state="expanded")
    except Exception:
        pass
    render()
