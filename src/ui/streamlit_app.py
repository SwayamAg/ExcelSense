"""Streamlit conversational interface for Excel_sense_v2.

A thin rendering layer over `Chatbot.ask`. It computes nothing: every figure it
shows was produced by the deterministic engine and arrives inside the evidence
package. That constraint is what makes the interface safe to iterate on - there
is no way for a formatting change to alter a number, because no number is
calculated here.

What the interface is responsible for is honesty of presentation:

  outcomes      an answer, a refusal, a request for clarification and an
                insufficient-evidence result look different, because they are
                different. A refusal styled like an answer is a lie told by CSS.
  provenance    every table says where it came from; every answer can be opened
                to show the formulas, rules and row counts behind it.
  limits        what the analysis cannot establish is rendered with the answer,
                not hidden behind a disclosure nobody opens.

Run:  streamlit run app.py
"""
from __future__ import annotations

import html
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from src.api.chatbot import ANSWERED, CLARIFY, FAILED, REFUSED, ChatResponse, Chatbot
from src.evidence import EvidenceTable
from src.visualization import ChartSpec

APP_TITLE = "Excel_sense_v2"
APP_SUBTITLE = ("Every figure below is computed by the deterministic analytics "
                "engine. The language model only explains what the engine found.")

MAX_HISTORY = 40
TABLE_PREVIEW_ROWS = 12

OUTCOME_STYLE = {
    ANSWERED: ("answer", "Answer"),
    CLARIFY: ("clarify", "Needs clarification"),
    REFUSED: ("refused", "Cannot answer"),
    FAILED: ("failed", "Insufficient evidence"),
}

STARTER_QUESTIONS = [
    "Which product has the highest lapse rate?",
    "What is the claim settlement ratio?",
    "How many policies are there?",
    "Show me persistency by product",
]

CSS = """
<style>
  .block-container { padding-top: 2.2rem; max-width: 1100px; }

  .tt-title { font-size: 2.1rem; font-weight: 700; letter-spacing: -0.02em;
              margin: 0 0 .35rem 0; }
  .tt-sub   { opacity: .70; font-size: .95rem; margin-bottom: 1.6rem; }

  .tt-q {
    border-left: 3px solid #4c8dff;
    background: rgba(76,141,255,.09);
    padding: .75rem 1rem; border-radius: 6px;
    font-weight: 600; margin: 1.4rem 0 .6rem 0;
  }

  .tt-a {
    border: 1px solid rgba(128,128,128,.28);
    border-radius: 8px; padding: 1rem 1.15rem; line-height: 1.6;
  }
  .tt-a.refused  { border-color: rgba(255,138,128,.55);
                   background: rgba(255,138,128,.07); }
  .tt-a.clarify  { border-color: rgba(255,196,84,.55);
                   background: rgba(255,196,84,.07); }
  .tt-a.failed   { border-color: rgba(160,160,160,.45);
                   background: rgba(128,128,128,.07); }

  .tt-badge {
    display: inline-block; font-size: .70rem; font-weight: 700;
    letter-spacing: .06em; text-transform: uppercase;
    padding: .15rem .5rem; border-radius: 4px; margin-bottom: .55rem;
    background: rgba(76,141,255,.16); color: #4c8dff;
  }
  .tt-badge.refused { background: rgba(255,138,128,.16); color: #ff8a80; }
  .tt-badge.clarify { background: rgba(255,196,84,.16);  color: #ffc454; }
  .tt-badge.failed  { background: rgba(160,160,160,.18); color: #b0b0b0; }

  .tt-meta { font-size: .78rem; opacity: .55; margin-top: .5rem; }
  .tt-limit { font-size: .85rem; opacity: .78; line-height: 1.55;
              border-left: 2px solid rgba(255,196,84,.5);
              padding: .1rem 0 .1rem .75rem; margin: .3rem 0; }
  .tt-src { font-size: .76rem; opacity: .55; margin-top: .35rem; }
  hr.tt-rule { border: none; border-top: 1px solid rgba(128,128,128,.18);
               margin: 2rem 0 .4rem 0; }
</style>
"""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the five datasets...")
def get_bot() -> Chatbot:
    """One chatbot per server process.

    Loading and typing 76,000 rows takes about half a second and the result is
    immutable, so it is shared across sessions. Conversation state is not: that
    lives in `st.session_state`, per browser session.
    """
    return Chatbot()


def _fmt(value: Any, unit: str, decimals: Optional[int]) -> str:
    """Format one cell in the unit the evidence layer declared for its column."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if unit == "rate":
            return "{:,.2f}%".format(value)
        if unit == "money":
            return "{:,.2f}".format(value)
        if unit == "count":
            return "{:,.0f}".format(value)
        if unit == "days":
            return "{:,.1f}".format(value)
        if unit == "identifier":
            return str(value)
        if decimals is not None:
            return "{:,.{}f}".format(value, decimals).rstrip("0").rstrip(".")
        return "{:,}".format(value) if isinstance(value, int) else str(value)
    return str(value)


def table_to_frame(table: EvidenceTable) -> pd.DataFrame:
    """Render an evidence table as text, so nothing is silently re-formatted.

    Streamlit will right-align and re-round a numeric column on its own. Since
    the evidence layer already decided each column's unit and precision, the
    values are formatted here and handed over as strings.
    """
    data = {}
    for column in table.columns:
        data[column.label] = [_fmt(row.get(column.key), column.unit, column.decimals)
                              for row in table.rows]
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_chart(chart: ChartSpec) -> None:
    """Draw the chart the evidence declared, or say why there is none."""
    from src.visualization.charts import (
        CATEGORY_CHARTS, DISTRIBUTION_CHARTS, POINT_CHARTS, SERIES_CHARTS,
        TILE_CHARTS,
    )

    if chart.chart_type in TILE_CHARTS:
        tiles = chart.tiles[:4]
        for column, tile in zip(st.columns(len(tiles) or 1), tiles):
            column.metric(tile.get("label", ""),
                          _fmt(tile.get("value"), tile.get("unit", "number"), None))
        return

    if not chart.series or not chart.categories:
        return

    frame = pd.DataFrame(
        {s.name: list(s.values) for s in chart.series},
        index=pd.Index([str(c) for c in chart.categories],
                       name=chart.category_label or chart.category_key))

    if chart.chart_type in CATEGORY_CHARTS:
        if chart.chart_type == "horizontal_bar":
            st.bar_chart(frame, horizontal=True, height=max(240, 34 * len(frame)))
        else:
            st.bar_chart(frame, height=320)
    elif chart.chart_type in SERIES_CHARTS:
        st.line_chart(frame, height=320)
    elif chart.chart_type in POINT_CHARTS:
        st.scatter_chart(frame, height=320)
    elif chart.chart_type in DISTRIBUTION_CHARTS:
        st.bar_chart(frame, height=320)
    else:
        st.bar_chart(frame, height=320)

    for annotation in chart.annotations:
        st.caption("{}: {}".format(annotation.label, _fmt(annotation.value, "rate", 2)
                                   if isinstance(annotation.value, float)
                                   else annotation.value))


def render_evidence(response: ChatResponse) -> None:
    """The tables, the chart, and the provenance behind the answer."""
    if response.chart and (response.chart.series or response.chart.tiles):
        with st.expander("Chart — {}".format(response.chart.title), expanded=True):
            render_chart(response.chart)
            if response.chart.source_note:
                st.markdown('<div class="tt-src">{}</div>'.format(
                    html.escape(response.chart.source_note)), unsafe_allow_html=True)
            for note in response.chart.footnotes[:4]:
                st.caption(note)

    for table in response.tables:
        # The limitations table is rendered inline below instead. What an analysis
        # cannot establish is part of the answer, and a collapsed expander is where
        # caveats go to be ignored.
        if table.key == "limitations":
            continue
        label = "{} ({:,} row{})".format(
            table.title, table.total_rows, "" if table.total_rows == 1 else "s")
        with st.expander(label, expanded=False):
            frame = table_to_frame(table)
            st.dataframe(frame.head(TABLE_PREVIEW_ROWS), width="stretch",
                         hide_index=True)
            if len(frame) > TABLE_PREVIEW_ROWS:
                st.caption("Showing {} of {:,} rows.".format(
                    TABLE_PREVIEW_ROWS, table.total_rows))
            if table.source_note:
                st.markdown('<div class="tt-src">{}</div>'.format(
                    html.escape(table.source_note)), unsafe_allow_html=True)
            for note in table.footnotes[:5]:
                st.caption(note)

    evidence = response.evidence
    if evidence is None:
        return

    limits = list(evidence.unsupported_conclusions)
    for text in evidence.limitations:
        if text not in limits:
            limits.append(text)
    if limits:
        st.markdown("**What this analysis cannot establish**")
        for text in limits[:5]:
            st.markdown('<div class="tt-limit">{}</div>'.format(html.escape(text)),
                        unsafe_allow_html=True)

    with st.expander("How this answer was produced", expanded=False):
        left, right = st.columns(2)
        with left:
            st.markdown("**Analysis**")
            st.write("{} — {}".format(evidence.detected_problem_id,
                                      evidence.problem_statement))
            st.markdown("**Population**")
            st.write("{} ({:,} rows)".format(
                evidence.population or "—", evidence.population_size or 0))
            if evidence.tables_used:
                st.markdown("**Datasets read**")
                st.write(", ".join(evidence.tables_used))
        with right:
            if evidence.formulas_applied:
                st.markdown("**Formulas applied**")
                st.write(", ".join(sorted(set(evidence.formulas_applied))))
            if evidence.business_rules_applied:
                st.markdown("**Business rules applied**")
                st.write(", ".join(sorted(
                    r.get("id", "?") for r in evidence.business_rules_applied)))
            st.markdown("**Routing confidence**")
            st.write("{:.0%}".format(response.confidence))
            if response.summary:
                st.markdown("**Answer written by**")
                st.write({"model": "language model, validated against the evidence",
                          "deterministic": "deterministic composer (no model call)"
                          }.get(response.summary.source, response.summary.source))

        if evidence.provenance.get("cache", {}).get("hit"):
            st.caption("Served from the result cache; figures computed at {}.".format(
                evidence.provenance["cache"]["computed_at_utc"]))


def render_turn(response: ChatResponse) -> None:
    """One exchange: the question, the answer, and everything behind it."""
    style, label = OUTCOME_STYLE.get(response.outcome, ("answer", "Answer"))

    st.markdown('<div class="tt-q">{}</div>'.format(html.escape(response.question)),
                unsafe_allow_html=True)
    st.markdown(
        '<div class="tt-a {style}"><span class="tt-badge {style}">{label}</span>'
        '<div>{text}</div></div>'.format(
            style=style, label=html.escape(label),
            text=html.escape(response.answer).replace("\n", "<br>")),
        unsafe_allow_html=True)

    meta = []
    if response.problem_id:
        meta.append(response.problem_id)
    meta.append("{:.2f}s".format(response.elapsed_seconds))
    if response.context_used:
        meta.append("carried forward: " + ", ".join(
            "{}={}".format(k, v) for k, v in response.context_used.items()))
    st.markdown('<div class="tt-meta">{}</div>'.format(html.escape(" · ".join(meta))),
                unsafe_allow_html=True)

    if response.security_findings:
        st.warning("The request was refused before any analysis ran: {}".format(
            "; ".join(f["detail"] for f in response.security_findings)))

    if response.outcome in (ANSWERED, FAILED):
        render_evidence(response)

    if response.suggestions and response.outcome in (CLARIFY, REFUSED, FAILED):
        st.markdown("**Questions I can answer from this data**")
        for index, suggestion in enumerate(response.suggestions[:3]):
            if st.button(suggestion["question"],
                         key="sugg-{}-{}".format(response.turn, index)):
                ask(suggestion["question"])
                st.rerun()

    st.markdown('<hr class="tt-rule">', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_state() -> None:
    st.session_state.setdefault("history", [])       # List[ChatResponse]
    st.session_state.setdefault("archive", [])       # List[Dict] past sessions
    st.session_state.setdefault("pending", None)


def ask(question: str) -> None:
    """Route one question through the chatbot and keep the result."""
    bot = get_bot()
    started = time.monotonic()
    try:
        response = bot.ask(question)
    except Exception as exc:  # pragma: no cover - surfaced in the UI, not swallowed
        st.error("The request could not be completed: {}".format(exc))
        return
    response.elapsed_seconds = response.elapsed_seconds or (time.monotonic() - started)
    st.session_state.history.append(response)
    if len(st.session_state.history) > MAX_HISTORY:
        st.session_state.history = st.session_state.history[-MAX_HISTORY:]


def archive_session() -> None:
    """Park the current conversation and start a fresh one."""
    if st.session_state.history:
        st.session_state.archive.insert(0, {
            "title": st.session_state.history[0].question,
            "turns": len(st.session_state.history),
            "questions": [r.question for r in st.session_state.history],
        })
        st.session_state.archive = st.session_state.archive[:12]
    st.session_state.history = []
    get_bot().reset()


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Conversation")
        st.caption("Follow-ups reuse the previous answer's subject. A refused turn "
                   "carries nothing forward.")

        left, right = st.columns(2)
        if left.button("New chat", width="stretch",
                       help="Archive this conversation and start a fresh one"):
            archive_session()
            st.rerun()
        if right.button("Clear", width="stretch",
                        help="Discard this conversation without archiving"):
            st.session_state.history = []
            get_bot().reset()
            st.rerun()

        st.divider()
        st.markdown("### Try a question")
        for index, question in enumerate(STARTER_QUESTIONS):
            if st.button(question, key="starter-{}".format(index), width="stretch"):
                st.session_state.pending = question
                st.rerun()

        if st.session_state.archive:
            st.divider()
            header, clear = st.columns([3, 1])
            header.markdown("### Past conversations")
            if clear.button("Clear", key="clear-archive"):
                st.session_state.archive = []
                st.rerun()
            for entry in st.session_state.archive:
                with st.expander("{} ({} turn{})".format(
                        entry["title"][:44], entry["turns"],
                        "" if entry["turns"] == 1 else "s")):
                    for question in entry["questions"]:
                        st.caption("• {}".format(question))

        st.divider()
        bot = get_bot()
        st.caption("{} analyses · {:,} policies · {} datasets".format(
            len(bot.registry), len(bot.dataset.frame("policy")),
            len(bot.dataset.table_keys)))
        st.caption("Answers written by: {}".format(
            "language model (validated)" if bot.summarizer.client.available
            and bot.summarizer.client.provider != "deterministic"
            else "deterministic composer"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    render_sidebar()

    st.markdown('<div class="tt-title">{}</div>'.format(APP_TITLE),
                unsafe_allow_html=True)
    st.markdown('<div class="tt-sub">{}</div>'.format(APP_SUBTITLE),
                unsafe_allow_html=True)

    pending = st.session_state.pop("pending", None)
    if pending:
        with st.spinner("Analysing..."):
            ask(pending)

    if not st.session_state.history:
        st.info("Ask a question about policies, products, agents, customers, claims "
                "or renewals. If the five datasets cannot answer it, the system will "
                "say so rather than estimate.")

    for response in st.session_state.history:
        render_turn(response)

    question = st.chat_input("Ask a question, or follow up on the answer above")
    if question:
        with st.spinner("Analysing..."):
            ask(question)
        st.rerun()


if __name__ == "__main__":
    main()
