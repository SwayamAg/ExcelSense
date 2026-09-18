"""Terminal interface.

Renders a chat response as text: the answer first, then the evidence that supports
it, then the chart data, then what the analysis cannot establish. No dependency
beyond the standard library, so the interface is as testable as everything under it.

Formatting follows the units the evidence layer declared, so a rate prints as a
percentage and a count never grows a decimal point.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional

from src.api.chatbot import ANSWERED, CLARIFY, FAILED, REFUSED, ChatResponse, Chatbot
from src.evidence import EvidenceTable
from src.visualization import ChartSpec

WIDTH = 96
BAR_WIDTH = 32

OUTCOME_LABEL = {
    ANSWERED: "ANSWER",
    CLARIFY: "NEEDS CLARIFICATION",
    REFUSED: "CANNOT ANSWER",
    FAILED: "INSUFFICIENT EVIDENCE",
}


def format_cell(value: Any, unit: str, decimals: Optional[int]) -> str:
    if value is None:
        return "-"
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
        if decimals is not None:
            return "{:,.{}f}".format(value, decimals).rstrip("0").rstrip(".")
        return str(value)
    text = str(value)
    return text if len(text) <= 44 else text[:41] + "..."


def render_table(table: EvidenceTable, max_rows: int = 12) -> str:
    """A plain-text table with its source note and footnotes."""
    columns = table.columns
    rows = table.rows[:max_rows]
    if not columns or not rows:
        return ""

    cells: List[List[str]] = [[c.label for c in columns]]
    for row in rows:
        cells.append([format_cell(row.get(c.key), c.unit, c.decimals) for c in columns])

    widths = [max(len(r[i]) for r in cells) for i in range(len(columns))]
    widths = [min(w, 44) for w in widths]

    def line(values: List[str]) -> str:
        return "  ".join(v.ljust(w)[:w] for v, w in zip(values, widths))

    out = ["  " + table.title.strip(), "  " + "-" * min(sum(widths) + 2 * len(widths), WIDTH)]
    out.append("  " + line(cells[0]))
    out.append("  " + "-" * min(sum(widths) + 2 * len(widths), WIDTH))
    for row in cells[1:]:
        out.append("  " + line(row))
    if table.truncated:
        out.append("  ... showing {} of {} rows".format(len(rows), table.total_rows))
    if table.source_note:
        out.append("  source: {}".format(table.source_note[:WIDTH - 12]))
    for note in table.footnotes[:3]:
        out.append("  note: {}".format(note[:WIDTH - 10]))
    return "\n".join(out)


def render_chart(chart: ChartSpec, max_bars: int = 12, compact: bool = False) -> str:
    """A text bar chart, drawn only from the values the chart spec carries."""
    if chart.tiles:
        lines = ["  {} ({})".format(chart.title.strip(), chart.chart_type)]
        for tile in chart.tiles[:8]:
            lines.append("    {:<34} {}".format(
                tile["label"][:34], format_cell(tile["value"], tile["unit"], 2)))
        return "\n".join(lines)

    if not chart.series or not chart.categories:
        return ""

    series = chart.series[0]
    pairs = [(str(c), v) for c, v in zip(chart.categories, series.values)
             if isinstance(v, (int, float))][:max_bars]
    if not pairs:
        return ""

    peak = max(abs(v) for _, v in pairs) or 1
    # The label column must fit the annotation names too, or "baseline" prints as "bas".
    label_candidates = [c for c, _ in pairs] + [a.kind for a in chart.annotations[:2]]
    label_width = min(max(len(c) for c in label_candidates), 30)

    # The compact header names the measure only; the question is already on screen.
    lines = ["  {}".format(series.name)] if compact else \
        ["  {} - {} ({})".format(chart.title.strip(), series.name, chart.chart_type)]
    for label, value in pairs:
        filled = int(round(BAR_WIDTH * abs(value) / peak))
        lines.append("    {:<{}}  {:<{}} {}".format(
            label[:label_width], label_width,
            "#" * filled, BAR_WIDTH,
            format_cell(value, series.unit, 2)))
    for annotation in chart.annotations[:2]:
        # Use the annotation kind, not its description: a truncated sentence reads
        # as a mangled category label rather than a reference line.
        lines.append("    {:<{}}  {} {}".format(
            annotation.kind[:label_width], label_width,
            "-" * 4, format_cell(annotation.value, series.unit, 2)))
    if chart.truncated:
        lines.append("    ... {} categories shown".format(len(pairs)))
    return "\n".join(lines)


def render(response: ChatResponse, show_tables: bool = True,
           show_chart: bool = True, show_detail: bool = False) -> str:
    """One turn, as text.

    The default is the executive view: the question, the answer, and a compact
    chart when there is one. The readers are executives, and a ten-column table
    with "% delta vs baseline 138.1714" under a one-sentence answer buries the
    answer. The evidence is not dropped - `--detail` (or `:detail` in a session)
    prints every table, the calculations, the limitations and the routing footer.
    """
    out: List[str] = ["-" * WIDTH, "Q: {}".format(response.question.strip())]
    out.append("{}: {}".format(OUTCOME_LABEL.get(response.outcome, response.outcome.upper()),
                               response.answer))

    if response.context_used:
        out.append("(using {} from the previous question)".format(
            ", ".join(str(v) for v in response.context_used.values())))

    if response.suggestions and response.outcome != ANSWERED:
        out.append("")
        out.append("You could ask:")
        for item in response.suggestions[:3]:
            out.append("  - {}".format(item["question"]))

    if show_chart and response.chart:
        chart_text = render_chart(response.chart, max_bars=12 if show_detail else 8,
                                  compact=not show_detail)
        if chart_text:
            out.append("")
            out.append(chart_text)

    if not show_detail:
        if response.answered and response.tables and show_tables:
            out.append("")
            out.append("(evidence tables and calculations: --detail, or :detail in a session)")
        return "\n".join(out)

    if show_tables and response.tables:
        for table in response.tables:
            rendered = render_table(table)
            if rendered:
                out.append("")
                out.append(rendered)

    if response.evidence and response.evidence.is_sufficient:
        footer = []
        if response.problem_id:
            footer.append("{} @ {:.2f}".format(response.problem_id, response.confidence))
        if response.summary:
            footer.append("answer via {}".format(response.summary.source))
        footer.append("{:.2f}s".format(response.elapsed_seconds))
        out.append("")
        out.append("[{}]".format(" | ".join(footer)))

    return "\n".join(out)


BANNER = """\
Excel_sense_v2 - life insurance portfolio analysis
Answers come from five CSV datasets and nothing else. Questions the data cannot
support are refused rather than guessed at.

  :help     what this can answer        :detail   show calculations and rules
  :reset    forget the conversation     :quit     exit
"""


def _help_text(chatbot: Chatbot) -> str:
    domains: Dict[str, List[str]] = {}
    for ps in chatbot.registry.all():
        domains.setdefault(ps.business_domain, []).append(ps.problem_statement)
    lines = ["I can answer questions in these areas:", ""]
    for domain in sorted(domains):
        lines.append("  {} ({} analyses)".format(domain, len(domains[domain])))
        lines.append("      e.g. {}".format(domains[domain][0]))
    lines.append("")
    lines.append("I cannot answer anything requiring data these five files do not hold, "
                 "such as commission, profitability, fund performance or smoker status.")
    return "\n".join(lines)


def repl(chatbot: Optional[Chatbot] = None, stream=None) -> None:
    """Interactive session."""
    stream = stream or sys.stdout
    chatbot = chatbot or Chatbot()
    show_detail = False

    print(BANNER, file=stream)
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("", file=stream)
            return

        if not question:
            continue
        command = question.lower()
        if command in (":quit", ":q", "quit", "exit"):
            return
        if command in (":help", "help", "?"):
            print(_help_text(chatbot), file=stream)
            continue
        if command == ":reset":
            chatbot.reset()
            print("Conversation cleared.", file=stream)
            continue
        if command == ":detail":
            show_detail = not show_detail
            print("Detail {}.".format("on" if show_detail else "off"), file=stream)
            continue

        response = chatbot.ask(question)
        print(render(response, show_detail=show_detail), file=stream)
        print("", file=stream)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask questions about the life insurance datasets.")
    parser.add_argument("question", nargs="*", help="a question to answer, or none for a session")
    parser.add_argument("--detail", action="store_true",
                        help="show every evidence table, the calculations and the limitations")
    parser.add_argument("--no-tables", action="store_true", help="answer only")
    parser.add_argument("--json", action="store_true", help="emit the response as JSON")
    args = parser.parse_args(argv)

    chatbot = Chatbot()
    if not args.question:
        repl(chatbot)
        return 0

    response = chatbot.ask(" ".join(args.question))
    if args.json:
        import json
        print(json.dumps(response.as_dict(), indent=2, default=str))
    else:
        print(render(response, show_tables=not args.no_tables, show_detail=args.detail))
    return 0 if response.answered else 1


if __name__ == "__main__":
    raise SystemExit(main())
