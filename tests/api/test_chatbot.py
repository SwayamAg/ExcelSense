"""Phase 8 - chatbot integration tests.

Covers the parts of the chatbot that are not any single scenario: how the layers
are wired together, how conversation memory behaves, what happens when something
below it fails, and how a turn is rendered.
"""
from __future__ import annotations

import io
import json

import pytest

import src.analysis.advanced_patterns  # noqa: F401
from src.api import ANSWERED, CLARIFY, FAILED, REFUSED, Chatbot, Conversation
from src.llm import ScriptedClient
from src.ui import render, render_chart, render_table
from src.ui.cli import _help_text, format_cell, main, repl


@pytest.fixture(scope="module")
def bot(dataset):
    return Chatbot(dataset=dataset)


@pytest.fixture(autouse=True)
def _clean(bot):
    bot.reset()
    yield
    bot.reset()


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------

def test_a_question_runs_every_layer(bot):
    response = bot.ask("Which product has the highest lapse rate?")
    assert response.routing is not None          # routed
    assert response.evidence is not None         # executed
    assert response.tables                       # evidence presented
    assert response.chart is not None            # visualised
    assert response.summary is not None          # summarised
    assert response.answer                       # answered


def test_the_answer_is_grounded_in_the_evidence(bot):
    response = bot.ask("What is the overall lapse rate?")
    assert response.summary.validation["valid"]
    assert response.evidence.aggregates["lapse_rate"] == 10.948


def test_a_turn_is_serialisable_with_and_without_evidence(bot):
    response = bot.ask("How many policies are there?")
    json.dumps(response.as_dict(), default=str)
    full = response.as_dict(include_evidence=True)
    assert "evidence" in full
    json.dumps(full, default=str)


def test_turns_are_numbered(bot):
    assert bot.ask("How many policies are there?").turn == 1
    assert bot.ask("What is the total APE?").turn == 2


def test_elapsed_time_is_recorded(bot):
    response = bot.ask("How many policies are there?")
    assert response.elapsed_seconds > 0


def test_the_chatbot_reports_which_analysis_answered(bot):
    response = bot.ask("What is the overall lapse rate?")
    assert response.problem_id == "PS-035"
    assert response.confidence > 0


# --------------------------------------------------------------------------
# Empty and malformed input
# --------------------------------------------------------------------------

@pytest.mark.parametrize("question", ["", "   ", "\n"])
def test_empty_input_is_handled(bot, question):
    response = bot.ask(question)
    assert response.outcome == REFUSED
    assert "Ask a question" in response.answer


def test_punctuation_only_input_is_refused(bot):
    response = bot.ask("???!!!")
    assert response.outcome in (REFUSED, CLARIFY)
    assert response.problem_id is None


def test_a_very_long_question_does_not_break_routing(bot):
    question = "What is the overall lapse rate? " + ("please " * 200)
    response = bot.ask(question)
    assert response.outcome in (ANSWERED, CLARIFY, REFUSED)


# --------------------------------------------------------------------------
# Conversation memory
# --------------------------------------------------------------------------

def test_a_fresh_conversation_carries_nothing(bot):
    response = bot.ask("What about P22?")
    assert response.context_used == {}


def test_an_entity_carries_into_an_elliptical_follow_up(bot):
    bot.ask("How is product P91 performing on lapse?")
    follow_up = bot.ask("What about P22?")
    assert follow_up.outcome in (ANSWERED, CLARIFY, REFUSED)
    # The follow-up names its own product, so nothing needed to be carried.
    assert "product" not in follow_up.context_used or \
        follow_up.context_used.get("product") == "P91"


def test_context_is_not_applied_to_a_self_contained_question(bot):
    bot.ask("Tell me about policy 11000000015")
    second = bot.ask("Which agents have the worst lapse rates?")
    assert second.context_used == {}
    assert second.problem_id == "PS-039"


def test_nothing_carries_forward_from_a_refusal(bot):
    """A rejected premise must not leak into the next answer."""
    refused = bot.ask("Show me commission for agent 977337")
    assert refused.outcome == REFUSED
    follow_up = bot.ask("and for that one?")
    assert follow_up.context_used == {}


def test_resetting_clears_the_conversation(bot):
    bot.ask("Tell me about policy 11000000015")
    assert bot.conversation.turns
    bot.reset()
    assert not bot.conversation.turns
    assert bot.ask("what about it?").context_used == {}


def test_conversation_memory_is_bounded():
    conversation = Conversation(max_turns=3)
    from src.api.chatbot import ChatResponse
    for i in range(6):
        conversation.record(ChatResponse(question="q{}".format(i), outcome=ANSWERED,
                                         answer="a"))
    assert len(conversation.turns) == 3


def test_a_follow_up_is_recognised_only_when_it_looks_like_one():
    conversation = Conversation()
    from src.api.chatbot import ChatResponse
    conversation.record(ChatResponse(question="first", outcome=ANSWERED, answer="a"))
    assert conversation.looks_like_follow_up("what about P22")
    assert conversation.looks_like_follow_up("and for Karnataka")
    assert not conversation.looks_like_follow_up(
        "Which agents have the worst lapse rates across the portfolio?")


# --------------------------------------------------------------------------
# Failure handling
# --------------------------------------------------------------------------

def test_an_llm_failure_still_answers(dataset):
    """The facts were settled before the model was called."""
    bot = Chatbot(dataset=dataset,
                  llm_client=ScriptedClient(error=TimeoutError("model timed out")))
    response = bot.ask("What is the overall lapse rate?")
    assert response.outcome == ANSWERED
    assert "10.95" in response.answer
    assert response.summary.fallback_reason


def test_an_llm_hallucination_is_replaced_not_shown(dataset):
    bot = Chatbot(dataset=dataset,
                  llm_client=ScriptedClient(replies=["The lapse rate is 73.4%."]))
    response = bot.ask("What is the overall lapse rate?")
    assert response.outcome == ANSWERED
    assert "73.4" not in response.answer
    assert "10.95" in response.answer


def test_a_valid_model_answer_is_used(dataset):
    bot = Chatbot(dataset=dataset, llm_client=ScriptedClient(
        replies=["The portfolio lapse rate is 10.95%, or 2,737 of 25,000 policies."]))
    response = bot.ask("What is the overall lapse rate?")
    assert response.summary.source == "model"
    assert "2,737" in response.answer


def test_no_question_ever_raises(bot):
    for question in ("", "???", "a" * 500, "SELECT * FROM policies",
                     "Tell me about policy 99999999999", "why", "commission"):
        assert bot.ask(question).answer


# --------------------------------------------------------------------------
# Refusal and clarification quality
# --------------------------------------------------------------------------

def test_a_refusal_offers_an_alternative(bot):
    response = bot.ask("Show me commission by agent")
    assert response.outcome == REFUSED
    assert response.suggestions
    assert all("question" in s for s in response.suggestions)


def test_an_ambiguous_question_asks_rather_than_guesses(bot):
    response = bot.ask("performance")
    assert response.outcome in (CLARIFY, REFUSED)
    assert response.problem_id is None
    assert response.suggestions


def test_an_unknown_identifier_is_named_in_the_refusal(bot):
    response = bot.ask("Tell me about policy 99999999999")
    assert response.outcome == REFUSED
    assert "99999999999" in response.answer


def test_no_refusal_produces_tables_or_charts(bot):
    for question in ("Show me commission by agent", "What is the capital of France?"):
        response = bot.ask(question)
        assert not response.tables
        assert response.chart is None


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def test_render_includes_question_and_answer(bot):
    text = render(bot.ask("What is the overall lapse rate?"))
    assert "What is the overall lapse rate?" in text
    assert "ANSWER" in text


def test_render_hides_audit_tables_unless_asked(bot):
    response = bot.ask("What is the overall lapse rate?")
    plain = render(response)
    detailed = render(response, show_detail=True)
    assert "How each figure was calculated" not in plain
    assert "How each figure was calculated" in detailed


def test_render_draws_a_chart_when_one_exists(bot):
    text = render(bot.ask("Which product has the highest lapse rate?"))
    assert "#" in text
    assert "baseline" in text


def test_render_shows_a_refusal_clearly(bot):
    text = render(bot.ask("Show me commission by agent"))
    assert "CANNOT ANSWER" in text
    assert "You could ask:" in text


def test_render_table_formats_units(bot):
    response = bot.ask("Which product has the highest lapse rate?")
    table = next(t for t in response.tables if t.key == "result")
    text = render_table(table)
    assert "26.07%" in text            # a rate carries its percent sign
    assert "4,721" in text             # a count is grouped and has no decimals


def test_render_chart_is_empty_for_no_chart():
    from src.visualization import ChartSpec
    assert render_chart(ChartSpec(chart_type="bar", title="empty")) == ""


@pytest.mark.parametrize("value,unit,expected", [
    (10.948, "rate", "10.95%"),
    (2306202661.89, "money", "2,306,202,661.89"),
    (25000, "count", "25,000"),
    (None, "count", "-"),
    (True, "text", "yes"),
])
def test_format_cell(value, unit, expected):
    assert format_cell(value, unit, 2) == expected


def test_help_lists_what_can_be_asked(bot):
    text = _help_text(bot)
    assert "Portfolio Overview" in text
    assert "cannot answer" in text


# --------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------

def test_cli_answers_a_question(capsys):
    code = main(["What", "is", "the", "overall", "lapse", "rate?"])
    captured = capsys.readouterr().out
    assert code == 0
    assert "10.95" in captured


def test_cli_returns_non_zero_when_it_cannot_answer(capsys):
    code = main(["Show", "me", "commission", "by", "agent"])
    assert code == 1
    assert "CANNOT ANSWER" in capsys.readouterr().out


def test_cli_can_emit_json(capsys):
    main(["--json", "How", "many", "policies", "are", "there?"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["outcome"] == ANSWERED
    assert payload["problem_id"] == "PS-001"


def test_repl_handles_commands(bot):
    stream = io.StringIO()
    questions = iter([":help", ":detail", ":reset", "How many policies are there?",
                      ":quit"])
    import builtins
    original = builtins.input
    builtins.input = lambda *_: next(questions)
    try:
        repl(bot, stream=stream)
    finally:
        builtins.input = original
    output = stream.getvalue()
    assert "I can answer questions in these areas" in output
    assert "Detail on" in output
    assert "Conversation cleared" in output
    assert "25,000" in output


def test_repl_exits_on_eof(bot):
    stream = io.StringIO()
    import builtins
    original = builtins.input
    builtins.input = lambda *_: (_ for _ in ()).throw(EOFError())
    try:
        repl(bot, stream=stream)
    finally:
        builtins.input = original
    assert "Excel_sense_v2" in stream.getvalue()
