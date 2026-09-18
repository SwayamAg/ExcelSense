"""The answer an executive actually reads.

Specification §8 sets the answer contract, §12 rule 9 requires answers that are
"short, crisp, direct, and business-friendly", and §14 makes concision part of
the definition of done. Those are requirements, not style preferences, and
nothing else in the suite tests them: every other test asks whether a figure is
correct, not whether the sentence carrying it is worth a CEO's attention.

The distinction that drives this file is between the *answer* and the *evidence*.
Sample sizes, exclusion thresholds, population definitions and caveats are all
required - §14 lists them - but they belong in the evidence tables and the
limitations channel, which the interfaces render separately. Padding them into
the answer prose does not make the answer more honest; it makes it unread.

So: the answer states the finding. Everything else has its own channel, and
these tests check that the split is respected in both directions - nothing
operational in the prose, and nothing lost from the evidence.
"""
from __future__ import annotations

import re

import pytest

from src.api.chatbot import ANSWERED, CLARIFY, Chatbot
from src.llm.client import DeterministicClient

# Questions a CEO or a distribution head would actually ask, in their own words.
EXECUTIVE_QUESTIONS = [
    "Which product has the highest lapse rate?",
    "What is the claim settlement ratio?",
    "How many policies are there?",
    "Show me persistency by product",
    "Which agents sold the most policies?",
    "How concentrated is premium among the largest policies?",
    "Give me a portfolio health summary",
    "Which branches underperform?",
    "What is driving lapses across the portfolio?",
    "Who are the top-producing agents?",
    "What is the average policy size?",
    "How many claims were settled?",
]

# The answer is prose for a business reader. None of this belongs in it.
MACHINERY = [
    (r"\bgroup[_ ]rows\b", "an internal row-count column name"),
    (r"\bminimum sample\b", "a sampling threshold"),
    (r"\bexcluded from ranking\b", "an internal exclusion rule"),
    (r"\bPS-\d{3}\b", "a problem statement id"),
    (r"\bF-[A-Z-]+\b", "a formula id"),
    (r"\bBR-\d{3}\b", "a business rule id"),
    (r"\bdropna\b|\bNaN\b|\bnull\b", "a data-processing term"),
    (r"_[a-z]+_", "a raw column name"),
    (r"\bevidence package\b|\bresult table\b", "internal vocabulary"),
    (r"\bdelta vs baseline\b", "a raw column label"),
]

MAX_ANSWER_WORDS = 45
MAX_ANSWER_SENTENCES = 4


def sentences(text: str) -> list:
    """Split into sentences without treating 26.07% as two of them."""
    protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", text)
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", protected) if s.strip()]
    return [p.replace("<DOT>", ".") for p in parts]


@pytest.fixture(scope="module")
def bot():
    """The deterministic composer: what a user sees with no API key configured."""
    return Chatbot(llm_client=DeterministicClient())


@pytest.fixture(scope="module")
def answers(bot):
    out = {}
    for question in EXECUTIVE_QUESTIONS:
        bot.reset()
        out[question] = bot.ask(question)
    return out


# --------------------------------------------------------------------------
# Short
# --------------------------------------------------------------------------

def test_answers_are_short(answers):
    """§12 rule 9: short, crisp, direct."""
    long_answers = []
    for question, response in answers.items():
        words = len(response.answer.split())
        if words > MAX_ANSWER_WORDS:
            long_answers.append("{!r}: {} words\n      {}".format(
                question, words, response.answer))
    assert not long_answers, (
        "{} answer(s) exceed {} words:\n    {}".format(
            len(long_answers), MAX_ANSWER_WORDS, "\n    ".join(long_answers)))


def test_answers_are_at_most_a_short_paragraph(answers):
    """§8: direct answer, the finding behind it, the supporting number."""
    verbose = []
    for question, response in answers.items():
        count = len(sentences(response.answer))
        if count > MAX_ANSWER_SENTENCES:
            verbose.append("{!r}: {} sentences\n      {}".format(
                question, count, response.answer))
    assert not verbose, "{} answer(s) run past {} sentences:\n    {}".format(
        len(verbose), MAX_ANSWER_SENTENCES, "\n    ".join(verbose))


# --------------------------------------------------------------------------
# Business-friendly
# --------------------------------------------------------------------------

def test_no_internal_machinery_reaches_the_answer(answers):
    """A reader should not need the schema to parse the sentence."""
    leaks = []
    for question, response in answers.items():
        for pattern, description in MACHINERY:
            match = re.search(pattern, response.answer, re.IGNORECASE)
            if match:
                leaks.append("{!r}: {} ({!r})\n      {}".format(
                    question, description, match.group(0), response.answer))
    assert not leaks, "{} answer(s) expose internals:\n    {}".format(
        len(leaks), "\n    ".join(leaks))


def test_operational_caveats_stay_out_of_the_prose(answers):
    """Sample thresholds are required evidence, but they are not the answer.

    §14 wants the limitation surfaced. The limitations channel is where the
    interfaces render it; repeating it in the prose costs the executive the
    sentence they came for.
    """
    padded = []
    for question, response in answers.items():
        if re.search(r"fell below|were excluded|are excluded|minimum sample",
                     response.answer, re.IGNORECASE):
            padded.append("{!r}\n      {}".format(question, response.answer))
    assert not padded, (
        "{} answer(s) carry an operational caveat in the prose:\n    {}".format(
            len(padded), "\n    ".join(padded)))


def test_answers_do_not_read_as_a_field_dump(answers):
    """"X is 1. Y is 2. Z is 3." states figures without saying anything."""
    dumps = []
    for question, response in answers.items():
        parts = sentences(response.answer)
        if len(parts) < 3:
            continue
        pattern = sum(1 for p in parts if re.match(r"^[A-Z][\w ]{1,28} is [\d.,]", p))
        if pattern >= 3:
            dumps.append("{!r}\n      {}".format(question, response.answer))
    assert not dumps, (
        "{} answer(s) read as a list of fields rather than a finding:\n    {}".format(
            len(dumps), "\n    ".join(dumps)))


def test_no_answer_contains_a_grammatical_artefact(answers):
    """Generated prose fails in recognisable ways. These are the ones seen."""
    broken = []
    for question, response in answers.items():
        text = response.answer
        for pattern, description in [
            (r"\b1 (?:groups|policies|claims|renewals|rows) ", "plural after 1"),
            (r"\bgroup rows\b", "an untranslated column name"),
            (r"\bis is\b|\bthe the\b|\ba a\b", "a duplicated word"),
            (r"\bat not available\b|\bis not available\b\.$", "a missing value in prose"),
            (r"\s{2,}", "doubled whitespace"),
            (r"\.\s*\.", "a doubled full stop"),
        ]:
            if re.search(pattern, text, re.IGNORECASE):
                broken.append("{!r}: {}\n      {}".format(question, description, text))
    assert not broken, "{} answer(s) contain a writing defect:\n    {}".format(
        len(broken), "\n    ".join(broken))


# --------------------------------------------------------------------------
# Direct
# --------------------------------------------------------------------------

def test_the_first_sentence_carries_the_answer(answers):
    """§8: the direct answer comes first, not after the setup."""
    buried = []
    for question, response in answers.items():
        if response.outcome != ANSWERED:
            continue
        first = sentences(response.answer)[0] if response.answer else ""
        if not re.search(r"\d", first):
            buried.append("{!r}\n      {}".format(question, response.answer))
    assert not buried, (
        "{} answer(s) open without a figure:\n    {}".format(
            len(buried), "\n    ".join(buried)))


def test_a_ranking_answer_names_the_leader_and_its_figure(bot):
    """§8, rankings: leader + KPI, then the peer comparison."""
    bot.reset()
    response = bot.ask("Which product has the highest lapse rate?")
    assert response.outcome == ANSWERED
    first = sentences(response.answer)[0]
    assert "26.07%" in first, (
        "the leading figure is not in the opening sentence: {!r}".format(first))
    assert re.search(r"10\.95|portfolio", response.answer), (
        "the answer gives a leader with nothing to compare it against")


def test_a_concentration_question_answers_about_concentration(bot):
    """A concentration question answered with a per-row ranking is the wrong answer.

    "How concentrated is premium?" is asking what share the largest policies
    hold. Naming the single largest policy and its 0.11% share answers a
    question nobody asked.
    """
    bot.reset()
    response = bot.ask("How concentrated is premium among the largest policies?")
    assert response.outcome == ANSWERED, response.answer
    assert re.search(r"top\s*\d+|largest\s*\d+|\d+%\s*of\b|share", response.answer,
                     re.IGNORECASE), (
        "the answer does not describe concentration: {!r}".format(response.answer))
    assert not re.match(r"^\d{8,}", response.answer.strip()), (
        "the answer opens with a raw policy identifier: {!r}".format(response.answer))


# --------------------------------------------------------------------------
# It answers the question that was asked
# --------------------------------------------------------------------------

def test_most_executive_questions_get_an_answer(answers):
    """Clarification is a correct outcome, but it is not a frequent one.

    §14: an executive asks a natural-language question and receives a concise
    answer. If half of plainly-phrased questions come back asking which of three
    analyses was meant, the system is not usable by the audience it is for.
    """
    answered = [q for q, r in answers.items() if r.outcome == ANSWERED]
    rate = 100.0 * len(answered) / len(answers)
    unanswered = ["{!r} -> {}".format(q, r.outcome)
                  for q, r in sorted(answers.items()) if r.outcome != ANSWERED]
    assert rate >= 80.0, (
        "only {:.0f}% of plainly-phrased executive questions were answered:\n"
        "    {}".format(rate, "\n    ".join(unanswered)))


def test_a_portfolio_summary_returns_a_summary(bot):
    """A one-number reply to "how is the portfolio doing" is a routing failure."""
    bot.reset()
    response = bot.ask("Give me a portfolio health summary")
    assert response.outcome == ANSWERED, response.answer
    assert len(response.answer.split()) >= 10, (
        "a portfolio summary came back as {!r}".format(response.answer))
    assert len(response.evidence.aggregates) >= 3, (
        "a portfolio summary rests on {} figure(s)".format(
            len(response.evidence.aggregates)))


def test_a_vague_question_is_never_answered(bot):
    """A vague question may be refused or clarified. It may not be answered.

    This test used to skip when the question stopped reading as ambiguous. That
    hid the regression it existed for: after a routing change, "Tell me about
    performance" was answered with a confident figure from an arbitrary analysis.
    Anything other than an answer is acceptable; an answer is the failure.
    """
    for question in ("Tell me about performance", "how are we doing",
                     "performance", "numbers"):
        bot.reset()
        response = bot.ask(question)
        assert response.outcome != ANSWERED, (
            "a vague question was answered as {}: {!r} -> {!r}".format(
                response.problem_id, question, response.answer))


def test_a_clarification_offers_real_alternatives(bot):
    """When the system does ask, the question must be answerable from the options."""
    bot.reset()
    response = bot.ask("How many claims were settled?")
    if response.outcome == ANSWERED:
        return
    assert response.outcome == CLARIFY, response.answer
    assert len(response.suggestions) >= 2, "a clarification offers no real choice"


# --------------------------------------------------------------------------
# The evidence keeps everything the answer dropped
# --------------------------------------------------------------------------

def test_nothing_dropped_from_the_prose_is_lost_from_the_evidence(bot):
    """Concision must not become omission.

    The sample-size caveat leaves the sentence; it must still reach the user
    through the channel built for it.
    """
    bot.reset()
    response = bot.ask("Which product has the highest lapse rate?")
    assert response.outcome == ANSWERED

    limitations = " ".join(response.evidence.limitations).lower()
    assert "minimum sample" in limitations or "excluded" in limitations, (
        "the exclusion caveat is in neither the answer nor the limitations")

    assert response.evidence.population_size, (
        "the population size left the prose and is not in the evidence either")
    assert response.tables, "the supporting table is missing"


def test_every_answered_question_still_carries_its_evidence(answers):
    """§14: a concise answer *plus* the calculated evidence, not instead of it."""
    bare = []
    for question, response in answers.items():
        if response.outcome != ANSWERED:
            continue
        if not response.tables:
            bare.append("{!r}: no evidence table".format(question))
        # A ranking carries its figures in the result table and the recorded
        # calculations rather than in scalar aggregates; all three are the
        # evidence, and an answer needs at least one of them.
        evidence = response.evidence
        if evidence and not (evidence.aggregates or evidence.result_table
                             or evidence.calculations):
            bare.append("{!r}: no computed figures".format(question))
    assert not bare, "\n    ".join(bare)


def test_diagnostic_answers_still_state_what_cannot_be_concluded(bot):
    """The one caveat that is not operational detail - it bounds the claim itself."""
    bot.reset()
    response = bot.ask("What is driving lapses across the portfolio?")
    if response.outcome != ANSWERED:
        pytest.skip("that question does not currently route to a diagnostic")
    assert response.evidence.unsupported_conclusions, (
        "a diagnostic answer declares no limit on what it can establish")


# --------------------------------------------------------------------------
# Defects found by reading the output as an executive would
#
# Every test above passed while these were live. Each one below names a defect
# that was actually shipped, so it cannot quietly return.
# --------------------------------------------------------------------------

def test_a_gap_between_rates_is_stated_in_points(bot):
    """26.07% against 10.95% is 15.13 points, not 15.13%.

    The relative difference is 138%. Writing "a gap of 15.13%" states neither,
    and an executive reading it will take it as the relative figure.
    """
    bot.reset()
    answer = bot.ask("Which product has the highest lapse rate?").answer
    assert "15.13 points" in answer, answer
    assert "15.13%" not in answer, answer


def test_products_are_named_not_just_coded(bot):
    bot.reset()
    answer = bot.ask("Which product has the highest lapse rate?").answer
    assert re.search(r"P91 \([A-Z][^)]+\)", answer), (
        "the product is given only as a code: {!r}".format(answer))
    assert "Canara HSBC" not in answer, "the plan name carries the company prefix"


def test_a_bare_identifier_never_opens_an_answer(answers):
    """"960773 leads" makes the reader guess that a number is an agent."""
    bare = ["{!r}\n      {}".format(q, r.answer) for q, r in answers.items()
            if re.match(r"^\d{4,}", r.answer.strip())]
    assert not bare, "\n    ".join(bare)


def test_a_leader_on_a_small_base_is_flagged(bot):
    """A branch 'worst' on 14 policies is not a finding to act on unqualified."""
    bot.reset()
    response = bot.ask("Which branches underperform?")
    assert response.outcome == ANSWERED
    policies = response.evidence.result_table[0].get("policies")
    if policies is not None and policies < 30:
        assert re.search(r"only {} polic".format(policies), response.answer), (
            "the leader rests on {} policies and the answer does not say so: "
            "{!r}".format(policies, response.answer))


def test_lowest_is_not_claimed_from_a_truncated_list(bot):
    """The last row of a top-10 is not the lowest of 1,956 agents."""
    bot.reset()
    response = bot.ask("Which agents sold the most policies?")
    ranked = (response.evidence.ranking or {}).get("entities_ranked", 0)
    if ranked > len(response.evidence.result_table):
        assert "lowest" not in response.answer.lower(), response.answer


def test_no_misspelled_plural(answers):
    wrong = ["{!r}: {}".format(q, r.answer) for q, r in answers.items()
             if re.search(r"\b\w+[^aeiou]ys\b", r.answer)]
    assert not wrong, "\n    ".join(wrong)


def test_a_diagnostic_states_the_size_and_direction_of_the_difference(bot):
    """"-56.17 on owner earned income" has no unit, no direction and no comparison."""
    bot.reset()
    response = bot.ask("What is driving lapses across the portfolio?")
    if response.outcome != ANSWERED:
        pytest.skip("that question does not currently route to a diagnostic")
    answer = response.answer
    assert re.search(r"\d+(\.\d+)?% (lower|higher)", answer), answer
    assert " vs " in answer, "the difference is not anchored to both values"
    assert "owner" not in answer.lower(), "schema vocabulary in the answer"
    assert re.search(r"not a proven cause|not established", answer), answer


# --------------------------------------------------------------------------
# The whole catalogue, not a sample
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def catalogue_answers(ps_executor):
    """Every analysis in the catalogue, composed as the fallback would compose it."""
    from scripts.build_regression_baseline import REQUIRED_PARAMETERS
    from src.llm.composer import compose_answer
    from src.llm.validator import validate_answer

    out = {}
    for ps in ps_executor.registry.executable():
        evidence = ps_executor.execute(ps.id, REQUIRED_PARAMETERS.get(ps.id))
        text = compose_answer(evidence)
        out[ps.id] = (text, validate_answer(text, evidence))
    return out


def test_every_catalogue_answer_is_grounded(catalogue_answers):
    """Rewording for executives must never cost the grounding guarantee."""
    failures = ["{}: {}\n      {}".format(pid, [i.detail for i in result.issues], text)
                for pid, (text, result) in sorted(catalogue_answers.items())
                if not result.is_valid]
    assert not failures, "{} composed answer(s) fail validation:\n    {}".format(
        len(failures), "\n    ".join(failures))


def test_every_catalogue_answer_is_short(catalogue_answers):
    long_ones = ["{}: {} words\n      {}".format(pid, len(text.split()), text)
                 for pid, (text, _) in sorted(catalogue_answers.items())
                 if len(text.split()) > 60]
    assert not long_ones, "{} of {} answers exceed 60 words:\n    {}".format(
        len(long_ones), len(catalogue_answers), "\n    ".join(long_ones))


def test_no_catalogue_answer_exposes_machinery(catalogue_answers):
    leaks = []
    for pid, (text, _) in sorted(catalogue_answers.items()):
        for pattern, description in MACHINERY:
            if re.search(pattern, text, re.IGNORECASE):
                leaks.append("{}: {}\n      {}".format(pid, description, text))
    assert not leaks, "{} leak(s):\n    {}".format(len(leaks), "\n    ".join(leaks))
