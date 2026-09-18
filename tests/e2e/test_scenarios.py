"""Phase 8 - the fourteen end-to-end scenarios.

The specification names fourteen scenarios the finished system must handle. Each is
asked as a person would ask it, and the whole stack runs: routing, execution,
evidence, tables, chart, summarization.

Two of the fourteen are only satisfied by a refusal. Channel comparison is impossible
because CHANNEL is 'Banca' for all 25,000 policies, and the insufficient-evidence
scenario asks about a policy that does not exist. Passing those means declining
clearly, not producing something.
"""
from __future__ import annotations

import json

import pytest

import src.analysis.advanced_patterns  # noqa: F401
from src.api import ANSWERED, CLARIFY, FAILED, REFUSED, Chatbot
from src.ui import render


@pytest.fixture(scope="module")
def bot(dataset):
    return Chatbot(dataset=dataset)


@pytest.fixture(autouse=True)
def _clean_conversation(bot):
    """Each scenario starts with no memory, so none can pass on a previous turn."""
    bot.reset()
    yield
    bot.reset()


def ask(bot, question):
    return bot.ask(question)


# --------------------------------------------------------------------------
# 1. Policy lookup
# --------------------------------------------------------------------------

def test_scenario_01_policy_lookup(bot):
    response = ask(bot, "Tell me about policy 11000000015")
    assert response.outcome == ANSWERED
    assert response.problem_id == "PS-031"
    assert "11000000015" in response.answer
    assert response.evidence.retrieved_records
    assert response.summary.validation["valid"]


def test_scenario_01_lookup_withholds_personal_data(bot):
    response = ask(bot, "Tell me about policy 11000000015")
    record = response.evidence.retrieved_records[0]
    assert not [k for k in record if "Given.Name" in k or "Postal.Code" in k]


# --------------------------------------------------------------------------
# 2. Claim lookup
# --------------------------------------------------------------------------

def test_scenario_02_claim_lookup(bot):
    response = ask(bot, "How many claims have been made and what are they worth?")
    assert response.outcome == ANSWERED
    assert response.evidence.aggregates["claim_count"] == 134
    assert "134" in response.answer


def test_scenario_02_largest_claims(bot):
    response = ask(bot, "Which are the largest claims?")
    assert response.outcome == ANSWERED
    assert response.problem_id == "PS-061"
    assert response.evidence.population_size == 134


# --------------------------------------------------------------------------
# 3. Lapse diagnosis
# --------------------------------------------------------------------------

def test_scenario_03_lapse_diagnosis(bot):
    response = ask(bot, "Why are policies lapsing?")
    assert response.outcome == ANSWERED
    assert response.problem_id == "PS-040"
    assert response.evidence.unsupported_conclusions


def test_scenario_03_diagnosis_refuses_to_claim_cause(bot):
    """A diagnostic answer that asserts cause would be the worst possible failure."""
    response = ask(bot, "Why are policies lapsing?")
    joined = " ".join(response.evidence.unsupported_conclusions).lower()
    assert "caus" in joined
    assert response.summary.validation["valid"]
    limitations = next(t for t in response.tables if t.key == "limitations")
    assert any("caus" in row["statement"].lower() for row in limitations.rows)


# --------------------------------------------------------------------------
# 4. Persistency lookup
# --------------------------------------------------------------------------

def test_scenario_04_persistency_lookup(bot):
    response = ask(bot, "How does persistency differ across the 13th, 25th and 37th month buckets?")
    assert response.outcome == ANSWERED
    assert response.problem_id == "PS-045"
    buckets = {row["Persistency_Bucket"] for row in response.evidence.result_table}
    assert buckets == {"13th Month", "25th Month", "37th Month"}


def test_scenario_04_persistency_states_its_population(bot):
    """Renewal metrics use a smaller denominator; the answer must not hide that."""
    response = ask(bot, "What is our renewal collection rate?")
    assert response.outcome in (ANSWERED, CLARIFY)
    if response.outcome == ANSWERED:
        assert "24,542" in response.evidence.population or \
               "persistency" in response.evidence.population.lower()


# --------------------------------------------------------------------------
# 5. Agent ranking
# --------------------------------------------------------------------------

def test_scenario_05_agent_ranking(bot):
    response = ask(bot, "Which agents have the worst lapse rates?")
    assert response.outcome == ANSWERED
    assert response.problem_id == "PS-039"
    assert response.evidence.ranking
    assert all(row["policies"] >= 30 for row in response.evidence.result_table)


def test_scenario_05_ranking_reports_the_sample_guard(bot):
    response = ask(bot, "Which agents have the worst lapse rates?")
    assert response.evidence.sample_sizes["minimum_group_size"] == 30
    result = next(t for t in response.tables if t.key == "result")
    assert any("minimum sample" in note or "threshold" in note
               for note in result.footnotes)


# --------------------------------------------------------------------------
# 6. Agent peer comparison
# --------------------------------------------------------------------------

def test_scenario_06_agent_peer_comparison(bot):
    response = ask(bot, "How does agent 977337 compare against their peers?")
    assert response.outcome == ANSWERED
    assert response.problem_id == "PS-069"
    assert response.routing.parameters.get("agent_id") == "977337"
    assert response.evidence.comparison_baseline


# --------------------------------------------------------------------------
# 7. "Why is this agent performing better?"
# --------------------------------------------------------------------------

def test_scenario_07_why_is_this_agent_better(bot):
    response = ask(bot, "Why does the top agent outperform their peers?")
    assert response.outcome == ANSWERED
    assert response.problem_id == "PS-068"
    assert response.evidence.ranking["leader"]
    assert response.evidence.comparison_baseline["description"].startswith("peer average")


def test_scenario_07_states_what_it_cannot_explain(bot):
    """The data holds nothing about how an agent works, and the answer must say so."""
    response = ask(bot, "Why does the top agent outperform their peers?")
    joined = " ".join(response.evidence.unsupported_conclusions).lower()
    assert "association" in joined or "not" in joined


# --------------------------------------------------------------------------
# 8. Product comparison
# --------------------------------------------------------------------------

def test_scenario_08_product_comparison(bot):
    response = ask(bot, "How do traditional and ULIP policies compare across key metrics?")
    assert response.outcome == ANSWERED
    assert response.problem_id == "PS-016"
    classes = {row["Trad.ULIP"] for row in response.evidence.result_table}
    assert classes == {"TRAD", "ULIP"}


def test_scenario_08_product_ranking(bot):
    response = ask(bot, "Which product has the highest lapse rate?")
    assert response.outcome == ANSWERED
    assert response.evidence.result_table[0]["Prod_No"] == "P91"
    assert response.chart is not None


# --------------------------------------------------------------------------
# 9. Channel comparison - impossible, and must be refused
# --------------------------------------------------------------------------

def test_scenario_09_channel_comparison_is_refused(bot):
    """CHANNEL is 'Banca' for all 25,000 policies. There is nothing to compare."""
    response = ask(bot, "Compare channels on persistency")
    assert response.outcome == REFUSED
    assert response.problem_id is None
    assert "Banca" in response.answer


def test_scenario_09_refusal_explains_and_redirects(bot):
    response = ask(bot, "Which channel performs best?")
    assert response.outcome in (REFUSED, CLARIFY)
    assert response.suggestions


# --------------------------------------------------------------------------
# 10. Customer-level question
# --------------------------------------------------------------------------

def test_scenario_10_customer_question(bot):
    response = ask(bot, "What is the age profile of our customers?")
    assert response.outcome == ANSWERED
    assert "owner" in response.evidence.tables_used
    assert response.evidence.aggregates.get("median") is not None


def test_scenario_10_customer_segmentation(bot):
    response = ask(bot, "Which occupations generate the most premium?")
    assert response.outcome in (ANSWERED, CLARIFY)
    if response.outcome == ANSWERED:
        assert response.problem_id == "PS-080"


# --------------------------------------------------------------------------
# 11. Multi-table question
# --------------------------------------------------------------------------

def test_scenario_11_multi_table_question(bot):
    response = ask(bot, "Give me a full performance scorecard for every product")
    assert response.outcome == ANSWERED
    assert response.problem_id == "PS-020"
    assert set(response.evidence.tables_used) >= {"policy", "claims", "persistency"}
    assert len(response.evidence.join_path) >= 2


def test_scenario_11_join_path_is_recorded(bot):
    response = ask(bot, "Give me a full performance scorecard for every product")
    joined = " ".join(response.evidence.join_path)
    assert "policy.AppID" in joined


# --------------------------------------------------------------------------
# 12. Formula-based question
# --------------------------------------------------------------------------

def test_scenario_12_formula_based_question(bot):
    response = ask(bot, "What is the overall lapse rate?")
    assert response.outcome == ANSWERED
    assert response.evidence.aggregates["lapse_rate"] == 10.948
    assert "F-RATE" in response.evidence.formulas_applied


def test_scenario_12_calculation_is_auditable(bot):
    """Every figure must be traceable to a formula and its inputs."""
    response = ask(bot, "What is the overall lapse rate?")
    calculations = next(t for t in response.tables if t.key == "calculations")
    rate_row = next(r for r in calculations.rows if r["formula"] == "F-RATE")
    assert rate_row["value"] == 10.948
    assert "numerator=2737" in rate_row["inputs"]


def test_scenario_12_answer_repeats_the_figure_exactly(bot):
    response = ask(bot, "What is the overall lapse rate?")
    assert "10.95" in response.answer or "10.948" in response.answer
    assert response.summary.validation["valid"]


# --------------------------------------------------------------------------
# 13. Unsupported question
# --------------------------------------------------------------------------

@pytest.mark.parametrize("question,concept", [
    ("How do smokers compare to non-smokers on premiums?", "smoking"),
    ("Show me commission paid to each agent", "commission"),
    ("Which product is most profitable?", "margin"),
    ("What is the NAV of the ULIP fund?", "fund"),
    ("Which agents missed their target?", "target"),
])
def test_scenario_13_unsupported_question_is_refused(bot, question, concept):
    response = ask(bot, question)
    assert response.outcome == REFUSED
    assert response.problem_id is None
    assert concept in response.answer.lower()
    assert "inventing the missing field" in response.answer


def test_scenario_13_refusal_names_what_is_missing(bot):
    response = ask(bot, "How do smokers compare on premiums?")
    assert response.routing.missing_concepts
    assert response.routing.missing_concepts[0]["concept"] == "smoker"


def test_scenario_13_out_of_domain_is_refused(bot):
    response = ask(bot, "What is the capital of France?")
    assert response.outcome in (REFUSED, CLARIFY)
    assert response.problem_id is None


# --------------------------------------------------------------------------
# 14. Insufficient-evidence question
# --------------------------------------------------------------------------

def test_scenario_14_unknown_policy_is_reported_not_invented(bot):
    response = ask(bot, "Tell me about policy 00000000000")
    assert response.outcome in (REFUSED, FAILED)
    lowered = response.answer.lower()
    assert "does not appear in the data" in response.answer or "cannot" in lowered
    assert not response.tables
    assert response.chart is None


def test_scenario_14_no_figures_are_offered_when_nothing_was_computed(bot):
    response = ask(bot, "Tell me about policy 00000000000")
    assert response.evidence is None or not response.evidence.aggregates


# --------------------------------------------------------------------------
# Every scenario, rendered
# --------------------------------------------------------------------------

SCENARIOS = [
    ("policy lookup", "Tell me about policy 11000000015", ANSWERED),
    ("claim lookup", "How many claims have been made and what are they worth?", ANSWERED),
    ("lapse diagnosis", "Why are policies lapsing?", ANSWERED),
    ("persistency lookup", "How does persistency differ across the 13th, 25th and 37th month buckets?", ANSWERED),
    ("agent ranking", "Which agents have the worst lapse rates?", ANSWERED),
    ("agent peer comparison", "How does agent 977337 compare against their peers?", ANSWERED),
    ("why this agent is better", "Why does the top agent outperform their peers?", ANSWERED),
    ("product comparison", "How do traditional and ULIP policies compare across key metrics?", ANSWERED),
    ("channel comparison", "Compare channels on persistency", REFUSED),
    ("customer question", "What is the age profile of our customers?", ANSWERED),
    ("multi-table question", "Give me a full performance scorecard for every product", ANSWERED),
    ("formula question", "What is the overall lapse rate?", ANSWERED),
    ("unsupported question", "Show me commission paid to each agent", REFUSED),
    ("insufficient evidence", "Tell me about policy 00000000000", REFUSED),
]


@pytest.mark.parametrize("name,question,expected", SCENARIOS)
def test_all_fourteen_scenarios(bot, name, question, expected):
    """The Phase 8 gate: all critical end-to-end scenarios pass."""
    response = ask(bot, question)
    assert response.outcome == expected, (
        "{}: expected {}, got {} ({})".format(name, expected, response.outcome,
                                              response.answer[:90]))
    assert response.answer, "{} produced no answer text".format(name)


@pytest.mark.parametrize("name,question,expected", SCENARIOS)
def test_every_scenario_renders(bot, name, question, expected):
    text = render(ask(bot, question))
    assert question.strip() in text
    assert len(text.splitlines()) >= 3


@pytest.mark.parametrize("name,question,expected", SCENARIOS)
def test_every_scenario_is_serialisable(bot, name, question, expected):
    json.dumps(ask(bot, question).as_dict(), default=str)


def test_no_scenario_ever_raises(bot):
    for name, question, _ in SCENARIOS:
        bot.ask(question)          # the assertion is that nothing propagates
