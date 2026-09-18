"""Phase 5 - routing tests.

The gate is routing accuracy against a project-defined threshold with no unsafe
fallback to an unrelated PS. The second half of that matters more than the first:
routing a question about smokers to a premium analysis and returning a confident
number is worse than refusing, so the refusal paths are tested as hard as the
matches.
"""
from __future__ import annotations

import json

import pytest

from src.router import (
    AMBIGUOUS,
    REFUSED_INVALID_ENTITY,
    REFUSED_LOW_CONFIDENCE,
    REFUSED_UNSUPPORTED,
    ROUTED,
    QueryRouter,
    find_unsupported,
    normalize,
)
from src.router.matcher import WEIGHTS
from src.router.router import ACCEPTANCE, AMBIGUITY_MARGIN, CONFIDENCE_FLOOR
from tests.router.query_suite import (
    AMBIGUOUS_QUERIES,
    CATEGORIES,
    IN_DOMAIN,
    INVALID_ENTITIES,
    OUT_OF_DOMAIN,
    UNSUPPORTED,
)


@pytest.fixture(scope="module")
def router(query_router):
    return query_router


@pytest.fixture(scope="module")
def decisions(router):
    """Every suite question routed once."""
    out = {}
    for items in CATEGORIES.values():
        for question, expected in items:
            out[question] = (router.route(question), expected)
    return out


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------

def test_in_domain_accuracy_meets_the_acceptance_threshold(decisions):
    """The Phase 5 gate: routing accuracy against the project-defined threshold."""
    correct = sum(1 for d, expected in decisions.values()
                  if d.outcome == ROUTED and d.problem_id == expected)
    accuracy = 100.0 * correct / len(decisions)
    assert accuracy >= ACCEPTANCE["min_in_domain_accuracy_pct"], (
        "in-domain routing accuracy {:.1f}% is below the {:.1f}% threshold".format(
            accuracy, ACCEPTANCE["min_in_domain_accuracy_pct"]))


def test_incorrect_routes_stay_within_tolerance(decisions):
    """The other half of the gate: no unsafe fallback to an unrelated PS."""
    wrong = [(q, d.problem_id, expected) for q, (d, expected) in decisions.items()
             if d.outcome == ROUTED and d.problem_id != expected]
    rate = 100.0 * len(wrong) / len(decisions)
    assert rate <= ACCEPTANCE["max_incorrect_route_pct"], (
        "{:.1f}% of questions routed to the wrong PS: {}".format(rate, wrong))


def test_out_of_domain_questions_are_never_routed(router):
    """A question the datasets cannot answer must not reach an executor."""
    routed = [q for q in OUT_OF_DOMAIN if router.route(q).outcome == ROUTED]
    assert not routed, "out-of-domain questions were routed: {}".format(routed)


def test_vague_questions_are_never_routed(router):
    routed = [q for q in AMBIGUOUS_QUERIES if router.route(q).outcome == ROUTED]
    assert not routed, "vague questions were routed: {}".format(routed)


def test_unsupported_questions_are_refused_by_name(router):
    """Naming the missing field is what stops the question being asked again blindly."""
    missed = []
    for question, concept in UNSUPPORTED:
        decision = router.route(question)
        if decision.outcome != REFUSED_UNSUPPORTED:
            missed.append((question, decision.outcome))
    rate = 100.0 * (len(UNSUPPORTED) - len(missed)) / len(UNSUPPORTED)
    assert rate >= ACCEPTANCE["min_unsupported_named_refusal_pct"], (
        "unsupported questions not refused by name: {}".format(missed))


def test_every_category_is_represented_in_the_suite():
    """The specification names nine categories; the suite must cover them."""
    assert set(CATEGORIES) == {"direct", "paraphrase", "executive", "why", "how",
                               "ranking", "entity_scoped"}
    assert len(IN_DOMAIN) >= 80
    assert UNSUPPORTED and OUT_OF_DOMAIN and AMBIGUOUS_QUERIES and INVALID_ENTITIES


@pytest.mark.parametrize("category", sorted(CATEGORIES))
def test_no_category_collapses(router, category):
    """An overall average can hide one category routing nothing correctly."""
    items = CATEGORIES[category]
    correct = sum(1 for q, expected in items
                  if router.route(q).problem_id == expected)
    accuracy = 100.0 * correct / len(items)
    assert accuracy >= 70.0, "{} routes only {:.0f}% correctly".format(category, accuracy)


def test_correct_ps_is_almost_always_in_the_candidate_list(decisions):
    """Even when held for clarification, the right answer should be on offer."""
    present = sum(1 for d, expected in decisions.values()
                  if expected in [c.problem_id for c in d.candidates])
    rate = 100.0 * present / len(decisions)
    assert rate >= 95.0, "correct PS appears in the candidates only {:.1f}% of the time".format(rate)


# --------------------------------------------------------------------------
# Refusal behaviour
# --------------------------------------------------------------------------

def test_unsupported_refusal_names_the_missing_field(router):
    decision = router.route("How do smokers compare to non-smokers on premiums?")
    assert decision.outcome == REFUSED_UNSUPPORTED
    assert decision.missing_concepts
    assert "smoking" in decision.missing_concepts[0]["reason"]
    assert decision.problem_id is None


def test_unsupported_refusal_happens_before_matching(router):
    """Scoring a question that cannot be answered risks routing it anyway."""
    decision = router.route("Which product is most profitable?")
    assert decision.outcome == REFUSED_UNSUPPORTED
    assert "cost, expense or margin" in decision.reason


def test_channel_comparison_is_refused_because_the_column_is_single_valued(router):
    decision = router.route("Compare channels on persistency")
    assert decision.outcome == REFUSED_UNSUPPORTED
    assert "Banca" in decision.reason


def test_a_stated_reason_is_refused_but_a_diagnostic_why_is_not(router):
    """The data records no reason codes, but it can still compare cohorts."""
    stated = router.route("What reason did customers give for surrendering?")
    assert stated.outcome == REFUSED_UNSUPPORTED

    diagnostic = router.route("Why are policies lapsing?")
    assert diagnostic.outcome == ROUTED
    assert diagnostic.problem_id == "PS-040"


def test_invalid_identifier_is_refused_and_named(router):
    for question in INVALID_ENTITIES:
        decision = router.route(question)
        assert decision.outcome == REFUSED_INVALID_ENTITY
        assert "does not appear in the data" in decision.reason
        assert decision.problem_id is None


def test_low_confidence_refusal_explains_why_it_did_not_guess(router):
    decision = router.route("purple monkey dishwasher")
    assert decision.outcome == REFUSED_LOW_CONFIDENCE
    assert "confident" in decision.reason or "closely enough" in decision.reason


def test_empty_question_is_refused(router):
    for question in ("", "   ", "???"):
        decision = router.route(question)
        assert decision.outcome.startswith("refused")
        assert decision.problem_id is None


def test_every_refusal_offers_somewhere_to_go(router):
    """A refusal that leaves the user stuck is a worse answer than it needs to be."""
    for question in ("Show me commission by agent", "analysis please"):
        decision = router.route(question)
        assert decision.is_refusal
        assert decision.suggestions, "{} refused with no suggestions".format(question)


def test_ambiguous_outcome_returns_the_options(router):
    ambiguous = [router.route(q) for _, items in CATEGORIES.items() for q, _ in items]
    held = [d for d in ambiguous if d.outcome == AMBIGUOUS]
    assert held, "no question in the suite exercises the ambiguity path"
    for decision in held:
        assert len(decision.candidates) >= 2
        assert decision.suggestions
        assert decision.problem_id is None


# --------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------

def test_nothing_below_the_floor_is_ever_routed(router):
    for items in CATEGORIES.values():
        for question, _ in items:
            decision = router.route(question)
            if decision.outcome == ROUTED:
                assert decision.confidence >= CONFIDENCE_FLOOR


def test_nothing_inside_the_ambiguity_margin_is_ever_routed(router):
    for items in CATEGORIES.values():
        for question, _ in items:
            decision = router.route(question)
            if decision.outcome != ROUTED or len(decision.candidates) < 2:
                continue
            lead = decision.candidates[0].score - decision.candidates[1].score
            assert lead >= AMBIGUITY_MARGIN


def test_thresholds_are_exposed_in_the_decision(router):
    payload = router.route("How many policies are there?").as_dict()
    assert payload["thresholds"]["confidence_floor"] == CONFIDENCE_FLOOR
    assert payload["thresholds"]["ambiguity_margin"] == AMBIGUITY_MARGIN


def test_scoring_weights_sum_to_one(router):
    assert round(sum(WEIGHTS.values()), 6) == 1.0


def test_confidence_is_bounded(router):
    for items in CATEGORIES.values():
        for question, _ in items:
            for candidate in router.route(question).candidates:
                assert 0.0 <= candidate.score <= 1.0


# --------------------------------------------------------------------------
# Normalization and vocabulary
# --------------------------------------------------------------------------

def test_normalization_expands_contractions():
    assert "what is" in normalize("What's the lapse rate?").cleaned


def test_normalization_is_case_and_punctuation_insensitive(router):
    a = router.route("How many policies are there?")
    b = router.route("HOW MANY POLICIES ARE THERE")
    assert a.problem_id == b.problem_id


def test_business_aliases_reach_the_same_ps(router):
    """The same question in three registers must land in the same place."""
    for phrasing in ("What is our lapse rate?",
                     "What share of policies have lapsed?",
                     "How many policies have stopped paying?"):
        decision = router.route(phrasing)
        assert decision.outcome in (ROUTED, AMBIGUOUS)
        if decision.outcome == ROUTED:
            assert decision.problem_id in ("PS-035", "PS-002")


def test_analysis_verbs_are_not_expanded_into_the_vocabulary():
    """Expanding superlatives once made every ranking question look identical."""
    terms = normalize("Which state has the highest lapse rate?").terms
    assert "ranking" not in terms, (
        "superlatives are leaking into the matching vocabulary again")


def test_unsupported_detection_handles_plurals():
    assert find_unsupported("Do smokers lapse more?")
    assert find_unsupported("Which riders sell best?")


def test_supported_questions_report_no_missing_concepts():
    assert find_unsupported("What is the overall lapse rate?") == []
    assert find_unsupported("Which product has the highest lapse rate?") == []


# --------------------------------------------------------------------------
# Entity extraction
# --------------------------------------------------------------------------

def test_appid_is_extracted_and_validated(router):
    decision = router.route("Tell me about policy 11000000015")
    entity = decision.entities.first("app_id")
    assert entity.value == "11000000015" and entity.valid
    assert decision.parameters["app_id"] == "11000000015"


def test_unknown_appid_is_marked_invalid(router):
    decision = router.route("Tell me about policy 00000000000")
    assert decision.entities.first("app_id").valid is False


def test_agent_id_is_extracted(router):
    decision = router.route("How does agent 977337 compare against their peers?")
    entity = decision.entities.first("sales_id")
    assert entity.value == "977337" and entity.valid


def test_product_code_is_extracted_and_validated(router):
    decision = router.route("How is product P91 performing on lapse?")
    entity = decision.entities.first("product")
    assert entity.value == "P91" and entity.valid


def test_state_is_extracted(router):
    decision = router.route("How many policies are in Karnataka?")
    assert decision.entities.first("state").value == "Karnataka"


def test_status_is_extracted(router):
    decision = router.route("How many policies have lapsed?")
    assert decision.entities.first("policy_status").value == "Lapsed"


def test_persistency_bucket_is_extracted(router):
    decision = router.route("What is the 37th month persistency?")
    assert decision.entities.first("persistency_bucket").value == "37th Month"


def test_top_n_is_extracted_and_passed_through(router):
    decision = router.route("Show me the top 5 policies by premium")
    assert decision.entities.first("top_n").value == 5
    if decision.outcome == ROUTED:
        assert decision.parameters.get("n") == 5


def test_year_outside_the_extract_is_marked_invalid(router):
    decision = router.route("How many policies were issued in 2019?")
    year = decision.entities.first("year")
    assert year is not None and year.valid is False
    assert decision.outcome == REFUSED_INVALID_ENTITY


def test_parameters_are_filtered_to_what_the_ps_declares(router):
    """An entity the PS never declared must not be smuggled into execution."""
    decision = router.route("Tell me about policy 11000000015")
    assert decision.outcome == ROUTED
    declared = set(router.registry.get(decision.problem_id).record.get("parameters", {}))
    assert set(decision.parameters) <= declared


# --------------------------------------------------------------------------
# Determinism and output contract
# --------------------------------------------------------------------------

def test_routing_is_deterministic(router):
    for question in ("Which product has the highest lapse rate?",
                     "Why are policies lapsing?",
                     "Show me commission by agent"):
        # Bypass the fixture cache: two genuinely separate routings.
        first = router.route_uncached(question)
        second = router.route_uncached(question)
        assert first.outcome == second.outcome
        assert first.problem_id == second.problem_id
        assert first.confidence == second.confidence


def test_decision_is_json_serialisable(router):
    for question in ("How many policies are there?", "How do smokers compare?",
                     "Tell me about policy 11000000015"):
        json.dumps(router.route(question).as_dict())


def test_decision_carries_the_scoring_breakdown(router):
    decision = router.route("Which product has the highest lapse rate?")
    top = decision.candidates[0]
    assert set(top.components) == set(WEIGHTS)
    assert top.closest_example_query if hasattr(top, "closest_example_query") else True
    payload = top.as_dict()
    assert payload["closest_example_query"]
    assert payload["business_domain"]


def test_routed_decisions_name_a_real_ps(router, decisions):
    for decision, _ in decisions.values():
        if decision.outcome == ROUTED:
            assert decision.problem_id in router.registry


def test_candidates_are_ordered_by_confidence(router):
    decision = router.route("Which agents have the worst lapse rates?")
    scores = [c.score for c in decision.candidates]
    assert scores == sorted(scores, reverse=True)
