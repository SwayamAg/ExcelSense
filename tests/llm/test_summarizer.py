"""Phase 7 - LLM reasoning and summarization tests.

The gate is that the same evidence produces factually consistent answers even when
the wording changes. Everything else here protects the same claim: the model may
choose the words, and nothing else.

A `ScriptedClient` stands in for the model so misbehaviour can be provoked
deliberately - a fabricated number, an invented product, an unhedged causal claim, a
timeout, a refusal - and the system's response to each is asserted.
"""
from __future__ import annotations

import json

import pytest

import src.analysis.advanced_patterns  # noqa: F401
from src.llm import (
    SOURCE_FALLBACK_ERROR,
    SOURCE_FALLBACK_REJECTED,
    SOURCE_FALLBACK_UNAVAILABLE,
    SOURCE_MODEL,
    AnthropicClient,
    DeterministicClient,
    GroqClient,
    LLMUnavailable,
    ScriptedClient,
    Summarizer,
    build_messages,
    compose_answer,
    contract_as_dict,
    extract_numbers,
    system_prompt_for,
    validate_answer,
)
from src.llm.client import DEFAULT_MODEL
from src.llm.contract import build_evidence_payload

REQUIRED_PARAMETERS = {
    "PS-031": {"app_id": "11000000015"},
    "PS-069": {"agent_id": "977337"},
}


@pytest.fixture(scope="module")
def executor(ps_executor):
    return ps_executor


@pytest.fixture(scope="module")
def lapse_evidence(executor):
    return executor.execute("PS-035")


@pytest.fixture(scope="module")
def ranked_evidence(executor):
    return executor.execute("PS-013")


@pytest.fixture(scope="module")
def diagnostic_evidence(executor):
    return executor.execute("PS-040")


@pytest.fixture(scope="module")
def insufficient_evidence(executor):
    return executor.execute("PS-031", {"app_id": "00000000000"})


def summarizer(*replies, error=None):
    return Summarizer(client=ScriptedClient(replies=list(replies), error=error))


# --------------------------------------------------------------------------
# The gate: same evidence, consistent facts, whatever the wording
# --------------------------------------------------------------------------

def test_same_evidence_yields_the_same_facts_in_different_wordings(lapse_evidence):
    """The Phase 7 gate."""
    wordings = [
        "The portfolio lapse rate is 10.95%, covering 2,737 of 25,000 policies.",
        "Of the 25,000 policies in the book, 2,737 have lapsed - a rate of 10.95%.",
        "Lapse stands at 10.95%. That is 2,737 policies out of 25,000.",
    ]
    answers = [summarizer(w).summarize(lapse_evidence) for w in wordings]

    assert all(a.source == SOURCE_MODEL for a in answers), (
        "a valid wording was rejected: {}".format(
            [a.fallback_reason for a in answers if a.source != SOURCE_MODEL]))

    fact_sets = [set(extract_numbers(a.text)) for a in answers]
    assert fact_sets[0] == fact_sets[1] == fact_sets[2], (
        "the same evidence produced different figures: {}".format(fact_sets))


def test_wording_may_change_but_a_wrong_figure_may_not(lapse_evidence):
    good = summarizer("Lapse is 10.95% across 25,000 policies.").summarize(lapse_evidence)
    bad = summarizer("Lapse is 14.20% across 25,000 policies.").summarize(lapse_evidence)
    assert good.source == SOURCE_MODEL
    assert bad.source == SOURCE_FALLBACK_REJECTED
    assert "10.95" in bad.text or "10.948" in bad.text


def test_every_ps_produces_a_valid_answer(executor):
    """Across all 103 analyses, the answer the user sees passes validation."""
    failures = {}
    summary = Summarizer(client=DeterministicClient())
    for ps in executor.registry.executable():
        evidence = executor.execute(ps.id, REQUIRED_PARAMETERS.get(ps.id))
        answer = summary.summarize(evidence)
        if not answer.validation["valid"]:
            failures[ps.id] = answer.validation["issues"]
    assert not failures, "answers failing validation: {}".format(failures)


# --------------------------------------------------------------------------
# Number fidelity
# --------------------------------------------------------------------------

def test_a_fabricated_number_is_rejected(lapse_evidence):
    answer = summarizer("The lapse rate is 42.7% across the book.").summarize(lapse_evidence)
    assert answer.source == SOURCE_FALLBACK_REJECTED
    assert "42.7" in answer.fallback_reason


def test_a_recomputed_number_is_rejected(lapse_evidence):
    """The model must not do arithmetic, even correct arithmetic."""
    answer = summarizer(
        "89.05% of policies have not lapsed.").summarize(lapse_evidence)
    assert answer.source == SOURCE_FALLBACK_REJECTED


def test_numbers_present_in_the_evidence_are_accepted(ranked_evidence):
    answer = summarizer(
        "P91 has the highest lapse rate at 26.07%, against a portfolio average of "
        "10.95%.").summarize(ranked_evidence)
    assert answer.source == SOURCE_MODEL


def test_rounding_to_fewer_places_is_accepted(lapse_evidence):
    """10.948 may honestly be written 10.95 or 10.9."""
    for text in ("Lapse is 10.948%.", "Lapse is 10.95%.", "Lapse is 10.9%."):
        assert summarizer(text).summarize(lapse_evidence).source == SOURCE_MODEL


def test_validator_counts_what_it_checked(lapse_evidence):
    result = validate_answer("Lapse is 10.95% over 25,000 policies.", lapse_evidence)
    assert result.numbers_checked >= 2
    assert result.is_valid


# --------------------------------------------------------------------------
# Entity fidelity
# --------------------------------------------------------------------------

def test_an_invented_product_is_rejected(ranked_evidence):
    answer = summarizer(
        "P77 has the highest lapse rate at 26.07%.").summarize(ranked_evidence)
    assert answer.source == SOURCE_FALLBACK_REJECTED
    assert "P77" in answer.fallback_reason


def test_a_real_product_is_accepted(ranked_evidence):
    answer = summarizer("P91 leads on lapse at 26.07%.").summarize(ranked_evidence)
    assert answer.source == SOURCE_MODEL


def test_an_invented_agent_id_is_rejected(executor):
    evidence = executor.execute("PS-066")
    answer = summarizer(
        "Agent 123456 is the top producer.").summarize(evidence)
    assert answer.source == SOURCE_FALLBACK_REJECTED


# --------------------------------------------------------------------------
# Causal safety
# --------------------------------------------------------------------------

def test_an_unhedged_causal_claim_is_rejected(diagnostic_evidence):
    answer = summarizer(
        "Policies lapse because of low customer income."
    ).summarize(diagnostic_evidence)
    assert answer.source == SOURCE_FALLBACK_REJECTED
    assert "causation" in answer.fallback_reason


def test_a_hedged_association_is_accepted(diagnostic_evidence):
    answer = summarizer(
        "Lapsed policies are associated with lower declared income and smaller "
        "premiums. The data cannot establish that either causes the lapse."
    ).summarize(diagnostic_evidence)
    assert answer.source == SOURCE_MODEL


def test_causal_language_is_allowed_where_nothing_is_disclaimed(lapse_evidence):
    """A descriptive analysis makes no causal disclaimer, so the check must not fire."""
    result = validate_answer(
        "The lapse rate is 10.95%, which results in 2,737 lapsed policies.",
        lapse_evidence)
    assert not [i for i in result.errors if i.kind == "causal_safety"]


def test_the_deterministic_diagnostic_answer_disclaims_causation(diagnostic_evidence):
    text = compose_answer(diagnostic_evidence)
    assert any(k in text.lower() for k in ["not established causes", "not a proven cause", "cannot", "observed difference"])
    assert validate_answer(text, diagnostic_evidence).is_valid


# --------------------------------------------------------------------------
# Insufficient evidence
# --------------------------------------------------------------------------

def test_insufficient_evidence_is_admitted_not_filled(insufficient_evidence):
    answer = Summarizer(client=DeterministicClient()).summarize(insufficient_evidence)
    assert answer.sufficient is False
    assert "cannot" in answer.text.lower()
    assert answer.validation["valid"]


def test_answering_anyway_is_rejected(insufficient_evidence):
    answer = summarizer(
        "The policy has an APE of 250,000 and is premium paying."
    ).summarize(insufficient_evidence)
    assert answer.source == SOURCE_FALLBACK_REJECTED


def test_insufficient_prompt_tells_the_model_not_to_speculate(insufficient_evidence):
    system = system_prompt_for(insufficient_evidence)
    assert "Do not speculate" in system


# --------------------------------------------------------------------------
# Concision
# --------------------------------------------------------------------------

def test_a_long_answer_is_rejected(lapse_evidence):
    rambling = " ".join(["The lapse rate is 10.95%."] * 12)
    assert summarizer(rambling).summarize(lapse_evidence).source == SOURCE_FALLBACK_REJECTED


def test_an_empty_answer_falls_back(lapse_evidence):
    answer = summarizer("").summarize(lapse_evidence)
    assert answer.source == SOURCE_FALLBACK_ERROR


def test_every_deterministic_answer_is_concise(executor):
    summary = Summarizer(client=DeterministicClient())
    for ps in executor.registry.executable():
        evidence = executor.execute(ps.id, REQUIRED_PARAMETERS.get(ps.id))
        text = summary.summarize(evidence).text
        assert len(text.split()) <= 190, "{} answer runs long".format(ps.id)


# --------------------------------------------------------------------------
# Failure and fallback
# --------------------------------------------------------------------------

def test_an_api_error_falls_back(lapse_evidence):
    answer = summarizer(error=TimeoutError("request timed out")).summarize(lapse_evidence)
    assert answer.source == SOURCE_FALLBACK_ERROR
    assert "timed out" in answer.fallback_reason
    assert answer.validation["valid"]
    assert "10.95" in answer.text


def test_a_connection_failure_falls_back(lapse_evidence):
    answer = summarizer(error=ConnectionError("network unreachable")).summarize(lapse_evidence)
    assert answer.source == SOURCE_FALLBACK_ERROR
    assert answer.text


def test_an_unavailable_client_falls_back(lapse_evidence):
    answer = summarizer(error=LLMUnavailable("no credentials")).summarize(lapse_evidence)
    assert answer.source == SOURCE_FALLBACK_ERROR


def test_a_model_refusal_falls_back(lapse_evidence):
    from src.llm.client import LLMResponse

    class RefusingClient(ScriptedClient):
        def complete(self, system, messages, max_tokens=1024):
            return LLMResponse(text="", model="m", provider="scripted",
                               stop_reason="refusal", refused=True,
                               error="the model declined to answer")

    answer = Summarizer(client=RefusingClient()).summarize(lapse_evidence)
    assert answer.source == SOURCE_FALLBACK_ERROR
    assert "declined" in answer.fallback_reason


def test_no_model_configured_uses_the_deterministic_path(lapse_evidence):
    answer = Summarizer(client=DeterministicClient()).summarize(lapse_evidence)
    assert answer.source == SOURCE_FALLBACK_UNAVAILABLE
    assert answer.validation["valid"]


def test_the_fallback_answer_is_itself_validated(lapse_evidence):
    answer = summarizer(error=TimeoutError("boom")).summarize(lapse_evidence)
    assert answer.validation is not None
    assert answer.validation["valid"]


def test_a_failure_never_raises_to_the_caller(executor):
    """Whatever the model does, the user gets an answer."""
    for error in (TimeoutError("t"), ConnectionError("c"), ValueError("v"),
                  RuntimeError("r"), LLMUnavailable("u")):
        answer = summarizer(error=error).summarize(executor.execute("PS-001"))
        assert answer.text


# --------------------------------------------------------------------------
# The prompt contract
# --------------------------------------------------------------------------

def test_the_prompt_carries_only_evidence(ranked_evidence):
    messages = build_messages(ranked_evidence)
    assert len(messages) == 1 and messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert ranked_evidence.question in content
    assert "EVIDENCE" in content


def test_the_prompt_withholds_provenance(ranked_evidence):
    payload = build_evidence_payload(ranked_evidence)
    assert "provenance" not in payload
    assert "__row_id" not in json.dumps(payload)


def test_the_prompt_supplies_the_limits(diagnostic_evidence):
    payload = build_evidence_payload(diagnostic_evidence)
    assert payload.get("cannot_be_concluded")
    assert payload.get("caveat")


def test_the_system_prompt_forbids_calculation_and_invention():
    system = system_prompt_for.__globals__["SYSTEM_PROMPT"]
    assert "Never perform a calculation" in system
    assert "Never invent an entity" in system
    assert "Never state a number that is not in the evidence" in system


def test_the_contract_is_exportable_and_matches_config():
    from pathlib import Path

    contract = contract_as_dict()
    exported = json.loads(
        (Path(__file__).resolve().parents[2] / "config" / "llm_prompts.json")
        .read_text(encoding="utf-8"))
    assert exported["system_prompt"] == contract["system_prompt"]
    assert exported["contract_version"] == contract["contract_version"]
    assert exported["model"]["id"] == DEFAULT_MODEL


def test_no_sampling_parameters_are_configured():
    """Temperature and top_p are rejected on this model family, and a summarizer
    over fixed evidence has no use for them."""
    from pathlib import Path

    exported = json.loads(
        (Path(__file__).resolve().parents[2] / "config" / "llm_prompts.json")
        .read_text(encoding="utf-8"))
    assert "none sent" in exported["model"]["sampling_parameters"]


# --------------------------------------------------------------------------
# The client
# --------------------------------------------------------------------------

def test_the_groq_client_reports_unavailability_rather_than_raising():
    client = GroqClient()
    assert isinstance(client.available, bool)
    if not client.available:
        assert client.unavailable_reason


def test_the_anthropic_client_reports_unavailability_rather_than_raising():
    client = AnthropicClient()
    assert isinstance(client.available, bool)
    if not client.available:
        assert client.unavailable_reason


def test_the_default_model_is_pinned_without_a_date_suffix():
    assert "llama" in DEFAULT_MODEL or "claude" in DEFAULT_MODEL


def test_the_deterministic_client_refuses_a_raw_prompt():
    with pytest.raises(LLMUnavailable):
        DeterministicClient().complete("system", [{"role": "user", "content": "hi"}])


def test_the_scripted_client_records_what_it_was_sent(lapse_evidence):
    client = ScriptedClient(replies=["Lapse is 10.95%."])
    Summarizer(client=client).summarize(lapse_evidence)
    assert len(client.calls) == 1
    assert "Never perform a calculation" in client.calls[0]["system"]


# --------------------------------------------------------------------------
# The answer object
# --------------------------------------------------------------------------

def test_the_answer_is_json_serialisable(lapse_evidence):
    answer = summarizer("Lapse is 10.95%.").summarize(lapse_evidence)
    json.dumps(answer.as_dict())


def test_the_answer_records_where_it_came_from(lapse_evidence):
    from_model = summarizer("Lapse is 10.95%.").summarize(lapse_evidence)
    assert from_model.as_dict()["source"] == SOURCE_MODEL
    assert from_model.as_dict()["fallback_reason"] is None

    fell_back = summarizer("Lapse is 99.9%.").summarize(lapse_evidence)
    assert fell_back.as_dict()["fallback_reason"]


def test_the_answer_names_the_problem_it_answers(lapse_evidence):
    answer = summarizer("Lapse is 10.95%.").summarize(lapse_evidence)
    assert answer.problem_id == "PS-035"
    assert answer.question == lapse_evidence.question
