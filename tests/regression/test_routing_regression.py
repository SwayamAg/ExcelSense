"""Routing still sends each question where it sent it.

Accuracy alone is too coarse to protect routing. Two questions can swap which
analysis they reach and leave the percentage exactly where it was, and the user
who asked about lapse rates gets a persistency breakdown with full provenance
attached. So the baseline records the decision for every one of the catalogue's
example queries, not just the score.

The asymmetry in these tests is deliberate. A question that stops routing
correctly is a failure. A question that *starts* routing correctly is reported,
not failed - the suite should not stand in the way of the router getting better,
only in the way of it getting quietly worse.
"""
from __future__ import annotations

import pytest

from src.router.router import ACCEPTANCE, AMBIGUITY_MARGIN, CONFIDENCE_FLOOR


@pytest.fixture(scope="module")
def recorded(baseline):
    return baseline["routing"]


@pytest.fixture(scope="module")
def current(query_router, recorded):
    """Every baselined question, routed now."""
    return {q: query_router.route(q) for q in recorded["routes"]}


# --------------------------------------------------------------------------
# The score
# --------------------------------------------------------------------------

def test_accuracy_has_not_fallen(recorded, current):
    correct = sum(1 for q, decision in current.items()
                  if decision.outcome == "routed"
                  and decision.problem_id == recorded["routes"][q]["expected"])
    accuracy = 100.0 * correct / len(current)
    assert accuracy >= recorded["accuracy_pct"] - 0.01, (
        "routing accuracy fell from {:.2f}% to {:.2f}% ({} of {} example queries "
        "correct)".format(recorded["accuracy_pct"], accuracy, correct, len(current)))


def test_accuracy_still_meets_the_phase_5_gate(recorded, current):
    """The regression floor and the gate are different numbers; both must hold."""
    correct = sum(1 for q, decision in current.items()
                  if decision.outcome == "routed"
                  and decision.problem_id == recorded["routes"][q]["expected"])
    accuracy = 100.0 * correct / len(current)
    assert accuracy >= ACCEPTANCE["min_in_domain_accuracy_pct"], (
        "{:.2f}% is below the {:.1f}% acceptance threshold".format(
            accuracy, ACCEPTANCE["min_in_domain_accuracy_pct"]))


# --------------------------------------------------------------------------
# The individual decisions
# --------------------------------------------------------------------------

def test_no_question_stopped_routing_correctly(recorded, current):
    """The regression that accuracy can hide."""
    lost = []
    for question, was in sorted(recorded["routes"].items()):
        if not was["correct"]:
            continue
        decision = current[question]
        if decision.outcome != "routed" or decision.problem_id != was["expected"]:
            lost.append("{!r}: {} -> {} ({})".format(
                question, was["expected"], decision.problem_id or "-",
                decision.outcome))
    assert not lost, (
        "{} question(s) no longer route where they did:\n  {}".format(
            len(lost), "\n  ".join(lost[:30])))


def test_no_question_started_routing_to_a_different_analysis(recorded, current):
    """A swap between two wrong answers is still a change worth seeing.

    A question that was refused and is now answered is an improvement and is not
    failed here. A question that was answered by one PS and is now answered by
    another is flagged regardless of which was correct, because someone chose
    that mapping.
    """
    swapped = []
    for question, was in sorted(recorded["routes"].items()):
        decision = current[question]
        if was["outcome"] != "routed" or decision.outcome != "routed":
            continue
        if decision.problem_id != was["problem_id"]:
            swapped.append("{!r}: {} -> {}".format(
                question, was["problem_id"], decision.problem_id))
    assert not swapped, (
        "{} question(s) now route to a different analysis:\n  {}\n\nIf the new "
        "mapping is better, regenerate the baseline.".format(
            len(swapped), "\n  ".join(swapped[:30])))


def test_improvements_are_reported_not_failed(recorded, current):
    gained = [q for q, was in recorded["routes"].items()
              if not was["correct"]
              and current[q].outcome == "routed"
              and current[q].problem_id == was["expected"]]
    if gained:
        pytest.skip("{} question(s) now route correctly that did not before. "
                    "Regenerate the baseline to lock the improvement in:\n  {}".format(
                        len(gained), "\n  ".join(sorted(gained)[:20])))


def test_confidence_has_not_collapsed(recorded, current):
    """A right answer reached with much less confidence is on its way to being wrong."""
    dropped = []
    for question, was in sorted(recorded["routes"].items()):
        decision = current[question]
        if was["outcome"] != "routed" or decision.outcome != "routed":
            continue
        if decision.confidence < was["confidence"] - 0.05:
            dropped.append("{!r}: {:.3f} -> {:.3f}".format(
                question, was["confidence"], decision.confidence))
    assert not dropped, (
        "{} question(s) route with materially lower confidence:\n  {}".format(
            len(dropped), "\n  ".join(dropped[:20])))


def test_every_routed_decision_clears_the_confidence_floor(current):
    """The floor is the refusal boundary. Nothing may be routed below it."""
    below = ["{!r}: {:.3f}".format(q, d.confidence) for q, d in sorted(current.items())
             if d.outcome == "routed" and d.confidence < CONFIDENCE_FLOOR]
    assert not below, (
        "{} decision(s) were routed below the {:.3f} confidence floor:\n  {}".format(
            len(below), CONFIDENCE_FLOOR, "\n  ".join(below)))


# --------------------------------------------------------------------------
# The thresholds themselves
# --------------------------------------------------------------------------

def test_the_routing_thresholds_are_unchanged():
    """Loosening a threshold turns refusals into guesses, and does it silently.

    These values were chosen by sweeping them against the query suite. Changing
    one is a legitimate thing to do; changing one without anybody noticing is not,
    so they are pinned here.
    """
    assert CONFIDENCE_FLOOR == 0.32, (
        "the confidence floor moved to {}. Below it the router refuses; raising or "
        "lowering it changes what the system is willing to answer.".format(
            CONFIDENCE_FLOOR))
    assert AMBIGUITY_MARGIN == 0.020, (
        "the ambiguity margin moved to {}. It decides when two analyses are too "
        "close to choose between.".format(AMBIGUITY_MARGIN))
    assert ACCEPTANCE["min_out_of_domain_refusal_pct"] == 100.0, (
        "the out-of-domain refusal requirement is no longer total. Anything less "
        "means the system will sometimes answer a question about data it does not "
        "have.")


def test_the_example_query_corpus_has_not_shrunk(recorded, registry_size=None):
    """Fewer example queries means less routing surface under test."""
    from src.ps.registry import get_registry

    total = sum(len(ps.record.get("example_queries", []))
                for ps in get_registry().all())
    assert total >= recorded["example_query_count"], (
        "the catalogue now carries {} example queries, down from {}. Each one is a "
        "routing case that stopped being checked.".format(
            total, recorded["example_query_count"]))
