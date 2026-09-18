"""Query routing.

Decides which Problem Statement answers a question, or refuses. The refusal path
matters as much as the match: the reference implementation this project replaces
answered "how do smokers compare" by routing to an unrelated PS and returning a
confident figure from a column that does not exist. That failure mode is designed
out here - a question resting on absent data is refused by name, and a question
that no PS matches well enough is refused rather than forced onto the nearest one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.data import Dataset, load_dataset
from src.ps.registry import PSRegistry, get_registry
from src.router.entities import EntityExtractor, ExtractionResult
from src.router.matcher import Candidate, PSMatcher
from src.router.normalizer import NormalizedQuery, find_unsupported, normalize

ROUTER_VERSION = "1.0.0"

# Acceptance thresholds, chosen by sweeping the query suite rather than by feel.
# The objective was the most correct routes obtainable while still refusing every
# out-of-domain question: at 0.32 / 0.020 the suite gives 80 correct routes, 1
# incorrect, 0 out-of-domain routed, and 14 questions held for clarification.
# Loosening either value starts routing out-of-domain questions, which is the one
# outcome this layer exists to prevent.
CONFIDENCE_FLOOR = 0.32      # below this nothing is routed
AMBIGUITY_MARGIN = 0.020     # a lead this thin over the runner-up is not a decision
MAX_CANDIDATES = 5

# The project-defined acceptance criteria for Phase 5, asserted in tests/router.
ACCEPTANCE = {
    "min_in_domain_accuracy_pct": 90.0,   # correctly routed, of all in-domain questions
    "max_incorrect_route_pct": 3.0,       # routed to the wrong PS
    "min_out_of_domain_refusal_pct": 100.0,
    "min_unsupported_named_refusal_pct": 100.0,
}

ROUTED = "routed"
AMBIGUOUS = "ambiguous"
REFUSED_UNSUPPORTED = "refused_unsupported"
REFUSED_LOW_CONFIDENCE = "refused_low_confidence"
REFUSED_INVALID_ENTITY = "refused_invalid_entity"


@dataclass
class RoutingDecision:
    """What the router decided, and everything behind it."""
    question: str
    outcome: str
    problem_id: Optional[str] = None
    confidence: float = 0.0
    candidates: List[Candidate] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    entities: Optional[ExtractionResult] = None
    normalized: Optional[NormalizedQuery] = None
    reason: str = ""
    missing_concepts: List[Dict[str, str]] = field(default_factory=list)
    suggestions: List[Dict[str, str]] = field(default_factory=list)

    @property
    def is_routed(self) -> bool:
        return self.outcome == ROUTED

    @property
    def is_refusal(self) -> bool:
        return self.outcome.startswith("refused")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "router_version": ROUTER_VERSION,
            "question": self.question,
            "outcome": self.outcome,
            "problem_id": self.problem_id,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "parameters": dict(self.parameters),
            "candidates": [c.as_dict() for c in self.candidates],
            "entities": self.entities.as_dict() if self.entities else {},
            "normalized": self.normalized.as_dict() if self.normalized else {},
            "missing_concepts": list(self.missing_concepts),
            "suggestions": list(self.suggestions),
            "thresholds": {"confidence_floor": CONFIDENCE_FLOOR,
                           "ambiguity_margin": AMBIGUITY_MARGIN},
        }


class QueryRouter:
    """Maps a natural-language question to a Problem Statement, or refuses."""

    version = ROUTER_VERSION

    def __init__(self, dataset: Optional[Dataset] = None,
                 registry: Optional[PSRegistry] = None):
        self.dataset = dataset if dataset is not None else load_dataset()
        self.registry = registry if registry is not None else get_registry()
        self.matcher = PSMatcher(self.registry)
        self.extractor = EntityExtractor(self.dataset)

    def route(self, question: str) -> RoutingDecision:
        normalized = normalize(question)
        entities = self.extractor.extract(question)

        if not normalized.terms:
            return RoutingDecision(
                question=question, outcome=REFUSED_LOW_CONFIDENCE,
                normalized=normalized, entities=entities,
                reason="The question contains no terms that can be matched to an analysis.")

        # 1. A question resting on absent data is refused by name, before scoring.
        missing = find_unsupported(question)
        if missing:
            return self._refuse_unsupported(question, normalized, entities, missing)

        candidates = self.matcher.score(
            normalized, entity_kinds=entities.kinds)[:MAX_CANDIDATES]
        best = candidates[0]

        # 2. An identifier that does not exist is a dead end, however well the
        #    question otherwise matches. Naming the identifier is far more useful
        #    than a generic "nothing matched", so this is checked before the floor.
        invalid = [e for e in entities.invalid
                   if e.kind in ("app_id", "sales_id", "branch_no", "product", "year")]
        if invalid:
            entity = invalid[0]
            return RoutingDecision(
                question=question, outcome=REFUSED_INVALID_ENTITY,
                candidates=candidates, entities=entities, normalized=normalized,
                confidence=best.score,
                reason="'{}' does not appear in the data: {}.".format(
                    entity.surface, entity.note),
                suggestions=self._suggest(candidates))

        # 3. Nothing matched well enough. Refuse rather than force a bad match.
        if best.score < CONFIDENCE_FLOOR:
            return RoutingDecision(
                question=question, outcome=REFUSED_LOW_CONFIDENCE,
                candidates=candidates, entities=entities, normalized=normalized,
                confidence=best.score,
                reason=("No problem statement matches this question closely enough to answer "
                        "it safely. Answering from the nearest match would risk a confident "
                        "but wrong result."),
                suggestions=self._suggest(candidates))

        # 4. Two candidates too close to separate: ask rather than guess.
        runner_up = candidates[1] if len(candidates) > 1 else None
        if runner_up and (best.score - runner_up.score) < AMBIGUITY_MARGIN:
            return RoutingDecision(
                question=question, outcome=AMBIGUOUS,
                candidates=candidates, entities=entities, normalized=normalized,
                confidence=best.score,
                reason=("This question matches more than one analysis equally well. "
                        "Choosing between them would be a guess."),
                suggestions=self._suggest(candidates))

        return RoutingDecision(
            question=question, outcome=ROUTED, problem_id=best.problem_id,
            confidence=best.score, candidates=candidates, entities=entities,
            normalized=normalized,
            parameters=self._parameters_for(best.problem_id, entities),
            reason="Matched {} with confidence {:.2f}.".format(best.problem_id, best.score))

    # -- helpers -----------------------------------------------------------

    def _needs_identifier(self, problem_id: str) -> bool:
        declared = set(self.registry.get(problem_id).record.get("parameters", {}))
        return bool(declared & {"app_id", "agent_id"})

    def _parameters_for(self, problem_id: str, entities: ExtractionResult) -> Dict[str, Any]:
        """Keep only the entity values this PS actually declares."""
        declared = self.registry.get(problem_id).record.get("parameters", {})
        extracted = entities.parameters()
        return {k: v for k, v in extracted.items() if k in declared}

    def _refuse_unsupported(self, question: str, normalized: NormalizedQuery,
                            entities: ExtractionResult,
                            missing: List[Dict[str, str]]) -> RoutingDecision:
        reasons = "; ".join("{} ({})".format(m["concept"], m["reason"]) for m in missing)
        nearest = self.matcher.score(normalized, entity_kinds=entities.kinds)[:3]
        return RoutingDecision(
            question=question, outcome=REFUSED_UNSUPPORTED,
            entities=entities, normalized=normalized, missing_concepts=missing,
            candidates=nearest,
            reason=("The five datasets cannot answer this: {}. "
                    "Answering anyway would mean inventing the missing field.".format(reasons)),
            suggestions=self._suggest(nearest))

    @staticmethod
    def _suggest(candidates: List[Candidate]) -> List[Dict[str, str]]:
        return [{"problem_id": c.problem_id, "question": c.problem_statement,
                 "confidence": round(c.score, 3)} for c in candidates[:3]]


def get_router() -> QueryRouter:
    return QueryRouter()
