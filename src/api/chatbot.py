"""The chatbot.

One entry point that joins every layer built so far:

    question -> route -> execute -> evidence -> tables + chart -> summarize -> answer

Each layer already refuses rather than guesses, and this one preserves that. A
question the router will not route never reaches an executor; an execution that
cannot produce evidence never reaches the summarizer; an answer that fails
validation never reaches the user. The chatbot's job is to carry those decisions
through faithfully and explain them, not to paper over them.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.data import Dataset, load_dataset
from src.evidence import EvidenceTable, build_tables
from src.llm import Answer, Summarizer
from src.llm.client import LLMClient
from src.ps.evidence import EvidencePackage
from src.ps.executor import PSExecutor
from src.ps.registry import PSRegistry, get_registry
from src.router import QueryRouter, RoutingDecision
from src.router.router import (
    AMBIGUOUS,
    REFUSED_INVALID_ENTITY,
    REFUSED_LOW_CONFIDENCE,
    REFUSED_UNSUPPORTED,
    ROUTED,
)
from src.security.audit import AuditLog
from src.security.guards import (
    MAX_RECORDS_RETURNED,
    SecurityError,
    MAX_RESULT_ROWS,
    find_personal_fields,
    screen_question,
)
from src.visualization import ChartSpec, build_chart

CHATBOT_VERSION = "1.0.0"

# What kind of turn the user got back.
ANSWERED = "answered"
CLARIFY = "needs_clarification"
REFUSED = "refused"
FAILED = "could_not_complete"


@dataclass
class ChatResponse:
    """Everything produced for one question."""
    question: str
    outcome: str
    answer: str
    problem_id: Optional[str] = None
    security_findings: List[Dict[str, str]] = field(default_factory=list)
    confidence: float = 0.0
    tables: List[EvidenceTable] = field(default_factory=list)
    chart: Optional[ChartSpec] = None
    evidence: Optional[EvidencePackage] = None
    routing: Optional[RoutingDecision] = None
    summary: Optional[Answer] = None
    suggestions: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    turn: int = 0
    context_used: Dict[str, Any] = field(default_factory=dict)

    @property
    def answered(self) -> bool:
        return self.outcome == ANSWERED

    def as_dict(self, include_evidence: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "chatbot_version": CHATBOT_VERSION,
            "turn": self.turn,
            "question": self.question,
            "outcome": self.outcome,
            "answer": self.answer,
            "problem_id": self.problem_id,
            "confidence": round(self.confidence, 4),
            "tables": [t.as_dict() for t in self.tables],
            "chart": self.chart.as_dict() if self.chart else None,
            "suggestions": list(self.suggestions),
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "context_used": dict(self.context_used),
            "security_findings": list(self.security_findings),
        }
        if self.summary:
            payload["answer_source"] = self.summary.source
            payload["validation"] = self.summary.validation
        if include_evidence and self.evidence:
            payload["evidence"] = self.evidence.as_dict()
        return payload


@dataclass
class Turn:
    """One exchange, kept so a follow-up can be understood."""
    question: str
    outcome: str
    problem_id: Optional[str]
    entities: Dict[str, Any] = field(default_factory=dict)


class Conversation:
    """Conversation memory, deliberately shallow.

    Only entities carry forward, and only into a question that plainly lacks its
    own. Carrying an analysis forward would let "and for P22?" silently re-run
    something the user has moved on from; carrying anything forward from a refused
    turn would let a rejected premise leak into the next answer.
    """

    CARRYABLE = ("product", "state", "sales_id", "app_id", "policy_status",
                 "persistency_bucket", "plan_name", "branch_no")

    # A follow-up is short and lacks a subject of its own.
    FOLLOW_UP_MARKERS = ("what about", "and for", "how about", "same for",
                         "and in", "what of", "for that", "and that")

    def __init__(self, max_turns: int = 10):
        self.turns: List[Turn] = []
        self.max_turns = max_turns

    def record(self, response: ChatResponse) -> None:
        entities: Dict[str, Any] = {}
        if response.routing and response.routing.entities:
            for entity in response.routing.entities.entities:
                if entity.valid and entity.kind in self.CARRYABLE:
                    entities[entity.kind] = entity.value
        self.turns.append(Turn(question=response.question, outcome=response.outcome,
                               problem_id=response.problem_id, entities=entities))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    @property
    def last_answered(self) -> Optional[Turn]:
        for turn in reversed(self.turns):
            if turn.outcome == ANSWERED:
                return turn
        return None

    def looks_like_follow_up(self, question: str) -> bool:
        lowered = question.lower().strip()
        if any(marker in lowered for marker in self.FOLLOW_UP_MARKERS):
            return True
        return len(lowered.split()) <= 4 and bool(self.last_answered)

    def resolve(self, question: str) -> Dict[str, Any]:
        """Entities to reuse for this question, and why.

        Returns an empty mapping unless the question reads as a follow-up and the
        previous turn actually succeeded.
        """
        previous = self.last_answered
        if not previous or not previous.entities:
            return {}
        if not self.looks_like_follow_up(question):
            return {}
        return dict(previous.entities)

    def reset(self) -> None:
        self.turns = []


class Chatbot:
    """Question in, grounded executive answer out."""

    version = CHATBOT_VERSION

    def __init__(self, dataset: Optional[Dataset] = None,
                 registry: Optional[PSRegistry] = None,
                 llm_client: Optional[LLMClient] = None,
                 max_table_rows: int = 25,
                 audit_log: Optional[AuditLog] = None):
        self.dataset = dataset if dataset is not None else load_dataset()
        self.registry = registry if registry is not None else get_registry()
        self.router = QueryRouter(dataset=self.dataset, registry=self.registry)
        self.executor = PSExecutor(dataset=self.dataset, registry=self.registry)
        self.summarizer = Summarizer(client=llm_client)
        self.conversation = Conversation()
        self.max_table_rows = min(max_table_rows, MAX_RESULT_ROWS)
        self.audit = audit_log if audit_log is not None else AuditLog()

    # -- public ------------------------------------------------------------

    def ask(self, question: str) -> ChatResponse:
        started = time.monotonic()
        turn = len(self.conversation.turns) + 1

        if not question or not question.strip():
            return self._finish(ChatResponse(
                question=question or "", outcome=REFUSED, turn=turn,
                answer="Ask a question about the policy, claims, agent, customer or "
                       "renewal data and I will answer it from those five datasets."),
                started)

        # Screen before anything else touches the text.
        screening = screen_question(question)
        if not screening.safe:
            self.audit.record_refusal(question, screening.reason,
                                      [f["detail"] for f in screening.findings])
            return self._finish(ChatResponse(
                question=question, outcome=REFUSED, turn=turn,
                security_findings=screening.findings,
                answer=("I can only answer analytical questions about the five "
                        "datasets. That request was not one: {}. Ask about policies, "
                        "products, agents, customers, claims or renewals and I will "
                        "answer it from the data.".format(screening.reason))),
                started)
        question = screening.sanitized

        context = self.conversation.resolve(question)
        decision = self.router.route(question)

        if decision.outcome != ROUTED and context:
            # A follow-up may be unroutable only because it is elliptical. Try again
            # with the previous turn's subject restored, and say that we did.
            enriched = self._apply_context(question, context)
            retry = self.router.route(enriched)
            if retry.outcome == ROUTED:
                decision = retry

        if decision.outcome != ROUTED:
            return self._finish(self._not_routed(question, decision, turn, context), started)

        parameters = dict(decision.parameters)
        for key, value in context.items():
            mapped = {"app_id": "app_id", "sales_id": "agent_id"}.get(key)
            if mapped and mapped not in parameters:
                declared = self.registry.get(decision.problem_id).record.get("parameters", {})
                if mapped in declared:
                    parameters[mapped] = value

        evidence = self.executor.execute(decision.problem_id, parameters, question=question)
        summary = self.summarizer.summarize(evidence)

        if not evidence.is_sufficient:
            response = ChatResponse(
                question=question, outcome=FAILED, answer=summary.text,
                problem_id=decision.problem_id, confidence=decision.confidence,
                evidence=evidence, routing=decision, summary=summary, turn=turn,
                context_used=context,
                suggestions=self._suggestions(decision))
            return self._finish(response, started)

        response = ChatResponse(
            question=question, outcome=ANSWERED, answer=summary.text,
            problem_id=decision.problem_id, confidence=decision.confidence,
            tables=build_tables(evidence, self.max_table_rows),
            chart=build_chart(evidence),
            evidence=evidence, routing=decision, summary=summary, turn=turn,
            context_used=context)
        return self._finish(response, started)

    def reset(self) -> None:
        self.conversation.reset()

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _apply_context(question: str, context: Dict[str, Any]) -> str:
        """Restore an elliptical follow-up's missing subject."""
        parts = [question.strip()]
        for kind in ("product", "state", "plan_name", "sales_id", "app_id"):
            if kind in context and str(context[kind]).lower() not in question.lower():
                parts.append(str(context[kind]))
        return " ".join(parts)

    def _not_routed(self, question: str, decision: RoutingDecision,
                    turn: int, context: Dict[str, Any]) -> ChatResponse:
        """Turn a routing refusal into something a person can act on."""
        outcome = CLARIFY if decision.outcome == AMBIGUOUS else REFUSED

        # Refusals are read by the same executives as answers, so they are held to
        # the same brevity. The reasoning behind a refusal is recorded on the
        # routing decision; the user needs the outcome and a way forward.
        if decision.outcome == REFUSED_UNSUPPORTED:
            # The router's reason already names the missing data in plain words
            # ("no cost, expense or margin data exists"); the concept alone is the
            # user's own word echoed back, which explains nothing.
            reasons = "; ".join(m["reason"] for m in decision.missing_concepts)
            if reasons[:1].islower():
                reasons = reasons[0].upper() + reasons[1:]
            answer = ("{}, so this can't be answered without inventing the missing "
                      "field.".format(reasons.rstrip(".")))
        elif decision.outcome == REFUSED_INVALID_ENTITY:
            answer = decision.reason
        elif decision.outcome == AMBIGUOUS:
            options = "; ".join('"{}"'.format(s["question"])
                                for s in decision.suggestions[:3])
            answer = "Did you mean: {}?".format(options)
        else:
            answer = ("I can't match that to an analysis reliably. Try naming a "
                      "measure, such as lapse rate, premium or claims.")

        return ChatResponse(
            question=question, outcome=outcome, answer=answer,
            confidence=decision.confidence, routing=decision, turn=turn,
            context_used=context, suggestions=self._suggestions(decision))

    def _suggestions(self, decision: RoutingDecision) -> List[Dict[str, Any]]:
        suggestions = []
        for item in (decision.suggestions or [])[:3]:
            suggestions.append({"problem_id": item["problem_id"],
                                "question": item["question"]})
        if not suggestions:
            for candidate in (decision.candidates or [])[:3]:
                suggestions.append({"problem_id": candidate.problem_id,
                                    "question": candidate.problem_statement})
        return suggestions

    def _finish(self, response: ChatResponse, started: float) -> ChatResponse:
        response.elapsed_seconds = time.monotonic() - started
        self._assert_no_personal_data(response)
        self.conversation.record(response)
        self.audit.record_turn(
            response, [f["detail"] for f in response.security_findings])
        return response

    @staticmethod
    def _assert_no_personal_data(response: ChatResponse) -> None:
        """Last line of defence before anything leaves the system.

        BR-006 removes personal fields during execution. This checks the finished
        response independently, so a new output path cannot quietly reintroduce
        what the rule was written to prevent.
        """
        surfaces = [t.as_dict() for t in response.tables]
        if response.chart:
            surfaces.append(response.chart.as_dict())
        if response.evidence:
            surfaces.append({"records": response.evidence.retrieved_records,
                             "table": response.evidence.result_table})
        leaked = find_personal_fields(surfaces)
        if leaked:
            raise SecurityError(
                "personal data reached the response: {}".format(sorted(set(leaked))[:5]))


def get_chatbot(llm_client: Optional[LLMClient] = None) -> Chatbot:
    return Chatbot(llm_client=llm_client)
