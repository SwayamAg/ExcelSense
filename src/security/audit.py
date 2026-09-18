"""Audit logging.

Records what was asked and what was decided, without recording anything that would
make the log itself a disclosure. An audit trail that copies personal data out of a
protected system and into a log file has moved the problem rather than solved it.

What is recorded: the analysis chosen, the outcome, timings, rule and formula ids,
and the shape of the result. What is not: the raw question, identifiers, names,
postal codes, or any row of source data.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.security.guards import PERSONAL_FIELDS

LOGGER_NAME = "talk_to_data.audit"

# Values that identify a policy, client, agent or branch. Hashed, never written.
IDENTIFIER_RE = re.compile(r"\b\d{5,11}\b")
LOG_FIELD_LIMIT = 200


def fingerprint(value: Any) -> str:
    """A stable, non-reversible handle for an identifier.

    Enough to correlate two log lines about the same entity; not enough to recover
    which entity it was.
    """
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return "id:{}".format(digest[:12])


def scrub(text: str) -> str:
    """Replace anything identifier-shaped with a fingerprint."""
    if not text:
        return ""
    scrubbed = IDENTIFIER_RE.sub(lambda m: fingerprint(m.group(0)), str(text))
    return scrubbed[:LOG_FIELD_LIMIT]


@dataclass
class AuditEvent:
    """One recorded decision."""
    event: str
    outcome: str
    problem_id: Optional[str] = None
    confidence: Optional[float] = None
    question_fingerprint: str = ""
    question_length: int = 0
    tables_used: List[str] = field(default_factory=list)
    formulas_applied: List[str] = field(default_factory=list)
    business_rules: List[str] = field(default_factory=list)
    answer_source: Optional[str] = None
    rows_returned: int = 0
    elapsed_seconds: Optional[float] = None
    security_findings: List[str] = field(default_factory=list)
    at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "at_utc": self.at_utc,
            "event": self.event,
            "outcome": self.outcome,
            "problem_id": self.problem_id,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "question_fingerprint": self.question_fingerprint,
            "question_length": self.question_length,
            "tables_used": list(self.tables_used),
            "formulas_applied": list(self.formulas_applied),
            "business_rules": list(self.business_rules),
            "answer_source": self.answer_source,
            "rows_returned": self.rows_returned,
            "elapsed_seconds": (round(self.elapsed_seconds, 4)
                                if self.elapsed_seconds is not None else None),
            "security_findings": list(self.security_findings),
        }


class AuditLog:
    """Collects audit events, in memory and optionally through `logging`."""

    def __init__(self, emit: bool = False, max_events: int = 1000):
        self.events: List[AuditEvent] = []
        self.max_events = max_events
        self._logger = logging.getLogger(LOGGER_NAME) if emit else None

    def record(self, event: AuditEvent) -> AuditEvent:
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
        if self._logger:
            self._logger.info("audit %s", event.as_dict())
        return event

    def record_turn(self, response: Any,
                    security_findings: Optional[List[str]] = None) -> AuditEvent:
        """Record a chatbot turn, taking only what is safe to keep."""
        evidence = getattr(response, "evidence", None)
        summary = getattr(response, "summary", None)
        return self.record(AuditEvent(
            event="question",
            outcome=getattr(response, "outcome", "unknown"),
            problem_id=getattr(response, "problem_id", None),
            confidence=getattr(response, "confidence", None),
            question_fingerprint=fingerprint(getattr(response, "question", "")),
            question_length=len(getattr(response, "question", "") or ""),
            tables_used=list(getattr(evidence, "tables_used", []) or []),
            formulas_applied=list(getattr(evidence, "formulas_applied", []) or []),
            business_rules=[r.get("rule_id") for r in
                            (getattr(evidence, "business_rules_applied", []) or [])
                            if r.get("rule_id")],
            answer_source=getattr(summary, "source", None),
            rows_returned=len(getattr(evidence, "result_table", []) or []),
            elapsed_seconds=getattr(response, "elapsed_seconds", None),
            security_findings=list(security_findings or []),
        ))

    def record_refusal(self, question: str, reason: str,
                       findings: Optional[List[str]] = None) -> AuditEvent:
        return self.record(AuditEvent(
            event="refused",
            outcome="blocked",
            question_fingerprint=fingerprint(question),
            question_length=len(question or ""),
            security_findings=list(findings or [reason]),
        ))

    # -- inspection --------------------------------------------------------

    @property
    def blocked(self) -> List[AuditEvent]:
        return [e for e in self.events if e.event == "refused"]

    def as_dicts(self) -> List[Dict[str, Any]]:
        return [e.as_dict() for e in self.events]

    def leaks(self) -> List[str]:
        """Anything in the log that should not be there.

        The audit trail is checked against the same standard as the answers: a log
        that has copied a name or a postal code out of the data has become a
        disclosure of its own.
        """
        problems = []
        for index, event in enumerate(self.events):
            payload = event.as_dict()
            text = str(payload)
            for field_name in PERSONAL_FIELDS:
                if field_name in text:
                    problems.append("event {} records {}".format(index, field_name))
            for match in IDENTIFIER_RE.finditer(text):
                problems.append("event {} records a raw identifier {!r}".format(
                    index, match.group(0)))
        return problems


_DEFAULT_LOG = AuditLog()


def get_audit_log() -> AuditLog:
    return _DEFAULT_LOG
