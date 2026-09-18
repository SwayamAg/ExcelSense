"""Answer validation.

The prompt asks the model to stay inside the evidence. This module checks that it
did, and is the reason the system can make a grounding claim at all: a prompt is a
request, a validator is a guarantee.

Four checks, in the order they matter:

  number fidelity     every figure in the answer appears in the evidence
  entity fidelity     every named product, agent or state appears in the evidence
  causal safety       no causal claim unless the evidence establishes one
  concision           an executive answer that runs long is a failed answer

A failing answer is rejected and the deterministic fallback is used instead, so a
misbehaving model degrades the prose rather than the facts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set

from src.ps.evidence import EvidencePackage

# Numbers written in prose: 1,234.56 / 12.5% / 87 / 2.5x
NUMBER_RE = re.compile(r"(?<![\w.])(-?\d[\d,]*(?:\.\d+)?)(?![\w])")

# Figures that are linguistic rather than quantitative, and small integers that
# appear in ordinary phrasing ("the top three", "both classes").
LINGUISTIC_NUMBERS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                      "100", "1000"}

CAUSAL_CLAIM = re.compile(
    r"\b(because of|caused by|causes|causing|due to|owing to|the reason (?:is|for)|"
    r"leads? to|results? in|drives?|driven by|explains? why|as a result of|"
    r"attributable to)\b", re.IGNORECASE)

# No closing \b: several of these are prefixes ("observ" must match "observed",
# "correlat" must match "correlated"). A trailing boundary made every one of them
# unmatchable, so a properly hedged answer was rejected as an unhedged causal claim.
HEDGED_CAUSAL = re.compile(
    r"\b(?:associated with|association|correlat|appears? to|suggests?|may\b|might\b|"
    r"cannot be established|could not be established|not established|not proven|"
    r"no causal|not.{0,20}caus|observ|rather than a cause)", re.IGNORECASE)

# Phrases that reveal the machinery rather than the finding.
LEAKED_INTERNALS = re.compile(
    r"\b(evidence package|problem statement ps-|formula f-|business rule br-|"
    r"result_table|aggregates|the prompt|system prompt|as an ai|language model|"
    r"provenance|data contract)\b", re.IGNORECASE)

MAX_SENTENCES = 7
MAX_WORDS = 190


@dataclass
class ValidationIssue:
    kind: str
    detail: str
    severity: str = "error"      # 'error' rejects the answer, 'warning' does not

    def as_dict(self) -> Dict[str, str]:
        return {"kind": self.kind, "detail": self.detail, "severity": self.severity}


@dataclass
class ValidationResult:
    issues: List[ValidationIssue] = field(default_factory=list)
    numbers_checked: int = 0
    entities_checked: int = 0

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def add(self, kind: str, detail: str, severity: str = "error") -> None:
        self.issues.append(ValidationIssue(kind, detail, severity))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.is_valid,
            "issues": [i.as_dict() for i in self.issues],
            "numbers_checked": self.numbers_checked,
            "entities_checked": self.entities_checked,
        }


# ---------------------------------------------------------------------------
# Gathering what the evidence actually contains
# ---------------------------------------------------------------------------

def _walk(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def _number_forms(value: float) -> Set[str]:
    """Every reasonable way an evidence figure might legitimately be written.

    Expansion happens here and nowhere else. Expanding the *answer's* figure as well
    would let a fabricated 99.9 round to 100 and match, so the answer is always
    compared as written against these forms.

    The unsigned spelling is included because prose often carries the sign as a word
    ("56.17 lower") rather than a minus.
    """
    forms = set()
    if value != value or value in (float("inf"), float("-inf")):
        return forms
    for candidate in {float(value), abs(float(value))}:
        forms.add("{:g}".format(candidate))
        forms.add(str(candidate))
        if candidate.is_integer():
            forms.add(str(int(candidate)))
            forms.add("{:,}".format(int(candidate)))
        for places in (0, 1, 2, 3, 4):
            rounded = round(candidate, places)
            forms.add("{:.{}f}".format(rounded, places))
            forms.add("{:,.{}f}".format(rounded, places))
            if float(rounded).is_integer():
                forms.add(str(int(rounded)))
                forms.add("{:,}".format(int(rounded)))
    return {f.rstrip(".") for f in forms}


def evidence_numbers(evidence: EvidencePackage) -> Set[str]:
    """Every numeric value the evidence contains, in every acceptable spelling."""
    allowed: Set[str] = set(LINGUISTIC_NUMBERS)
    for value in _walk(evidence.as_dict()):
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            allowed |= _number_forms(float(value))
        elif isinstance(value, str):
            # Figures embedded in caveat and population text are legitimate to quote.
            # So are the bounds inside a bucket label: "18_29" and "11_15y" contain
            # 18, 29, 11 and 15, which an answer may speak as "18-29" or "11-15
            # years". Underscores and unit suffixes are word characters to the
            # number pattern, so the label is spaced out before it is scanned.
            spaced = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", value.replace("_", " "))
            for match in list(NUMBER_RE.finditer(value)) + list(NUMBER_RE.finditer(spaced)):
                literal = match.group(1)
                allowed.add(literal)
                allowed.add(literal.replace(",", ""))
                try:
                    allowed |= _number_forms(float(literal.replace(",", "")))
                except ValueError:
                    pass
    return allowed


def evidence_entities(evidence: EvidencePackage) -> Set[str]:
    """Identifier-like strings the evidence names."""
    entities: Set[str] = set()
    for value in _walk(evidence.as_dict()):
        if isinstance(value, str) and value:
            entities.add(value.strip().lower())
            for token in re.split(r"[\s,;:/]+", value):
                if token:
                    entities.add(token.strip().lower())
    return entities


# Tokens in an answer that look like they name a specific entity.
ENTITY_TOKEN = re.compile(r"\b(P\d{2}|\d{6,11})\b")


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------

def check_number_fidelity(answer: str, evidence: EvidencePackage,
                          result: ValidationResult) -> None:
    allowed = evidence_numbers(evidence)
    for match in NUMBER_RE.finditer(answer):
        literal = match.group(1)
        result.numbers_checked += 1
        bare = literal.replace(",", "")
        # Compared as written: the evidence side already holds every rounding the
        # answer may legitimately have used.
        candidates = {literal, bare, bare.lstrip("-")}
        if not (candidates & allowed):
            result.add("number_fidelity",
                       "the answer states {!r}, which is not in the evidence".format(
                           literal))


def check_entity_fidelity(answer: str, evidence: EvidencePackage,
                          result: ValidationResult) -> None:
    known = evidence_entities(evidence)
    for match in ENTITY_TOKEN.finditer(answer):
        token = match.group(1)
        result.entities_checked += 1
        if token.lower() not in known:
            result.add("entity_fidelity",
                       "the answer names {!r}, which the evidence does not".format(token))


def check_causal_safety(answer: str, evidence: EvidencePackage,
                        result: ValidationResult) -> None:
    """A causal claim is allowed only where the evidence did not disclaim one."""
    if not CAUSAL_CLAIM.search(answer):
        return
    disclaims = bool(evidence.unsupported_conclusions) or bool(
        [l for l in evidence.limitations if "caus" in l.lower()])
    if not disclaims:
        return
    if HEDGED_CAUSAL.search(answer):
        return
    result.add("causal_safety",
               "the answer asserts causation although the evidence states that cause "
               "cannot be established")


def check_no_leaked_internals(answer: str, result: ValidationResult) -> None:
    match = LEAKED_INTERNALS.search(answer)
    if match:
        result.add("leaked_internals",
                   "the answer mentions {!r}, which is machinery rather than a "
                   "finding".format(match.group(0)), severity="warning")


def check_concision(answer: str, result: ValidationResult) -> None:
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s]
    words = answer.split()
    if len(sentences) > MAX_SENTENCES:
        result.add("concision",
                   "the answer runs to {} sentences; the limit is {}".format(
                       len(sentences), MAX_SENTENCES))
    if len(words) > MAX_WORDS:
        result.add("concision",
                   "the answer runs to {} words; the limit is {}".format(
                       len(words), MAX_WORDS))
    if not words:
        result.add("empty", "the answer is empty")


def check_insufficient_is_honoured(answer: str, evidence: EvidencePackage,
                                   result: ValidationResult) -> None:
    """When the analysis failed, the answer must say so rather than fill the gap."""
    if evidence.is_sufficient:
        return
    lowered = answer.lower()
    admits = any(phrase in lowered for phrase in (
        "cannot", "can not", "could not", "unable", "not available", "no record",
        "does not", "insufficient", "not answer", "no data"))
    if not admits:
        result.add("insufficient_evidence",
                   "the evidence was insufficient but the answer does not say so")
    if result.numbers_checked == 0 and NUMBER_RE.search(answer):
        result.add("insufficient_evidence",
                   "the answer supplies figures although no analysis was completed")


def validate_answer(answer: str, evidence: EvidencePackage) -> ValidationResult:
    """Check an answer against the evidence it claims to rest on."""
    result = ValidationResult()
    if answer is None:
        result.add("empty", "no answer was produced")
        return result

    check_concision(answer, result)
    check_number_fidelity(answer, evidence, result)
    check_entity_fidelity(answer, evidence, result)
    check_causal_safety(answer, evidence, result)
    check_no_leaked_internals(answer, result)
    check_insufficient_is_honoured(answer, evidence, result)
    return result


def extract_numbers(text: str) -> List[str]:
    """The figures an answer asserts, for comparing two wordings of the same evidence."""
    return [m.group(1).replace(",", "") for m in NUMBER_RE.finditer(text or "")]
