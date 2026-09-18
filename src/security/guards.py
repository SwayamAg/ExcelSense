"""Input guards and resource limits.

The architecture does most of the security work before this module is reached. User
text never becomes a query: it selects a pre-written analysis from a fixed catalogue,
and that analysis runs against five files loaded at startup. There is no query string
to inject into, no file path to traverse, and no code path from user text to the
filesystem.

What remains is worth guarding explicitly rather than relying on that argument:

  injection     user text reaching the LLM must be framed as data, not instruction
  isolation     the loader must read the five datasets and nothing else
  exposure      personal fields must not leave the system, by any route
  limits        a question must not be able to cost an unbounded amount of work
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.data.contract import CSV_DIR, TABLE_FILES

# Resource limits. A question is a sentence, not a payload.
MAX_QUESTION_CHARS = 500
MAX_QUESTION_WORDS = 80
MAX_RESULT_ROWS = 200
MAX_RECORDS_RETURNED = 50

# Fields that identify a person. Never returned, by any route.
PERSONAL_FIELDS = ("Owner.Given.Name", "Owner.Comm.Postal.Code")

# Text that tries to issue instructions rather than ask a question. Detection is not
# the defence - the evidence validator is - but a detected attempt is worth refusing
# outright and recording, rather than passing to a model and relying on one layer.
INJECTION_PATTERNS = [
    (re.compile(r"\bignore\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\b",
                re.IGNORECASE), "instruction to ignore prior instructions"),
    (re.compile(r"\bdisregard\s+(?:all\s+|the\s+)?(?:previous|prior|above|rules?|"
                r"instructions?|constraints?)\b", re.IGNORECASE),
     "instruction to disregard the rules"),
    (re.compile(r"\b(?:you\s+are\s+now|from\s+now\s+on\s+you|act\s+as|pretend\s+(?:to\s+be|"
                r"you)|roleplay\s+as)\b", re.IGNORECASE), "attempt to reassign the role"),
    (re.compile(r"\b(?:system\s+prompt|your\s+instructions|your\s+prompt|"
                r"initial\s+instructions)\b", re.IGNORECASE),
     "attempt to read or alter the system prompt"),
    (re.compile(r"\b(?:developer|system|admin(?:istrator)?)\s*(?:mode|override|message)\b",
                re.IGNORECASE), "claim of elevated authority"),
    (re.compile(r"\bjailbreak|\bDAN\b|\bdo\s+anything\s+now\b", re.IGNORECASE),
     "known jailbreak phrasing"),
    (re.compile(r"</?(?:system|instructions?|admin)>", re.IGNORECASE),
     "forged instruction markup"),
    # Qualifying words may sit between the verb and the noun: "set the minimum
    # sample threshold to 1" is the same attempt as "set the threshold".
    (re.compile(r"\b(?:set|change|override|redefine|replace|adjust|lower|raise|"
                r"disable|remove)\s+(?:the\s+)?(?:\w+\s+){0,3}"
                r"(?:formula|rule|threshold|baseline|weighting|minimum|cutoff|guard)s?\b",
                re.IGNORECASE),
     "attempt to change a formula, threshold or business rule"),
    (re.compile(r"\b(?:say|report|answer|output|respond|tell\s+me)\s+that\s+"
                r".{0,40}\b(?:is|are|was|equals?)\b\s*\d", re.IGNORECASE),
     "instruction to state a specific figure"),
    (re.compile(r"\b(?:regardless|no\s+matter|even\s+if)\s+(?:of\s+)?(?:the\s+)?"
                r"(?:data|evidence|what)\b", re.IGNORECASE),
     "instruction to answer contrary to the evidence"),
]

# Code and query fragments. Nothing here can execute, because no user text ever
# reaches an interpreter or a query engine - but an attempt is still a signal.
CODE_PATTERNS = [
    (re.compile(r"\b(?:DROP|DELETE|TRUNCATE|INSERT|UPDATE|ALTER)\s+(?:TABLE|FROM|INTO)\b",
                re.IGNORECASE), "SQL statement"),
    (re.compile(r"(?:^|\s);\s*(?:SELECT|DROP|DELETE)\b", re.IGNORECASE),
     "chained SQL statement"),
    (re.compile(r"\b(?:__import__|eval|exec|compile|globals|locals|getattr|setattr)\s*\(",
                re.IGNORECASE), "Python execution"),
    (re.compile(r"\bos\.(?:system|popen|remove|unlink)|subprocess\.", re.IGNORECASE),
     "shell execution"),
    (re.compile(r"\.\./|\.\.\\\\|/etc/passwd|C:\\\\Windows", re.IGNORECASE),
     "path traversal"),
    (re.compile(r"<script\b|javascript:", re.IGNORECASE), "script injection"),
]


class SecurityError(Exception):
    """Raised when a request must not be served at all."""


@dataclass
class GuardResult:
    """The outcome of screening one question."""
    question: str
    safe: bool = True
    sanitized: str = ""
    findings: List[Dict[str, str]] = field(default_factory=list)

    def add(self, kind: str, detail: str) -> None:
        self.findings.append({"kind": kind, "detail": detail})
        self.safe = False

    @property
    def reason(self) -> str:
        if self.safe:
            return ""
        return "; ".join(f["detail"] for f in self.findings)

    def as_dict(self) -> Dict[str, Any]:
        return {"safe": self.safe, "findings": list(self.findings),
                "sanitized_length": len(self.sanitized)}


def _strip_control_characters(text: str) -> str:
    """Remove characters that can hide content from a reader or a check.

    Bidirectional overrides and zero-width joiners let text render one way and parse
    another, which is how a question can look benign and read as an instruction.
    """
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(
        ch for ch in normalized
        if ch in ("\n", "\t", " ") or unicodedata.category(ch)[0] != "C")


def screen_question(question: str) -> GuardResult:
    """Screen a user question before anything else touches it."""
    result = GuardResult(question=question or "")

    if question is None:
        result.add("empty", "no question was supplied")
        return result

    cleaned = _strip_control_characters(question)
    if cleaned != question:
        result.findings.append({"kind": "sanitized",
                                "detail": "control or bidirectional characters removed"})
    result.sanitized = cleaned.strip()

    if len(result.sanitized) > MAX_QUESTION_CHARS:
        result.add("too_long", "the question exceeds {} characters".format(
            MAX_QUESTION_CHARS))
        result.sanitized = result.sanitized[:MAX_QUESTION_CHARS]
    if len(result.sanitized.split()) > MAX_QUESTION_WORDS:
        result.add("too_long", "the question exceeds {} words".format(MAX_QUESTION_WORDS))

    for pattern, description in INJECTION_PATTERNS:
        if pattern.search(result.sanitized):
            result.add("prompt_injection", description)

    for pattern, description in CODE_PATTERNS:
        if pattern.search(result.sanitized):
            result.add("code_injection", description)

    return result


def neutralize_for_prompt(question: str) -> str:
    """Render a question inert for inclusion in a prompt.

    Even a screened question is quoted and labelled as data. Anything that could be
    read as a delimiter or an instruction marker is defanged, so no user text can
    close the evidence block and open an instruction.
    """
    cleaned = _strip_control_characters(question or "").strip()
    cleaned = re.sub(r"[<>]", " ", cleaned)
    cleaned = re.sub(r"\b(?:system|assistant|user)\s*:", " ", cleaned,
                     flags=re.IGNORECASE)
    cleaned = re.sub(r"```+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:MAX_QUESTION_CHARS]


# ---------------------------------------------------------------------------
# Dataset isolation
# ---------------------------------------------------------------------------

def allowed_dataset_paths(csv_dir: Optional[Path] = None) -> Set[Path]:
    """The only files this system may read."""
    base = (csv_dir or CSV_DIR).resolve()
    return {(base / name).resolve() for name in TABLE_FILES.values()}


def assert_path_allowed(path: Path, csv_dir: Optional[Path] = None) -> Path:
    """Refuse any path outside the five datasets."""
    resolved = Path(path).resolve()
    if resolved not in allowed_dataset_paths(csv_dir):
        raise SecurityError(
            "reading {} is not permitted; this system reads only the five datasets "
            "in data/csv".format(resolved.name))
    return resolved


def check_dataset_isolation(dataset) -> List[str]:
    """Confirm the loaded dataset is exactly the five declared files."""
    problems = []
    loaded = set(dataset.table_keys)
    expected = set(TABLE_FILES)
    if loaded != expected:
        problems.append("loaded tables {} do not match the five datasets {}".format(
            sorted(loaded), sorted(expected)))
    for key in loaded & expected:
        contract_file = dataset.contract.table(key).file
        if not contract_file.startswith("data/csv/"):
            problems.append("{} is sourced from outside data/csv: {}".format(
                key, contract_file))
    return problems


# ---------------------------------------------------------------------------
# Personal data
# ---------------------------------------------------------------------------

def find_personal_fields(payload: Any, path: str = "") -> List[str]:
    """Locate any personal field anywhere in a structure about to be returned."""
    found: List[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_text = str(key)
            if any(field in key_text for field in PERSONAL_FIELDS):
                found.append("{}.{}".format(path, key_text).lstrip("."))
            found.extend(find_personal_fields(value, "{}.{}".format(path, key_text)))
    elif isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            found.extend(find_personal_fields(item, "{}[{}]".format(path, index)))
    return found


def redact(payload: Any) -> Any:
    """Remove personal fields from a structure, wherever they appear."""
    if isinstance(payload, dict):
        return {k: redact(v) for k, v in payload.items()
                if not any(field in str(k) for field in PERSONAL_FIELDS)}
    if isinstance(payload, list):
        return [redact(item) for item in payload]
    if isinstance(payload, tuple):
        return tuple(redact(item) for item in payload)
    return payload
