"""Entity extraction.

Pulls the identifiers, codes and values a question supplies, and validates each one
against the actual data. An AppID that is not in Policy_Details is reported as
not-found rather than passed on to an executor that will simply return nothing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.data import Dataset

# Identifier shapes, taken from the profiled data rather than assumed:
# AppID is 11 digits, SALES_ID 6, ClientID 9-11, Branch_No 5, Prod_No 'P' + 2 digits.
APPID_RE = re.compile(r"\b(\d{11})\b")
SALES_ID_RE = re.compile(r"\b(\d{6})\b")
BRANCH_RE = re.compile(r"\bbranch\s*(?:no\.?|number|#)?\s*(\d{5})\b", re.IGNORECASE)
PROD_RE = re.compile(r"\b(P\d{2})\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(20\d{2})\b")
QUARTER_RE = re.compile(r"\b(20\d{2})\s*[-\s]?q([1-4])\b", re.IGNORECASE)
MONTH_RE = re.compile(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])\b")
TOP_N_RE = re.compile(r"\b(?:top|bottom|first|last|best|worst)\s+(\d{1,3})\b", re.IGNORECASE)

STATUS_TERMS = {
    "lapsed": "Lapsed", "lapse": "Lapsed",
    "surrendered": "Surrendered", "surrender": "Surrendered",
    "death": "Death", "died": "Death", "deceased": "Death",
    "in force": "Premium paying (regular)", "inforce": "Premium paying (regular)",
    "premium paying": "Premium paying (regular)", "active": "Premium paying (regular)",
}

RENEWAL_TERMS = {"paid": "Paid", "pending": "Pending", "unpaid": "Pending",
                 "outstanding": "Pending"}

BUCKET_TERMS = {"13th": "13th Month", "25th": "25th Month", "37th": "37th Month"}


@dataclass
class Entity:
    kind: str
    value: Any
    surface: str
    valid: bool
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "value": self.value, "surface": self.surface,
                "valid": self.valid, "note": self.note}


@dataclass
class ExtractionResult:
    entities: List[Entity] = field(default_factory=list)

    def of_kind(self, kind: str) -> List[Entity]:
        return [e for e in self.entities if e.kind == kind]

    def first(self, kind: str) -> Optional[Entity]:
        found = self.of_kind(kind)
        return found[0] if found else None

    @property
    def kinds(self) -> List[str]:
        return sorted({e.kind for e in self.entities})

    @property
    def invalid(self) -> List[Entity]:
        return [e for e in self.entities if not e.valid]

    def as_dict(self) -> Dict[str, Any]:
        return {"entities": [e.as_dict() for e in self.entities], "kinds": self.kinds}

    def parameters(self) -> Dict[str, Any]:
        """Entity values in the shape the executor's parameters expect."""
        params: Dict[str, Any] = {}
        app = self.first("app_id")
        if app and app.valid:
            params["app_id"] = app.value
        agent = self.first("sales_id")
        if agent and agent.valid:
            params["agent_id"] = agent.value
        n = self.first("top_n")
        if n:
            params["n"] = n.value
        grain = self.first("granularity")
        if grain:
            params["granularity"] = grain.value
        return params


class EntityExtractor:
    """Finds and validates entities, using the loaded data as the authority."""

    def __init__(self, dataset: Dataset):
        self._ds = dataset
        policy = dataset.table("policy")
        self._app_ids = set(policy.ident("AppID"))
        self._sales_ids = set(dataset.table("sales").ident("SALES_ID"))
        self._branches = set(policy.ident("Branch_No"))
        self._products = set(policy.col("Prod_No"))
        self._plans = {p.lower(): p for p in set(policy.col("Plan.Name"))}
        self._states = {s.lower(): s for s in set(dataset.table("owner").col("Owner.Comm.State"))
                        if isinstance(s, str)}
        self._client_ids = set(dataset.table("owner").ident("ClientID"))

    def extract(self, text: str) -> ExtractionResult:
        result = ExtractionResult()
        lowered = text.lower()

        for match in APPID_RE.finditer(text):
            value = match.group(1)
            if value in self._app_ids:
                result.entities.append(Entity("app_id", value, value, True))
            elif value in self._client_ids:
                result.entities.append(Entity("client_id", value, value, True))
            else:
                result.entities.append(Entity(
                    "app_id", value, value, False,
                    "no policy or client carries this 11-digit identifier"))

        for match in BRANCH_RE.finditer(text):
            value = match.group(1)
            result.entities.append(Entity(
                "branch_no", value, match.group(0), value in self._branches,
                "" if value in self._branches else "no branch carries this number"))

        # A six-digit number is only an agent id if it is not already claimed as a
        # branch number in this question.
        claimed = {e.value for e in result.entities}
        for match in SALES_ID_RE.finditer(text):
            value = match.group(1)
            if value in claimed:
                continue
            result.entities.append(Entity(
                "sales_id", value, value, value in self._sales_ids,
                "" if value in self._sales_ids else "no agent carries this SALES_ID"))

        for match in PROD_RE.finditer(text):
            value = match.group(1).upper()
            result.entities.append(Entity(
                "product", value, match.group(0), value in self._products,
                "" if value in self._products else "no product carries this code"))

        for plan_lower, plan in self._plans.items():
            if plan_lower in lowered:
                result.entities.append(Entity("plan_name", plan, plan, True))

        for state_lower, state in self._states.items():
            if re.search(r"\b{}\b".format(re.escape(state_lower)), lowered):
                result.entities.append(Entity("state", state, state, True))

        for term, status in STATUS_TERMS.items():
            if re.search(r"\b{}\b".format(re.escape(term)), lowered):
                if not any(e.kind == "policy_status" and e.value == status
                           for e in result.entities):
                    result.entities.append(Entity("policy_status", status, term, True))

        for term, status in RENEWAL_TERMS.items():
            if re.search(r"\brenewal[s]?\b.*\b{}\b|\b{}\b.*\brenewal".format(
                    re.escape(term), re.escape(term)), lowered):
                result.entities.append(Entity("renewal_status", status, term, True))
                break

        for term, bucket in BUCKET_TERMS.items():
            if re.search(r"\b{}\s*month\b".format(re.escape(term)), lowered):
                result.entities.append(Entity("persistency_bucket", bucket, term, True))

        for match in QUARTER_RE.finditer(text):
            value = "{}-Q{}".format(match.group(1), match.group(2))
            result.entities.append(Entity("period", value, match.group(0), True))
            result.entities.append(Entity("granularity", "quarter", match.group(0), True))
        for match in MONTH_RE.finditer(text):
            value = "{}-{:02d}".format(match.group(1), int(match.group(2)))
            result.entities.append(Entity("period", value, match.group(0), True))
            result.entities.append(Entity("granularity", "month", match.group(0), True))
        for match in YEAR_RE.finditer(text):
            year = match.group(1)
            if any(e.kind == "period" and e.value.startswith(year) for e in result.entities):
                continue
            valid = 2023 <= int(year) <= 2026
            result.entities.append(Entity(
                "year", year, year, valid,
                "" if valid else "the extract covers 2023 to 2026 only"))

        for match in TOP_N_RE.finditer(text):
            result.entities.append(Entity("top_n", int(match.group(1)), match.group(0), True))

        for term, grain in (("quarterly", "quarter"), ("monthly", "month"),
                            ("yearly", "year"), ("annually", "year")):
            if term in lowered and not result.of_kind("granularity"):
                result.entities.append(Entity("granularity", grain, term, True))

        return result
