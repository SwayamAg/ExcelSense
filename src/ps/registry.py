"""The Problem Statement registry.

Loads the catalogue, resolves a PS by id, and validates parameters against the
declarations each PS makes. A PS whose parameters cannot be validated never
reaches the executor.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
CATALOGUE_PATH = ROOT / "config" / "problem_statements.json"
MAPPING_PATH = ROOT / "config" / "ps_logic_mapping.json"
SHARD_INDEX_PATH = ROOT / "data" / "PS" / "_index.json"


class PSNotFound(Exception):
    """Raised when a problem id is not in the catalogue."""


class ParameterError(Exception):
    """Raised when supplied parameters do not satisfy a PS's declaration."""


class ProblemStatement:
    """One PS, with convenient access to the parts the executor needs."""

    def __init__(self, record: Dict[str, Any], logic: Optional[Dict[str, Any]] = None):
        self.record = record
        self.logic = logic or {}

    def __getattr__(self, name: str) -> Any:
        try:
            return self.record[name]
        except KeyError:
            raise AttributeError(name)

    @property
    def id(self) -> str:
        return self.record["problem_id"]

    @property
    def execution(self) -> Dict[str, Any]:
        plan = self.record.get("execution")
        if not plan:
            raise PSNotFound("{} has no execution plan".format(self.id))
        return plan

    @property
    def has_execution(self) -> bool:
        return bool(self.record.get("execution"))

    @property
    def formula_ids(self) -> List[str]:
        return self.record.get("formula_ids", [])

    @property
    def business_rule_ids(self) -> List[str]:
        return self.logic.get("business_rule_ids", [])

    def validate_parameters(self, supplied: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Check supplied values against the PS's declared parameters and apply defaults."""
        supplied = dict(supplied or {})
        declared = self.record.get("parameters", {})

        unknown = set(supplied) - set(declared)
        if unknown:
            raise ParameterError(
                "{} does not accept {}; it declares {}".format(
                    self.id, sorted(unknown), sorted(declared) or "no parameters"))

        resolved: Dict[str, Any] = {}
        for name, spec in declared.items():
            if name in supplied and supplied[name] is not None:
                resolved[name] = _coerce(self.id, name, spec, supplied[name])
            elif "default" in spec:
                resolved[name] = spec["default"]
            elif spec.get("required"):
                raise ParameterError(
                    "{} requires parameter {!r}: {}".format(
                        self.id, name, spec.get("description", "")))
        return resolved


def _coerce(ps_id: str, name: str, spec: Dict[str, Any], value: Any) -> Any:
    kind = spec.get("type")
    try:
        if kind == "integer":
            value = int(value)
        elif kind == "number":
            value = float(value)
        elif kind == "string":
            value = str(value)
    except (TypeError, ValueError):
        raise ParameterError(
            "{}: parameter {!r} must be a {}, got {!r}".format(ps_id, name, kind, value))

    if kind == "enum" and value not in spec.get("values", []):
        raise ParameterError(
            "{}: parameter {!r} must be one of {}, got {!r}".format(
                ps_id, name, spec.get("values"), value))
    if "min" in spec and value < spec["min"]:
        raise ParameterError(
            "{}: parameter {!r} must be at least {}, got {}".format(
                ps_id, name, spec["min"], value))
    if "max" in spec and value > spec["max"]:
        raise ParameterError(
            "{}: parameter {!r} must be at most {}, got {}".format(
                ps_id, name, spec["max"], value))
    return value


class PSRegistry:
    """All problem statements, indexed for lookup and routing."""

    def __init__(self, catalogue: Dict[str, Any], mapping: Optional[Dict[str, Any]] = None):
        self.catalogue = catalogue
        self.version = catalogue.get("catalogue_version", "unknown")
        logic = {}
        if mapping:
            logic = {m["problem_id"]: m for m in mapping["mappings"]}
        self._problems = {
            r["problem_id"]: ProblemStatement(r, logic.get(r["problem_id"]))
            for r in catalogue["problem_statements"]
        }

    @classmethod
    def load(cls, catalogue_path: Optional[Path] = None,
             mapping_path: Optional[Path] = None) -> "PSRegistry":
        catalogue_path = catalogue_path or CATALOGUE_PATH
        mapping_path = mapping_path or MAPPING_PATH
        if not catalogue_path.exists():
            raise PSNotFound(
                "PS catalogue missing at {}; run scripts/build_ps_catalogue.py".format(
                    catalogue_path))
        mapping = None
        if mapping_path.exists():
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        return cls(json.loads(catalogue_path.read_text(encoding="utf-8")), mapping)

    def __len__(self) -> int:
        return len(self._problems)

    def __contains__(self, problem_id: str) -> bool:
        return problem_id in self._problems

    @property
    def ids(self) -> List[str]:
        return sorted(self._problems)

    def get(self, problem_id: str) -> ProblemStatement:
        if problem_id not in self._problems:
            raise PSNotFound("unknown problem id {!r}".format(problem_id))
        return self._problems[problem_id]

    def all(self) -> List[ProblemStatement]:
        return [self._problems[i] for i in self.ids]

    def executable(self) -> List[ProblemStatement]:
        return [ps for ps in self.all() if ps.has_execution]

    def by_domain(self, domain: str) -> List[ProblemStatement]:
        return [ps for ps in self.all() if ps.business_domain == domain]

    def by_tier(self, tier: str) -> List[ProblemStatement]:
        return [ps for ps in self.all() if ps.analysis_tier == tier]

    def by_table(self, table: str) -> List[ProblemStatement]:
        return [ps for ps in self.all() if table in ps.relevant_tables]

    @property
    def domains(self) -> List[str]:
        return sorted({ps.business_domain for ps in self.all()})


def get_registry() -> PSRegistry:
    return PSRegistry.load()
