"""The evidence package - the factual contract handed to the LLM.

Specification section 7 defines what an execution must record. This module builds
that package and nothing else: it holds facts the deterministic layer established,
and it is the only thing the LLM is ever allowed to reason over.

The package distinguishes four kinds of statement, because collapsing them is how
an analysis turns an observation into a claim it cannot support:

    observed     read directly from a CSV
    calculated   produced by the formula engine, with its inputs
    compared     a difference against a declared baseline
    unsupported  explicitly named as beyond what the data establishes
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0.0"


@dataclass
class Insufficient:
    """Why an execution could not produce an answer."""
    reason: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"reason": self.reason, "detail": self.detail}


@dataclass
class EvidencePackage:
    """Everything an answer rests on."""

    question: str
    detected_problem_id: str
    problem_statement: str

    # What was touched
    tables_used: List[str] = field(default_factory=list)
    columns_used: List[str] = field(default_factory=list)
    filters: List[Dict[str, Any]] = field(default_factory=list)
    join_path: List[str] = field(default_factory=list)

    # What was found
    population: str = ""
    population_size: Optional[int] = None
    retrieved_records: List[Dict[str, Any]] = field(default_factory=list)
    aggregates: Dict[str, Any] = field(default_factory=dict)
    result_table: List[Dict[str, Any]] = field(default_factory=list)
    result_columns: List[str] = field(default_factory=list)

    # How it was computed
    calculations: List[Dict[str, Any]] = field(default_factory=list)
    formulas_applied: List[str] = field(default_factory=list)
    business_rules_applied: List[Dict[str, Any]] = field(default_factory=list)

    # Analytical framing
    comparison_baseline: Optional[Dict[str, Any]] = None
    ranking: Optional[Dict[str, Any]] = None
    supporting_metrics: Dict[str, Any] = field(default_factory=dict)
    sample_sizes: Dict[str, Any] = field(default_factory=dict)

    # Honesty
    data_quality_notes: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    unsupported_conclusions: List[str] = field(default_factory=list)
    caveat: str = ""

    # Presentation hints (consumed in Phase 6)
    expected_answer_type: str = ""
    answer_template: str = ""
    visualization: Optional[Dict[str, Any]] = None

    # Execution record
    provenance: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    insufficient: Optional[Insufficient] = None
    warnings: List[str] = field(default_factory=list)
    generated_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    @property
    def is_sufficient(self) -> bool:
        return self.insufficient is None

    def add_calculation(self, label: str, result: Any, note: str = "") -> Any:
        """Record a FormulaResult and return its value, so callers cannot skip recording."""
        evidence = result.as_evidence()
        entry = {"label": label, **evidence}
        if note:
            entry["note"] = note
        self.calculations.append(entry)
        if evidence["formula_id"] not in self.formulas_applied:
            self.formulas_applied.append(evidence["formula_id"])
        for warning in evidence.get("warnings", []):
            message = "{}: {}".format(label, warning)
            if message not in self.warnings:
                self.warnings.append(message)
        return result.value

    def add_rule(self, application: Any) -> None:
        self.business_rules_applied.append(application.as_evidence())

    def add_limitation(self, text: str) -> None:
        if text and text not in self.limitations:
            self.limitations.append(text)

    def add_unsupported(self, text: str) -> None:
        if text and text not in self.unsupported_conclusions:
            self.unsupported_conclusions.append(text)

    def mark_insufficient(self, reason: str, **detail) -> None:
        self.insufficient = Insufficient(reason=reason, detail=detail)

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": self.generated_at_utc,
            "question": self.question,
            "detected_problem_id": self.detected_problem_id,
            "problem_statement": self.problem_statement,
            "tables_used": list(self.tables_used),
            "columns_used": list(self.columns_used),
            "filters": list(self.filters),
            "join_path": list(self.join_path),
            "population": self.population,
            "population_size": self.population_size,
            "retrieved_records": list(self.retrieved_records),
            "aggregates": dict(self.aggregates),
            "result_table": list(self.result_table),
            "result_columns": list(self.result_columns),
            "calculations": list(self.calculations),
            "formulas_applied": list(self.formulas_applied),
            "business_rules_applied": list(self.business_rules_applied),
            "comparison_baseline": self.comparison_baseline,
            "ranking": self.ranking,
            "supporting_metrics": dict(self.supporting_metrics),
            "sample_sizes": dict(self.sample_sizes),
            "data_quality_notes": list(self.data_quality_notes),
            "limitations": list(self.limitations),
            "unsupported_conclusions": list(self.unsupported_conclusions),
            "caveat": self.caveat,
            "expected_answer_type": self.expected_answer_type,
            "answer_template": self.answer_template,
            "visualization": self.visualization,
            "parameters": dict(self.parameters),
            "provenance": dict(self.provenance),
            "warnings": list(self.warnings),
            "sufficient": self.is_sufficient,
        }
        if self.insufficient is not None:
            payload["insufficient"] = self.insufficient.as_dict()
        return payload

    def summary(self) -> str:
        """One line for logs; never the user-facing answer."""
        if not self.is_sufficient:
            return "{}: insufficient evidence - {}".format(
                self.detected_problem_id, self.insufficient.reason)
        return "{}: {} rows, {} calculations, {} rules".format(
            self.detected_problem_id, len(self.result_table) or self.population_size or 0,
            len(self.calculations), len(self.business_rules_applied))
