"""The PS execution engine.

Turns a Problem Statement into an evidence package, deterministically and without
any LLM involvement. Given the same data and parameters it produces the same
evidence every time.

Failure is a first-class outcome: when the data cannot support the question, the
executor returns an evidence package marked insufficient with the reason, rather
than an empty result that reads like an answer.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from src.analysis.patterns import PATTERNS, PatternError
from src.data.query_engine import QueryError
from src.logic import FormulaError, RuleError
from src.data import Dataset, load_dataset
from src.ps.cache import ResultCache
from src.ps.evidence import EvidencePackage
from src.ps.registry import ParameterError, PSRegistry, ProblemStatement, get_registry

EXECUTOR_VERSION = "1.0.0"


class ExecutionError(Exception):
    """Raised when a PS cannot be executed because its plan is malformed."""


class PSExecutor:
    """Executes problem statements against the loaded datasets."""

    version = EXECUTOR_VERSION

    def __init__(self, dataset: Optional[Dataset] = None,
                 registry: Optional[PSRegistry] = None,
                 cache: Optional[ResultCache] = None,
                 use_cache: bool = True):
        self.dataset = dataset if dataset is not None else load_dataset()
        self.registry = registry if registry is not None else get_registry()
        if cache is not None:
            self.cache = cache
        else:
            self.cache = ResultCache.for_dataset(
                self.dataset, registry=self.registry,
                executor_version=self.version, enabled=use_cache)

    # -- execution ---------------------------------------------------------

    def execute(self, problem_id: str, parameters: Optional[Dict[str, Any]] = None,
                question: str = "") -> EvidencePackage:
        """Produce the evidence for one problem statement.

        A cache hit is only taken after the parameters have been validated, so an
        invalid call is still rejected by the PS that declared them rather than
        being answered from a neighbouring entry.
        """
        ps = self.registry.get(problem_id)
        try:
            validated = ps.validate_parameters(parameters)
        except ParameterError:
            validated = None
        if validated is not None:
            cached = self.cache.get(problem_id, validated, question=question)
            if cached is not None:
                return cached
        evidence = self._execute_uncached(ps, parameters, question)
        if validated is not None and evidence.is_sufficient:
            self.cache.put(problem_id, validated, evidence)
        return evidence

    def _execute_uncached(self, ps: ProblemStatement,
                          parameters: Optional[Dict[str, Any]],
                          question: str) -> EvidencePackage:
        evidence = self._new_package(ps, question)

        try:
            evidence.parameters = ps.validate_parameters(parameters)
        except ParameterError as exc:
            evidence.mark_insufficient(str(exc), kind="parameter_error")
            return evidence

        if not ps.has_execution:
            evidence.mark_insufficient(
                "{} has no execution plan, so it cannot be run".format(ps.id),
                kind="no_execution_plan")
            return evidence

        plan = self._resolve_plan(ps, evidence.parameters)
        pattern = plan.get("pattern")
        if pattern not in PATTERNS:
            evidence.mark_insufficient(
                "unknown analytical pattern {!r}".format(pattern), kind="unknown_pattern")
            return evidence

        try:
            PATTERNS[pattern](self.dataset, plan, evidence)
        except (PatternError, QueryError, RuleError, FormulaError,
                KeyError, ValueError, TypeError) as exc:
            evidence.mark_insufficient(
                "the analysis could not be completed: {}".format(exc),
                kind="execution_error", pattern=pattern)
            return evidence

        self._finalise(ps, evidence)
        return evidence

    def execute_fresh(self, problem_id: str,
                      parameters: Optional[Dict[str, Any]] = None,
                      question: str = "") -> EvidencePackage:
        """Execute without consulting or populating the cache.

        Used by the reproducibility and regression tests, which must observe the
        computation itself rather than a record of an earlier one.
        """
        return self._execute_uncached(self.registry.get(problem_id), parameters,
                                      question)

    def execute_all(self, parameters: Optional[Dict[str, Dict[str, Any]]] = None
                    ) -> Dict[str, EvidencePackage]:
        """Run every executable PS. Used by the regression suite."""
        parameters = parameters or {}
        return {ps.id: self.execute(ps.id, parameters.get(ps.id))
                for ps in self.registry.executable()}

    # -- internals ---------------------------------------------------------

    def _new_package(self, ps: ProblemStatement, question: str) -> EvidencePackage:
        return EvidencePackage(
            question=question or ps.problem_statement,
            detected_problem_id=ps.id,
            problem_statement=ps.problem_statement,
            population=ps.population,
            caveat=ps.caveat,
            expected_answer_type=ps.expected_answer_type,
            answer_template=ps.answer_template,
            columns_used=list(ps.relevant_columns),
        )

    def _resolve_plan(self, ps: ProblemStatement, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute parameter placeholders into the plan.

        A placeholder is the string '$name'. Only declared parameters can be
        substituted, so a plan cannot reach for a value the PS never declared.
        """
        plan = copy.deepcopy(ps.execution)
        return _substitute(plan, parameters)

    def _finalise(self, ps: ProblemStatement, evidence: EvidencePackage) -> None:
        """Attach the PS's declared framing to the computed evidence."""
        for limitation in ps.record.get("limitations", []):
            evidence.add_limitation(limitation)

        declared = ps.record.get("evidence_requirements", {})
        for note in declared.get("data_quality_notes", []):
            if note not in evidence.data_quality_notes:
                evidence.data_quality_notes.append(note)
        for limitation in declared.get("limitations", []):
            evidence.add_limitation(limitation)

        for finding_id in evidence.data_quality_notes:
            try:
                finding = self.dataset.contract.finding(finding_id)
            except Exception:
                continue
            evidence.supporting_metrics.setdefault("data_quality_detail", {})[finding_id] = {
                "severity": finding["severity"], "finding": finding["finding"],
                "handling": finding["handling"],
            }

        if ps.analysis_tier == "diagnostic":
            evidence.add_unsupported(
                "This analysis reports observable differences. The five datasets contain no "
                "reason codes, payment history, contact records or intervention history, so no "
                "causal explanation can be established from them.")

        if ps.record.get("visualization_recommended"):
            evidence.visualization = {
                "type": ps.record.get("visualization_type"),
                "recommended": True,
                "source": "evidence.result_table",
            }

        for table in ps.relevant_tables:
            if table not in evidence.tables_used:
                evidence.tables_used.append(table)

        evidence.provenance.setdefault("executor_version", self.version)
        evidence.provenance.setdefault("catalogue_version", self.registry.version)
        evidence.provenance.setdefault("contract_version", self.dataset.contract.version)


def _substitute(node: Any, parameters: Dict[str, Any]) -> Any:
    if isinstance(node, dict):
        return {k: _substitute(v, parameters) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute(v, parameters) for v in node]
    if isinstance(node, tuple):
        return tuple(_substitute(v, parameters) for v in node)
    if isinstance(node, str) and node.startswith("$"):
        name = node[1:]
        if name in parameters:
            return parameters[name]
        return None
    return node


def get_executor() -> PSExecutor:
    return PSExecutor()
