"""Executive summarization.

Orchestrates the last step: evidence in, a short grounded answer out.

    evidence -> prompt -> model -> validate -> answer
                                       |
                                       +-- rejected -> deterministic answer

The validator sits between the model and the user, so an answer that strays outside
the evidence never reaches a reader. When it rejects, the deterministic composer
answers instead — the facts were settled before the model was called, so the cost of
a bad or absent model is prose quality, never correctness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.llm.client import (
    DEFAULT_MAX_TOKENS,
    DeterministicClient,
    LLMClient,
    LLMResponse,
    LLMUnavailable,
    default_client,
)
from src.llm.composer import compose_answer
from src.llm.contract import CONTRACT_VERSION, build_messages, system_prompt_for
from src.llm.validator import ValidationResult, validate_answer
from src.ps.evidence import EvidencePackage

SUMMARIZER_VERSION = "1.0.0"

SOURCE_MODEL = "model"
SOURCE_FALLBACK_REJECTED = "deterministic_after_rejection"
SOURCE_FALLBACK_ERROR = "deterministic_after_error"
SOURCE_FALLBACK_UNAVAILABLE = "deterministic_no_model"


@dataclass
class Answer:
    """The user-facing answer, with everything needed to defend it."""
    text: str
    source: str
    problem_id: str
    question: str
    sufficient: bool
    validation: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    latency_seconds: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    fallback_reason: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    @property
    def from_model(self) -> bool:
        return self.source == SOURCE_MODEL

    def as_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.text,
            "source": self.source,
            "problem_id": self.problem_id,
            "question": self.question,
            "sufficient": self.sufficient,
            "model": self.model,
            "provider": self.provider,
            "latency_seconds": self.latency_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "fallback_reason": self.fallback_reason,
            "validation": self.validation,
            "notes": list(self.notes),
            "summarizer_version": SUMMARIZER_VERSION,
            "contract_version": CONTRACT_VERSION,
        }


class Summarizer:
    """Turns an evidence package into a short, grounded executive answer."""

    version = SUMMARIZER_VERSION

    def __init__(self, client: Optional[LLMClient] = None,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 max_rows: int = 25):
        self.client = client if client is not None else default_client()
        self.max_tokens = max_tokens
        self.max_rows = max_rows

    def summarize(self, evidence: EvidencePackage) -> Answer:
        deterministic = compose_answer(evidence)

        if isinstance(self.client, DeterministicClient):
            return self._deterministic_answer(
                evidence, deterministic, SOURCE_FALLBACK_UNAVAILABLE,
                "no language model is configured; the answer is composed from the "
                "evidence directly")

        system = system_prompt_for(evidence)
        messages = build_messages(evidence, self.max_rows)

        try:
            response = self.client.complete(system, messages, self.max_tokens)
        except (LLMUnavailable, Exception) as exc:          # never propagate to the user
            return self._deterministic_answer(
                evidence, deterministic, SOURCE_FALLBACK_ERROR,
                "the language model could not be reached: {}".format(exc))

        if not response.ok:
            reason = response.error or "the model returned no text"
            if response.refused:
                reason = "the model declined to answer"
            return self._deterministic_answer(
                evidence, deterministic, SOURCE_FALLBACK_ERROR, reason, response)

        validation = validate_answer(response.text, evidence)
        if not validation.is_valid:
            return self._deterministic_answer(
                evidence, deterministic, SOURCE_FALLBACK_REJECTED,
                "the model's answer failed validation: {}".format(
                    "; ".join(i.detail for i in validation.errors)),
                response, validation)

        return Answer(
            text=response.text,
            source=SOURCE_MODEL,
            problem_id=evidence.detected_problem_id,
            question=evidence.question,
            sufficient=evidence.is_sufficient,
            validation=validation.as_dict(),
            model=response.model,
            provider=response.provider,
            latency_seconds=response.latency_seconds,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            notes=[i.detail for i in validation.warnings],
        )

    def _deterministic_answer(self, evidence: EvidencePackage, text: str,
                              source: str, reason: str,
                              response: Optional[LLMResponse] = None,
                              validation: Optional[ValidationResult] = None) -> Answer:
        """The fallback path. Its own output is validated too."""
        own_check = validate_answer(text, evidence)
        return Answer(
            text=text,
            source=source,
            problem_id=evidence.detected_problem_id,
            question=evidence.question,
            sufficient=evidence.is_sufficient,
            validation=own_check.as_dict(),
            model=response.model if response else None,
            provider=response.provider if response else "deterministic",
            latency_seconds=response.latency_seconds if response else None,
            fallback_reason=reason,
            notes=([i.detail for i in (validation.errors if validation else [])]
                   + [i.detail for i in own_check.warnings]),
        )


def get_summarizer(client: Optional[LLMClient] = None) -> Summarizer:
    return Summarizer(client=client)
