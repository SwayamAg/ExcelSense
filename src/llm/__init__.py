"""LLM reasoning layer: explains evidence, never establishes it."""
from .client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    AnthropicClient,
    DeterministicClient,
    GroqClient,
    LLMClient,
    LLMResponse,
    LLMUnavailable,
    ScriptedClient,
    default_client,
)
from .composer import compose_answer
from .contract import CONTRACT_VERSION, build_messages, contract_as_dict, system_prompt_for
from .summarizer import (
    SOURCE_FALLBACK_ERROR,
    SOURCE_FALLBACK_REJECTED,
    SOURCE_FALLBACK_UNAVAILABLE,
    SOURCE_MODEL,
    Answer,
    Summarizer,
    get_summarizer,
)
from .validator import ValidationResult, extract_numbers, validate_answer

__all__ = [
    "GroqClient", "AnthropicClient", "DeterministicClient", "ScriptedClient", "LLMClient",
    "LLMResponse", "LLMUnavailable", "default_client", "DEFAULT_MODEL",
    "DEFAULT_MAX_TOKENS",
    "compose_answer", "build_messages", "system_prompt_for", "contract_as_dict",
    "CONTRACT_VERSION",
    "Summarizer", "Answer", "get_summarizer",
    "SOURCE_MODEL", "SOURCE_FALLBACK_REJECTED", "SOURCE_FALLBACK_ERROR",
    "SOURCE_FALLBACK_UNAVAILABLE",
    "validate_answer", "ValidationResult", "extract_numbers",
]
