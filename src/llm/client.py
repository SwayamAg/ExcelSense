"""LLM clients.

Four implementations behind one interface:

  GroqClient            Groq API (Llama 3.3 70B Versatile), ultra-fast reasoning
  AnthropicClient       Claude API, used when Anthropic SDK & key are present
  DeterministicClient   composes an answer from the evidence with no model at all
  ScriptedClient        returns canned text, for testing the layers around the model

The deterministic client is not a stub. It is the production fallback the
specification requires for API failures, and it is what keeps the system usable when
the LLM is unavailable: the facts were already established before any model was
called, so an outage costs polish, not correctness.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.ps.evidence import EvidencePackage

# Groq default model: Llama 3.3 70B Versatile
DEFAULT_GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
DEFAULT_MODEL = DEFAULT_GROQ_MODEL

DEFAULT_MAX_TOKENS = 1024        # deliberately small: executive answers are short
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_EFFORT = "low"


class LLMUnavailable(Exception):
    """Raised when a client cannot serve a request at all."""


@dataclass
class LLMResponse:
    """What a client returned, and what it cost."""
    text: str
    model: str
    provider: str
    stop_reason: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_seconds: Optional[float] = None
    error: Optional[str] = None
    refused: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text, "model": self.model, "provider": self.provider,
            "stop_reason": self.stop_reason, "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_seconds": self.latency_seconds,
            "error": self.error, "refused": self.refused,
        }


class LLMClient:
    """Interface every client honours."""

    provider = "base"
    model = "none"

    @property
    def available(self) -> bool:
        raise NotImplementedError

    def complete(self, system: str, messages: List[Dict[str, str]],
                 max_tokens: int = DEFAULT_MAX_TOKENS) -> LLMResponse:
        raise NotImplementedError


class GroqClient(LLMClient):
    """Groq API for fast LLM reasoning and summarization.

    Constructed lazily and never at import time, so a missing SDK or absent
    credentials is a runtime fallback rather than a broken application.
    """

    provider = "groq"

    def __init__(self, model: str = DEFAULT_GROQ_MODEL,
                 timeout: float = DEFAULT_TIMEOUT_SECONDS,
                 max_retries: int = DEFAULT_MAX_RETRIES,
                 api_key: Optional[str] = None):
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._api_key = api_key or os.environ.get("GROQ_API_KEY")
        self._client = None
        self._unavailable_reason: Optional[str] = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self._api_key:
            self._unavailable_reason = "GROQ_API_KEY environment variable is not set"
            raise LLMUnavailable(self._unavailable_reason)
        try:
            import groq
        except ImportError:
            self._unavailable_reason = (
                "the groq SDK is not installed (pip install groq)")
            raise LLMUnavailable(self._unavailable_reason)
        try:
            self._client = groq.Groq(
                api_key=self._api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        except Exception as exc:
            self._unavailable_reason = f"could not construct Groq client: {exc}"
            raise LLMUnavailable(self._unavailable_reason)
        return self._client

    @property
    def available(self) -> bool:
        try:
            self._ensure_client()
        except LLMUnavailable:
            return False
        return True

    @property
    def unavailable_reason(self) -> Optional[str]:
        return self._unavailable_reason

    def complete(self, system: str, messages: List[Dict[str, str]],
                 max_tokens: int = DEFAULT_MAX_TOKENS) -> LLMResponse:
        import time

        client = self._ensure_client()
        started = time.monotonic()

        formatted_messages = []
        if system:
            formatted_messages.append({"role": "system", "content": system})
        for msg in messages:
            formatted_messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
        except Exception as exc:
            return self._error_response(exc, time.monotonic() - started)

        elapsed = time.monotonic() - started

        choice = response.choices[0] if response.choices else None
        text = choice.message.content.strip() if (choice and choice.message and choice.message.content) else ""
        finish_reason = getattr(choice, "finish_reason", None) if choice else None
        usage = getattr(response, "usage", None)

        return LLMResponse(
            text=text,
            model=getattr(response, "model", self.model),
            provider=self.provider,
            stop_reason=finish_reason,
            input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            latency_seconds=elapsed,
        )

    def _error_response(self, exc: Exception, elapsed: float) -> LLMResponse:
        name = type(exc).__name__
        mapping = {
            "APITimeoutError": "the request timed out",
            "APIConnectionError": "could not reach Groq API",
            "RateLimitError": "rate limited by Groq",
            "AuthenticationError": "the Groq API key was rejected",
            "PermissionDeniedError": "the Groq API key lacks permission",
            "NotFoundError": "the requested Groq model was not found",
            "BadRequestError": "the Groq request was rejected",
        }
        detail = mapping.get(name, "Groq call failed")
        return LLMResponse(
            text="", model=self.model, provider=self.provider,
            latency_seconds=elapsed,
            error=f"{name}: {detail} ({exc})",
        )


class AnthropicClient(LLMClient):
    """Claude via the official SDK.

    Constructed lazily and never at import time, so a missing SDK or absent
    credentials is a runtime fallback rather than a broken application.
    """

    provider = "anthropic"

    def __init__(self, model: str = DEFAULT_ANTHROPIC_MODEL,
                 timeout: float = DEFAULT_TIMEOUT_SECONDS,
                 max_retries: int = DEFAULT_MAX_RETRIES,
                 effort: str = DEFAULT_EFFORT,
                 api_key: Optional[str] = None):
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.effort = effort
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None
        self._unavailable_reason: Optional[str] = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self._api_key:
            self._unavailable_reason = "ANTHROPIC_API_KEY environment variable is not set"
            raise LLMUnavailable(self._unavailable_reason)
        try:
            import anthropic
        except ImportError:
            self._unavailable_reason = (
                "the anthropic SDK is not installed (pip install anthropic)")
            raise LLMUnavailable(self._unavailable_reason)
        try:
            kwargs: Dict[str, Any] = {"timeout": self.timeout,
                                      "max_retries": self.max_retries}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = anthropic.Anthropic(**kwargs)
        except Exception as exc:                      # missing or invalid credentials
            self._unavailable_reason = "could not construct the client: {}".format(exc)
            raise LLMUnavailable(self._unavailable_reason)
        return self._client

    @property
    def available(self) -> bool:
        try:
            self._ensure_client()
        except LLMUnavailable:
            return False
        return True

    @property
    def unavailable_reason(self) -> Optional[str]:
        return self._unavailable_reason

    def complete(self, system: str, messages: List[Dict[str, str]],
                 max_tokens: int = DEFAULT_MAX_TOKENS) -> LLMResponse:
        import time

        client = self._ensure_client()
        started = time.monotonic()
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                output_config={"effort": self.effort},
            )
        except Exception as exc:
            return self._error_response(exc, time.monotonic() - started)

        elapsed = time.monotonic() - started

        # A policy decline returns HTTP 200; check before reading content.
        if getattr(response, "stop_reason", None) == "refusal":
            return LLMResponse(
                text="", model=self.model, provider=self.provider,
                stop_reason="refusal", latency_seconds=elapsed, refused=True,
                error="the model declined to answer")

        text = "".join(block.text for block in response.content
                       if getattr(block, "type", None) == "text").strip()
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=text,
            model=getattr(response, "model", self.model),
            provider=self.provider,
            stop_reason=getattr(response, "stop_reason", None),
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
            latency_seconds=elapsed,
        )

    def _error_response(self, exc: Exception, elapsed: float) -> LLMResponse:
        """Classify the failure so the caller can log it usefully."""
        name = type(exc).__name__
        mapping = {
            "APITimeoutError": "the request timed out",
            "APIConnectionError": "could not reach the API",
            "RateLimitError": "rate limited",
            "AuthenticationError": "the API key was rejected",
            "PermissionDeniedError": "the API key lacks permission",
            "NotFoundError": "the model or endpoint was not found",
            "BadRequestError": "the request was rejected",
        }
        detail = mapping.get(name, "call failed")
        return LLMResponse(
            text="", model=self.model, provider=self.provider,
            latency_seconds=elapsed,
            error="{}: {} ({})".format(name, detail, exc))


class DeterministicClient(LLMClient):
    """Composes the answer from the evidence, with no model involved.

    This is the fallback path, and it is deliberately capable rather than a
    placeholder: because every figure was established before any model was called,
    a competent answer can be written from the evidence alone. Losing the LLM costs
    fluency, not facts.
    """

    provider = "deterministic"
    model = "template"

    def __init__(self, composer: Optional[Callable[[EvidencePackage], str]] = None):
        self._composer = composer

    @property
    def available(self) -> bool:
        return True

    def complete(self, system: str, messages: List[Dict[str, str]],
                 max_tokens: int = DEFAULT_MAX_TOKENS) -> LLMResponse:
        raise LLMUnavailable(
            "the deterministic client composes from an evidence package, not a prompt; "
            "call compose() instead")

    def compose(self, evidence: EvidencePackage) -> LLMResponse:
        from src.llm.composer import compose_answer

        text = (self._composer or compose_answer)(evidence)
        return LLMResponse(text=text, model=self.model, provider=self.provider,
                           stop_reason="end_turn")


@dataclass
class ScriptedClient(LLMClient):
    """Returns prepared text. Used to test the layers around the model."""

    replies: List[str] = field(default_factory=list)
    error: Optional[Exception] = None
    provider: str = "scripted"
    model: str = "scripted"
    calls: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return True

    def complete(self, system: str, messages: List[Dict[str, str]],
                 max_tokens: int = DEFAULT_MAX_TOKENS) -> LLMResponse:
        self.calls.append({"system": system, "messages": messages,
                           "max_tokens": max_tokens})
        if self.error is not None:
            raise self.error
        if not self.replies:
            return LLMResponse(text="", model=self.model, provider=self.provider,
                               error="no scripted reply remaining")
        text = self.replies.pop(0)
        return LLMResponse(text=text, model=self.model, provider=self.provider,
                           stop_reason="end_turn", input_tokens=0, output_tokens=0)


def default_client() -> LLMClient:
    """The client the application uses: Groq when available, Anthropic as alternative, deterministic fallback otherwise."""
    if os.environ.get("TALK_TO_DATA_DISABLE_LLM") == "1":
        return DeterministicClient()

    # 1. Groq API with GROQ_API_KEY
    if os.environ.get("GROQ_API_KEY"):
        groq_client = GroqClient()
        if groq_client.available:
            return groq_client

    # 2. Anthropic API with ANTHROPIC_API_KEY
    if os.environ.get("ANTHROPIC_API_KEY"):
        anthropic_client = AnthropicClient()
        if anthropic_client.available:
            return anthropic_client

    # 3. Fallback to Groq if SDK & default setup works
    groq_client = GroqClient()
    if groq_client.available:
        return groq_client

    # 4. Fallback to Anthropic if available
    anthropic_client = AnthropicClient()
    if anthropic_client.available:
        return anthropic_client

    # 5. Deterministic composer fallback
    return DeterministicClient()
