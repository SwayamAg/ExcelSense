"""
ollama_client.py
================
Unified LLM client supporting both local Ollama and Groq cloud API.

Priority order:
  1. Local Ollama  (http://localhost:11434)  — zero-latency, private
  2. Groq Cloud API (GROQ_API_KEY env / st.secrets) — free-tier fallback

All existing callers (import ollama_client as _oc) continue to work
unchanged. New helpers:
  - is_groq_available()        → bool
  - is_llm_available()         → bool  (Ollama OR Groq)
  - get_active_backend()       → "ollama" | "groq" | "none"
  - get_groq_model()           → str
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import List, Optional

# ---------------------------------------------------------------------------
# Ollama config
# ---------------------------------------------------------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# ---------------------------------------------------------------------------
# Groq config
# ---------------------------------------------------------------------------
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def _get_groq_key() -> Optional[str]:
    """Read Groq API key from env or Streamlit secrets (if available)."""
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------

def is_ollama_available(timeout: float = 2.0) -> bool:
    """Check if local Ollama daemon is reachable."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def is_groq_available() -> bool:
    """Return True if a Groq API key is configured."""
    return bool(_get_groq_key())


def is_llm_available(timeout: float = 2.0) -> bool:
    """Return True if any LLM backend (Ollama or Groq) is available."""
    return is_ollama_available(timeout=timeout) or is_groq_available()


def get_active_backend(timeout: float = 2.0) -> str:
    """Return which backend will be used: 'ollama', 'groq', or 'none'."""
    if is_ollama_available(timeout=timeout):
        return "ollama"
    if is_groq_available():
        return "groq"
    return "none"


# ---------------------------------------------------------------------------
# Ollama model helpers
# ---------------------------------------------------------------------------

def get_available_models(timeout: float = 2.0) -> List[str]:
    """Return list of model names available locally in Ollama."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return [
                    m.get("name") or m.get("model")
                    for m in data.get("models", [])
                    if m.get("name") or m.get("model")
                ]
    except Exception:
        pass
    return []


def get_default_model() -> str:
    """Get preferred or first available local Ollama model."""
    env_model = os.environ.get("OLLAMA_MODEL")
    if env_model:
        return env_model
    models = get_available_models()
    if models:
        for preferred in (
            "qwen2.5:14b", "qwen2.5:7b", "gemma4-e2b-it-qat:latest",
            "llama3:latest", "mistral:latest",
        ):
            if preferred in models:
                return preferred
        return models[0]
    return "qwen2.5:14b"


def get_groq_model() -> str:
    """Return the configured Groq model name."""
    return GROQ_DEFAULT_MODEL


DEFAULT_MODEL = get_default_model()


# ---------------------------------------------------------------------------
# Groq inference
# ---------------------------------------------------------------------------

def _generate_groq_response(
    prompt: str,
    system_prompt: Optional[str] = None,
    timeout: int = 60,
) -> Optional[str]:
    """Call Groq /openai/v1/chat/completions and return response string."""
    key = _get_groq_key()
    if not key:
        return None

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": GROQ_DEFAULT_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1024,
    }

    try:
        req = urllib.request.Request(
            GROQ_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"].strip()
            return content if content else None
    except Exception as e:
        print(f"[Groq Error] {e}")
        return None


# ---------------------------------------------------------------------------
# Unified inference — Ollama first, Groq fallback
# ---------------------------------------------------------------------------

def generate_llm_response(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    timeout: int = 120,
) -> Optional[str]:
    """Generate a response using the best available backend.

    Priority: local Ollama → Groq cloud API → None.
    All callers are unchanged; Groq kicks in transparently when Ollama
    is offline (e.g., on Streamlit Community Cloud).
    """
    # 1. Try Ollama
    if is_ollama_available():
        result = _generate_ollama_response(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            timeout=timeout,
        )
        if result:
            return result

    # 2. Groq fallback
    if is_groq_available():
        print("[LLM] Ollama unavailable — falling back to Groq.")
        return _generate_groq_response(
            prompt=prompt,
            system_prompt=system_prompt,
            timeout=min(timeout, 60),
        )

    return None


def _generate_ollama_response(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    timeout: int = 120,
) -> Optional[str]:
    """Internal: call Ollama /api/chat."""
    target_model = model or get_default_model()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": target_model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 1024,
        },
    }

    try:
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            msg = data.get("message", {})
            content = msg.get("content", "").strip()
            if not content and msg.get("thinking"):
                content = msg.get("thinking", "").strip()
            return content if content else None
    except Exception as e:
        print(f"[Ollama Error with model {target_model}] {e}")
        if model and model != get_default_model():
            fallback = get_default_model()
            if fallback != target_model:
                print(f"[Ollama] Retrying with fallback model {fallback}...")
                return _generate_ollama_response(
                    prompt, system_prompt=system_prompt,
                    model=fallback, timeout=timeout,
                )
        return None

