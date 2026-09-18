"""Phase 7 - export the LLM prompt contract for inspection.

Generated from src/llm/contract.py so the published contract can never drift from
the prompt the application actually sends.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm.client import (  # noqa: E402
    DEFAULT_EFFORT, DEFAULT_MAX_RETRIES, DEFAULT_MAX_TOKENS, DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SECONDS)
from src.llm.contract import contract_as_dict  # noqa: E402
from src.llm.validator import MAX_SENTENCES, MAX_WORDS  # noqa: E402


def main() -> None:
    payload = {
        **contract_as_dict(),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "scripts/build_llm_config.py",
        "source_of_truth": "src/llm/contract.py",
        "model": {
            "id": DEFAULT_MODEL,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "effort": DEFAULT_EFFORT,
            "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
            "max_retries": DEFAULT_MAX_RETRIES,
            "sampling_parameters": (
                "none sent - temperature, top_p and top_k are rejected on this model "
                "family, and a deterministic summarizer has no use for them"),
        },
        "validation": {
            "checks": ["number_fidelity", "entity_fidelity", "causal_safety",
                       "leaked_internals", "concision", "insufficient_evidence"],
            "max_sentences": MAX_SENTENCES,
            "max_words": MAX_WORDS,
            "on_failure": "the answer is rejected and composed from the evidence instead",
        },
        "fallback": {
            "triggers": ["no model configured", "SDK missing", "credentials missing",
                         "API error or timeout", "model refusal", "validation failure"],
            "behaviour": (
                "the answer is composed deterministically from the evidence package; "
                "because every figure was established before the model was called, a "
                "model outage costs prose quality rather than correctness"),
        },
    }
    out = ROOT / "config" / "llm_prompts.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote {} ({:,} bytes)".format(out.name, out.stat().st_size))


if __name__ == "__main__":
    main()
