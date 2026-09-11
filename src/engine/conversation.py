"""
conversation.py
===============
Multi-turn conversation state and follow-up resolution.

The business engine is stateless: `ask(question)` takes one string and has no
memory. A chat surface needs a follow-up like "why is *their* persistency so
low?" to resolve "their" against what the previous turn was about. This module
supplies that layer, and nothing else.

Two resolution paths:

  rules   inherit the previous turn's focus entity when the new question names
          no entity of its own and reads as a continuation. Always available,
          no dependencies, used for turn 1 and whenever the LLM path fails.

  llm     a local Ollama model reads the question plus the conversation state
          and returns *constrained JSON* naming a metric, dimension and
          operation drawn from the registries. It never sees raw data and never
          emits a number, so a model error surfaces as a visibly wrong
          interpretation the user can correct - never as a wrong figure.

Every LLM field is validated against `analytics_core`'s registries before it is
trusted. Anything unrecognised drops the whole response back to the rule path
and marks the turn degraded, which the UI is expected to tell the user about.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
for _sub in ["src", "src/analytics", "src/engine", "src/retrieval", "src/llm"]:
    _p = os.path.join(ROOT_DIR, _sub)
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import analytics_core as ac

# Auto-execute an LLM resolution at or above this confidence. Below it the
# engine asks the user rather than guessing.
CONFIDENCE_THRESHOLD = 0.75

# Canonical operation names the resolver is allowed to choose. Deliberately the
# short list from `run_operation`'s primary spellings, not every alias.
RESOLVABLE_OPS = [
    "metric_by_dimension", "compare_to_baseline", "rank_entities",
    "composite_score", "driver_scan", "concentration", "outliers",
    "crosstab_metric", "portfolio_overview", "entity_shortlist",
    "segment_profile", "multi_metric_table", "top_entity_concentration",
    "correlation_analysis",
]

# Entity id shapes that actually occur in the 5 production tables. The legacy
# EntityResolver still carries CUST/POL/AGT/CLM patterns from the previous
# dataset, none of which match anything here.
ENTITY_PATTERNS: List[Tuple[str, str, str]] = [
    ("agent", r"\b(?:sales[\s_]?id|salesperson|sales person|agent|advisor|rep)\s*#?\s*(\d{4,7})\b", "Agent"),
    ("policy", r"\b(?:appid|app id|policy(?:\s*(?:no|number|id))?)\s*#?\s*(\d{8,12})\b", "Policy"),
    ("customer", r"\b(?:clientid|client id|client|customer|owner)\s*#?\s*(\d{8,12})\b", "Customer"),
    ("claim", r"\b(?:claim(?:\s*(?:no|number|id))?)\s*#?\s*(\d{1,8})\b", "Claim"),
]

# A question carrying one of these, and no entity of its own, is a continuation.
_CONTINUATION_CUES = re.compile(
    r"\b(they|them|their|theirs|he|him|his|she|her|hers|it|its|this|that|those|these|"
    r"the same|same one|that one|instead|as well|too|also|then|what about|how about|"
    r"compared|versus|vs)\b",
    re.IGNORECASE,
)

# Leading verbs that imply "keep doing what we were doing".
_ELLIPSIS_CUES = re.compile(
    r"^\s*(and |but |ok(?:ay)? |now |just |only |what about|how about|why|show|list|"
    r"break|split|drill|compare|rank|sort)\b",
    re.IGNORECASE,
)

# Phrases that explicitly move AWAY from the focus entity. "how do the other
# agents compare" is about everyone except the one in focus, so carrying the
# focus entity forward answers the opposite question. These override both the
# continuation cues and the model's own inherit_entity flag.
_CONTRAST_CUES = re.compile(
    r"\b(other|others|rest of|everyone else|every one else|remaining|"
    r"different|another|across all|all agents|all sales|whole book|"
    r"rest of the|anyone else|someone else|peers|peer group)\b",
    re.IGNORECASE,
)


def reads_as_contrast(question: str) -> bool:
    """Whether the question points away from the focus entity rather than at it."""
    return bool(_CONTRAST_CUES.search(question or ""))


# =====================================================================
# STATE
# =====================================================================

@dataclass
class FocusEntity:
    """The record a turn was about, carried forward to the next turn."""
    type: str                      # agent | policy | customer | claim | segment
    id: str
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = f"{self.type.title()} {self.id}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> Optional["FocusEntity"]:
        if not d or not d.get("id"):
            return None
        return cls(type=str(d.get("type", "segment")), id=str(d["id"]),
                   label=str(d.get("label", "")))


@dataclass
class TurnState:
    """What one turn established, and what the next turn may inherit."""
    turn_id: int = 0
    question: str = ""
    focus_entity: Optional[FocusEntity] = None
    focus_metrics: List[str] = field(default_factory=list)
    active_filters: Dict[str, Any] = field(default_factory=dict)
    cohort: Dict[str, Any] = field(default_factory=dict)
    ops_run: List[str] = field(default_factory=list)
    result_ref: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["focus_entity"] = self.focus_entity.to_dict() if self.focus_entity else None
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TurnState":
        return cls(
            turn_id=int(d.get("turn_id", 0)),
            question=str(d.get("question", "")),
            focus_entity=FocusEntity.from_dict(d.get("focus_entity")),
            focus_metrics=list(d.get("focus_metrics") or []),
            active_filters=dict(d.get("active_filters") or {}),
            cohort=dict(d.get("cohort") or {}),
            ops_run=list(d.get("ops_run") or []),
            result_ref=d.get("result_ref"),
        )


@dataclass
class ResolvedFollowUp:
    """The resolver's verdict on one question.

    `interpretation` is the only free text and contains no figures, so a bad
    resolution is visible to the user rather than silently wrong.
    """
    question: str
    entity: Optional[FocusEntity] = None
    inherited_from: Optional[str] = None
    intent: Optional[str] = None
    metric: Optional[str] = None
    dimension: Optional[str] = None
    op: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    interpretation: str = ""
    ambiguity: Optional[str] = None
    confidence: float = 0.0
    source: str = "rules"              # rules | llm
    degraded: bool = False
    degraded_reason: Optional[str] = None
    rejected_fields: List[str] = field(default_factory=list)

    @property
    def needs_clarification(self) -> bool:
        """True when the engine should ask instead of guessing."""
        return bool(self.ambiguity) or self.confidence < CONFIDENCE_THRESHOLD

    def proof_rows(self) -> List[Tuple[str, str]]:
        """The 'How I read your question' panel, as label/value pairs."""
        rows: List[Tuple[str, str]] = []
        if self.entity:
            src = f"carried from {self.inherited_from}" if self.inherited_from else "named in this question"
            rows.append((f'"{_referring_phrase(self.question)}"', f"{self.entity.label}  ({src})"))
        if self.metric:
            spec = ac.METRICS.get(self.metric)
            rows.append(("metric", spec.label if spec else self.metric))
        if self.dimension:
            dim = ac.DIMENSIONS.get(self.dimension)
            rows.append(("broken down by", dim.label if dim else self.dimension))
        if self.op:
            rows.append(("analysis", self.op))
        if self.ambiguity:
            rows.append(("ambiguity", self.ambiguity))
        rows.append(("confidence", f"{self.confidence:.2f}"
                     + ("" if self.confidence >= CONFIDENCE_THRESHOLD
                        else f" - below the {CONFIDENCE_THRESHOLD:.2f} threshold, so I'm asking")))
        return rows

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["entity"] = self.entity.to_dict() if self.entity else None
        return d


def _referring_phrase(question: str) -> str:
    """The pronoun or phrase the entity was resolved from, for the proof panel."""
    m = _CONTINUATION_CUES.search(question or "")
    return m.group(0).lower() if m else "this"


class Conversation:
    """An ordered list of turns, plus the state the next turn inherits."""

    def __init__(self, turns: Optional[List[TurnState]] = None):
        self.turns: List[TurnState] = list(turns or [])

    def __len__(self) -> int:
        return len(self.turns)

    @property
    def is_first_turn(self) -> bool:
        return not self.turns

    def last(self) -> Optional[TurnState]:
        return self.turns[-1] if self.turns else None

    def append(self, state: TurnState) -> TurnState:
        state.turn_id = len(self.turns) + 1
        self.turns.append(state)
        return state

    def reset(self):
        self.turns = []

    def to_list(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.turns]

    @classmethod
    def from_list(cls, rows: Optional[List[Dict[str, Any]]]) -> "Conversation":
        return cls([TurnState.from_dict(r) for r in (rows or []) if isinstance(r, dict)])


# =====================================================================
# ENTITY EXTRACTION
# =====================================================================

def extract_entity(question: str) -> Optional[FocusEntity]:
    """Pull an explicitly named record out of the question, if there is one."""
    if not question:
        return None
    for etype, pattern, label_prefix in ENTITY_PATTERNS:
        m = re.search(pattern, question, re.IGNORECASE)
        if m:
            ident = m.group(1)
            return FocusEntity(type=etype, id=ident, label=f"{label_prefix} {ident}")
    return None


def reads_as_continuation(question: str) -> bool:
    """Whether a question with no entity of its own is following on."""
    q = (question or "").strip()
    if not q:
        return False
    return bool(_CONTINUATION_CUES.search(q)) or bool(_ELLIPSIS_CUES.match(q))


# =====================================================================
# AMBIGUITY CHECK (deterministic, against the data)
# =====================================================================

# Surface words that name a dimension in ordinary speech. Used to run the
# ambiguity check against what the *question* said, not only against what the
# model chose - a weak local model often picks the wrong dimension, and the
# ambiguity in "their branch" is present either way.
_DIMENSION_WORDS: List[Tuple[str, str]] = [
    (r"\bbranch(?:es)?\b", "branch_no"),
    (r"\bcit(?:y|ies)\b", "city"),
    (r"\bstates?\b", "state"),
    (r"\bregions?\b", "state"),
    (r"\bplans?\b", "plan_name"),
    (r"\bproducts?\b", "plan_name"),
    (r"\boccupations?\b", "occupation"),
    (r"\bpayment modes?\b", "payment_mode"),
    (r"\bbuckets?\b", "persistency_bucket"),
]


def dimensions_named_in(question: str) -> List[str]:
    """Dimension names the question itself refers to, in plain words."""
    q = (question or "").lower()
    found: List[str] = []
    for pattern, dim in _DIMENSION_WORDS:
        if re.search(pattern, q) and dim not in found:
            found.append(dim)
    return found


def check_dimension_ambiguity(ctx: ac.DataContext,
                              entity: Optional[FocusEntity],
                              dimension: Optional[str]) -> Optional[str]:
    """Flag a breakdown that reads as singular but isn't.

    "their branch" presumes one branch. Most agents here sell across several,
    so the phrase has no referent and a modal-value guess would answer a
    different question than the one asked.
    """
    if entity is None or not dimension:
        return None
    dim = ac.DIMENSIONS.get(dimension)
    if dim is None or dim.cardinality_hint != "high":
        return None
    try:
        frame = ctx.frame("policy")
    except Exception:
        return None
    key_col = {"agent": "sales_id", "customer": "client_id", "policy": "app_id"}.get(entity.type)
    if not key_col or key_col not in frame.columns or dim.column not in frame.columns:
        return None
    sub = frame[frame[key_col].astype(str) == str(entity.id)]
    if sub.empty:
        return None
    n_values = sub[dim.column].nunique(dropna=True)
    if n_values > 1:
        biggest = int(sub[dim.column].value_counts().iloc[0])
        return (f"{entity.label} spans {n_values} {dim.label.lower()} values across "
                f"{len(sub)} policies; the largest holds {biggest}")
    return None


# =====================================================================
# RESOLVER
# =====================================================================

_LLM_SYSTEM = """You resolve follow-up questions in a conversation about an insurance analytics dataset.

You NEVER answer the question and you NEVER state a number, figure, percentage or count. Another system computes every value. Your only job is to say WHAT should be computed.

Return a single JSON object, nothing else. No prose, no markdown fences.

{
  "entity_type": "agent" | "policy" | "customer" | "claim" | null,
  "entity_id": "<id the question refers to, or null to reuse the previous turn's>",
  "inherit_entity": true | false,
  "intent": "why" | "comparison" | "ranking" | "profile" | "list" | "overview" | "forecast" | "other",
  "metric": "<one metric name from the allowed list, or null>",
  "dimension": "<one dimension name from the allowed list, or null>",
  "op": "<one operation name from the allowed list, or null>",
  "interpretation": "<one plain sentence, no numbers, restating what you think is being asked>",
  "ambiguity": "<a phrase in the question that presumes something the data may not support, or null>",
  "confidence": <0.0 to 1.0>
}

Rules:
- Use ONLY names from the allowed lists. Never invent a metric, dimension or operation name.
- If the question refers back with a pronoun ("they", "their", "it", "that one") set inherit_entity true and entity_id null.
- If the question asks about the future, predictions, or anything not measurable from historical records, set intent to "forecast".
- Set confidence below 0.75 when the question is vague or when you had to guess which metric was meant.
- "interpretation" must contain no digits."""


def _build_llm_prompt(question: str, prev: Optional[TurnState]) -> str:
    metrics = ", ".join(sorted(ac.METRICS.keys()))
    dims = ", ".join(sorted(ac.DIMENSIONS.keys()))
    ops = ", ".join(RESOLVABLE_OPS)

    if prev is not None:
        ent = prev.focus_entity.label if prev.focus_entity else "none"
        prev_block = (
            f"PREVIOUS TURN\n"
            f"  question: {prev.question}\n"
            f"  focus entity: {ent}\n"
            f"  metrics in play: {', '.join(prev.focus_metrics) or 'none'}\n"
            f"  filters: {json.dumps(prev.active_filters) if prev.active_filters else 'none'}\n"
            f"  analyses run: {', '.join(prev.ops_run) or 'none'}\n"
        )
    else:
        prev_block = "PREVIOUS TURN\n  (none - this is the first question)\n"

    return (
        f"{prev_block}\n"
        f"FOLLOW-UP QUESTION\n  {question}\n\n"
        f"ALLOWED METRICS\n  {metrics}\n\n"
        f"ALLOWED DIMENSIONS\n  {dims}\n\n"
        f"ALLOWED OPERATIONS\n  {ops}\n\n"
        f"Return the JSON object now."
    )


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first JSON object out of a model response."""
    if not text:
        return None
    cleaned = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(cleaned[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(cleaned[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


class FollowUpResolver:
    """Resolves a question against the conversation state.

    Turn 1 is always rule-based: there is no state to resolve, and keeping the
    opening question free of a model call keeps it fast.
    """

    def __init__(self, ctx: Optional[ac.DataContext] = None, use_llm: bool = True,
                 model: Optional[str] = None, timeout: int = 30):
        self.ctx = ctx
        self.use_llm = use_llm
        self.model = model
        self.timeout = timeout

    # ---- public -------------------------------------------------------
    def resolve(self, question: str, conversation: Conversation) -> ResolvedFollowUp:
        prev = conversation.last()

        # Turn 1: rules only, by design.
        if prev is None:
            return self._resolve_by_rules(question, None)

        if not self.use_llm:
            return self._resolve_by_rules(question, prev)

        llm_result = self._resolve_by_llm(question, prev)
        if llm_result is not None:
            return llm_result

        fallback = self._resolve_by_rules(question, prev)
        fallback.degraded = True
        fallback.degraded_reason = (
            "The local model was unreachable or returned something unusable, so this "
            "follow-up was resolved by simple carry-forward rules instead.")
        return fallback

    # ---- rule path ----------------------------------------------------
    def _resolve_by_rules(self, question: str, prev: Optional[TurnState]) -> ResolvedFollowUp:
        named = extract_entity(question)
        entity, inherited_from = named, None

        if entity is None and prev is not None and prev.focus_entity is not None \
                and reads_as_continuation(question) and not reads_as_contrast(question):
            entity = prev.focus_entity
            inherited_from = f"turn {prev.turn_id}"

        res = ResolvedFollowUp(
            question=question,
            entity=entity,
            inherited_from=inherited_from,
            source="rules",
            confidence=0.9 if named is not None else (0.8 if entity is not None else 0.5),
        )
        if entity is not None:
            res.filters = self._entity_filters(entity)
            res.interpretation = (
                f"about {entity.label}"
                + (f", carried over from the previous question" if inherited_from else "")
            )
        else:
            res.interpretation = "a new question with no carried-over subject"
        return res

    # ---- llm path -----------------------------------------------------
    def _resolve_by_llm(self, question: str, prev: TurnState) -> Optional[ResolvedFollowUp]:
        try:
            import ollama_client as oc
        except Exception:
            return None
        if not oc.is_ollama_available():
            return None

        raw = oc.generate_llm_response(
            prompt=_build_llm_prompt(question, prev),
            system_prompt=_LLM_SYSTEM,
            model=self.model,
            timeout=self.timeout,
        )
        obj = _extract_json(raw or "")
        if obj is None:
            return None
        return self._validate(obj, question, prev)

    # ---- validation ---------------------------------------------------
    def _validate(self, obj: Dict[str, Any], question: str,
                  prev: TurnState) -> Optional[ResolvedFollowUp]:
        """Accept only registry-known names; drop anything else and say so.

        A rejected field is blanked rather than fatal - the engine can still
        run with fewer hints. A response with nothing usable returns None so
        the caller falls back to rules.
        """
        rejected: List[str] = []

        metric = _clean_str(obj.get("metric"))
        if metric and metric not in ac.METRICS and metric not in ac.SHARE_METRICS:
            rejected.append(f"metric '{metric}'")
            metric = None

        dimension = _clean_str(obj.get("dimension"))
        if dimension and dimension not in ac.DIMENSIONS:
            rejected.append(f"dimension '{dimension}'")
            dimension = None

        op = _clean_str(obj.get("op"))
        if op and op not in RESOLVABLE_OPS:
            rejected.append(f"operation '{op}'")
            op = None

        intent = _clean_str(obj.get("intent")) or None

        # Entity: an id written in the question wins over anything the model says.
        entity = extract_entity(question)
        inherited_from = None
        contrast = reads_as_contrast(question)
        if entity is None:
            ent_id = _clean_str(obj.get("entity_id"))
            ent_type = _clean_str(obj.get("entity_type")) or "segment"
            carried = prev.focus_entity

            if contrast:
                # "the other agents", "compared with everyone else" - the question
                # is about the complement of the focus entity. Carrying it forward
                # would answer the opposite of what was asked, so no inheritance
                # happens here regardless of what the model claimed.
                entity = None
            elif carried is not None and ent_id and str(ent_id) == str(carried.id):
                # A small model will echo the previous turn's id back with the
                # wrong type ("Customer 800812" for what is an agent). When the id
                # matches the one in focus, keep the type the previous turn
                # established - that came from the data, not from the model.
                entity = carried
                inherited_from = f"turn {prev.turn_id}"
            elif ent_id:
                entity = FocusEntity(type=ent_type, id=ent_id)
            elif carried is not None and (obj.get("inherit_entity")
                                          or reads_as_continuation(question)):
                entity = carried
                inherited_from = f"turn {prev.turn_id}"

        interpretation = _clean_str(obj.get("interpretation")) or ""
        if re.search(r"\d", interpretation):
            # the model was told not to state figures; drop the sentence rather
            # than show an unverified number next to verified ones
            rejected.append("interpretation (contained digits)")
            interpretation = ""

        try:
            confidence = float(obj.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        if not any([metric, dimension, op, entity]):
            return None

        confidence = _calibrate_confidence(
            model_confidence=confidence, question=question, prev=prev,
            metric=metric, dimension=dimension,
            entity_inherited=inherited_from is not None,
            rejected=rejected,
        )

        res = ResolvedFollowUp(
            question=question,
            entity=entity,
            inherited_from=inherited_from,
            intent=intent,
            metric=metric,
            dimension=dimension,
            op=op,
            interpretation=interpretation or (f"about {entity.label}" if entity else ""),
            ambiguity=_clean_str(obj.get("ambiguity")),
            confidence=confidence,
            source="llm",
            rejected_fields=rejected,
        )
        if entity is not None:
            res.filters = self._entity_filters(entity)

        # A data-backed ambiguity check beats the model's guess at one. Check
        # the dimensions the *question* named as well as the one the model
        # chose: a small model often picks the wrong dimension, but "their
        # branch" is ambiguous regardless of what it picked.
        if self.ctx is not None and entity is not None:
            candidates = [dimension] + [d for d in dimensions_named_in(question)
                                        if d != dimension]
            for cand in candidates:
                found = check_dimension_ambiguity(self.ctx, entity, cand)
                if found:
                    res.ambiguity = found
                    res.dimension = res.dimension or cand
                    break

        if rejected:
            res.degraded = True
            res.degraded_reason = (
                "Part of the follow-up could not be matched to known metrics or "
                "dimensions and was ignored: " + "; ".join(rejected) + ".")
        return res

    # ---- helpers ------------------------------------------------------
    @staticmethod
    def _entity_filters(entity: FocusEntity) -> Dict[str, Any]:
        col = {"agent": "sales_id", "policy": "app_id",
               "customer": "client_id", "claim": "claim_no"}.get(entity.type)
        return {col: entity.id} if col else {}


def _calibrate_confidence(model_confidence: float, question: str,
                          prev: Optional[TurnState], metric: Optional[str],
                          dimension: Optional[str], entity_inherited: bool,
                          rejected: List[str]) -> float:
    """Damp the model's self-reported confidence with deterministic evidence.

    Small local models report 1.0 almost unconditionally, which would make the
    clarification threshold dead code. Rather than trust that number, treat it
    as a ceiling and subtract for each thing the model asserted that nothing in
    the question or the previous turn actually supports.
    """
    q = (question or "").lower()
    score = max(0.0, min(1.0, model_confidence))

    # A metric nobody asked for: not named in the question, not carried over.
    if metric:
        spec = ac.METRICS.get(metric)
        words = set(re.findall(r"[a-z]+", metric.replace("_", " ")))
        if spec:
            words |= set(re.findall(r"[a-z]+", spec.label.lower()))
        words -= {"rate", "total", "avg", "average", "of", "per", "the", "and", "to"}
        mentioned = any(w in q for w in words if len(w) > 3)
        carried = bool(prev and metric in (prev.focus_metrics or []))
        if not mentioned and not carried:
            score -= 0.30

    # A breakdown nobody asked for.
    if dimension:
        dim = ac.DIMENSIONS.get(dimension)
        label_words = set(re.findall(r"[a-z]+", (dim.label if dim else dimension).lower()))
        label_words |= set(re.findall(r"[a-z]+", dimension.replace("_", " ")))
        if not any(w in q for w in label_words if len(w) > 3):
            score -= 0.20

    if entity_inherited:
        score -= 0.05
    if rejected:
        score -= 0.10 * len(rejected)

    return max(0.0, min(1.0, round(score, 2)))


def _clean_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("null", "none", "n/a", ""):
        return None
    return s


__all__ = [
    "CONFIDENCE_THRESHOLD", "RESOLVABLE_OPS",
    "FocusEntity", "TurnState", "ResolvedFollowUp", "Conversation",
    "FollowUpResolver", "extract_entity", "reads_as_continuation",
    "check_dimension_ambiguity",
]
