"""
business_retrieval.py
=====================
Hybrid retriever over the combined Problem Statement library (legacy 270
SQL-template PS + 146 business PS).

The legacy retriever was a single TF-IDF cosine over `problem_statement +
example_queries + intent + tables + columns`, with a hard 0.30 floor. Business
phrasing ("where should management focus retention effort?") shares almost no
vocabulary with a PS worded "What is the lapse rate by product?", so it fell
straight through to the weak dynamic-Cypher fallback.

This retriever keeps TF-IDF as one signal and adds four more, all offline:

  1. lexical      - TF-IDF cosine over the PS document (word 1-2 grams)
  2. expanded     - TF-IDF cosine over a concept-expanded version of the query
  3. concept      - blended coverage/Jaccard overlap between the concepts
                    detected in the question and the concepts each PS declares
  4. intent       - trigger-phrase score for the PS's business intent
  5. operation    - whether the phrasing implies analytical machinery (a ranked
                    list, an outlier hunt, a concentration measure) that the PS's
                    plan actually contains. This is what separates Problem
                    Statements that share a topic but need different analysis.

Scores are combined with fixed weights, and business PS get a small prior when
the question is phrased as a business question (recommendation, diagnosis,
comparison, prioritisation) rather than a lookup.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import business_intents as bi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_ps_path(filename: str) -> str:
    candidates = [
        os.path.join(BASE_DIR, "data", "problem_statements", filename),
        os.path.join(BASE_DIR, "..", "..", "data", "problem_statements", filename),
        os.path.join(BASE_DIR, "..", "data", "problem_statements", filename),
        os.path.join(BASE_DIR, filename),
        os.path.join(BASE_DIR, "data", filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.join(BASE_DIR, filename)


LEGACY_PS_PATH = _resolve_ps_path("problem_statements.json")
BUSINESS_PS_PATH = _resolve_ps_path("business_problem_statements.json")

# Weights for the five signals. Tuned on the held-out business evaluation set.
W_LEXICAL = 0.25
W_EXPANDED = 0.25
W_CONCEPT = 0.19
W_INTENT = 0.10
W_OPERATION = 0.21

# Phrasings that imply a particular analytical operation must be in the plan.
# This is the signal that separates Problem Statements sharing a topic but
# needing different machinery: "which claims dominate the payout" and "what is
# driving claims risk" are both about claims, but only the first needs
# concentration analysis.
OPERATION_CUES: Dict[str, Tuple[str, ...]] = {
    "entity_shortlist": (
        "list", "call list", "shortlist", "target list", "who should", "name them",
        "which customers should", "which policies should", "which agents should",
        "give me the", "candidates", "contact", "least afford to lose", "win back",
        "cross-sell", "cross sell", "room to buy", "underinsured", "top 10 customers",
        "target", "prioritise for", "prioritize for", "who are the",
    ),
    "concentration": (
        "concentrat", "diversif", "handful", "a few", "account for most", "bulk of",
        "majority of", "rests on", "depend on", "dependent on", "over-exposed",
        "hhi", "top 10", "largest few", "how much of the book",
    ),
    "top_entity_concentration": (
        "handful", "a few", "account for most", "biggest policies", "largest claims",
        "top 10", "single policy", "biggest claims", "bulk of",
    ),
    "outliers": (
        "unusual", "odd", "outlier", "anomal", "abnormal", "statistically",
        "looks wrong", "stands out", "irregular", "suspicious", "surprising",
    ),
    "driver_scan": (
        "why", "what is driving", "what drives", "what sits behind", "root cause",
        "what explains", "unpack", "decompose", "factors", "reasons", "diagnose",
        "talk me through",
    ),
    "composite_score": (
        "priorit", "rank", "ranking", "which should we", "where should",
        "best balance", "scorecard", "worth", "first", "agenda", "tightest ship",
        "visit first", "focus",
    ),
    "crosstab_metric": (
        "combination", "combinations", "cells", "pockets", "for each product",
        "equally well", "interact", "by product and", "across both",
        "different products", "different regions", "different segments",
    ),
    "segment_profile": (
        "look like", "profile", "picture of", "describe", "characteristics",
        "distinguish", "differ from", "compared with the rest", "full read",
        "what does a", "against the rest",
    ),
    "compare_to_baseline": (
        "compared with", "versus the book", "above average", "below average",
        "more than they should", "than it should", "elevated",
    ),
    "portfolio_overview": (
        "overall", "portfolio", "board-level", "board level", "executive",
        "state of the book", "kpi", "headline", "whole book", "the book",
    ),
}

# A question needs at least this combined score before a business PS is used;
# below it the engine falls back to an ad-hoc analytical plan or the legacy engine.
BUSINESS_FLOOR = 0.16


def load_ps_library(include_legacy: bool = True,
                    include_business: bool = True) -> List[Dict[str, Any]]:
    """Load the combined library. Legacy records are returned unmodified so
    existing consumers (query_workflow, regression tests) keep working."""
    records: List[Dict[str, Any]] = []
    if include_legacy and os.path.exists(LEGACY_PS_PATH):
        with open(LEGACY_PS_PATH) as f:
            records.extend(json.load(f))
    if include_business and os.path.exists(BUSINESS_PS_PATH):
        with open(BUSINESS_PS_PATH) as f:
            records.extend(json.load(f))
    return records


def is_business_ps(rec: Dict[str, Any]) -> bool:
    return bool(rec.get("analytical_steps"))


@dataclass
class RetrievalHit:
    record: Dict[str, Any]
    score: float
    lexical: float
    expanded: float
    concept: float
    intent: float
    operation: float = 0.0

    @property
    def problem_id(self) -> str:
        return self.record["problem_id"]

    @property
    def is_business(self) -> bool:
        return is_business_ps(self.record)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "problem_statement": self.record["problem_statement"],
            "intent_type": self.record.get("intent_type"),
            "business_intent": self.record.get("business_intent"),
            "tier": self.record.get("analysis_tier", "simple_retrieval"),
            "kind": "business" if self.is_business else "legacy",
            "score": round(self.score, 4),
            "signals": {"lexical": round(self.lexical, 4),
                        "expanded": round(self.expanded, 4),
                        "concept": round(self.concept, 4),
                        "intent": round(self.intent, 4),
                        "operation": round(self.operation, 4)},
        }


# Phrases that mark a question as business-analytical rather than a lookup.
_BUSINESS_MARKERS = (
    "why", "should", "recommend", "compare", "prioriti", "focus", "attention",
    "healthy", "health", "risk", "driving", "drives", "driver", "explain",
    "assessment", "strategy", "unusual", "anomal", "concentrat", "best",
    "worst", "strongest", "weakest", "investigate", "economics", "attractive",
    "management", "perform", "distinguish", "candidate", "balance", "across",
)
# Phrases that mark a question as a plain lookup, where the legacy engine is right.
_LOOKUP_MARKERS = (
    "how many", "what is the total", "what is the average", "list all",
    "show me the details", "count of", "sum of", "how much is",
    "are there", "how many records", "what is the number of", "details of",
    "show me the record", "what is the value of",
)


class BusinessRetriever:
    _singleton: Optional["BusinessRetriever"] = None

    @classmethod
    def get(cls, records: Optional[Sequence[Dict[str, Any]]] = None) -> "BusinessRetriever":
        if cls._singleton is None or records is not None:
            cls._singleton = cls(records)
        return cls._singleton

    def __init__(self, records: Optional[Sequence[Dict[str, Any]]] = None):
        self.records = list(records or load_ps_library())
        self._build_index()

    # ---- index -------------------------------------------------------
    def _doc(self, rec: Dict[str, Any]) -> str:
        parts = [rec.get("problem_statement", "")]
        parts.extend(rec.get("example_queries", []) or [])
        parts.append(rec.get("intent_type", "") or "")
        parts.append(rec.get("business_objective", "") or "")
        parts.extend(rec.get("relevant_tables", []) or [])
        parts.extend(rec.get("relevant_columns", []) or [])
        parts.extend(rec.get("required_metrics", []) or [])
        parts.extend(rec.get("required_operations", []) or [])
        concepts = rec.get("concepts") or []
        for c in concepts:
            parts.append(c)
            parts.extend(bi.CONCEPT_ONTOLOGY.get(c, [])[:6])
        return " ".join(str(p) for p in parts).lower()

    def _build_index(self):
        self.corpus = [self._doc(r) for r in self.records]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english",
                                          sublinear_tf=True, min_df=1)
        self.matrix = self.vectorizer.fit_transform(self.corpus)

        self.ps_concepts: List[set] = []
        for r in self.records:
            declared = set(r.get("concepts") or [])
            if not declared:
                declared = set(bi.concepts_in(
                    r.get("problem_statement", "") + " " +
                    " ".join(r.get("example_queries", []) or [])))
            self.ps_concepts.append(declared)

        self.ps_intent = [r.get("business_intent") for r in self.records]
        self.business_mask = np.array([is_business_ps(r) for r in self.records])
        self.ps_ops: List[set] = [
            {s["op"] for s in (r.get("analytical_steps") or [])} for r in self.records]

    @staticmethod
    def cued_operations(query: str) -> set:
        q = f" {query.lower()} "
        return {op for op, cues in OPERATION_CUES.items()
                if any(c in q for c in cues)}

    # ---- scoring -----------------------------------------------------
    def question_style(self, query: str) -> str:
        """lookup | business | neutral.

        A question is a lookup only when it asks for one fact and carries no
        analytical signal at all. Everything with two business markers, an
        implied analytical operation, or three or more domain concepts is
        treated as business - a plain keyword count was routing questions like
        "which of our products is quietly bleeding policyholders?" to the
        lookup engine.
        """
        q = query.lower()
        lookup = any(m in q for m in _LOOKUP_MARKERS)
        markers = sum(1 for m in _BUSINESS_MARKERS if m in q)
        concepts = len(bi.concepts_in(query))
        cued = len(self.cued_operations(query))

        if lookup and markers == 0 and cued == 0 and concepts <= 2:
            return "lookup"
        if markers >= 2 or cued >= 1 or concepts >= 3:
            return "business"
        if lookup:
            return "lookup"
        return "business" if markers else "neutral"

    def score(self, query: str) -> np.ndarray:
        q = query.lower()
        expanded = bi.expand_query(query)

        lex = cosine_similarity(self.vectorizer.transform([q]), self.matrix).flatten()
        exp = cosine_similarity(self.vectorizer.transform([expanded]), self.matrix).flatten()

        # Concept score blends coverage (how much of the question the PS covers)
        # with Jaccard (how specific the PS is). Pure Jaccard systematically
        # penalised broad executive Problem Statements: an eight-concept
        # portfolio-health PS could never out-score a two-concept KPI PS on a
        # question that touched three concepts.
        qc = set(bi.concepts_in(query))
        if qc:
            con = np.array([
                0.55 * (len(qc & pc) / len(qc)) +
                0.45 * (len(qc & pc) / len(qc | pc) if (qc | pc) else 0.0)
                for pc in self.ps_concepts])
        else:
            con = np.zeros(len(self.records))

        intent_scores = bi.score_intents(query)
        if intent_scores:
            top = max(intent_scores.values())
            itn = np.array([
                (intent_scores.get(i, 0.0) / top) if i else 0.0
                for i in self.ps_intent])
        else:
            itn = np.zeros(len(self.records))

        cued = self.cued_operations(query)
        if cued:
            ops = np.array([len(cued & po) / len(cued) if po else 0.0
                            for po in self.ps_ops])
        else:
            ops = np.zeros(len(self.records))

        combined = (W_LEXICAL * lex + W_EXPANDED * exp + W_CONCEPT * con +
                    W_INTENT * itn + W_OPERATION * ops)

        style = self.question_style(query)
        if style == "business":
            combined = combined + 0.05 * self.business_mask
        elif style == "lookup":
            combined = combined - 0.04 * self.business_mask

        self._last = {"lexical": lex, "expanded": exp, "concept": con,
                      "intent": itn, "operation": ops}
        return combined

    def retrieve(self, query: str, top_k: int = 5,
                 business_only: bool = False) -> List[RetrievalHit]:
        combined = self.score(query)
        sig = self._last
        order = np.argsort(combined)[::-1]
        hits: List[RetrievalHit] = []
        for idx in order:
            if business_only and not self.business_mask[idx]:
                continue
            hits.append(RetrievalHit(
                record=self.records[idx], score=float(combined[idx]),
                lexical=float(sig["lexical"][idx]), expanded=float(sig["expanded"][idx]),
                concept=float(sig["concept"][idx]), intent=float(sig["intent"][idx]),
                operation=float(sig["operation"][idx])))
            if len(hits) >= top_k:
                break
        return hits

    def best_business(self, query: str) -> Tuple[Optional[RetrievalHit], List[RetrievalHit]]:
        """Top business PS above the floor, plus the top-5 for the trace panel."""
        top5 = self.retrieve(query, top_k=5)
        biz = self.retrieve(query, top_k=1, business_only=True)
        best = biz[0] if biz and biz[0].score >= BUSINESS_FLOOR else None
        return best, top5


__all__ = ["BusinessRetriever", "RetrievalHit", "load_ps_library",
           "is_business_ps", "BUSINESS_FLOOR", "OPERATION_CUES"]
