"""PS matching.

Scores a normalized question against every candidate Problem Statement. The scoring
is deterministic and inspectable: each component contributes a named, bounded amount,
and the breakdown travels with the result so a match can be argued with.
"""
from __future__ import annotations

import difflib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from src.ps.registry import ProblemStatement, PSRegistry
from src.router.normalizer import (
    STOPWORDS, NormalizedQuery, clean, content_terms, expand_aliases, tokenize)

# Component weights. They sum to 1.0 so a score reads as a confidence.
WEIGHTS = {
    "example_similarity": 0.14,   # closeness to the PS's own example questions
    "phrase_signal": 0.16,        # distinctive multi-word phrases shared with the PS
    "rare_term_signal": 0.10,     # shared words weighted by how rare they are
    "statement_similarity": 0.06,  # closeness to the PS statement itself
    "concept_overlap": 0.07,      # declared concepts the question mentions
    "domain_signal": 0.06,        # vocabulary tied to the PS's tables and columns
    "dimension_signal": 0.18,     # what the question asks to be grouped by
    "intent_signal": 0.17,        # what shape of answer the question asks for
    "tier_signal": 0.05,          # descriptive, analytical, diagnostic or executive
    "entity_fit": 0.01,           # identifiers the question supplies
}

# "Which states...", "by occupation", "per agent" - the grouping dimension is the
# sharpest way to separate PSs that share a subject and a shape. Ranking states by
# volume and ranking individual policies by value read almost identically without it.
DIMENSION_TERMS: Dict[str, Set[str]] = {
    "state": {"state", "states", "statewise"},
    "city": {"city", "cities", "town", "towns"},
    "branch": {"branch", "branches"},
    "agent": {"agent", "agents", "advisor", "advisors", "adviser", "producer",
              "producers", "producing", "salesperson"},
    "product": {"product", "products", "plan", "plans", "scheme", "schemes"},
    "occupation": {"occupation", "occupations", "profession", "professions", "job"},
    "education": {"education", "qualification", "educated"},
    "gender": {"gender", "male", "female", "men", "women"},
    "age": {"age", "ages", "old", "young", "age_band", "demographic"},
    "income": {"income", "earnings", "salary", "affluent", "earning"},
    "status": {"status", "statuses", "lapse", "lapsed", "surrender", "surrendered",
               "inforce", "active", "death", "died", "deceased"},
    "bucket": {"bucket", "buckets", "13th", "25th", "37th"},
    "payment_mode": {"payment", "mode", "modes", "autodebit", "cheque", "upi", "cash",
                     "card", "ecs"},
    "cause": {"cause", "causes", "mortality", "dying"},
    "period": {"month", "monthly", "quarter", "quarterly", "year", "yearly", "period",
               "periods", "vintage", "cohort", "cohorts"},
    "client": {"client", "clients", "customer", "customers", "policyholder",
               "policyholders"},
    "term": {"term", "terms", "duration", "tenure"},
    "trad_ulip": {"ulip", "traditional", "trad", "linked"},
}

# Dimensions whose terms name a *value* rather than a grouping axis. Mentioning
# "unit linked" says something about which policies are meant; mentioning "age" does not.
VALUE_DIMENSIONS = {"trad_ulip", "status", "bucket", "payment_mode"}

# Phrasings that actually request a grouping. Without this, any mention of a
# dimension word would be read as a request to group by it.
GROUPING_PATTERNS = ("which {}", "what {}", "by {}", "per {}", "across {}", "each {}",
                     "{} wise", "top {}", "best {}", "worst {}", "{} with the",
                     "{} have the", "{} generate", "{} carry", "{} are our")


# The column a dimension corresponds to in a PS's group_by.
DIMENSION_COLUMNS: Dict[str, Set[str]] = {
    "state": {"owner_owner.comm.state", "owner.comm.state"},
    "city": {"owner_owner.comm.city.name", "owner.comm.city.name"},
    "branch": {"branch_no"},
    "agent": {"sales_id"},
    "product": {"prod_no", "plan.name"},
    "occupation": {"owner_owner.occupation", "owner.occupation"},
    "education": {"owner_owner.education", "owner.education"},
    "gender": {"owner.gender", "owner_owner.gender"},
    "age": {"age_band"},
    "income": {"income_band"},
    "status": {"policy.status", "persistency_renewal_status", "renewal_status"},
    "bucket": {"persistency_bucket", "persistency_persistency_bucket"},
    "payment_mode": {"payment_mode"},
    "cause": {"cause.of.death"},
    "period": {"period", "cohort_year"},
    "client": {"owner_clientid", "clientid"},
    "term": {"term_band"},
    "trad_ulip": {"trad.ulip", "prod_no"},
}

# Cues that decide the shape of an answer outright. When one of these appears, a PS
# that offers that shape is right and one that does not is wrong - the subject matter
# they share is not the question being asked.
# Phrase triggers paired with the intent tokens that satisfy them. A token may be an
# intent structure ("Time-Series") or a measure ("Trend"): "each month" wants a time
# series whose measure happens to be a count.
DECISIVE_CUES: List[tuple] = [
    ({"why", "explain", "root cause", "diagnose", "what is driving", "what drives",
      "reason for"}, {"Diagnostic", "Driver"}),
    ({"how concentrated", "concentration", "dependent on", "reliance on"},
     {"Concentration"}),
    ({"snapshot", "overview", "brief me", "business review", "dashboard",
      "complete picture", "summarise", "summarize"}, {"Overview"}),
    ({"reconcile", "reconciliation", "data quality", "cannot answer", "can this data",
      "not answer", "limitations"}, {"Integrity", "Reconciliation"}),
    ({"tell me about policy", "details for", "look up", "full record", "policy details"},
     {"Record", "Lookup"}),
    ({"over time", "trend", "each month", "by month", "monthly", "quarterly",
      "shifted over"}, {"Trend", "Time-Series", "Seasonality"}),
    ({"outlier", "outliers", "anomalous", "statistical outlier"},
     {"Outliers", "Anomaly"}),
]

# Vocabulary that signals the depth of analysis wanted.
TIER_CUES: Dict[str, Set[str]] = {
    "diagnostic": {"why", "explain", "root cause", "diagnose", "what is driving",
                   "what drives", "and why"},
    "multi_step_business": {"snapshot", "brief me", "business review", "scorecard",
                            "should i act", "should we focus", "focus on", "priorities",
                            "segmentation", "complete picture", "act on", "remediation",
                            "composite", "quality score", "biggest impact", "losing",
                            "leaking", "costing us", "value loss", "at risk",
                            "worry about"},
    "descriptive": {"how many", "what is the total", "list", "show me the"},
}

# Softer cues, paired with every intent token that would satisfy them. A PS declaring
# GroupBy/Ranking/Count answers "which branches write the most business" perfectly
# well without ever using the token "TopN".
INTENT_CUES: List[tuple] = [
    ({"how many", "number of", "count"}, {"Count", "Distinct"}),
    ({"total", "sum", "altogether"}, {"Sum", "Count"}),
    ({"average", "mean", "typical", "avg"}, {"Average", "Median"}),
    ({"median"}, {"Median", "Average"}),
    ({"rate", "percentage", "percent", "proportion", "share of"},
     {"Ratio", "Rate", "Distribution"}),
    ({"rank", "ranking", "league", "order by"}, {"Ranking", "TopN", "BottomN"}),
    ({"top", "best", "highest", "largest", "biggest", "leading", "most"},
     {"TopN", "Ranking", "Max", "Sum"}),
    ({"worst", "lowest", "bottom", "least"}, {"BottomN", "Ranking", "Min"}),
    ({"compare", "versus", " vs ", "against", "difference between", "how do", "how does"},
     {"Comparison"}),
    ({"distribution", "breakdown", "spread", "split", "mix"},
     {"Distribution", "Bucketing", "Profile"}),
    ({"band", "bands", "buckets", "grouped"}, {"Bucketing", "Distribution"}),
    ({"profile", "segment", "segmentation", "scorecard", "score"},
     {"Profile", "Distribution", "Overview"}),
    ({"unusual", "suspicious", "flag", "anomaly"}, {"Anomaly", "Outliers"}),
    ({"seasonal", "seasonality", "by month of"}, {"Seasonality", "Trend"}),
]

# Vocabulary that points at a table, used as a weak domain signal.
TABLE_TERMS: Dict[str, Set[str]] = {
    "policy": {"policy", "policies", "product", "plan", "premium", "ape", "sum_assured",
               "term", "branch", "inforce", "lapse", "lapsed", "surrender", "status"},
    "owner": {"customer", "client", "owner", "policyholder", "age", "gender", "income",
              "occupation", "education", "state", "city", "demographic", "geography"},
    "sales": {"agent", "advisor", "sales", "producer", "persistency", "complaint",
              "conduct", "rcu", "pivc", "freelook", "misconduct"},
    "claims": {"claim", "claims", "death", "died", "mortality", "settlement", "payout",
               "intimation", "cause"},
    "persistency": {"renewal", "renewals", "persistency", "bucket", "collection",
                    "payment mode", "due", "pending", "paid"},
}


@dataclass
class Candidate:
    """One scored PS, with the breakdown that produced the score."""
    problem_id: str
    problem_statement: str
    score: float
    components: Dict[str, float] = field(default_factory=dict)
    matched_terms: List[str] = field(default_factory=list)
    best_example: str = ""
    business_domain: str = ""
    analysis_tier: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "problem_statement": self.problem_statement,
            "confidence": round(self.score, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "matched_terms": self.matched_terms,
            "closest_example_query": self.best_example,
            "business_domain": self.business_domain,
            "analysis_tier": self.analysis_tier,
        }


def _similarity(query_terms: Set[str], target: str) -> float:
    """Blend term overlap with sequence similarity.

    Overlap alone rewards long targets that happen to contain the query's words;
    sequence similarity alone is brittle to word order. Together they behave.
    """
    target_terms = content_terms(tokenize(target.lower()))
    if not query_terms or not target_terms:
        return 0.0
    overlap = len(query_terms & target_terms) / len(query_terms | target_terms)
    sequence = difflib.SequenceMatcher(
        None, " ".join(sorted(query_terms)), " ".join(sorted(target_terms))).ratio()
    return 0.65 * overlap + 0.35 * sequence


def _bigrams(text: str) -> Set[str]:
    words = [w for w in tokenize(text.lower()) if w not in STOPWORDS and len(w) > 2]
    return {"{} {}".format(a, b) for a, b in zip(words, words[1:])}


def _phrase_overlap(query_text: str, targets: Sequence[str]) -> float:
    """Distinctive two-word phrases shared with the PS.

    Single words are weak: "agents" and "best" appear in a dozen PSs. "renewal
    collection" or "age profile" appear in one, and that is what separates PSs which
    share both subject and shape.
    """
    query_pairs = _bigrams(query_text)
    if not query_pairs:
        return 0.0
    best = 0.0
    for target in targets:
        shared = query_pairs & _bigrams(target)
        if shared:
            best = max(best, len(shared) / len(query_pairs))
    return best


def _requested_dimensions(cleaned: str, terms: Set[str]) -> Set[str]:
    """Dimensions the question asks to group by, not merely mentions."""
    requested = set()
    for name, words in DIMENSION_TERMS.items():
        if not (terms & words):
            continue
        for word in words & terms:
            if any(pattern.format(word) in cleaned for pattern in GROUPING_PATTERNS):
                requested.add(name)
                break
    return requested


# Evidence mass, in average-IDF words, that the rare-term signal is shrunk by.
# Chosen by sweep: see the Phase 10 routing notes in PROJECT_STATUS.md.
RARE_TERM_PRIOR_WORDS = 0.8

class PSMatcher:
    """Scores problem statements against a question."""

    def __init__(self, registry: PSRegistry):
        self.registry = registry
        self._prepared = {ps.id: self._prepare(ps) for ps in registry.all()}
        self._idf = self._build_idf()
        self._mean_idf = (sum(self._idf.values()) / len(self._idf)) if self._idf else 1.0

    def _build_idf(self) -> Dict[str, float]:
        """How rare each word is across the catalogue.

        Every signal above this one treats "policies" and "underperform" as equally
        informative. They are not. "policies" appears in most of the 103 problem
        statements and separates nothing; "underperform", "settled" and "health"
        appear in one or two, and are the entire reason a reader would say the
        question belongs to a particular analysis.

        Inverse document frequency is the standard way to say that, and it is
        computed from the catalogue itself rather than a hand-kept stop list, so it
        stays true as problem statements are added.
        """
        frequency: Dict[str, int] = {}
        for prepared in self._prepared.values():
            words = set()
            for text in [prepared["raw_statement"]] + prepared["raw_examples"]:
                words |= content_terms(tokenize(text.lower()))
            words |= prepared["concepts"]
            for word in words:
                frequency[word] = frequency.get(word, 0) + 1

        total = max(1, len(self._prepared))
        return {word: math.log(1.0 + total / count)
                for word, count in frequency.items()}

    def _rare_term_overlap(self, terms: Set[str], prepared: Dict[str, Any]) -> float:
        """Shared vocabulary, weighted by how much each shared word tells you.

        Scored against the question's own terms rather than the PS's, so a long
        problem statement gains nothing from length alone.
        """
        if not terms:
            return 0.0
        target = prepared["searchable"]
        # A word absent from the whole catalogue is maximally rare, but it is also
        # not evidence for any particular PS - it can only be shared if it appears,
        # so the default is only used for words this PS does contain.
        default = math.log(1.0 + len(self._prepared))
        budget = sum(self._idf.get(t, default) for t in terms)
        if budget <= 0:
            return 0.0
        earned = sum(self._idf.get(t, default) for t in terms & target)
        # Shrink toward zero by a fixed amount of evidence. Normalising by the
        # question's own terms alone lets a one-word question that shares its one
        # word score a perfect 1.0 - maximum confidence from minimum evidence,
        # which is how "performance" came to be routed rather than refused. The
        # prior is measured in average-IDF words, so it costs a one-word question
        # most of its score and a five-word question very little.
        prior = RARE_TERM_PRIOR_WORDS * self._mean_idf
        return min(1.0, earned / (budget + prior))

    @staticmethod
    def _prepare(ps: ProblemStatement) -> Dict[str, Any]:
        concept_terms: Set[str] = set()
        for concept in ps.concepts:
            concept_terms |= content_terms(tokenize(concept.lower()))
        column_terms: Set[str] = set()
        for column in ps.relevant_columns:
            column_terms |= content_terms(tokenize(column.split(".", 1)[-1].lower()))
        plan = ps.record.get("execution") or {}
        group_by = plan.get("group_by")
        keys = [group_by] if isinstance(group_by, str) else list(group_by or [])
        if plan.get("entity_column"):
            keys.append(plan["entity_column"])
        grouping_keys = {str(k).lower() for k in keys}

        # Columns the PS keys on without grouping: conditions, filters and the
        # columns a derived flag is built from.
        keyed = set(grouping_keys)
        for spec in (plan.get("condition_counts") or {}).values():
            keyed.add(str(spec.get("column", "")).lower())
        for spec in plan.get("filters") or []:
            if spec:
                keyed.add(str(spec[0]).lower())
        for spec in plan.get("derive") or []:
            if spec.get("kind") == "flag" and spec.get("column"):
                keyed.add(str(spec["column"]).lower())
        if plan.get("column"):
            keyed.add(str(plan["column"]).lower())

        dimensions = {name for name, columns in DIMENSION_COLUMNS.items()
                      if grouping_keys & columns}
        keyed_dimensions = {name for name, columns in DIMENSION_COLUMNS.items()
                            if keyed & columns}

        searchable = set(concept_terms) | set(column_terms)
        for text in [ps.problem_statement] + list(ps.example_queries):
            searchable |= content_terms(tokenize(clean(text).lower()))

        return {
            "ps": ps,
            "searchable": searchable,
            "dimensions": dimensions,
            "keyed_dimensions": keyed_dimensions,
            "groups": bool(keys),
            "examples": [expand_aliases(clean(q)) for q in ps.example_queries],
            "raw_examples": [clean(q) for q in ps.example_queries],
            "statement": expand_aliases(clean(ps.problem_statement)),
            "raw_statement": clean(ps.problem_statement),
            "concepts": concept_terms,
            "columns": column_terms,
            "tables": set(ps.relevant_tables),
            "measures": set(ps.intent_type.split("/")),
        }

    def score(self, query: NormalizedQuery,
              entity_kinds: Optional[Sequence[str]] = None,
              shortlist: Optional[Sequence[str]] = None) -> List[Candidate]:
        entity_kinds = set(entity_kinds or [])
        ids = list(shortlist) if shortlist else self.registry.ids
        results = [self._score_one(self._prepared[i], query, entity_kinds) for i in ids]
        results.sort(key=lambda c: c.score, reverse=True)
        return results

    def _score_one(self, prepared: Dict[str, Any], query: NormalizedQuery,
                   entity_kinds: Set[str]) -> Candidate:
        ps: ProblemStatement = prepared["ps"]
        terms = query.terms

        best_example, example_score = "", 0.0
        for example in prepared["examples"]:
            value = _similarity(terms, example)
            if value > example_score:
                best_example, example_score = example, value

        statement_score = _similarity(terms, prepared["statement"])
        phrase_score = _phrase_overlap(
            query.cleaned, prepared["raw_examples"] + [prepared["raw_statement"]])

        concepts = prepared["concepts"]
        concept_score = len(terms & concepts) / len(concepts) if concepts else 0.0

        column_hits = terms & prepared["columns"]
        table_hits = set()
        for table in prepared["tables"]:
            if terms & TABLE_TERMS.get(table, set()):
                table_hits.add(table)
        table_coverage = len(table_hits) / len(prepared["tables"]) if prepared["tables"] else 0.0
        domain_score = min(1.0, 0.6 * table_coverage + 0.4 * min(1.0, len(column_hits) / 3.0))

        cue_text = query.cleaned
        measures = prepared["measures"]

        decisive = set()
        for triggers, accepted in DECISIVE_CUES:
            if any(trigger in cue_text for trigger in triggers):
                decisive |= accepted
        if decisive:
            # A decisive cue settles the shape of the answer. Offering it scores near
            # full; not offering it scores near zero, however similar the wording.
            intent_score = 1.0 if decisive & measures else 0.05
        else:
            matched, satisfied = 0, 0
            for triggers, accepted in INTENT_CUES:
                if any(trigger in cue_text for trigger in triggers):
                    matched += 1
                    if accepted & measures:
                        satisfied += 1
            intent_score = (satisfied / matched) if matched else 0.35

        asked = _requested_dimensions(query.cleaned, terms)
        mentioned = {name for name, words in DIMENSION_TERMS.items() if terms & words}
        offered = prepared["dimensions"]
        if not asked:
            # A value-bearing mention still tilts: "how many unit linked policies"
            # favours the TRAD/ULIP split. Naming a dimension that merely describes
            # the subject ("how old are our customers") does not, because it says
            # nothing about whether a grouping is wanted.
            values = mentioned & VALUE_DIMENSIONS
            keyed = prepared["keyed_dimensions"]
            if values and keyed & values:
                dimension_score = 0.85
            elif values and keyed:
                dimension_score = 0.4
            elif values:
                # The question names a specific status or class; a PS that neither
                # groups nor filters on it answers a different question.
                dimension_score = 0.3
            else:
                dimension_score = 0.5
        elif offered == asked:
            dimension_score = 1.0     # groups by exactly what was asked
        elif asked <= offered:
            dimension_score = 0.78    # offers it, but splits further than asked
        elif offered & asked:
            dimension_score = 0.6     # partial overlap
        else:
            # The question named a grouping this PS does not offer. Ranking individual
            # policies cannot answer "which states write the most business".
            dimension_score = 0.03

        wanted_tiers = {tier for tier, cues in TIER_CUES.items()
                        if any(cue in cue_text for cue in cues)}
        if wanted_tiers:
            tier_score = 1.0 if ps.analysis_tier in wanted_tiers else 0.1
        else:
            tier_score = 0.5

        entity_score = self._entity_fit(ps, entity_kinds)

        components = {
            "example_similarity": example_score,
            "phrase_signal": phrase_score,
            "rare_term_signal": self._rare_term_overlap(terms, prepared),
            "statement_similarity": statement_score,
            "concept_overlap": concept_score,
            "domain_signal": domain_score,
            "dimension_signal": dimension_score,
            "intent_signal": intent_score,
            "tier_signal": tier_score,
            "entity_fit": entity_score,
        }
        total = sum(WEIGHTS[k] * v for k, v in components.items())

        return Candidate(
            problem_id=ps.id,
            problem_statement=ps.problem_statement,
            score=total,
            components=components,
            matched_terms=sorted(terms & (concepts | prepared["columns"])),
            best_example=best_example,
            business_domain=ps.business_domain,
            analysis_tier=ps.analysis_tier,
        )

    @staticmethod
    def _entity_fit(ps: ProblemStatement, entity_kinds: Set[str]) -> float:
        """Reward a PS that can use the identifiers supplied, penalise one that cannot."""
        declared = set(ps.record.get("parameters", {}))
        if "app_id" in entity_kinds:
            return 1.0 if "app_id" in declared else 0.0
        if "sales_id" in entity_kinds:
            return 1.0 if "agent_id" in declared else 0.3
        if not entity_kinds:
            return 0.5
        return 0.5
