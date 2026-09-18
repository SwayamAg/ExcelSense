"""Query normalization and business vocabulary.

Executives do not speak in column names. This module maps the language they do use
onto the concepts the catalogue is built from, without ever inventing a concept the
data cannot support.
"""
from __future__ import annotations

import re
from typing import Dict, List, Set

CONTRACTIONS = {
    "what's": "what is", "whats": "what is", "who's": "who is", "whos": "who is",
    "how's": "how is", "where's": "where is", "isn't": "is not", "aren't": "are not",
    "don't": "do not", "doesn't": "does not", "didn't": "did not", "can't": "cannot",
    "won't": "will not", "we're": "we are", "i'm": "i am", "let's": "let us",
    "there's": "there is", "that's": "that is", "it's": "it is", "you're": "you are",
}

# Business vocabulary -> canonical concept terms. Multi-word phrases are matched
# first so "sum assured" does not decompose into "sum" and "assured".
ALIASES: Dict[str, str] = {
    # portfolio
    "the book": "portfolio", "our book": "portfolio", "book of business": "portfolio",
    "whole portfolio": "portfolio", "overall portfolio": "portfolio",

    # premium
    "annualised premium equivalent": "ape premium", "annualized premium equivalent": "ape premium",
    "annual premium": "ape premium", "annualised premium": "ape premium",
    "premium income": "ape premium", "new business premium": "ape premium",
    "case size": "ape premium average", "ticket size": "ape premium average",
    "average premium": "ape premium average",

    # cover
    "sum assured": "sum_assured cover", "sum insured": "sum_assured cover",
    "cover amount": "sum_assured cover", "coverage": "sum_assured cover",
    "face amount": "sum_assured cover", "exposure": "sum_assured cover exposure",

    # status
    "in force": "inforce active", "in-force": "inforce active",
    "premium paying": "inforce active", "still paying": "inforce active",
    "active policies": "inforce active", "live policies": "inforce active",
    "lapsed policies": "lapse lapsed", "lapse rate": "lapse rate",
    "fallen off": "lapse lapsed", "stopped paying": "lapse lapsed",
    "surrendered policies": "surrender surrendered", "cashed in": "surrender surrendered",
    "early exit": "surrender surrendered",

    # persistency
    "persistency": "persistency renewal", "renewal rate": "persistency renewal rate",
    "13th month": "persistency bucket 13th", "25th month": "persistency bucket 25th",
    "37th month": "persistency bucket 37th",
    "collection": "renewal collection", "collections": "renewal collection",
    "retention": "retention persistency", "stickiness": "retention persistency",

    # distribution
    "agent": "agent sales", "agents": "agent sales", "advisor": "agent sales",
    "advisors": "agent sales", "adviser": "agent sales", "seller": "agent sales",
    "sales person": "agent sales", "salesperson": "agent sales",
    "producers": "agent sales production volume",
    "branch": "branch", "branches": "branch", "office": "branch",

    # product
    "product": "product plan", "products": "product plan", "plan": "product plan",
    "plans": "product plan", "scheme": "product plan",
    "unit linked": "ulip", "unit-linked": "ulip", "traditional": "trad traditional",

    # customer
    "customer": "customer owner client", "customers": "customer owner client",
    "client": "customer owner client", "clients": "customer owner client",
    "policyholder": "customer owner client", "policyholders": "customer owner client",
    "policy holder": "customer owner client", "owner": "customer owner client",

    # claims
    "death claim": "claim death", "death claims": "claim death",
    "payout": "claim payout paid", "payouts": "claim payout paid",
    "settlement": "claim settlement settled", "settled": "claim settlement settled",
    "cause of death": "cause death mortality", "mortality": "cause death mortality",

    # Deliberately absent: superlatives ("top", "best", "largest"), comparison words,
    # "why", "trend", "snapshot" and other analysis verbs. Expanding those would add
    # the same tokens to every question of a given shape and swamp the subject matter.
    # They are read as intent cues in matcher.py instead.

    # quality / conduct
    "complaint": "complaint conduct", "complaints": "complaint conduct",
    "mis-selling": "conduct misconduct", "misselling": "conduct misconduct",
    "conduct": "conduct quality", "quality score": "composite quality score",
    "free look": "freelook cancellation", "freelook": "freelook cancellation",

    # production and volume - nouns only, no ranking words
    "producing": "production volume",
    "producer": "agent sales production",
    "business": "policies volume",
    "how big": "count number size",
    "how large": "count number size",
    "how old": "age profile demographic",
    "distinct": "distinct unique count",
    "unique": "distinct unique count",
    "typically pay": "average premium ape",
    "typically buy": "average premium ape",
    "value segmentation": "customer value segment tier",
    "remediation": "agent conduct quality action",
    "retention effort": "retention persistency lapse",

    # integrity
    "data quality": "data quality integrity", "reconcile": "reconciliation integrity",
    "reconciliation": "reconciliation integrity", "integrity": "reconciliation integrity",
    "limitations": "scope limitations", "cannot answer": "scope limitations unsupported",
}

# Concepts the five CSVs cannot support, mapped to what is actually missing.
# A question resting on one of these is refused rather than approximated.
UNSUPPORTED_CONCEPTS: Dict[str, str] = {
    "smoker": "no smoking or lifestyle field exists",
    "smoking": "no smoking or lifestyle field exists",
    "bmi": "no height, weight or BMI field exists",
    "medical": "no medical or underwriting data exists",
    "underwriting decision": "no underwriting outcome is recorded",
    "underwriting risk": "no risk score or underwriting rating exists",
    "risk score": "no risk score is recorded for any policy or customer",
    "health condition": "no health or morbidity data exists",
    "commission": "no commission or incentive data exists",
    "incentive": "no commission or incentive data exists",
    "profitability": "no cost, expense or margin data exists",
    "profit": "no cost, expense or margin data exists",
    "profitable": "no cost, expense or margin data exists",
    "unprofitable": "no cost, expense or margin data exists",
    "loss making": "no cost, expense or margin data exists",
    "margin": "no cost, expense or margin data exists",
    "expense": "no cost or expense data exists",
    "reserve": "no reserving data exists",
    "solvency": "no capital or solvency data exists",
    "nav": "no fund, NAV or unit-price data exists",
    "fund performance": "no fund, NAV or unit-price data exists",
    "fund value": "no fund, NAV or unit-price data exists",
    "unit price": "no fund, NAV or unit-price data exists",
    "rider": "no rider or benefit-structure data exists",
    "nominee": "no nominee or beneficiary data exists",
    "maturity": "no maturity event or maturity benefit data exists",
    "zone": "no zone or regional hierarchy exists; geography is the owner address only",
    "region": "no zone or regional hierarchy exists; geography is the owner address only",
    "territory": "no sales territory is recorded",
    "target": "no agent target or production goal is recorded",
    "quota": "no agent target or production goal is recorded",
    "agent tenure": "no agent tenure, joining date or experience is recorded",
    "agent age": "no agent demographics are recorded",
    "training": "no training or certification data exists",
    "surrender value": "no surrender value or surrender charge is recorded",
    "surrender charge": "no surrender value or surrender charge is recorded",
    "surrender reason": "no surrender reason is recorded",
    "lapse reason": "no lapse reason is recorded",
    "reason did": "no reason code or stated reason is captured anywhere in the data",
    "reason given": "no reason code or stated reason is captured anywhere in the data",
    "reason code": "no reason code or stated reason is captured anywhere in the data",
    "stated reason": "no reason code or stated reason is captured anywhere in the data",
    "reason customers give": "no reason code or stated reason is captured in the data",
    "reason for surrendering": "no surrender reason is recorded",
    "reason for lapsing": "no lapse reason is recorded",
    "lapse date": "no lapse date is recorded; lapse is a status only",
    "payment history": "no premium payment transaction history exists",
    "premiums paid": "no cumulative premium-paid figure exists",
    "marital status": "no marital status is recorded",
    "channel comparison": "CHANNEL is 'Banca' for every policy, so it cannot be compared",
    "compare channels": "CHANNEL is 'Banca' for every policy, so it cannot be compared",
    "by channel": "CHANNEL is 'Banca' for every policy, so it cannot be compared",
    "complaint case": "only an aggregate agent-level complaint rate exists, with no case records",
    "customer service": "no servicing or contact history exists",
    "call centre": "no servicing or contact history exists",
    "email": "no contact details or communication history exists",
    "phone": "no contact details or communication history exists",
}

STOPWORDS: Set[str] = {
    # Conversational filler. "Tell me about performance" and "how are we doing"
    # carry no more analytical content than "performance" alone, but counting
    # these as terms let them clear the confidence floor as two-word questions.
    "about", "how", "doing",
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "to", "in", "on",
    "for", "with", "and", "or", "do", "does", "did", "have", "has", "had", "we", "our",
    "us", "i", "me", "my", "you", "your", "it", "its", "this", "that", "these", "those",
    "there", "here", "what", "which", "show", "give", "get", "tell", "please", "can",
    "could", "would", "should", "at", "by", "from", "as", "so", "than", "then",
}

_WORD = re.compile(r"[a-z0-9_.]+")


class NormalizedQuery:
    """A question reduced to the terms the catalogue can be matched against."""

    def __init__(self, raw: str, cleaned: str, expanded: str,
                 tokens: List[str], terms: Set[str]):
        self.raw = raw
        self.cleaned = cleaned
        self.expanded = expanded
        self.tokens = tokens
        self.terms = terms

    def as_dict(self) -> Dict[str, object]:
        return {"raw": self.raw, "cleaned": self.cleaned, "expanded": self.expanded,
                "tokens": list(self.tokens), "terms": sorted(self.terms)}


def clean(text: str) -> str:
    """Lowercase, expand contractions, strip punctuation that carries no meaning."""
    lowered = text.lower().strip()
    for contraction, expansion in CONTRACTIONS.items():
        lowered = re.sub(r"\b{}\b".format(re.escape(contraction)), expansion, lowered)
    lowered = lowered.replace("%", " percent ").replace("&", " and ")
    lowered = re.sub(r"[?!,;:\"'()\[\]]+", " ", lowered)
    lowered = re.sub(r"\s*-\s*", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def expand_aliases(text: str) -> str:
    """Append canonical concept terms for any business phrasing present.

    The original wording is kept: an alias adds vocabulary rather than replacing it,
    so a query never loses a term that might match something else.
    """
    additions: List[str] = []
    for phrase in sorted(ALIASES, key=len, reverse=True):
        if re.search(r"\b{}\b".format(re.escape(phrase)), text):
            additions.append(ALIASES[phrase])
    return (text + " " + " ".join(additions)).strip() if additions else text


def tokenize(text: str) -> List[str]:
    return _WORD.findall(text)


def content_terms(tokens: List[str]) -> Set[str]:
    return {t for t in tokens if t not in STOPWORDS and len(t) > 1}


def normalize(text: str) -> NormalizedQuery:
    cleaned = clean(text)
    expanded = expand_aliases(cleaned)
    tokens = tokenize(expanded)
    return NormalizedQuery(text, cleaned, expanded, tokens, content_terms(tokens))


def find_unsupported(text: str) -> List[Dict[str, str]]:
    """Concepts the question depends on that the datasets do not contain."""
    cleaned = clean(text)
    found = []
    seen = set()
    for phrase in sorted(UNSUPPORTED_CONCEPTS, key=len, reverse=True):
        # Allow a simple plural on the phrase: "smokers" and "riders" must be caught
        # by "smoker" and "rider". Without this a question sails past the refusal and
        # gets answered from whatever happens to match next, which is the exact
        # failure this check exists to prevent.
        if not re.search(r"\b{}(?:s|es)?\b".format(re.escape(phrase)), cleaned):
            continue
        reason = UNSUPPORTED_CONCEPTS[phrase]
        if reason in seen:
            continue
        seen.add(reason)
        found.append({"concept": phrase, "reason": reason})
    return found
