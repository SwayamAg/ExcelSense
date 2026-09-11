"""
lookup.py
=========
Direct-lookup answers over the five production tables.

This replaces the graph route. The previous implementation built a NetworkX
DiGraph *out of the pandas fact frames*, translated the question into Cypher,
and interpreted that Cypher with a hand-written parser - roughly 1,170 lines to
answer two question shapes:

    instance lookup     "show details for AppID 1000000015"
    simple aggregate    "how many policies are lapsed"

Both are one pandas expression. The graph added no reachability that a join did
not already give: every table is one hop from `Policy_Details`, 0.96% of clients
hold more than one policy, and there is no free text to mine into relationships.
A depth-1 graph is a join.

Everything analytical continues to go through `analytics_core`; this module only
covers the lookups that never needed a planner.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
for _sub in ["src", "src/analytics", "src/engine", "src/retrieval", "src/llm"]:
    _p = os.path.join(ROOT_DIR, _sub)
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

import analytics_core as ac

# Identifier shapes that occur in these tables. The old resolver carried
# CUST/POL/AGT/CLM patterns from an earlier dataset and matched nothing here.
ENTITY_PATTERNS: List[Tuple[str, str]] = [
    ("app_id", r"\b(?:appid|app id|policy(?:\s*(?:no|number|id))?)\s*#?\s*(\d{8,12})\b"),
    ("client_id", r"\b(?:clientid|client id|client|customer|owner)\s*#?\s*(\d{8,12})\b"),
    ("sales_id", r"\b(?:sales_id|sales id|salesperson|sales person|agent|advisor|rep)\s*#?\s*(\d{4,7})\b"),
    ("claim_no", r"\b(?:claim(?:\s*(?:no|number|id))?)\s*#?\s*(\d{1,8})\b"),
]

ENTITY_FRAME = {"app_id": "policy", "client_id": "customer",
                "sales_id": "agent", "claim_no": "claim"}

# Columns worth showing for each entity, in reading order.
ENTITY_COLUMNS = {
    "app_id": ["app_id", "plan_name", "policy_status", "trad_ulip", "ape",
               "sum_assured", "premium_paying_term", "policy_term", "branch_no",
               "state", "inforce_date", "sales_id"],
    "client_id": ["client_id", "app_id", "owner_name", "age", "gender", "education",
                  "occupation", "earned_income", "city", "state"],
    "sales_id": ["sales_id", "sales_type", "policies_sold", "sales_first_year_persistency",
                 "sales_early_claim_rate", "sales_surrender_rate", "rcu_rejection_rate",
                 "pivc_mismatch_rate", "complaint_rate", "total_ape"],
    "claim_no": ["claim_no", "app_id", "cause_of_death", "claim_amount", "amount_paid",
                 "claim_status", "settlement_tat_days"],
}

# Status words -> the value stored in POLICY.STATUS.
STATUS_WORDS = {
    "lapsed": "Lapsed", "lapse": "Lapsed",
    "surrendered": "Surrendered", "surrender": "Surrendered",
    "death": "Death", "died": "Death",
    "premium paying": "Premium paying (regular)", "in force": "Premium paying (regular)",
    "in-force": "Premium paying (regular)", "inforce": "Premium paying (regular)",
    "active": "Premium paying (regular)", "paying": "Premium paying (regular)",
}

COUNT_CUES = ("how many", "count", "number of", "total number")
SUM_CUES = ("total ", "sum of", "combined")
AVG_CUES = ("average", "avg", "mean")

VALUE_COLUMNS = {
    "ape": ("ape", "policy"), "premium": ("ape", "policy"),
    "sum assured": ("sum_assured", "policy"), "sum_assured": ("sum_assured", "policy"),
    "claim amount": ("claim_amount", "claim"),
    "income": ("earned_income", "customer"), "age": ("age", "customer"),
}


@dataclass
class LookupResult:
    kind: str                       # entity | aggregate | unsupported
    answer: str = ""
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.kind != "unsupported"

    def to_markdown(self, max_rows: int = 20) -> str:
        lines = [self.answer] if self.answer else []
        if not self.frame.empty:
            lines.append("")
            try:
                lines.append(self.frame.head(max_rows).to_markdown(index=False))
            except Exception:
                lines.append("```\n" + self.frame.head(max_rows).to_string(index=False) + "\n```")
        return "\n".join(lines) if lines else "No matching records found."


def resolve_entity(question: str) -> Optional[Tuple[str, str]]:
    """(key, id) named in the question, or None."""
    for key, pattern in ENTITY_PATTERNS:
        m = re.search(pattern, question or "", re.IGNORECASE)
        if m:
            return key, m.group(1)
    return None


def _status_filter(question: str) -> Optional[str]:
    q = (question or "").lower()
    for word, value in sorted(STATUS_WORDS.items(), key=lambda kv: -len(kv[0])):
        if word in q:
            return value
    return None


def _match_id(frame: pd.DataFrame, key: str, ident: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """Find a record, tolerating the documented AppID form.

    Every AppID in the CSVs is 11 digits beginning with `1`
    (`15000635514`), while the metadata document quotes 10-digit samples
    without it (`5000635514`). Someone reading the documentation - or the app's
    own example questions - types the shorter form and gets nothing back. An
    exact match is tried first; the prefixed form second, and the answer says
    which one matched.
    """
    exact = frame[frame[key].astype(str) == str(ident)]
    if not exact.empty:
        return exact, None
    # Stored AppIDs are 11 digits; a documented 10-digit one is the same record
    # with the leading 1 dropped. This holds whatever the second digit is, so
    # the test is the length, not the first character.
    if key == "app_id" and len(str(ident)) == 10:
        prefixed = f"1{ident}"
        alt = frame[frame[key].astype(str) == prefixed]
        if not alt.empty:
            return alt, prefixed
    return exact, None


def lookup_entity(ctx: ac.DataContext, key: str, ident: str) -> LookupResult:
    """One record, shown as a field/value table."""
    frame = ctx.frame(ENTITY_FRAME[key])
    if key not in frame.columns:
        return LookupResult("unsupported",
                            f"`{key}` is not present in the {ENTITY_FRAME[key]} table.")
    sub, matched_as = _match_id(frame, key, ident)
    if sub.empty:
        return LookupResult("entity", f"No record found for {key} {ident}.")
    if matched_as:
        ident = matched_as

    cols = [c for c in ENTITY_COLUMNS.get(key, []) if c in sub.columns]
    row = sub.iloc[0]
    table = pd.DataFrame(
        [{"Field": c.replace("_", " ").title(), "Value": _fmt(row[c])} for c in cols])

    label = {"app_id": "Policy", "client_id": "Customer",
             "sales_id": "Agent", "claim_no": "Claim"}[key]
    extra = f" ({len(sub):,} rows)" if len(sub) > 1 else ""
    note = ""
    if matched_as:
        note = (f"\n\n> Matched as `{matched_as}`. Every AppID in these files is "
                f"11 digits beginning with `1`; the metadata document quotes the "
                f"10-digit form.")
    return LookupResult("entity", f"**{label} {ident}**{extra}{note}", table,
                        {"key": key, "id": ident, "matches": len(sub),
                         "matched_as": matched_as})


def lookup_aggregate(ctx: ac.DataContext, question: str) -> LookupResult:
    """A count, sum or average over one fact frame, with an optional filter."""
    q = (question or "").lower()
    status = _status_filter(question)

    value_col, base = None, "policy"
    for phrase, (col, frame_name) in VALUE_COLUMNS.items():
        if phrase in q:
            value_col, base = col, frame_name
            break

    frame = ctx.frame(base)
    filtered = frame
    if status and "policy_status" in frame.columns:
        filtered = frame[frame["policy_status"] == status]

    scope = f" with status {status}" if status else ""
    n_total = len(frame)

    if any(c in q for c in AVG_CUES) and value_col and value_col in filtered.columns:
        val = float(filtered[value_col].mean())
        answer = (f"Average {value_col.replace('_', ' ')}{scope} is "
                  f"**{val:,.2f}** across {len(filtered):,} records.")
    elif any(c in q for c in SUM_CUES) and value_col and value_col in filtered.columns:
        val = float(filtered[value_col].sum())
        answer = (f"Total {value_col.replace('_', ' ')}{scope} is "
                  f"**{val:,.0f}** across {len(filtered):,} records.")
    elif any(c in q for c in COUNT_CUES) or status:
        pct = (len(filtered) / n_total * 100) if n_total else 0.0
        answer = (f"**{len(filtered):,}** of {n_total:,} records{scope} "
                  f"({pct:.1f}%).")
    else:
        return LookupResult("unsupported", "")

    return LookupResult("aggregate", answer, pd.DataFrame(),
                        {"base": base, "status": status, "n": len(filtered)})


def lookup_top_records(ctx: ac.DataContext, question: str) -> LookupResult:
    """List top N individual records (claims, policies, agents) with optional filters."""
    q = (question or "").lower()
    is_top_request = bool(re.search(r"\b(top|highest|largest|list|show|first)\s*(\d+)?\b", q)) and any(
        w in q for w in ("claim", "policy", "policies", "agent", "advisor", "customer")
    )
    if not is_top_request and not ("highest value" in q or "top value" in q or "largest claim" in q):
        return LookupResult("unsupported", "")

    m_n = re.search(r"\b(?:top|first|largest|highest)\s*(\d+)\b", q)
    n = int(m_n.group(1)) if m_n else 10
    n = min(max(n, 1), 100)

    # 1. Claims
    if "claim" in q:
        df = ctx.frame("claim").copy()
        if df.empty:
            return LookupResult("entity", "No claims records found.")
        
        # Check zone/state filter
        scope_parts = []
        if "western" in q or "west zone" in q or "west region" in q or ("west" in q and "west bengal" not in q):
            zone_match = df["zone"].astype(str).str.lower().isin(["west", "western"]) if "zone" in df.columns else pd.Series(False, index=df.index)
            state_match = df["state"].isin(["Maharashtra", "Gujarat", "Goa", "Rajasthan", "Madhya Pradesh", "Chhattisgarh"]) & (df["state"] != "West Bengal")
            df = df[zone_match | state_match]
            scope_parts.append("Western zone")
        elif "north" in q:
            zone_match = df["zone"].astype(str).str.lower().isin(["north", "northern"]) if "zone" in df.columns else pd.Series(False, index=df.index)
            state_match = df["state"].isin(["Delhi", "Punjab", "Haryana", "Uttar Pradesh", "Himachal Pradesh", "Uttarakhand", "Jammu & Kashmir"])
            df = df[zone_match | state_match]
            scope_parts.append("Northern zone")
        elif "south" in q:
            zone_match = df["zone"].astype(str).str.lower().isin(["south", "southern"]) if "zone" in df.columns else pd.Series(False, index=df.index)
            state_match = df["state"].isin(["Karnataka", "Tamil Nadu", "Kerala", "Andhra Pradesh", "Telangana"])
            df = df[zone_match | state_match]
            scope_parts.append("Southern zone")
        elif "east" in q:
            zone_match = df["zone"].astype(str).str.lower().isin(["east", "eastern"]) if "zone" in df.columns else pd.Series(False, index=df.index)
            state_match = df["state"].isin(["West Bengal", "Bihar", "Odisha", "Jharkhand", "Assam", "Sikkim", "Tripura", "Meghalaya"])
            df = df[zone_match | state_match]
            scope_parts.append("Eastern zone")

        status = _status_filter(question)
        if status and "claim_status" in df.columns:
            df = df[df["claim_status"] == status]
            scope_parts.append(f"status '{status}'")

        if df.empty:
            return LookupResult("entity", f"No claims found matching {' and '.join(scope_parts)}.")

        sorted_df = df.sort_values(by="claim_amount", ascending=False).head(n)
        cols_present = [c for c in ["claim_no", "app_id", "cause_of_death", "claim_amount", "amount_paid", "claim_status", "state"] if c in sorted_df.columns]
        table_df = sorted_df[cols_present].copy()
        
        total_top_amt = float(table_df["claim_amount"].sum())
        scope_str = f" in {' and '.join(scope_parts)}" if scope_parts else ""
        answer_text = f"**Top {len(table_df)} highest value claims**{scope_str}, totaling **₹{total_top_amt:,.0f}** across records (highest is Claim #{table_df.iloc[0]['claim_no']} at ₹{table_df.iloc[0]['claim_amount']:,.0f})."
        
        # Format table columns for presentation
        col_rename = {
            "claim_no": "Claim No",
            "app_id": "App ID",
            "cause_of_death": "Cause of Death",
            "claim_amount": "Claim Amount",
            "amount_paid": "Amount Paid",
            "claim_status": "Status",
            "state": "State",
        }
        table_df = table_df.rename(columns=col_rename)
        for col in ["Claim Amount", "Amount Paid"]:
            if col in table_df.columns:
                table_df[col] = table_df[col].apply(lambda v: f"₹{v:,.0f}" if pd.notna(v) else "-")

        return LookupResult("entity", answer_text, table_df, {"base": "claim", "n": len(table_df), "scope": scope_parts})

    # 2. Policies
    if "policy" in q or "policies" in q:
        df = ctx.frame("policy").copy()
        sort_col = "ape" if "sum assured" not in q else "sum_assured"
        sorted_df = df.sort_values(by=sort_col, ascending=False).head(n)
        cols_present = [c for c in ["app_id", "plan_name", "policy_status", "trad_ulip", "ape", "sum_assured", "state"] if c in sorted_df.columns]
        table_df = sorted_df[cols_present].copy()
        col_rename = {
            "app_id": "App ID",
            "plan_name": "Plan Name",
            "policy_status": "Status",
            "trad_ulip": "Category",
            "ape": "Annual Premium (APE)",
            "sum_assured": "Sum Assured",
            "state": "State",
        }
        table_df = table_df.rename(columns=col_rename)
        answer_text = f"**Top {len(table_df)} policies** by {sort_col.replace('_', ' ').upper()} across the portfolio."
        return LookupResult("entity", answer_text, table_df, {"base": "policy", "n": len(table_df)})

    return LookupResult("unsupported", "")


def answer(ctx: ac.DataContext, question: str) -> LookupResult:
    """Entity lookup first, then top-records listing, then simple aggregate."""
    hit = resolve_entity(question)
    if hit:
        return lookup_entity(ctx, hit[0], hit[1])
    top_res = lookup_top_records(ctx, question)
    if top_res.ok:
        return top_res
    return lookup_aggregate(ctx, question)


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    if isinstance(v, float):
        return f"{v:,.2f}" if abs(v) < 1e6 else f"{v:,.0f}"
    return str(v)


__all__ = ["LookupResult", "answer", "resolve_entity",
           "lookup_entity", "lookup_top_records", "lookup_aggregate"]
