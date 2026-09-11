"""
analytics_core.py
=================
Deterministic analytical engine for the HSBC Canara Life Insurance Round 2 Dataset.

This module is the *numerical truth layer*. Nothing in here calls an LLM and
nothing in here guesses: every number produced by this module is computed by
pandas from the raw CSVs, with an explicitly declared denominator.

Three layers:
  1. DataContext   - loads the 5 CSVs and builds five analysis-ready
                     "fact frames" (policy / claim / persistency / customer / agent),
                     each carrying the same set of business dimensions so any
                     metric can be grouped by any dimension.

  2. MetricSpec    - a named business metric with an explicit numerator,
     + METRICS       denominator, population definition, unit and polarity.
                     `METRICS` is the registry the planner selects from.

  3. Operations    - the executable verbs an analytical plan is made of:
                     metric_by_dimension, compare_to_baseline, rank,
                     composite_score, driver_scan, concentration, outliers,
                     crosstab, portfolio_overview, entity_shortlist, segment_profile.

Design rules honoured here:
  * Denominators are declared, not implied. `persistency_13m_rate` is measured only
    over the 13th Month cohort; `claim_settlement_ratio` is measured only over claims.
  * Small groups are flagged (`reliable=False`) rather than silently ranked.
  * Zero denominators return NaN, never a divide-by-zero or a fake 0.
"""

from __future__ import annotations

import os
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Sequence

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Terminal status
TERMINAL_STATUSES = ("Death", "Surrendered")

# Below this many units in the denominator we still report the number but flag it.
DEFAULT_MIN_N = 20


# =====================================================================
# 1. DATA CONTEXT
# =====================================================================

def _band(series: pd.Series, edges: Sequence[float], labels: Sequence[str]) -> pd.Series:
    return pd.cut(series, bins=list(edges), labels=list(labels), include_lowest=True).astype(object)


def _clean_num(val: Any) -> float:
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).replace(",", "").replace("$", "").replace("₹", "").strip()
    try:
        return float(s)
    except Exception:
        return np.nan


class DataContext:
    """Loads the raw CSVs and derives the analysis-ready fact frames."""

    _singleton: Optional["DataContext"] = None

    @classmethod
    def get(cls, data_dir: str = DATA_DIR) -> "DataContext":
        if cls._singleton is None or cls._singleton.data_dir != data_dir:
            cls._singleton = cls(data_dir)
        return cls._singleton

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self._load_raw()
        self._build_policy_fact()
        self._build_claim_fact()
        self._build_persistency_fact()
        self._build_customer_fact()
        self._build_agent_fact()

    # -- raw -----------------------------------------------------------
    def _read(self, fname: str) -> pd.DataFrame:
        candidates = [
            os.path.join(self.data_dir, "csv", fname),
            os.path.join(self.data_dir, fname),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "csv", fname),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", fname),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "csv", fname),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", fname),
        ]
        for p in candidates:
            if os.path.exists(p):
                return pd.read_csv(p, low_memory=False)
        return pd.read_csv(fname, low_memory=False)

    def _load_raw(self):
        self.raw_policies = self._read("Policy_Details.csv")
        self.raw_owners = self._read("Owner_Details.csv")
        self.raw_sales = self._read("Sales_Details.csv")
        self.raw_claims = self._read("Claims_Details.csv")
        self.raw_persistency = self._read("Persistency_Details.csv")

    # -- policy fact ---------------------------------------------------
    def _build_policy_fact(self):
        p = self.raw_policies.copy()
        
        # Standardize column names
        p = p.rename(columns={
            "AppID": "app_id",
            "SALES_ID": "sales_id",
            "Branch_No": "branch_no",
            "Prod_No": "prod_no",
            "POLICY.STATUS": "policy_status",
            "CHANNEL": "channel",
            "Login.Date": "login_date",
            "Inforce.Date": "inforce_date",
            "Inforce.month": "inforce_month",
            "Trad.ULIP": "trad_ulip",
            "Plan.Name": "plan_name",
            "Mode": "mode",
            "PREMIUM.PAYING.TERM": "premium_paying_term",
            "Policy.Term": "policy_term",
            "Sum.Assured": "sum_assured",
            "APE": "ape",
        })
        p["app_id"] = p["app_id"].astype("int64")
        p["sales_id"] = p["sales_id"].astype("int64")
        p["branch_no"] = p["branch_no"].astype("int64")
        p["sum_assured"] = pd.to_numeric(p["sum_assured"], errors="coerce").fillna(0.0)
        p["ape"] = pd.to_numeric(p["ape"], errors="coerce").fillna(0.0)
        p["annual_premium"] = p["ape"]  # APE is annualised premium equivalent

        # Merge Owner details
        o = self.raw_owners.copy().rename(columns={
            "AppID": "app_id",
            "ClientID": "client_id",
            "Owner.Given.Name": "owner_name",
            "Owner.Gender": "gender",
            "Owner.Age": "age",
            "Owner.Education": "education",
            "Owner.Earned.Income": "earned_income",
            "Owner.Comm.City.Name": "city",
            "Owner.Comm.State": "state",
            "Owner.Comm.Country": "country",
            "Owner.Comm.Postal.Code": "postal_code",
            "Owner.Occupation": "occupation",
        })
        o["app_id"] = pd.to_numeric(o["app_id"], errors="coerce").fillna(0).astype("int64")
        o["client_id"] = o["client_id"].astype(str)
        o["earned_income"] = pd.to_numeric(o["earned_income"], errors="coerce").fillna(0.0)
        o["age"] = pd.to_numeric(o["age"], errors="coerce")
        o["annual_income"] = o["earned_income"]
        o["gender"] = o["gender"].fillna("Unknown")
        o["occupation"] = o["occupation"].fillna("Others")
        o["state"] = o["state"].fillna("Unknown")
        o["city"] = o["city"].fillna("Unknown")
        o["country"] = o["country"].fillna("India")
        o["is_nri"] = o["country"].ne("India")
        
        # Merge owner
        p = p.merge(o, on="app_id", how="left")
        p["earned_income"] = p["earned_income"].fillna(0.0)
        p["annual_income"] = p["earned_income"]
        p["age"] = p["age"].fillna(p["age"].median())

        # Merge Sales quality indicators
        s = self.raw_sales.copy().rename(columns={
            "SALES_ID": "sales_id",
            "Surrender Rate": "sales_surrender_rate",
            "Early Claim Rate": "sales_early_claim_rate",
            "PIVC Number Mismatch Rate": "pivc_mismatch_rate",
            "PIVC Concern Raised Rate": "pivc_concern_rate",
            "RCU Rejections Rate": "rcu_rejection_rate",
            "Free Look  Cancellation Rate": "free_look_rate",
            "Sales Related Complaint": "complaint_rate",
            "First Year Persistency Rate": "sales_first_year_persistency",
            "Type": "sales_type",
        })
        s["sales_id"] = s["sales_id"].astype("int64")
        for col in ["sales_surrender_rate", "sales_early_claim_rate", "pivc_mismatch_rate",
                    "pivc_concern_rate", "rcu_rejection_rate", "free_look_rate",
                    "complaint_rate", "sales_first_year_persistency"]:
            s[col] = pd.to_numeric(s[col], errors="coerce")
        p = p.merge(s, on="sales_id", how="left")

        # Claims rollup
        cl = self.raw_claims.copy().rename(columns={
            "AppID": "app_id",
            "Claim_No": "claim_no",
            "Date.of.Death": "date_of_death",
            "Cause.of.Death": "cause_of_death",
            "Claim.Amount": "claim_amount_raw",
            "Date.of.Intimation": "date_of_intimation",
            "Status.of.Claim": "claim_status",
            "Amount.Paid": "amount_paid_raw",
            "Settlement.Date": "settlement_date",
        })
        cl["app_id"] = pd.to_numeric(cl["app_id"], errors="coerce").fillna(0).astype("int64")
        cl["claim_amount"] = cl["claim_amount_raw"].apply(_clean_num).fillna(0.0)
        cl["amount_paid"] = cl["amount_paid_raw"].apply(_clean_num).fillna(0.0)
        cl["is_settled_paid"] = cl["claim_status"].eq("Paid")
        cl["is_wip"] = cl["claim_status"].eq("WIP")
        
        cl_agg = cl.groupby("app_id").agg(
            claim_count=("claim_no", "count"),
            claim_amount_total=("claim_amount", "sum"),
            amount_paid_total=("amount_paid", "sum"),
            claim_status_latest=("claim_status", "first"),
            cause_of_death_latest=("cause_of_death", "first"),
        ).reset_index()
        p = p.merge(cl_agg, on="app_id", how="left")
        p["claim_count"] = p["claim_count"].fillna(0)
        p["claim_amount_total"] = p["claim_amount_total"].fillna(0.0)
        p["amount_paid_total"] = p["amount_paid_total"].fillna(0.0)
        p["has_claim"] = p["claim_count"] > 0

        # Persistency rollup
        r = self.raw_persistency.copy().rename(columns={
            "AppID": "app_id",
            "SALES_ID": "sales_id_pers",
            "POLICY.STATUS": "policy_status_pers",
            "CHANNEL": "channel_pers",
            "Inforce.Date": "inforce_date_pers",
            "PREMIUM.PAYING.TERM": "ppt_pers",
            "Renewal_Due_Date": "renewal_due_date",
            "Renewal_Paid_Date": "renewal_paid_date",
            "Renewal_Status": "renewal_status",
            "Renewal_Premium_Amount": "renewal_premium_amount",
            "Persistency_Bucket": "persistency_bucket",
            "Payment_Mode": "payment_mode",
        })
        r["app_id"] = pd.to_numeric(r["app_id"], errors="coerce").fillna(0).astype("int64")
        r["renewal_premium_amount"] = pd.to_numeric(r["renewal_premium_amount"], errors="coerce").fillna(0.0)
        r["is_renewed_paid"] = r["renewal_status"].eq("Paid")
        r["is_renewed_pending"] = r["renewal_status"].eq("Pending")

        r_agg = r.groupby("app_id").agg(
            persistency_bucket=("persistency_bucket", "first"),
            renewal_status=("renewal_status", "first"),
            renewal_premium_amount=("renewal_premium_amount", "first"),
            payment_mode=("payment_mode", "first"),
            is_renewed_paid=("is_renewed_paid", "first"),
            renewal_due_date=("renewal_due_date", "first"),
            renewal_paid_date=("renewal_paid_date", "first"),
        ).reset_index()
        p = p.merge(r_agg, on="app_id", how="left")
        p["has_persistency_record"] = p["persistency_bucket"].notna()
        p["is_renewed_paid"] = p["is_renewed_paid"].eq(True)

        # Status Flags
        st = p["policy_status"]
        p["is_inforce"] = st.eq("Premium paying (regular)")
        p["is_lapsed"] = st.eq("Lapsed")
        p["is_surrendered"] = st.eq("Surrendered")
        p["is_death"] = st.eq("Death")
        p["is_attrition"] = p["is_lapsed"] | p["is_surrendered"]
        p["is_terminal_by_design"] = p["is_death"]

        # Ratios
        p["ape_to_income"] = p["ape"] / p["earned_income"].replace(0, np.nan)
        p["sa_to_income"] = p["sum_assured"] / p["earned_income"].replace(0, np.nan)
        p["sa_to_ape"] = p["sum_assured"] / p["ape"].replace(0, np.nan)

        # Dates / Vintage
        p["inforce_date_dt"] = pd.to_datetime(p["inforce_date"], errors="coerce", format="mixed")
        p["inforce_year"] = p["inforce_date_dt"].dt.year
        ref = p["inforce_date_dt"].max()
        if pd.notna(ref):
            p["policy_vintage_years"] = ((ref - p["inforce_date_dt"]).dt.days / 365.25).round(2)
        else:
            p["policy_vintage_years"] = 0.0

        # Segment Bands
        p["income_band"] = _band(p["earned_income"],
                                 [0, 500_000, 1_000_000, 2_500_000, 5_000_000, 1e12],
                                 ["<5L", "5-10L", "10-25L", "25-50L", "50L+"])
        p["age_band"] = _band(p["age"], [0, 30, 40, 50, 60, 200],
                              ["<30", "30-39", "40-49", "50-59", "60+"])
        p["sa_band"] = _band(p["sum_assured"],
                             [0, 250_000, 500_000, 1_000_000, 2_500_000, 1e12],
                             ["<2.5L", "2.5-5L", "5-10L", "10-25L", "25L+"])
        p["ape_band"] = _band(p["ape"],
                              [0, 25_000, 50_000, 100_000, 250_000, 1e12],
                              ["<25k", "25-50k", "50k-1L", "1-2.5L", "2.5L+"])
        p["ppt_band"] = _band(p["premium_paying_term"], [0, 5, 7, 10, 12, 100],
                              ["<=5y", "6-7y", "8-10y", "11-12y", "12y+"])
        p["term_band"] = _band(p["policy_term"], [0, 10, 15, 20, 25, 100],
                               ["<=10y", "11-15y", "16-20y", "21-25y", "25y+"])
        p["ape_to_income_band"] = _band(p["ape_to_income"],
                                        [-1, 0.05, 0.10, 0.20, 0.40, 1e6],
                                        ["<5%", "5-10%", "10-20%", "20-40%", "40%+"])

        # Aliases for compatibility with legacy queries
        p["policy_number"] = p["app_id"].astype(str)
        p["customer_id"] = p["client_id"]
        p["product_name"] = p["plan_name"]
        p["product_category"] = p["trad_ulip"]
        p["sales_channel"] = p["channel"]
        p["zone"] = p["state"]  # Geographic region
        p["branch_id"] = p["branch_no"].astype(str)
        p["servicing_agent_id"] = p["sales_id"].astype(str)
        p["agent_name"] = "Agent_" + p["sales_id"].astype(str)
        p["total_premium_paid"] = p["ape"] * p["premium_paying_term"].clip(lower=1)
        p["rider_flag"] = "Standard"

        self.policy_fact = p

    # -- dimension columns shared by all fact frames --------------------
    POLICY_DIM_COLS = [
        "app_id", "policy_number", "client_id", "customer_id", "sales_id", "servicing_agent_id",
        "branch_no", "branch_id", "prod_no", "plan_name", "product_name", "trad_ulip",
        "product_category", "channel", "sales_channel", "policy_status", "mode",
        "premium_paying_term", "policy_term", "sum_assured", "ape", "annual_premium",
        "total_premium_paid", "owner_name", "gender", "age", "education", "earned_income",
        "annual_income", "city", "state", "zone", "country", "postal_code", "occupation",
        "is_nri", "income_band", "age_band", "sa_band", "ape_band", "ppt_band", "term_band",
        "sales_type", "pivc_mismatch_rate", "pivc_concern_rate", "rcu_rejection_rate",
        "free_look_rate", "complaint_rate", "sales_first_year_persistency",
        "persistency_bucket", "payment_mode", "renewal_status", "is_renewed_paid",
    ]

    def _attach_dims(self, df: pd.DataFrame, join_key: str = "app_id") -> pd.DataFrame:
        dims = self.policy_fact[self.POLICY_DIM_COLS].drop_duplicates(join_key)
        return df.merge(dims, on=join_key, how="left", suffixes=("", "_pol"))

    def _build_claim_fact(self):
        c = self.raw_claims.copy().rename(columns={
            "AppID": "app_id",
            "Claim_No": "claim_no",
            "Date.of.Death": "date_of_death",
            "Cause.of.Death": "cause_of_death",
            "Claim.Amount": "claim_amount_raw",
            "Date.of.Intimation": "date_of_intimation",
            "Status.of.Claim": "claim_status",
            "Amount.Paid": "amount_paid_raw",
            "Settlement.Date": "settlement_date",
        })
        c["app_id"] = pd.to_numeric(c["app_id"], errors="coerce").fillna(0).astype("int64")
        c["claim_amount"] = c["claim_amount_raw"].apply(_clean_num).fillna(0.0)
        c["amount_paid"] = c["amount_paid_raw"].apply(_clean_num).fillna(0.0)
        c["is_settled"] = c["claim_status"].eq("Paid")
        c["is_paid"] = c["is_settled"]
        c["is_wip"] = c["claim_status"].eq("WIP")
        
        # Parse Dates & TAT
        c["date_of_death_dt"] = pd.to_datetime(c["date_of_death"], errors="coerce", format="mixed")
        c["date_of_intimation_dt"] = pd.to_datetime(c["date_of_intimation"], errors="coerce", format="mixed")
        c["settlement_date_dt"] = pd.to_datetime(c["settlement_date"], errors="coerce", format="mixed")
        c["settlement_tat_days"] = (c["settlement_date_dt"] - c["date_of_intimation_dt"]).dt.days
        c["intimation_tat_days"] = (c["date_of_intimation_dt"] - c["date_of_death_dt"]).dt.days

        self.claim_fact = self._attach_dims(c, "app_id")

    def _build_persistency_fact(self):
        r = self.raw_persistency.copy().rename(columns={
            "AppID": "app_id",
            "SALES_ID": "sales_id",
            "POLICY.STATUS": "policy_status_src",
            "CHANNEL": "channel_src",
            "Inforce.Date": "inforce_date_src",
            "PREMIUM.PAYING.TERM": "ppt_src",
            "Renewal_Due_Date": "renewal_due_date",
            "Renewal_Paid_Date": "renewal_paid_date",
            "Renewal_Status": "renewal_status",
            "Renewal_Premium_Amount": "renewal_premium_amount",
            "Persistency_Bucket": "persistency_bucket",
            "Payment_Mode": "payment_mode",
        })
        r["app_id"] = pd.to_numeric(r["app_id"], errors="coerce").fillna(0).astype("int64")
        r["renewal_premium_amount"] = pd.to_numeric(r["renewal_premium_amount"], errors="coerce").fillna(0.0)
        r["is_paid"] = r["renewal_status"].eq("Paid")
        r["is_pending"] = r["renewal_status"].eq("Pending")
        r["paid_premium"] = np.where(r["is_paid"], r["renewal_premium_amount"], 0.0)
        r["pending_premium"] = np.where(r["is_pending"], r["renewal_premium_amount"], 0.0)

        # Dates
        r["renewal_due_date_dt"] = pd.to_datetime(r["renewal_due_date"], errors="coerce", format="mixed")
        r["renewal_paid_date_dt"] = pd.to_datetime(r["renewal_paid_date"], errors="coerce", format="mixed")
        r["paid_delay_days"] = (r["renewal_paid_date_dt"] - r["renewal_due_date_dt"]).dt.days

        # Attach policy dimensions
        self.persistency_fact = self._attach_dims(r, "app_id")

    def _build_customer_fact(self):
        p = self.policy_fact
        agg = p.groupby("client_id").agg(
            policy_count=("app_id", "count"),
            total_sum_assured=("sum_assured", "sum"),
            total_ape=("ape", "sum"),
            lapsed_policies=("is_lapsed", "sum"),
            surrendered_policies=("is_surrendered", "sum"),
            inforce_policies=("is_inforce", "sum"),
            death_policies=("is_death", "sum"),
            claim_count=("claim_count", "sum"),
            claim_amount_total=("claim_amount_total", "sum"),
            main_plan=("plan_name", lambda s: s.mode().iat[0] if len(s.mode()) else None),
            main_trad_ulip=("trad_ulip", lambda s: s.mode().iat[0] if len(s.mode()) else None),
            main_state=("state", lambda s: s.mode().iat[0] if len(s.mode()) else None),
        ).reset_index()

        o = self.raw_owners.copy().rename(columns={
            "ClientID": "client_id",
            "Owner.Given.Name": "owner_name",
            "Owner.Gender": "gender",
            "Owner.Age": "age",
            "Owner.Education": "education",
            "Owner.Earned.Income": "earned_income",
            "Owner.Comm.City.Name": "city",
            "Owner.Comm.State": "state",
            "Owner.Comm.Country": "country",
            "Owner.Comm.Postal.Code": "postal_code",
            "Owner.Occupation": "occupation",
        })
        o["client_id"] = o["client_id"].astype(str)
        o["earned_income"] = pd.to_numeric(o["earned_income"], errors="coerce").fillna(0.0)
        o["age"] = pd.to_numeric(o["age"], errors="coerce")
        o["annual_income"] = o["earned_income"]

        c = o.drop_duplicates("client_id").merge(agg, on="client_id", how="left")
        c["income_band"] = _band(c["earned_income"],
                                 [0, 500_000, 1_000_000, 2_500_000, 5_000_000, 1e12],
                                 ["<5L", "5-10L", "10-25L", "25-50L", "50L+"])
        c["age_band"] = _band(c["age"], [0, 30, 40, 50, 60, 200],
                              ["<30", "30-39", "40-49", "50-59", "60+"])
        c["has_attrition"] = (c["lapsed_policies"] + c["surrendered_policies"]) > 0
        c["attrition_ratio"] = (c["lapsed_policies"] + c["surrendered_policies"]) / c["policy_count"]
        c["value_score"] = c["total_ape"]
        c["customer_id"] = c["client_id"]
        c["full_name"] = c["owner_name"]
        c["location"] = c["city"] + ", " + c["state"]
        c["product_name"] = c["main_plan"]
        c["sales_channel"] = "Banca"
        c["zone"] = c["state"]
        self.customer_fact = c

    def _build_agent_fact(self):
        p = self.policy_fact
        agg = p.groupby("sales_id").agg(
            policies_sold=("app_id", "count"),
            total_sum_assured=("sum_assured", "sum"),
            total_ape=("ape", "sum"),
            avg_sum_assured=("sum_assured", "mean"),
            avg_ape=("ape", "mean"),
            lapsed=("is_lapsed", "sum"),
            surrendered=("is_surrendered", "sum"),
            inforce=("is_inforce", "sum"),
            death=("is_death", "sum"),
            claims=("claim_count", "sum"),
            claim_amount=("claim_amount_total", "sum"),
            main_plan=("plan_name", lambda s: s.mode().iat[0] if len(s.mode()) else None),
            main_branch=("branch_no", lambda s: s.mode().iat[0] if len(s.mode()) else None),
        ).reset_index()

        s = self.raw_sales.copy().rename(columns={
            "SALES_ID": "sales_id",
            "Surrender Rate": "sales_surrender_rate",
            "Early Claim Rate": "sales_early_claim_rate",
            "PIVC Number Mismatch Rate": "pivc_mismatch_rate",
            "PIVC Concern Raised Rate": "pivc_concern_rate",
            "RCU Rejections Rate": "rcu_rejection_rate",
            "Free Look  Cancellation Rate": "free_look_rate",
            "Sales Related Complaint": "complaint_rate",
            "First Year Persistency Rate": "sales_first_year_persistency",
            "Type": "sales_type",
        })
        s["sales_id"] = s["sales_id"].astype("int64")
        for col in ["sales_surrender_rate", "sales_early_claim_rate", "pivc_mismatch_rate",
                    "pivc_concern_rate", "rcu_rejection_rate", "free_look_rate",
                    "complaint_rate", "sales_first_year_persistency"]:
            s[col] = pd.to_numeric(s[col], errors="coerce")

        a = s.merge(agg, on="sales_id", how="left")
        denom = a["policies_sold"].replace(0, np.nan)
        a["policies_serviced"] = a["policies_sold"]
        a["target_achievement_pct"] = a["sales_first_year_persistency"].fillna(80.0)
        a["observed_lapse_rate"] = a["lapsed"] / denom
        a["observed_surrender_rate"] = a["surrendered"] / denom
        a["observed_persistency_rate"] = a["inforce"] / denom
        a["servicing_agent_id"] = a["sales_id"].astype(str)
        a["agent_name"] = "Agent_" + a["sales_id"].astype(str)
        a["branch_id"] = a["main_branch"].astype(str)
        a["state"] = "India"
        a["zone"] = "India"
        self.agent_fact = a

    # -- accessor ------------------------------------------------------
    def frame(self, base: str) -> pd.DataFrame:
        mapping = {
            "policy": self.policy_fact,
            "claim": self.claim_fact,
            "retention": self.persistency_fact,
            "persistency": self.persistency_fact,
            "customer": self.customer_fact,
            "agent": self.agent_fact,
            "sales": self.agent_fact,
        }
        if base not in mapping:
            raise KeyError(f"Unknown fact frame base '{base}'. Valid bases: {list(mapping.keys())}")
        return mapping[base]


# =====================================================================
# 2. METRIC SPEC REGISTRY
# =====================================================================

@dataclass
class MetricSpec:
    name: str
    label: str
    base: str  # "policy" | "claim" | "persistency" | "customer" | "agent"
    unit: str  # "%" | "₹" | "count" | "ratio" | "days"
    definition: str
    numerator: Callable[[pd.DataFrame], float]
    denominator: Callable[[pd.DataFrame], float]
    population: Callable[[pd.DataFrame], pd.DataFrame] = lambda d: d
    higher_is_better: bool = True
    scale: float = 1.0
    denominator_is_count: bool = True
    display_format: str = ".2f"

    def format(self, val: Any) -> str:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "n/a"
        try:
            f = self.display_format
            if self.unit == "₹":
                return f"₹{float(val):{f}}"
            elif self.unit == "%":
                return f"{float(val):{f}}%"
            return f"{float(val):{f}} {self.unit}".strip()
        except Exception:
            return str(val)

    def compute(self, df: pd.DataFrame) -> Dict[str, Any]:
        sub = self.population(df)
        denom = float(self.denominator(sub)) if len(sub) else 0.0
        if denom == 0.0 or math.isnan(denom):
            return {
                "value": np.nan,
                "numerator": np.nan,
                "denominator": 0.0,
                "n": len(sub),
                "reliable": False,
                "unit": self.unit,
                "label": self.label,
            }
        num = float(self.numerator(sub))
        val = (num / denom) * self.scale
        return {
            "value": val,
            "numerator": num,
            "denominator": denom,
            "n": len(sub) if self.denominator_is_count else int(denom),
            "reliable": len(sub) >= DEFAULT_MIN_N,
            "unit": self.unit,
            "label": self.label,
        }


# Metrics that were registered against fields this dataset does not contain.
# They are kept here, rather than silently deleted, so a Problem Statement that
# still asks for one gets told why it cannot be answered instead of failing with
# a bare KeyError - and so the reason is reviewable in one place.
RETIRED_METRICS: Dict[str, str] = {
    "smoker_share":
        "Owner_Details has no smoking field. The previous implementation "
        "estimated it as 15% of male policyholders, which was not a measurement.",
    "medical_underwriting_share":
        "No medical-underwriting field exists in the 5 tables. The previous "
        "implementation returned 100% unconditionally.",
    "rider_attachment_rate":
        "No rider field exists in the 5 tables. The previous implementation "
        "returned 100% unconditionally.",
    "total_surrender_value_paid":
        "No surrender-value field exists. The previous implementation used APE "
        "as a stand-in, which measures premium, not surrender proceeds.",
    "avg_surrender_charge":
        "No surrender-charge field exists. The previous implementation assumed a "
        "flat 10% of APE.",
    "avg_retention_incentive":
        "No incentive or commission data exists. The previous implementation "
        "assumed a flat 2% of renewal premium.",
    "total_retention_incentive":
        "No incentive or commission data exists. The previous implementation "
        "assumed a flat 2% of collected premium.",
    "incentive_per_retained_policy":
        "No incentive or commission data exists. The previous implementation "
        "assumed a flat 2% of collected premium.",
    "retention_success_rate":
        "Renamed: this was identical to `persistency_rate`. There is no retention "
        "programme in the dataset, only renewal status.",
    "retention_contacts":
        "Renamed to `renewal_record_count`. Persistency_Details holds one row per "
        "policy, so it never counted outreach contacts.",
    "retention_loss_to_lapse_rate":
        "Renamed to `renewal_pending_rate`. A pending renewal is an outstanding "
        "premium; the dataset does not record whether it later lapsed.",
}

# Where a retired metric has an honest replacement, name it.
METRIC_REPLACEMENTS: Dict[str, str] = {
    "retention_success_rate": "persistency_rate",
    "retention_contacts": "renewal_record_count",
    "retention_loss_to_lapse_rate": "renewal_pending_rate",
}


class _MetricRegistry(Dict[str, MetricSpec]):
    """Registry that explains a retired metric instead of raising KeyError.

    Subclassing dict means every existing `METRICS[name]` call site gets the
    better error without being touched.
    """

    def __missing__(self, key: str):
        reason = RETIRED_METRICS.get(key)
        if reason:
            swap = METRIC_REPLACEMENTS.get(key)
            hint = f" Use `{swap}` instead." if swap else ""
            raise KeyError(
                f"Metric '{key}' is not available for this dataset. {reason}{hint}")
        raise KeyError(f"Unknown metric '{key}'.")


METRICS: Dict[str, MetricSpec] = _MetricRegistry()

def register(m: MetricSpec) -> MetricSpec:
    METRICS[m.name] = m
    return m

# ---- Volume & Portfolio Metrics --------------------------------------
register(MetricSpec(
    "policy_count", "Total policies", base="policy", unit="count",
    definition="Total number of policy records.",
    numerator=lambda d: len(d),
    denominator=lambda d: 1.0,
    higher_is_better=True, scale=1.0, display_format=",.0f"))

register(MetricSpec(
    "total_ape", "Total APE", base="policy", unit="₹",
    definition="Sum of Annualised Premium Equivalent (APE) across policies.",
    numerator=lambda d: d["ape"].sum(),
    denominator=lambda d: 1.0,
    higher_is_better=True, scale=1.0, denominator_is_count=False, display_format=",.0f"))

register(MetricSpec(
    "avg_ape", "Average APE", base="policy", unit="₹",
    definition="Average Annualised Premium Equivalent (APE) per policy.",
    numerator=lambda d: d["ape"].sum(),
    denominator=lambda d: len(d),
    higher_is_better=True, scale=1.0, display_format=",.0f"))

register(MetricSpec(
    "total_sum_assured", "Total Sum Assured", base="policy", unit="₹",
    definition="Total sum assured across policies.",
    numerator=lambda d: d["sum_assured"].sum(),
    denominator=lambda d: 1.0,
    higher_is_better=True, scale=1.0, denominator_is_count=False, display_format=",.0f"))

register(MetricSpec(
    "avg_sum_assured", "Average Sum Assured", base="policy", unit="₹",
    definition="Average sum assured per policy.",
    numerator=lambda d: d["sum_assured"].sum(),
    denominator=lambda d: len(d),
    higher_is_better=True, scale=1.0, display_format=",.0f"))

register(MetricSpec(
    "avg_ppt", "Average Premium Paying Term", base="policy", unit="years",
    definition="Average premium paying term in years.",
    numerator=lambda d: d["premium_paying_term"].sum(),
    denominator=lambda d: len(d),
    higher_is_better=True, scale=1.0, display_format=".1f"))

register(MetricSpec(
    "avg_policy_term", "Average Policy Term", base="policy", unit="years",
    definition="Average policy term in years.",
    numerator=lambda d: d["policy_term"].sum(),
    denominator=lambda d: len(d),
    higher_is_better=True, scale=1.0, display_format=".1f"))

# ---- Policy Status & Attrition Metrics --------------------------------
register(MetricSpec(
    "inforce_rate", "In-force rate", base="policy", unit="%",
    definition="Active premium paying policies divided by total policies.",
    numerator=lambda d: d["is_inforce"].sum(),
    denominator=lambda d: len(d),
    higher_is_better=True, scale=100.0))

register(MetricSpec(
    "lapse_rate", "Lapse rate", base="policy", unit="%",
    definition="Lapsed policies divided by total policies.",
    numerator=lambda d: d["is_lapsed"].sum(),
    denominator=lambda d: len(d),
    higher_is_better=False, scale=100.0))

register(MetricSpec(
    "surrender_rate", "Surrender rate", base="policy", unit="%",
    definition="Surrendered policies divided by total policies.",
    numerator=lambda d: d["is_surrendered"].sum(),
    denominator=lambda d: len(d),
    higher_is_better=False, scale=100.0))

register(MetricSpec(
    "death_rate", "Death claim incidence rate", base="policy", unit="%",
    definition="Policies terminated due to death claim divided by total policies.",
    numerator=lambda d: d["is_death"].sum(),
    denominator=lambda d: len(d),
    higher_is_better=False, scale=100.0))

register(MetricSpec(
    "attrition_rate", "Total attrition rate", base="policy", unit="%",
    definition="Policies lost to lapse or surrender divided by total policies.",
    numerator=lambda d: (d["is_lapsed"] | d["is_surrendered"]).sum(),
    denominator=lambda d: len(d),
    higher_is_better=False, scale=100.0))

# ---- Persistency Metrics ---------------------------------------------
register(MetricSpec(
    "persistency_rate", "Overall persistency rate", base="persistency", unit="%",
    definition="Paid renewal records divided by total renewal records due.",
    numerator=lambda d: d["is_paid"].sum(),
    denominator=lambda d: len(d),
    higher_is_better=True, scale=100.0))

register(MetricSpec(
    "persistency_13m_rate", "13th Month Persistency rate", base="persistency", unit="%",
    definition="Paid renewals in 13th Month cohort divided by total 13th Month records.",
    population=lambda d: d[d["persistency_bucket"].astype(str).str.contains("13", na=False)],
    numerator=lambda d: d["is_paid"].sum(),
    denominator=lambda d: len(d),
    higher_is_better=True, scale=100.0))

register(MetricSpec(
    "persistency_25m_rate", "25th Month Persistency rate", base="persistency", unit="%",
    definition="Paid renewals in 25th Month cohort divided by total 25th Month records.",
    population=lambda d: d[d["persistency_bucket"].astype(str).str.contains("25", na=False)],
    numerator=lambda d: d["is_paid"].sum(),
    denominator=lambda d: len(d),
    higher_is_better=True, scale=100.0))

register(MetricSpec(
    "persistency_37m_rate", "37th Month Persistency rate", base="persistency", unit="%",
    definition="Paid renewals in 37th Month cohort divided by total 37th Month records.",
    population=lambda d: d[d["persistency_bucket"].astype(str).str.contains("37", na=False)],
    numerator=lambda d: d["is_paid"].sum(),
    denominator=lambda d: len(d),
    higher_is_better=True, scale=100.0))

register(MetricSpec(
    "renewal_premium_collection_rate", "Renewal collection efficiency", base="persistency", unit="%",
    definition="Renewal premium amount collected (Paid) divided by total renewal premium due.",
    numerator=lambda d: d["paid_premium"].sum(),
    denominator=lambda d: d["renewal_premium_amount"].sum(),
    higher_is_better=True, scale=100.0, denominator_is_count=False))

register(MetricSpec(
    "total_renewal_premium_due", "Total renewal premium due", base="persistency", unit="₹",
    definition="Total renewal premium due across cohort records.",
    numerator=lambda d: d["renewal_premium_amount"].sum(),
    denominator=lambda d: 1.0,
    higher_is_better=True, scale=1.0, denominator_is_count=False, display_format=",.0f"))

register(MetricSpec(
    "pending_renewal_premium", "Pending renewal premium at risk", base="persistency", unit="₹",
    definition="Uncollected renewal premium amount on pending records.",
    numerator=lambda d: d["pending_premium"].sum(),
    denominator=lambda d: 1.0,
    higher_is_better=False, scale=1.0, denominator_is_count=False, display_format=",.0f"))

# ---- Sales Quality & Risk Indicators ---------------------------------
register(MetricSpec(
    "pivc_mismatch_rate", "Average PIVC mismatch rate", base="policy", unit="%",
    definition="Average PIVC contact number mismatch rate of servicing sales agent.",
    numerator=lambda d: d["pivc_mismatch_rate"].dropna().sum(),
    denominator=lambda d: d["pivc_mismatch_rate"].dropna().count(),
    higher_is_better=False, scale=1.0))

register(MetricSpec(
    "pivc_concern_rate", "Average PIVC concern rate", base="policy", unit="%",
    definition="Average PIVC customer concern raised rate of servicing sales agent.",
    numerator=lambda d: d["pivc_concern_rate"].dropna().sum(),
    denominator=lambda d: d["pivc_concern_rate"].dropna().count(),
    higher_is_better=False, scale=1.0))

register(MetricSpec(
    "rcu_rejection_rate", "Average RCU rejection rate", base="policy", unit="%",
    definition="Average Risk Containment Unit (RCU) rejection rate of servicing sales agent.",
    numerator=lambda d: d["rcu_rejection_rate"].dropna().sum(),
    denominator=lambda d: d["rcu_rejection_rate"].dropna().count(),
    higher_is_better=False, scale=1.0))

register(MetricSpec(
    "free_look_rate", "Average Free Look cancellation rate", base="policy", unit="%",
    definition="Average Free Look cancellation rate of servicing sales agent.",
    numerator=lambda d: d["free_look_rate"].dropna().sum(),
    denominator=lambda d: d["free_look_rate"].dropna().count(),
    higher_is_better=False, scale=1.0))

register(MetricSpec(
    "complaint_rate", "Average sales complaint rate", base="policy", unit="%",
    definition="Average sales-related complaint rate of servicing sales agent.",
    numerator=lambda d: d["complaint_rate"].dropna().sum(),
    denominator=lambda d: d["complaint_rate"].dropna().count(),
    higher_is_better=False, scale=1.0))

register(MetricSpec(
    "sales_persistency_benchmark", "First year persistency benchmark", base="policy", unit="%",
    definition="Average first-year persistency rate across sales identifiers.",
    numerator=lambda d: d["sales_first_year_persistency"].dropna().sum(),
    denominator=lambda d: d["sales_first_year_persistency"].dropna().count(),
    higher_is_better=True, scale=1.0))

# ---- Claims & Underwriting Metrics -----------------------------------
register(MetricSpec(
    "claim_count", "Total claims", base="claim", unit="count",
    definition="Total number of claim records intimated.",
    numerator=lambda d: len(d),
    denominator=lambda d: 1.0,
    higher_is_better=False, scale=1.0, display_format=",.0f"))

register(MetricSpec(
    "total_claim_amount", "Total claim amount assessed", base="claim", unit="₹",
    definition="Total claim amount claimed across cases.",
    numerator=lambda d: d["claim_amount"].sum(),
    denominator=lambda d: 1.0,
    higher_is_better=False, scale=1.0, denominator_is_count=False, display_format=",.0f"))

register(MetricSpec(
    "total_claim_paid", "Total claim amount paid", base="claim", unit="₹",
    definition="Total claim amount successfully disbursed.",
    numerator=lambda d: d["amount_paid"].sum(),
    denominator=lambda d: 1.0,
    higher_is_better=True, scale=1.0, denominator_is_count=False, display_format=",.0f"))

register(MetricSpec(
    "claim_settlement_ratio", "Claim settlement ratio", base="claim", unit="%",
    definition="Settled (Paid) claims divided by total claims intimated.",
    numerator=lambda d: d["is_paid"].sum(),
    denominator=lambda d: len(d),
    higher_is_better=True, scale=100.0))

register(MetricSpec(
    "avg_settlement_tat", "Average claim settlement TAT", base="claim", unit="days",
    definition="Average turnaround time in days from claim intimation to settlement.",
    population=lambda d: d[d["is_paid"] & d["settlement_tat_days"].notna()],
    numerator=lambda d: d["settlement_tat_days"].sum(),
    denominator=lambda d: len(d),
    higher_is_better=False, scale=1.0, display_format=".1f"))

# ---- Customer Profile & Demographics ---------------------------------
register(MetricSpec(
    "avg_owner_age", "Average owner age", base="policy", unit="years",
    definition="Average age of policy owner.",
    numerator=lambda d: d["age"].dropna().sum(),
    denominator=lambda d: d["age"].dropna().count(),
    higher_is_better=True, scale=1.0, display_format=".1f"))

register(MetricSpec(
    "avg_earned_income", "Average earned income", base="policy", unit="₹",
    definition="Average annual earned income of policy owner.",
    numerator=lambda d: d["earned_income"].dropna().sum(),
    denominator=lambda d: d["earned_income"].dropna().count(),
    higher_is_better=True, scale=1.0, display_format=",.0f"))

register(MetricSpec(
    "salaried_share", "Salaried customer share", base="policy", unit="%",
    definition="Percentage of policyholders with salaried occupation.",
    numerator=lambda d: d["occupation"].eq("Salaried").sum(),
    denominator=lambda d: len(d),
    higher_is_better=True, scale=100.0))

register(MetricSpec(
    "business_owner_share", "Business owner customer share", base="policy", unit="%",
    definition="Percentage of policyholders who are business owners.",
    numerator=lambda d: d["occupation"].eq("Business Owner").sum(),
    denominator=lambda d: len(d),
    higher_is_better=True, scale=100.0))

register(MetricSpec(
    "female_share", "Female policyholder share", base="policy", unit="%",
    definition="Percentage of policies held by female owners.",
    numerator=lambda d: d["gender"].eq("F").sum(),
    denominator=lambda d: len(d),
    higher_is_better=True, scale=100.0))

register(MetricSpec(
    "trad_share", "Traditional product share", base="policy", unit="%",
    definition="Percentage of policies under TRAD classification.",
    numerator=lambda d: d["trad_ulip"].eq("TRAD").sum(),
    denominator=lambda d: len(d),
    higher_is_better=True, scale=100.0))

register(MetricSpec(
    "ulip_share", "ULIP product share", base="policy", unit="%",
    definition="Percentage of policies under ULIP classification.",
    numerator=lambda d: d["trad_ulip"].eq("ULIP").sum(),
    denominator=lambda d: len(d),
    higher_is_better=True, scale=100.0))

# ---- Aliases & Compatibility Metrics ---------------------------------
register(MetricSpec(
    "avg_annual_premium", "Average annual premium (APE)", base="policy", unit="₹",
    definition="Average annualised premium equivalent per policy.",
    numerator=lambda d: d["ape"].sum(),
    denominator=lambda d: len(d),
    higher_is_better=True, scale=1.0, display_format=",.0f"))

register(MetricSpec(
    "total_annual_premium", "Total annual premium (APE)", base="policy", unit="₹",
    definition="Sum of annualised premium equivalent across policies.",
    numerator=lambda d: d["ape"].sum(),
    denominator=lambda d: 1.0,
    higher_is_better=True, scale=1.0, denominator_is_count=False, display_format=",.0f"))

register(MetricSpec(
    "total_premium_collected", "Total premium collected", base="policy", unit="₹",
    definition="Total premium collected across policies.",
    numerator=lambda d: d["ape"].sum(),
    denominator=lambda d: 1.0,
    higher_is_better=True, scale=1.0, denominator_is_count=False, display_format=",.0f"))

register(MetricSpec(
    "avg_customer_income", "Average customer income", base="policy", unit="₹",
    definition="Average earned income of policyholders.",
    numerator=lambda d: d["earned_income"].dropna().sum(),
    denominator=lambda d: d["earned_income"].dropna().count(),
    higher_is_better=True, scale=1.0, display_format=",.0f"))

register(MetricSpec(
    "avg_customer_age", "Average customer age", base="policy", unit="years",
    definition="Average age of policyholders.",
    numerator=lambda d: d["age"].dropna().sum(),
    denominator=lambda d: d["age"].dropna().count(),
    higher_is_better=True, scale=1.0, display_format=".1f"))

register(MetricSpec(
    "avg_premium_to_income", "Average premium to income ratio", base="policy", unit="%",
    definition="Average ratio of annualised premium to earned income.",
    numerator=lambda d: d["ape_to_income"].dropna().sum(),
    denominator=lambda d: d["ape_to_income"].dropna().count(),
    higher_is_better=False, scale=100.0))

register(MetricSpec(
    "loss_ratio", "Portfolio Loss Ratio", base="policy", unit="%",
    definition="Total claim amount assessed divided by total annualised premium.",
    numerator=lambda d: d["claim_amount_total"].sum(),
    denominator=lambda d: d["ape"].sum(),
    higher_is_better=False, scale=100.0, denominator_is_count=False))

register(MetricSpec(
    "risk_loss_ratio", "Risk Loss Ratio", base="policy", unit="%",
    definition="Total claim amount paid divided by total annualised premium.",
    numerator=lambda d: d["amount_paid_total"].sum(),
    denominator=lambda d: d["ape"].sum(),
    higher_is_better=False, scale=100.0, denominator_is_count=False))

register(MetricSpec(
    "claim_rate", "Claim incidence rate", base="policy", unit="%",
    definition="Policies with claims divided by total policies.",
    numerator=lambda d: d["has_claim"].sum(),
    denominator=lambda d: len(d),
    higher_is_better=False, scale=100.0))

register(MetricSpec(
    "risk_claim_rate", "Death claim rate", base="policy", unit="%",
    definition="Policies terminated due to death divided by total policies.",
    numerator=lambda d: d["is_death"].sum(),
    denominator=lambda d: len(d),
    higher_is_better=False, scale=100.0))

register(MetricSpec(
    "renewal_record_count", "Renewal records", base="persistency", unit="count",
    definition="Renewal records in Persistency_Details. One row per AppID, so this "
               "counts policies with a renewal record - not renewal attempts or contacts.",
    numerator=lambda d: len(d),
    denominator=lambda d: 1.0,
    higher_is_better=True, scale=1.0, display_format=",.0f"))

register(MetricSpec(
    "avg_claim_amount", "Average claim amount", base="claim", unit="₹",
    definition="Average claim amount claimed per case.",
    numerator=lambda d: d["claim_amount"].sum(),
    denominator=lambda d: len(d),
    higher_is_better=False, scale=1.0, display_format=",.0f"))

register(MetricSpec(
    "early_claim_rate", "Early claim rate", base="policy", unit="%",
    definition="Average early claim rate across servicing sales reps.",
    numerator=lambda d: d["sales_early_claim_rate"].dropna().sum(),
    denominator=lambda d: d["sales_early_claim_rate"].dropna().count(),
    higher_is_better=False, scale=1.0))

register(MetricSpec(
    "repudiation_rate", "WIP / Repudiation rate", base="claim", unit="%",
    definition="Percentage of claims WIP or repudiated.",
    numerator=lambda d: d["is_wip"].sum(),
    denominator=lambda d: len(d),
    higher_is_better=False, scale=100.0))

register(MetricSpec(
    "median_sum_assured", "Median sum assured", base="policy", unit="₹",
    definition="Median sum assured across policies.",
    numerator=lambda d: d["sum_assured"].median() if len(d) else 0.0,
    denominator=lambda d: 1.0,
    higher_is_better=True, scale=1.0, denominator_is_count=False, display_format=",.0f"))

register(MetricSpec(
    "renewal_pending_rate", "Renewal pending rate", base="persistency", unit="%",
    definition="Renewal records with Renewal_Status = 'Pending', over all renewal "
               "records. Pending means the due premium is outstanding; the dataset "
               "does not record whether a pending renewal later lapsed.",
    numerator=lambda d: d["is_pending"].sum(),
    denominator=lambda d: len(d),
    higher_is_better=False, scale=100.0))





# =====================================================================
# 3. ANALYTICAL OPERATIONS (14 VERBS)
# =====================================================================

def _apply_filters(df: pd.DataFrame, filters: Optional[Dict[str, Any]]) -> pd.DataFrame:
    if not filters:
        return df
    sub = df
    for k, v in filters.items():
        if k not in sub.columns:
            continue
        if isinstance(v, (list, tuple, set)):
            sub = sub[sub[k].isin(list(v))]
        elif isinstance(v, str) and v.startswith("~"):
            sub = sub[~sub[k].astype(str).str.contains(v[1:], case=False, na=False)]
        elif isinstance(v, str) and "*" in v:
            pattern = re.escape(v).replace(r"\*", ".*")
            sub = sub[sub[k].astype(str).str.match(pattern, na=False)]
        else:
            sub = sub[sub[k] == v]
    return sub


def _resolve_dim_col(dim: str) -> str:
    if dim in DIMENSIONS:
        return DIMENSIONS[dim].column
    return dim


def check_dimension(dim: str) -> None:
    """Refuse a breakdown the dataset cannot support, and say why.

    Two cases produce a confident-looking table that answers nothing: a
    dimension aliased onto an unrelated column (asking "by smoker" and being
    answered by gender), and a dimension with one distinct value across the
    whole book (a "channel comparison" over 25,000 Banca policies).
    """
    if dim in RETIRED_DIMENSIONS:
        raise KeyError(f"Dimension '{dim}' is not available for this dataset. "
                       f"{RETIRED_DIMENSIONS[dim]}")
    col = _resolve_dim_col(dim)
    if col in DEGENERATE_DIMENSIONS:
        raise InsufficientDataError(
            f"'{dim}' cannot be compared: {DEGENERATE_DIMENSIONS[col]} A breakdown "
            f"would return a single row, not a comparison.")


def metric_by_dimension(
    ctx: DataContext,
    metric: str,
    dimension: str,
    filters: Optional[Dict[str, Any]] = None,
    min_n: int = DEFAULT_MIN_N,
    sort_descending: Optional[bool] = None,
) -> pd.DataFrame:
    """Computes a single metric broken down by a dimension."""
    if metric in DIMENSIONS and dimension in METRICS:
        metric, dimension = dimension, metric
    check_dimension(dimension)
    spec = METRICS[metric]
    df = _apply_filters(ctx.frame(spec.base), filters)
    
    col = _resolve_dim_col(dimension)
    if col not in df.columns:
        # Check if dimension is in other fact frames or policy_fact
        if col in ctx.policy_fact.columns:
            df = df.merge(ctx.policy_fact[["app_id", col]].drop_duplicates("app_id"), on="app_id", how="left")
        else:
            raise ValueError(f"Dimension '{dimension}' (column '{col}') not found in fact frame '{spec.base}'. Available: {list(df.columns)}")

    rows = []
    for val, grp in df.groupby(col, observed=True):
        res = spec.compute(grp)
        rows.append({
            dimension: val,
            col: val,
            "metric": metric,
            "metric_label": spec.label,
            "value": res["value"],
            "numerator": res["numerator"],
            "denominator": res["denominator"],
            "sample_size": res["n"],
            "reliable": res["n"] >= min_n and res["reliable"],
            "unit": spec.unit,
        })
    res_df = pd.DataFrame(rows)
    if res_df.empty:
        return pd.DataFrame(columns=[dimension, "metric", "metric_label", "value", "sample_size", "reliable", "unit"])
    
    ascending = not (spec.higher_is_better if sort_descending is None else sort_descending)
    return res_df.sort_values(by="value", ascending=ascending).reset_index(drop=True)


def compare_to_baseline(
    ctx: DataContext,
    metric: str,
    dimension: str,
    filters: Optional[Dict[str, Any]] = None,
    min_n: int = DEFAULT_MIN_N,
) -> pd.DataFrame:
    """Compares each segment against the portfolio overall baseline."""
    if metric in DIMENSIONS and dimension in METRICS:
        metric, dimension = dimension, metric
    spec = METRICS[metric]
    base_df = _apply_filters(ctx.frame(spec.base), filters)
    baseline = spec.compute(base_df)
    baseline_val = baseline["value"]

    by_dim = metric_by_dimension(ctx, metric, dimension, filters, min_n=min_n)
    if by_dim.empty:
        return by_dim

    by_dim[metric] = by_dim["value"]
    by_dim["portfolio_baseline"] = baseline_val
    by_dim["baseline"] = baseline_val
    by_dim["absolute_diff"] = by_dim["value"] - baseline_val
    by_dim["diff_vs_baseline"] = by_dim["absolute_diff"]
    by_dim["relative_diff_pct"] = np.where(
        baseline_val != 0, (by_dim["absolute_diff"] / baseline_val) * 100.0, np.nan
    )
    by_dim["lift"] = by_dim["relative_diff_pct"]
    by_dim["n"] = by_dim["sample_size"]
    by_dim[f"{metric}_n"] = by_dim["sample_size"]
    by_dim["is_adverse"] = np.where(
        spec.higher_is_better,
        by_dim["value"] < baseline_val,
        by_dim["value"] > baseline_val,
    )
    return by_dim


def rank_entities(
    ctx: DataContext,
    dimension: str,
    metric: str,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    worst: bool = False,
    min_n: int = DEFAULT_MIN_N,
    order: Optional[str] = None,
) -> pd.DataFrame:
    """Returns top-K best or worst entities on a metric."""
    if dimension in METRICS and metric in DIMENSIONS:
        dimension, metric = metric, dimension
    spec = METRICS[metric]
    if order is not None:
        sort_desc = (order.lower() == "desc")
    else:
        sort_desc = not worst if spec.higher_is_better else worst
    df = metric_by_dimension(ctx, metric, dimension, filters, min_n=min_n, sort_descending=sort_desc)
    filtered = df[df["reliable"]] if len(df[df["reliable"]]) >= top_k else df
    return filtered.head(top_k).reset_index(drop=True)


def _share_metric_by_dimension(
    ctx: DataContext,
    metric: str,
    dimension: str,
    filters: Optional[Dict[str, Any]] = None,
    min_n: int = DEFAULT_MIN_N,
) -> pd.DataFrame:
    """Compute share-type pseudo-metrics (policy_share, sum_assured_share, premium_share).

    These express each segment's percentage of the portfolio total and are NOT registered
    in METRICS because they require the full-portfolio denominator at runtime.
    """
    _num_fn = {
        "policy_share": lambda d: float(len(d)),
        "sum_assured_share": lambda d: float(d["sum_assured"].sum()),
        "premium_share": lambda d: float(d["ape"].sum()),
    }
    if metric not in _num_fn:
        return pd.DataFrame()

    num_fn = _num_fn[metric]
    base_df = _apply_filters(ctx.frame("policy"), filters)
    col = _resolve_dim_col(dimension)
    if col not in base_df.columns:
        return pd.DataFrame()

    total = num_fn(base_df)
    if total == 0:
        return pd.DataFrame()

    rows = []
    for val, grp in base_df.groupby(col, observed=True):
        seg_val = num_fn(grp)
        n = len(grp)
        rows.append({
            dimension: val,
            "metric": metric,
            "value": (seg_val / total) * 100.0,
            "sample_size": n,
            "reliable": n >= min_n,
        })
    return pd.DataFrame(rows)


def composite_score(
    ctx: DataContext,
    dimension: str,
    components: List[Dict[str, Any]],  # [{"metric": ..., "weight": ..., "direction": "high"|"low"}]
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
    min_n: int = DEFAULT_MIN_N,
) -> pd.DataFrame:
    """Computes a multi-metric normalised composite score."""
    frames = []
    for c in components:
        m = c["metric"]
        w = c.get("weight", 1.0)
        direction = c.get("direction", "high")
        # Handle share-type pseudo-metrics that aren't in METRICS registry
        if m in ("policy_share", "sum_assured_share", "premium_share"):
            df_m = _share_metric_by_dimension(ctx, m, dimension, filters, min_n)
        else:
            df_m = metric_by_dimension(ctx, m, dimension, filters, min_n=min_n)
        if df_m.empty:
            continue
        v_min, v_max = df_m["value"].min(), df_m["value"].max()
        denom = (v_max - v_min) if v_max != v_min else 1.0
        if direction == "high":
            df_m[f"norm_{m}"] = (df_m["value"] - v_min) / denom
        else:
            df_m[f"norm_{m}"] = (v_max - df_m["value"]) / denom
        df_m[f"w_{m}"] = df_m[f"norm_{m}"] * w
        frames.append((m, df_m[[dimension, "value", f"w_{m}", "sample_size", "reliable"]]))

    if not frames:
        return pd.DataFrame()

    merged = frames[0][1].rename(columns={"value": frames[0][0], "sample_size": f"{frames[0][0]}_n", "reliable": f"{frames[0][0]}_reliable"})
    total_score = merged[f"w_{frames[0][0]}"].copy()
    all_reliable = merged[f"{frames[0][0]}_reliable"].copy()

    for m, f in frames[1:]:
        renamed = f.rename(columns={"value": m, "sample_size": f"{m}_n", "reliable": f"{m}_reliable"})
        merged = merged.merge(renamed, on=dimension, how="inner")
        total_score += merged[f"w_{m}"]
        all_reliable &= merged[f"{m}_reliable"]

    merged["composite_score"] = total_score
    merged["priority_score"] = total_score
    merged["reliable"] = all_reliable
    merged = merged.sort_values(by="composite_score", ascending=False).reset_index(drop=True)
    merged["rank"] = range(1, len(merged) + 1)
    return merged.head(top_k)


def driver_scan(
    ctx: DataContext,
    metric: str,
    focus_filters: Optional[Dict[str, Any]] = None,
    candidate_dimensions: Optional[List[str]] = None,
    min_n: int = DEFAULT_MIN_N,
    top_k: int = 10,
) -> pd.DataFrame:
    """Scans multiple dimensions to find the largest variance drivers of a metric."""
    spec = METRICS[metric]
    dims = candidate_dimensions or [
        "plan_name", "trad_ulip", "payment_mode", "persistency_bucket",
        "state", "occupation", "income_band", "age_band", "ppt_band", "term_band"
    ]
    base_frame = ctx.frame(spec.base)
    valid_dims = [d for d in dims if d in base_frame.columns]

    rows = []
    for d in valid_dims:
        df = compare_to_baseline(ctx, metric, d, focus_filters, min_n=min_n)
        if df.empty:
            continue
        rel = df[df["reliable"]] if len(df[df["reliable"]]) > 0 else df
        dcol = _resolve_dim_col(d)
        for _, r in rel.iterrows():
            diff = r.get("diff_vs_baseline", r["value"] - r.get("baseline", 0))
            is_worse = (diff < 0) if spec.higher_is_better else (diff > 0)
            direction = "worse" if is_worse else "better"
            n = r.get("n", r.get("sample_size", 0)) or 0
            rows.append({
                "dimension": d,
                "dimension_label": DIMENSIONS[d].label if d in DIMENSIONS else d,
                "segment": r[dcol],
                "metric": metric,
                "value": r["value"],
                "focus_baseline": r.get("baseline", r.get("portfolio_baseline", 0)),
                "diff_vs_baseline": diff,
                "lift": r.get("lift", r.get("relative_diff_pct", 0)),
                "n": n,
                "direction": direction,
                "reliable": r.get("reliable", True),
                "abs_diff": abs(diff),
                "_dim_total": len(_apply_filters(base_frame, focus_filters)),
            })
    res = pd.DataFrame(rows)
    if res.empty:
        return res

    # Rank by how much a segment moves the book, not by how extreme its rate is.
    #
    #     contribution = (segment rate - baseline) x segment share of the book
    #
    # Sorting on the raw deviation puts a 23-policy state above a 6,000-policy
    # age band, which answers "where is the rate most extreme" - not "what is
    # driving the number". For a "why" question those are different questions,
    # and only the second one is being asked.
    total = res["_dim_total"].replace(0, pd.NA)
    res["share_pct"] = (res["n"] / total * 100.0).astype(float)
    res["contribution"] = res["diff_vs_baseline"] * (res["n"] / total).astype(float)
    res["abs_contribution"] = res["contribution"].abs()
    res = res.drop(columns=["_dim_total"])

    ranked = res.sort_values(by="abs_contribution", ascending=False)
    return ranked.head(top_k).reset_index(drop=True)


def explain_gap(
    ctx: DataContext,
    metric: str,
    dimension: str,
    filters: Optional[Dict[str, Any]] = None,
    candidate_confounders: Optional[List[str]] = None,
    min_n: int = DEFAULT_MIN_N,
) -> Dict[str, Any]:
    """Answer "why is <metric> higher for some <dimension> than others?".

    A "why" question needs three things in order, and the planner previously
    supplied none of them:

      1. **The gap itself.** State that it exists and how big it is, before
         explaining anything. Asking "why do younger customers lapse more"
         without first computing lapse by age explains a gap nobody established.

      2. **A confounder check.** A segment may differ on the metric because it
         differs on something else. Under-30s hold 92% traditional policies
         against 69% for the over-60s, and ULIP almost never lapses - so part of
         an age gap is really a product gap.

      3. **Stratification.** Recompute the gap *within* a level of the
         confounder. If it survives, the original dimension carries an
         independent association; if it collapses, the gap was composition.

    Returns a structured verdict. Nothing here claims causation: the dataset
    records no interventions, so "associated with" is as far as it goes.
    """
    spec = METRICS[metric]
    check_dimension(dimension)

    by_dim = metric_by_dimension(ctx, metric, dimension, filters, min_n=min_n)
    reliable = by_dim[by_dim["reliable"]] if "reliable" in by_dim.columns else by_dim
    if len(reliable) < 2:
        raise InsufficientDataError(
            f"Not enough reliable {dimension} groups to compare "
            f"{spec.label.lower()} across.")

    base = _apply_filters(ctx.frame(spec.base), filters)

    # The two groups being contrasted carry the whole answer, so they need more
    # behind them than the display threshold. A 23-policy state anchoring a
    # 19-point gap is an artefact of the smallest group, not a finding about
    # geography. Anchors must hold at least 1% of the population (floor
    # `min_n`); if that leaves fewer than two groups, fall back rather than fail.
    anchor_floor = max(min_n, int(len(base) * 0.01))
    anchors = reliable[reliable["sample_size"] >= anchor_floor]
    if len(anchors) < 2:
        anchors = reliable
        anchor_floor = min_n

    ordered = anchors.sort_values("value", ascending=False)
    high, low = ordered.iloc[0], ordered.iloc[-1]
    dcol = _resolve_dim_col(dimension)
    gap = float(high["value"]) - float(low["value"])

    # ---- 2. which other dimension's mix varies most across this one?
    if candidate_confounders is None:
        # Registered dimensions only: an unregistered column has no label and
        # its raw values (7.0) surface instead of readable bands.
        candidate_confounders = ["trad_ulip", "plan_name", "payment_mode",
                                 "ppt_band", "income_band", "state",
                                 "age_band", "persistency_bucket"]
    confounders: List[Dict[str, Any]] = []
    for cand in candidate_confounders:
        if cand == dimension:
            continue
        cdim = DIMENSIONS.get(cand)
        ccol = cdim.column if cdim else cand
        if ccol not in base.columns or ccol == dcol:
            continue
        try:
            mix = pd.crosstab(base[dcol], base[ccol], normalize="index") * 100.0
            if mix.empty or mix.shape[1] < 2:
                continue
            hi_row = mix.loc[high[dimension]] if high[dimension] in mix.index else None
            lo_row = mix.loc[low[dimension]] if low[dimension] in mix.index else None
            if hi_row is None or lo_row is None:
                continue
            spread = (hi_row - lo_row).abs()
            top_level = spread.idxmax()
            confounders.append({
                "dimension": cand,
                "label": cdim.label if cdim else cand,
                "level": top_level,
                "high_group_pct": float(hi_row[top_level]),
                "low_group_pct": float(lo_row[top_level]),
                "mix_spread_pts": float(spread.max()),
            })
        except Exception:
            continue
    confounders.sort(key=lambda c: -c["mix_spread_pts"])

    # ---- 3. stratify on the strongest *usable* confounder
    #
    # The stratum has to stay wide enough to hold a comparison. Holding one
    # specific plan constant collapsed a 12.3-point age gap to 0.2 - not because
    # age stopped mattering, but because that plan barely lapses at all, so
    # every age band read near zero. A confounder is only usable here if it has
    # few levels; the gap is then measured inside *every* qualifying level and
    # size-weighted, which is standardisation rather than one arbitrary slice.
    stratified: List[Dict[str, Any]] = []
    verdict, survives, stratified_meta = "", None, {}
    MAX_STRATA = 6

    usable = []
    for c in confounders:
        cdim = DIMENSIONS.get(c["dimension"])
        ccol = cdim.column if cdim else c["dimension"]
        if ccol in base.columns and base[ccol].nunique(dropna=True) <= MAX_STRATA:
            usable.append((c, ccol))

    if usable:
        top, ccol = usable[0]
        per_stratum, weights = [], []
        for level, level_n in base[ccol].value_counts().items():
            try:
                within = metric_by_dimension(
                    ctx, metric, dimension, {**(filters or {}), ccol: level}, min_n=min_n)
            except Exception:
                continue
            wr = within[within["reliable"]] if "reliable" in within.columns else within
            if len(wr) < 2:
                continue
            wo = wr.sort_values("value", ascending=False)
            w_gap = float(wo.iloc[0]["value"]) - float(wo.iloc[-1]["value"])
            per_stratum.append({
                "stratum": str(level), "n": int(level_n), "gap_pts": w_gap,
                "high": f"{wo.iloc[0][dimension]} {wo.iloc[0]['value']:.1f}",
                "low": f"{wo.iloc[-1][dimension]} {wo.iloc[-1]['value']:.1f}",
            })
            weights.append(level_n)

        # One stratum holds nothing constant; the "within groups" claim needs at
        # least two to mean anything.
        if len(per_stratum) >= 2:
            total_w = sum(weights) or 1
            weighted_gap = sum(s["gap_pts"] * w for s, w in zip(per_stratum, weights)) / total_w
            retained = (weighted_gap / gap * 100.0) if gap else 0.0
            survives = retained >= 50.0
            stratified = per_stratum
            stratified_meta = {"held_dimension": top["label"], "strata": len(per_stratum),
                               "weighted_gap": weighted_gap, "retained_pct": retained}
            dim_label = (DIMENSIONS[dimension].label.lower()
                         if dimension in DIMENSIONS else dimension)
            held = top["label"].lower()
            if retained > 110.0:
                # Wider inside the strata than overall: the confounder's mix was
                # masking the effect, which is worth saying plainly rather than
                # printing a retention figure above 100%.
                verdict = (
                    f"Holding {held} constant across its {len(per_stratum)} groups, "
                    f"the gap widens to {weighted_gap:.1f} points, against "
                    f"{gap:.1f} overall. {top['label']} mix was masking the "
                    f"difference, so {dim_label} matters more than the headline "
                    f"figure suggests.")
            elif survives:
                verdict = (
                    f"Holding {held} constant across its {len(per_stratum)} groups, "
                    f"the gap is {weighted_gap:.1f} points - {retained:.0f}% of the "
                    f"original {gap:.1f}. The pattern holds within groups, so "
                    f"{dim_label} carries an association of its own.")
            else:
                verdict = (
                    f"Holding {held} constant across its {len(per_stratum)} groups, "
                    f"the gap falls to {weighted_gap:.1f} points - {retained:.0f}% of "
                    f"the original {gap:.1f}. Most of it was {held} mix rather than "
                    f"{dim_label} itself.")

    # A one-point spread between two 50% groups is not a finding. Materiality is
    # relative to the higher group, with an absolute floor so tiny rates do not
    # register as dramatic.
    relative = (gap / float(high["value"]) * 100.0) if float(high["value"]) else 0.0
    material = gap >= 2.0 and relative >= 20.0

    return {
        "metric": metric,
        "metric_label": spec.label,
        "material": bool(material),
        "relative_gap_pct": relative,
        "anchor_floor": int(anchor_floor),
        "dimension": dimension,
        "dimension_label": DIMENSIONS[dimension].label if dimension in DIMENSIONS else dimension,
        "high_group": str(high[dimension]),
        "high_value": float(high["value"]),
        "high_n": int(high["sample_size"]),
        "low_group": str(low[dimension]),
        "low_value": float(low["value"]),
        "low_n": int(low["sample_size"]),
        "gap_pts": gap,
        # A zero denominator makes the multiple meaningless; the point gap still
        # stands on its own, so report None rather than inf/nan.
        "ratio": (float(high["value"]) / float(low["value"]))
                 if float(low["value"]) > 0 else None,
        "groups": ordered[[dimension, "value", "sample_size"]].to_dict("records"),
        "confounders": confounders[:3],
        "stratified": stratified,
        "stratified_meta": stratified_meta,
        "survives_stratification": survives,
        "verdict": verdict,
        "unit": spec.unit,
    }


def concentration_analysis(
    ctx: DataContext,
    dimension: str,
    value_metric: str = "total_ape",
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Analyzes Pareto / concentration of value across an entity dimension."""
    df = metric_by_dimension(ctx, value_metric, dimension, filters, min_n=1, sort_descending=True)
    if df.empty:
        return df
    total_val = df["value"].sum()
    df["share_pct"] = (df["value"] / total_val) * 100.0 if total_val else 0.0
    df["cumulative_share_pct"] = df["share_pct"].cumsum()
    # Compute HHI
    shares = df["share_pct"] / 100.0
    hhi = float((shares ** 2).sum())
    df.attrs["hhi"] = hhi
    df.attrs["effective_n"] = round(1.0 / hhi, 1) if hhi > 0 else len(df)
    df.attrs["interpretation"] = "highly concentrated" if hhi > 0.25 else ("moderately concentrated" if hhi > 0.15 else "well distributed")
    return df.head(top_k).reset_index(drop=True)


def top_entity_concentration(
    ctx: DataContext,
    column: str = "sum_assured",
    top_n: int = 10,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Measures single-record exposure (e.g. largest top N policies by sum_assured / annual_premium)."""
    df = _apply_filters(ctx.policy_fact, filters)
    if column not in df.columns:
        if column in ctx.claim_fact.columns:
            df = _apply_filters(ctx.claim_fact, filters)
        else:
            column = "sum_assured"
    valid = df.dropna(subset=[column])
    sorted_df = valid.sort_values(by=column, ascending=False)
    top_records = sorted_df.head(top_n)
    total_val = float(valid[column].sum())
    top_n_val = float(top_records[column].sum())
    top_n_share = (top_n_val / total_val * 100.0) if total_val else 0.0
    return {
        "column": column,
        "top_n": top_n,
        "n_records": len(valid),
        "total_value": total_val,
        "top_n_value": top_n_val,
        "top_n_share_pct": top_n_share,
    }


def outliers_analysis(
    ctx: DataContext,
    metric: str,
    dimension: str,
    threshold_sigma: float = 2.0,
    filters: Optional[Dict[str, Any]] = None,
    min_n: int = DEFAULT_MIN_N,
) -> pd.DataFrame:
    """Detects statistical outlier segments on a metric."""
    df = metric_by_dimension(ctx, metric, dimension, filters, min_n=min_n)
    if df.empty or len(df) < 3:
        return pd.DataFrame()
    mean = df["value"].mean()
    std = df["value"].std()
    if std == 0 or pd.isna(std):
        return pd.DataFrame()
    df[metric] = df["value"]
    df[f"{metric}_n"] = df["sample_size"]
    df["z_score"] = (df["value"] - mean) / std
    df["is_outlier"] = df["z_score"].abs() >= threshold_sigma
    df["method"] = f"z-score > {threshold_sigma}σ"
    return df[df["is_outlier"]].sort_values(by="z_score", ascending=False).reset_index(drop=True)


def crosstab_metric(
    ctx: DataContext,
    metric: str,
    row_dimension: str,
    col_dimension: str,
    filters: Optional[Dict[str, Any]] = None,
    min_n: int = DEFAULT_MIN_N,
) -> pd.DataFrame:
    """Builds a 2D matrix of a metric broken down by two dimensions."""
    spec = METRICS[metric]
    df = _apply_filters(ctx.frame(spec.base), filters)
    if row_dimension not in df.columns or col_dimension not in df.columns:
        raise ValueError(f"Dimensions '{row_dimension}', '{col_dimension}' not both present in fact frame.")

    res_matrix = {}
    for (r_val, c_val), grp in df.groupby([row_dimension, col_dimension], observed=True):
        val = spec.compute(grp)["value"]
        if r_val not in res_matrix:
            res_matrix[r_val] = {}
        res_matrix[r_val][c_val] = val
    return pd.DataFrame.from_dict(res_matrix, orient="index")


def _kpi_display(value: Any, spec: "MetricSpec") -> str:
    """Render one KPI the way an executive reads it.

    Rupee amounts go to crore/lakh rather than eleven digits, percentages lose
    the space before the sign, and counts carry no unit word at all.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    v = float(value)
    if spec.unit == "₹":
        if abs(v) >= 1e7:
            return f"Rs {v / 1e7:,.2f} cr"
        if abs(v) >= 1e5:
            return f"Rs {v / 1e5:,.2f} lakh"
        return f"Rs {v:,.0f}"
    if spec.unit == "%":
        return f"{v:.2f}%"
    if spec.unit == "days":
        return f"{v:,.1f} days"
    return f"{v:{spec.display_format}}"


def portfolio_overview(ctx: DataContext, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generates an executive KPI summary card across all domains."""
    kpi_keys = [
        "policy_count", "total_ape", "avg_ape", "total_sum_assured",
        "inforce_rate", "lapse_rate", "surrender_rate", "persistency_rate",
        "persistency_13m_rate", "persistency_25m_rate", "pivc_mismatch_rate",
        "rcu_rejection_rate", "claim_settlement_ratio", "avg_settlement_tat"
    ]
    res = {}
    for k in kpi_keys:
        if k in METRICS:
            spec = METRICS[k]
            sub = _apply_filters(ctx.frame(spec.base), filters)
            c = spec.compute(sub)
            res[k] = {
                "label": spec.label,
                "value": c["value"],
                "numerator": c["numerator"],
                "denominator": c["denominator"],
                "n": c["n"],
                "unit": spec.unit,
                "reliable": c["reliable"],
                # Appending the unit produced "21,026,553,570 ₹" and "50.68 %".
                # An executive KPI strip needs "Rs 2,102.66 cr" and "50.68%".
                "display": _kpi_display(c["value"], spec),
            }
    return res


def entity_shortlist(
    ctx: DataContext,
    dimension: str,
    conditions: List[Dict[str, Any]],  # [{"metric": ..., "op": ">=" | "<=", "threshold": ...}]
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
) -> pd.DataFrame:
    """Finds entities matching multiple strict multi-metric criteria."""
    candidates = None
    for cond in conditions:
        m = cond["metric"]
        op = cond.get("op", ">=")
        thresh = cond["threshold"]
        df = metric_by_dimension(ctx, m, dimension, filters, min_n=1)
        if df.empty:
            continue
        if op == ">=":
            matched = df[df["value"] >= thresh][[dimension, "value"]].rename(columns={"value": f"{m}_val"})
        elif op == "<=":
            matched = df[df["value"] <= thresh][[dimension, "value"]].rename(columns={"value": f"{m}_val"})
        elif op == ">":
            matched = df[df["value"] > thresh][[dimension, "value"]].rename(columns={"value": f"{m}_val"})
        else:
            matched = df[df["value"] < thresh][[dimension, "value"]].rename(columns={"value": f"{m}_val"})
        
        candidates = matched if candidates is None else candidates.merge(matched, on=dimension, how="inner")
    
    return candidates.head(top_k) if candidates is not None else pd.DataFrame()


def segment_profile(
    ctx: DataContext,
    segment_filters: Dict[str, Any],
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Provides a 360-degree KPI profile of a specific segment vs portfolio."""
    target_metrics = metrics or [
        "policy_count", "total_ape", "avg_ape", "avg_sum_assured",
        "lapse_rate", "surrender_rate", "persistency_13m_rate", "pivc_mismatch_rate"
    ]
    res = {}
    for m in target_metrics:
        if m in METRICS:
            spec = METRICS[m]
            base_df = ctx.frame(spec.base)
            seg_df = _apply_filters(base_df, segment_filters)
            base_c = spec.compute(base_df)
            seg_c = spec.compute(seg_df)
            res[m] = {
                "label": spec.label,
                "segment_value": seg_c["value"],
                "portfolio_baseline": base_c["value"],
                "diff": seg_c["value"] - base_c["value"] if pd.notna(seg_c["value"]) and pd.notna(base_c["value"]) else np.nan,
                "unit": spec.unit,
                "reliable": seg_c["reliable"],
            }
    return res


class InsufficientDataError(Exception):
    """Raised when an operation cannot be completed due to missing data."""
    pass


def multi_metric_table(
    ctx: DataContext,
    dimension: str,
    metrics: Sequence[str],
    filters: Optional[Dict[str, Any]] = None,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    min_n: int = DEFAULT_MIN_N,
) -> pd.DataFrame:
    """Builds a multi-metric summary table grouped by a dimension."""
    _share_metrics = ("policy_share", "sum_assured_share", "premium_share")
    frames = []
    for m in metrics:
        if m in _share_metrics:
            df = _share_metric_by_dimension(ctx, m, dimension, filters, min_n)
        elif m not in METRICS:
            continue
        else:
            df = metric_by_dimension(ctx, m, dimension, filters, min_n=min_n)
        if df.empty:
            continue
        frames.append(df[[dimension, "value", "sample_size", "reliable"]].rename(
            columns={"value": m, "sample_size": f"{m}_n", "reliable": f"{m}_rel"}))

    if not frames:
        return pd.DataFrame()

    res = frames[0]
    for f in frames[1:]:
        res = res.merge(f, on=dimension, how="outer")

    if sort_by and sort_by in res.columns:
        res = res.sort_values(by=sort_by, ascending=ascending)
    return res.reset_index(drop=True)



def correlation_analysis(
    ctx: DataContext,
    metric_a: str,
    metric_b: str,
    dimension: str,
    filters: Optional[Dict[str, Any]] = None,
    min_n: int = DEFAULT_MIN_N,
) -> pd.DataFrame:
    """Computes both metrics per segment of a shared dimension and returns
    a side-by-side table with Pearson correlation metadata."""
    spec_a = METRICS.get(metric_a)
    spec_b = METRICS.get(metric_b)
    if spec_a is None or spec_b is None:
        raise ValueError(f"Unknown metric(s): '{metric_a}', '{metric_b}'.")

    df_a = metric_by_dimension(ctx, metric_a, dimension, filters, min_n=min_n)
    df_b = metric_by_dimension(ctx, metric_b, dimension, filters, min_n=min_n)

    if df_a.empty or df_b.empty:
        return pd.DataFrame()

    dcol_a = _resolve_dim_col(dimension)
    dcol_b = _resolve_dim_col(dimension)
    # Use whichever dim column is actually present
    dcol = dcol_a if dcol_a in df_a.columns else dimension

    merged = df_a[[dcol, "value", "sample_size", "reliable"]].rename(
        columns={"value": metric_a, "sample_size": f"{metric_a}_n", "reliable": f"{metric_a}_reliable"}
    ).merge(
        df_b[[dcol, "value", "sample_size", "reliable"]].rename(
            columns={"value": metric_b, "sample_size": f"{metric_b}_n", "reliable": f"{metric_b}_reliable"}
        ),
        on=dcol,
        how="inner",
    )

    if merged.empty or len(merged) < 3:
        return merged

    vals_a = merged[metric_a].dropna()
    vals_b = merged[metric_b].dropna()
    common_idx = vals_a.index.intersection(vals_b.index)
    if len(common_idx) >= 3:
        pearson_r = float(np.corrcoef(vals_a[common_idx], vals_b[common_idx])[0, 1])
    else:
        pearson_r = float("nan")

    direction = "positive" if pearson_r > 0.2 else ("negative" if pearson_r < -0.2 else "weak/no")

    merged["pearson_r"] = pearson_r
    merged["direction"] = direction
    merged["n_segments"] = len(merged)

    # Adverse flag: both metrics are adverse for their respective higher_is_better setting
    def is_adverse_val(m: str, val: float) -> bool:
        s = METRICS.get(m)
        if s is None or pd.isna(val):
            return False
        avg = merged[m].mean()
        return val < avg if s.higher_is_better else val > avg

    merged["both_adverse"] = merged.apply(
        lambda r: is_adverse_val(metric_a, r[metric_a]) and is_adverse_val(metric_b, r[metric_b]),
        axis=1,
    )
    merged["either_adverse"] = merged.apply(
        lambda r: is_adverse_val(metric_a, r[metric_a]) or is_adverse_val(metric_b, r[metric_b]),
        axis=1,
    )

    # Sort by combined risk: both adverse first, then by metric_a value
    sort_asc_a = spec_a.higher_is_better  # ascending = worst first if higher_is_better
    merged = merged.sort_values(
        by=["both_adverse", metric_a],
        ascending=[False, sort_asc_a]
    ).reset_index(drop=True)

    return merged


def run_operation(ctx: DataContext, op: str, params: Dict[str, Any]) -> Any:
    """Dispatches operation string to the corresponding analytics_core function."""
    op_clean = op.lower().strip()
    
    # Map synonyms
    if op_clean in ("metric_by_dimension", "groupby_metric"):
        return metric_by_dimension(
            ctx,
            metric=params["metric"],
            dimension=params.get("dimension") or params.get("dim", "plan_name"),
            filters=params.get("filters"),
            min_n=int(params.get("min_n", DEFAULT_MIN_N)),
            sort_descending=params.get("sort_descending"),
        )
    elif op_clean in ("compare_to_baseline", "baseline_compare"):
        return compare_to_baseline(
            ctx,
            metric=params["metric"],
            dimension=params.get("dimension") or params.get("dim", "plan_name"),
            filters=params.get("filters"),
            min_n=int(params.get("min_n", DEFAULT_MIN_N)),
        )
    elif op_clean in ("rank", "rank_entities", "top_k"):
        return rank_entities(
            ctx,
            metric=params["metric"],
            dimension=params.get("dimension") or params.get("dim", "plan_name"),
            filters=params.get("filters"),
            top_k=int(params.get("top_k", 5)),
            worst=bool(params.get("worst", False)),
            min_n=int(params.get("min_n", DEFAULT_MIN_N)),
        )
    elif op_clean in ("composite_score", "risk_index"):
        return composite_score(
            ctx,
            dimension=params.get("dimension") or params.get("dim", "plan_name"),
            components=params["components"],
            filters=params.get("filters"),
            top_k=int(params.get("top_k", 10)),
            min_n=int(params.get("min_n", DEFAULT_MIN_N)),
        )
    elif op_clean in ("driver_scan", "variance_drivers"):
        return driver_scan(
            ctx,
            metric=params["metric"],
            focus_filters=params.get("focus_filters") or params.get("filters"),
            candidate_dimensions=params.get("candidate_dimensions") or params.get("dimensions"),
            min_n=int(params.get("min_n", DEFAULT_MIN_N)),
            top_k=int(params.get("top_k", 5)),
        )
    elif op_clean in ("concentration", "concentration_analysis", "pareto"):
        return concentration_analysis(
            ctx,
            dimension=params.get("dimension") or params.get("dim", "plan_name"),
            value_metric=params.get("value_metric", "total_ape"),
            top_k=int(params.get("top_k", 5)),
            filters=params.get("filters"),
        )
    elif op_clean in ("top_entity_concentration", "single_policy_concentration", "entity_concentration"):
        return top_entity_concentration(
            ctx,
            column=params.get("column", "sum_assured"),
            top_n=int(params.get("top_n", 10)),
            filters=params.get("filters"),
        )
    elif op_clean in ("outliers", "outliers_analysis", "anomaly_detection"):
        return outliers_analysis(
            ctx,
            metric=params["metric"],
            dimension=params.get("dimension") or params.get("dim", "plan_name"),
            threshold_sigma=float(params.get("threshold_sigma", 2.0)),
            filters=params.get("filters"),
            min_n=int(params.get("min_n", DEFAULT_MIN_N)),
        )
    elif op_clean in ("crosstab", "crosstab_metric", "matrix"):
        return crosstab_metric(
            ctx,
            metric=params["metric"],
            row_dimension=params.get("row_dimension") or params.get("dim_a", "plan_name"),
            col_dimension=params.get("col_dimension") or params.get("dim_b", "payment_mode"),
            filters=params.get("filters"),
            min_n=int(params.get("min_n", DEFAULT_MIN_N)),
        )
    elif op_clean in ("explain_gap", "why_gap", "gap_explanation"):
        return explain_gap(
            ctx,
            metric=params["metric"],
            dimension=params.get("dimension") or params.get("dim", "age_band"),
            filters=params.get("filters"),
            candidate_confounders=params.get("candidate_confounders"),
            min_n=int(params.get("min_n", DEFAULT_MIN_N)),
        )
    elif op_clean in ("portfolio_overview", "kpi_overview"):
        return portfolio_overview(ctx, filters=params.get("filters"))
    elif op_clean in ("entity_shortlist", "target_list"):
        return entity_shortlist(
            ctx,
            dimension=params.get("dimension") or params.get("dim", "client_id"),
            conditions=params["conditions"],
            filters=params.get("filters"),
            top_k=int(params.get("top_k", 10)),
        )
    elif op_clean in ("segment_profile", "360_profile"):
        return segment_profile(
            ctx,
            segment_filters=params.get("segment_filters") or params.get("filters", {}),
            metrics=params.get("metrics"),
        )
    elif op_clean in ("multi_metric_table", "metrics_table"):
        return multi_metric_table(
            ctx,
            dimension=params.get("dimension") or params.get("dim", "plan_name"),
            metrics=params.get("metrics", ["policy_count", "total_ape"]),
            filters=params.get("filters"),
            sort_by=params.get("sort_by"),
            ascending=bool(params.get("ascending", False)),
            min_n=int(params.get("min_n", DEFAULT_MIN_N)),
        )
    elif op_clean in ("correlation_analysis", "cross_domain_correlation", "metric_correlation"):
        return correlation_analysis(
            ctx,
            metric_a=params["metric_a"],
            metric_b=params["metric_b"],
            dimension=params.get("dimension") or params.get("dim", "plan_name"),
            filters=params.get("filters"),
            min_n=int(params.get("min_n", DEFAULT_MIN_N)),
        )
    else:
        raise ValueError(f"Unknown analytics operation '{op}'.")


# =====================================================================
# 4. DIMENSION REGISTRY
# =====================================================================

@dataclass
class DimensionSpec:
    name: str
    label: str
    column: str
    base: str = "policy"
    cardinality_hint: str = "low"  # "low" | "medium" | "high"


DIMENSIONS: Dict[str, DimensionSpec] = {
    "product": DimensionSpec("product", "Plan / Product", "plan_name"),
    "plan_name": DimensionSpec("plan_name", "Plan Name", "plan_name"),
    "product_category": DimensionSpec("product_category", "Trad vs ULIP", "trad_ulip"),
    "trad_ulip": DimensionSpec("trad_ulip", "Trad vs ULIP", "trad_ulip"),
    "channel": DimensionSpec("channel", "Sales Channel", "channel"),
    "sales_channel": DimensionSpec("sales_channel", "Sales Channel", "channel"),
    "state": DimensionSpec("state", "State", "state"),
    "city": DimensionSpec("city", "City", "city", cardinality_hint="high"),
    "occupation": DimensionSpec("occupation", "Occupation", "occupation"),
    "gender": DimensionSpec("gender", "Gender", "gender"),
    "education": DimensionSpec("education", "Education Level", "education"),
    "payment_mode": DimensionSpec("payment_mode", "Payment Mode", "payment_mode", base="persistency"),
    "persistency_bucket": DimensionSpec("persistency_bucket", "Persistency Bucket", "persistency_bucket", base="persistency"),
    "policy_status": DimensionSpec("policy_status", "Policy Status", "policy_status"),
    "cause_of_death": DimensionSpec("cause_of_death", "Cause of Death", "cause_of_death", base="claim"),
    "sales_type": DimensionSpec("sales_type", "Sales Agent Type", "sales_type", base="agent"),
    "income_band": DimensionSpec("income_band", "Income Band", "income_band"),
    "age_band": DimensionSpec("age_band", "Age Band", "age_band"),
    "sa_band": DimensionSpec("sa_band", "Sum Assured Band", "sa_band"),
    "ape_band": DimensionSpec("ape_band", "APE Band", "ape_band"),
    "ppt_band": DimensionSpec("ppt_band", "Premium Paying Term Band", "ppt_band"),
    "term_band": DimensionSpec("term_band", "Policy Term Band", "term_band"),
    "branch": DimensionSpec("branch", "Branch", "branch_no", cardinality_hint="high"),
    "branch_no": DimensionSpec("branch_no", "Branch Number", "branch_no", cardinality_hint="high"),
    "agent": DimensionSpec("agent", "Sales Rep", "sales_id", base="agent", cardinality_hint="high"),
    "sales_id": DimensionSpec("sales_id", "Sales ID", "sales_id", base="agent", cardinality_hint="high"),
    "issuance_year": DimensionSpec("issuance_year", "In-force Year", "inforce_year"),
    "policy_age_band": DimensionSpec("policy_age_band", "Policy Vintage / Term", "term_band"),
    "tenure_band": DimensionSpec("tenure_band", "Policy Term Band", "term_band"),
    "premium_to_income_band": DimensionSpec("premium_to_income_band", "APE to Income Band", "ape_to_income_band"),
    "customer_id": DimensionSpec("customer_id", "Client ID", "client_id", cardinality_hint="high"),
    "policy_number": DimensionSpec("policy_number", "App ID", "app_id", cardinality_hint="high"),
    "plan": DimensionSpec("plan", "Plan Name", "plan_name"),
    "renewal_status": DimensionSpec("renewal_status", "Renewal Status", "renewal_status", base="persistency"),
    "claim_status": DimensionSpec("claim_status", "Claim Status", "claim_status", base="claim"),
}

# Dimensions the previous registry exposed that this dataset cannot support.
# Several were aliased onto an unrelated column, so a question asking to break
# down "by smoker" was silently answered by gender. They are listed rather than
# deleted so the reason reaches the user.
RETIRED_DIMENSIONS: Dict[str, str] = {
    "smoker": "Owner_Details records no smoking status. This was aliased to `gender`.",
    "marital_status": "Owner_Details records no marital status. This was aliased to `gender`.",
    "surrender_reason": "No surrender-reason field exists. This was aliased to `policy_status`, "
                        "which says a policy surrendered, not why.",
    "medical_underwriting": "No medical-underwriting field exists.",
    "rider": "No rider field exists.",
    "rider_flag": "`rider_flag` is a synthesised constant with a single value.",
    "offer": "No offer or campaign field exists. This was aliased to `payment_mode`.",
    "contact_channel": "No outreach-contact field exists. This was aliased to `payment_mode`.",
    "agent_qualification": "Sales_Details.Type records Active/Inactive status, not qualification.",
    "zone": "No zone field exists. This was aliased to `state` (Owner.Comm.State).",
    "claim_type": "Every claim in Claims_Details is a death claim. This was aliased to "
                  "`cause_of_death`, which is a different concept.",
    "premium_payment_type": "Policy_Details.Mode is 'Annually' for all 25,000 rows.",
    "retention_status": "Renamed to `renewal_status` - the column is Renewal_Status, and "
                        "the dataset holds no retention programme.",
}

# Columns that carry a single value across the whole dataset. Breaking a metric
# down by one of these produces a one-row table that looks like a comparison.
DEGENERATE_DIMENSIONS: Dict[str, str] = {
    "channel": "CHANNEL is 'Banca' for all 25,000 policies.",
    "sales_channel": "CHANNEL is 'Banca' for all 25,000 policies.",
    "mode": "Policy_Details.Mode is 'Annually' for all 25,000 policies.",
}

OPERATIONS = [
    "metric_by_dimension", "compare_to_baseline", "rank", "rank_entities",
    "composite_score", "driver_scan", "concentration", "concentration_analysis",
    "outliers", "outliers_analysis", "crosstab", "crosstab_metric",
    "portfolio_overview", "entity_shortlist", "segment_profile", "multi_metric_table",
    "top_entity_concentration", "pareto_frontier", "waterfall", "metrics_table",
    "explain_gap", "why_gap", "gap_explanation",
    "correlation_analysis", "cross_domain_correlation", "metric_correlation"
]

SHARE_METRICS = ["policy_share", "ape_share", "sum_assured_share", "claim_share", "premium_share"]

def overall_metric(ctx: DataContext, metric: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Computes an un-grouped, portfolio-wide metric value with its declared denominator."""
    if metric not in METRICS:
        raise InsufficientDataError(f"Metric '{metric}' not found in registry.")
    spec = METRICS[metric]
    df = _apply_filters(ctx.frame(spec.base), filters)
    return spec.compute(df)

# Aliases for operational verbs
rank = rank_entities
concentration = concentration_analysis
outliers = outliers_analysis



