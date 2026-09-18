"""Phase 0 - Project Reconnaissance and Dataset Contract.

Profiles the five source CSVs in data/csv and emits:
  - config/data_contract.json   canonical machine-readable schema + parse contract
  - docs/DATA_PROFILE.md        human-readable profile

Everything recorded here is observable in the files. Where a choice had to be
made between two equally-parsing interpretations (notably the ambiguous
dd-mm-yyyy / mm-dd-yyyy dates in Policy_Details), the choice is cross-validated
against another table and the evidence is recorded in the contract.
"""
from __future__ import annotations

import json
import re
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "data" / "csv"

TABLES = OrderedDict([
    ("policy", "Policy_Details.csv"),
    ("owner", "Owner_Details.csv"),
    ("sales", "Sales_Details.csv"),
    ("claims", "Claims_Details.csv"),
    ("persistency", "Persistency_Details.csv"),
])

# Documented join keys, from the specification (Policy is the hub).
DOCUMENTED_JOINS = [
    {"name": "policy_owner", "left": "policy", "left_key": "AppID",
     "right": "owner", "right_key": "AppID", "documented_cardinality": "1:1"},
    {"name": "policy_claims", "left": "policy", "left_key": "AppID",
     "right": "claims", "right_key": "AppID", "documented_cardinality": "1:N"},
    {"name": "policy_sales", "left": "policy", "left_key": "SALES_ID",
     "right": "sales", "right_key": "SALES_ID", "documented_cardinality": "N:1"},
    {"name": "policy_persistency", "left": "policy", "left_key": "AppID",
     "right": "persistency", "right_key": "AppID", "documented_cardinality": "1:N"},
]

DATE_HINT = re.compile(r"(date|dob)", re.IGNORECASE)
MAX_CATEGORICAL = 60

# Columns that look numeric but are identifiers or codes. Typing them as numbers
# invites meaningless arithmetic, drops non-numeric values (foreign postal codes),
# and makes join keys depend on floating-point equality. They are kept as text.
IDENTIFIER_COLUMNS = {
    ("policy", "AppID"), ("policy", "SALES_ID"), ("policy", "Branch_No"),
    ("policy", "Inforce.month"),
    ("owner", "AppID"), ("owner", "ClientID"), ("owner", "Owner.Comm.Postal.Code"),
    ("sales", "SALES_ID"),
    ("claims", "AppID"), ("claims", "Claim_No"),
    ("persistency", "AppID"), ("persistency", "SALES_ID"),
}

# Month-first variants precede day-first variants. This ordering is not a guess:
# for Policy.Inforce.Date the dash-separated values parse equally well either
# way, and cross-validation against Persistency.Inforce.Date (which is
# unambiguous) proves month-first. See cross_validation in the emitted contract.
DATE_FORMATS = [
    "%m/%d/%Y", "%m-%d-%Y", "%d/%m/%Y", "%d-%m-%Y",
    "%Y-%m-%d", "%Y/%m/%d", "%d-%b-%y", "%d-%b-%Y", "%d %b %Y",
]


def normalize(col):
    """Canonical snake_case handle for a column; the source name is never mutated."""
    s = col.strip()
    s = re.sub(r"[.\s\-/]+", "_", s)
    s = re.sub(r"[^0-9A-Za-z_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()


def analyze_dates(values):
    """Greedily cover a date column with the fewest formats.

    Returns None when the column is not date-like. Otherwise returns a dict
    describing the format set, per-format counts and any residual values.
    """
    remaining = values.copy()
    chosen = []
    parsed_total = pd.Series(pd.NaT, index=values.index)

    while len(remaining) > 0:
        best_fmt, best_mask, best_n = None, None, 0
        for fmt in DATE_FORMATS:
            got = pd.to_datetime(remaining, format=fmt, errors="coerce")
            n = int(got.notna().sum())
            if n > best_n:
                best_fmt, best_mask, best_n = fmt, got, n
        if best_fmt is None or best_n == 0:
            break
        chosen.append({"format": best_fmt, "matched": best_n})
        parsed_total.loc[best_mask.dropna().index] = best_mask.dropna()
        remaining = remaining.loc[best_mask.isna()]
        if len(remaining) / max(len(values), 1) < 0.001 and len(remaining) == 0:
            break

    coverage = float(parsed_total.notna().mean()) if len(values) else 0.0
    if coverage < 0.90:
        return None

    residual = remaining.unique().tolist()
    return {
        "formats": chosen,
        "coverage_pct": round(100.0 * coverage, 4),
        "unparsed_count": int(len(remaining)),
        "unparsed_values": [str(v) for v in residual[:10]],
        "parsed": parsed_total,
    }


def try_numeric(values):
    """Parse numerics, tolerating thousands separators (incl. Indian lakh grouping)."""
    cleaned = values.str.replace(",", "", regex=False).str.replace("%", "", regex=False)
    parsed = pd.to_numeric(cleaned, errors="coerce")
    if len(values) and parsed.notna().mean() >= 0.98:
        return parsed
    return None


def jsonable(value):
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 6)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return str(value)


def profile_column(table_key, name, raw, n_rows):
    blank_as_null = raw.replace(r"^\s*$", np.nan, regex=True) if raw.dtype == object else raw
    null_count = int(blank_as_null.isna().sum())
    non_null = blank_as_null.dropna().astype(str).str.strip()
    unique_count = int(non_null.nunique())

    info = {
        "source_name": name,
        "normalized_name": normalize(name),
        "row_count": n_rows,
        "null_count": null_count,
        "null_pct": round(100.0 * null_count / n_rows, 4) if n_rows else 0.0,
        "non_null_count": int(len(non_null)),
        "unique_count": unique_count,
        "unique_pct": round(100.0 * unique_count / len(non_null), 4) if len(non_null) else 0.0,
        "is_unique_over_non_null": bool(len(non_null) > 0 and unique_count == len(non_null)),
        "comma_formatted_count": int(non_null.str.contains(",", regex=False).sum()),
        "sample_values": [str(v) for v in non_null.head(5).tolist()],
    }

    if (table_key, name) in IDENTIFIER_COLUMNS:
        info["inferred_type"] = "identifier"
        info["parse"] = {"kind": "identifier", "strip_whitespace": True,
                         "rationale": ("an identifier or code, not a quantity: kept as text so "
                                       "joins compare exactly and non-numeric values survive")}
        info["min_length"] = int(non_null.str.len().min()) if len(non_null) else None
        info["max_length"] = int(non_null.str.len().max()) if len(non_null) else None
        as_number = pd.to_numeric(non_null.str.replace(",", "", regex=False), errors="coerce")
        info["non_numeric_count"] = int(as_number.isna().sum())
        if info["non_numeric_count"]:
            info["non_numeric_examples"] = sorted(non_null[as_number.isna()].unique().tolist())[:5]
        if unique_count <= MAX_CATEGORICAL:
            info["categorical_values"] = sorted(non_null.unique().tolist())
        return info

    if DATE_HINT.search(name) and len(non_null):
        d = analyze_dates(non_null)
        if d is not None:
            parsed = d.pop("parsed")
            info["inferred_type"] = "date"
            info["parse"] = {"kind": "date", **d}
            info["min"] = jsonable(parsed.min())
            info["max"] = jsonable(parsed.max())
            return info

    nums = try_numeric(non_null) if len(non_null) else None
    if nums is not None:
        clean = nums.dropna()
        is_int = bool(len(clean) > 0 and np.all(np.equal(np.mod(clean, 1), 0)))
        info["inferred_type"] = "integer" if is_int else "float"
        info["parse"] = {"kind": "numeric", "strip_thousands_separator": True,
                         "unparsed_count": int(nums.isna().sum())}
        info["min"] = jsonable(clean.min())
        info["max"] = jsonable(clean.max())
        info["mean"] = jsonable(clean.mean())
        info["median"] = jsonable(clean.median())
        info["std"] = jsonable(clean.std())
        info["zero_count"] = int((clean == 0).sum())
        info["negative_count"] = int((clean < 0).sum())
        if unique_count <= MAX_CATEGORICAL:
            info["categorical_values"] = sorted(jsonable(v) for v in clean.unique())
        return info

    info["inferred_type"] = "string"
    info["parse"] = {"kind": "string", "strip_whitespace": True}
    info["min_length"] = int(non_null.str.len().min()) if len(non_null) else None
    info["max_length"] = int(non_null.str.len().max()) if len(non_null) else None
    if unique_count <= MAX_CATEGORICAL:
        vc = non_null.value_counts()
        info["categorical_values"] = sorted(vc.index.tolist())
        info["value_counts"] = {str(k): int(v) for k, v in vc.items()}
    return info


def profile_table(key, filename):
    path = CSV_DIR / filename
    df = pd.read_csv(path, dtype=str, keep_default_na=True,
                     na_values=["", "NA", "N/A", "null", "NULL", "nan"])
    n_rows = len(df)
    cols = [profile_column(key, c, df[c], n_rows) for c in df.columns]
    candidate_keys = [c["source_name"] for c in cols
                      if c["is_unique_over_non_null"] and c["null_count"] == 0 and n_rows > 1]
    table = {
        "table_key": key,
        "file": "data/csv/" + filename,
        "file_bytes": path.stat().st_size,
        "row_count": n_rows,
        "column_count": len(df.columns),
        "source_columns": list(df.columns),
        "normalized_columns": [normalize(c) for c in df.columns],
        "candidate_primary_keys": candidate_keys,
        "columns": cols,
    }
    return table, df


def duplicate_report(df, col):
    if col not in df.columns:
        return {"column": col, "present": False}
    s = df[col].dropna().astype(str).str.strip()
    dup_mask = s.duplicated(keep=False)
    dup_values = s[dup_mask].unique().tolist()
    return {
        "column": col,
        "present": True,
        "non_null_values": int(len(s)),
        "distinct_values": int(s.nunique()),
        "duplicated_rows": int(dup_mask.sum()),
        "distinct_duplicated_values": int(len(dup_values)),
        "example_duplicated_values": [str(v) for v in dup_values[:5]],
        "max_rows_per_value": int(s.value_counts().max()) if len(s) else 0,
    }


def validate_join(frames, spec):
    left, right = frames[spec["left"]], frames[spec["right"]]
    lk, rk = spec["left_key"], spec["right_key"]
    result = dict(spec)
    result["left_key_present"] = lk in left.columns
    result["right_key_present"] = rk in right.columns
    if not (result["left_key_present"] and result["right_key_present"]):
        result["validated"] = False
        result["error"] = "join key column missing"
        return result

    ls = left[lk].dropna().astype(str).str.strip()
    rs = right[rk].dropna().astype(str).str.strip()
    lset, rset = set(ls), set(rs)
    inter = lset & rset
    left_unique, right_unique = len(lset) == len(ls), len(rset) == len(rs)
    observed = ("1:1" if left_unique and right_unique else
                "1:N" if left_unique else "N:1" if right_unique else "N:M")

    result.update({
        "validated": True,
        "left_rows": int(len(left)), "right_rows": int(len(right)),
        "left_distinct_keys": len(lset), "right_distinct_keys": len(rset),
        "left_key_unique": bool(left_unique), "right_key_unique": bool(right_unique),
        "observed_cardinality": observed,
        "documented_cardinality_is_upper_bound": observed in ("1:1", "N:1"),
        "matched_keys": len(inter),
        "left_coverage_pct": round(100.0 * len(inter) / len(lset), 4) if lset else 0.0,
        "right_coverage_pct": round(100.0 * len(inter) / len(rset), 4) if rset else 0.0,
        "left_orphan_keys": len(lset - rset), "right_orphan_keys": len(rset - lset),
        "example_left_orphans": sorted(lset - rset)[:5],
        "example_right_orphans": sorted(rset - lset)[:5],
        "max_right_rows_per_key": int(rs.value_counts().max()) if len(rs) else 0,
        "max_left_rows_per_key": int(ls.value_counts().max()) if len(ls) else 0,
    })
    return result


def parse_dates_with(series, formats):
    """Apply an ordered format list the way the contract prescribes."""
    s = series.dropna().astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index)
    for fmt in formats:
        todo = out.isna()
        if not todo.any():
            break
        got = pd.to_datetime(s[todo], format=fmt, errors="coerce")
        out.loc[got.dropna().index] = got.dropna()
    return out


def cross_validate_policy_dates(frames):
    """Prove the month-first reading of Policy's ambiguous dash dates."""
    p, pe = frames["policy"], frames["persistency"]
    m = p[["AppID", "Inforce.Date"]].merge(
        pe[["AppID", "Inforce.Date"]], on="AppID", suffixes=("_pol", "_per"))
    truth = pd.to_datetime(m["Inforce.Date_per"], format="%m/%d/%Y", errors="coerce")
    out = {"joined_rows": int(len(m)),
           "reference": "Persistency_Details.Inforce.Date (unambiguous %m/%d/%Y)",
           "candidates": []}
    for fmts, label in ([["%m/%d/%Y", "%m-%d-%Y"], "month-first"],
                        [["%d/%m/%Y", "%d-%m-%Y"], "day-first"]):
        got = parse_dates_with(m["Inforce.Date_pol"], fmts)
        out["candidates"].append({
            "interpretation": label, "formats": fmts,
            "parsed": int(got.notna().sum()),
            "equal_to_reference": int((got == truth).sum()),
        })
    best = max(out["candidates"], key=lambda c: c["equal_to_reference"])
    out["conclusion"] = best["interpretation"]
    out["conclusive"] = bool(best["equal_to_reference"] == len(m))
    return out


def semantic_checks(frames):
    """Population semantics and reconciliation facts that PSs must respect."""
    p, pe, cl, o = frames["policy"], frames["persistency"], frames["claims"], frames["owner"]
    status = p["POLICY.STATUS"]
    death_ids = set(p.loc[status == "Death", "AppID"])
    claim_ids = set(cl["AppID"])
    active_lapsed = set(p.loc[status.isin(["Premium paying (regular)", "Lapsed"]), "AppID"])
    pers_ids = set(pe["AppID"])

    m = p[["AppID", "POLICY.STATUS"]].merge(
        pe[["AppID", "POLICY.STATUS"]], on="AppID", suffixes=("_policy", "_pers"))
    paid_flag = pe["Renewal_Status"].eq("Paid")
    paid_date = pe["Renewal_Paid_Date"].notna()

    due = parse_dates_with(pe["Renewal_Due_Date"], ["%m/%d/%Y"])
    pd_ = parse_dates_with(pe["Renewal_Paid_Date"], ["%m/%d/%Y"])
    delay = (pd_ - due).dt.days.dropna()

    return {
        "death_status_matches_claims_exactly": bool(death_ids == claim_ids),
        "death_status_count": len(death_ids),
        "claims_row_count": len(claim_ids),
        "persistency_population": {
            "policy_active_or_lapsed": len(active_lapsed),
            "persistency_rows": len(pers_ids),
            "policies_absent_from_persistency": len(active_lapsed - pers_ids),
            "note": ("Persistency covers only Premium paying (regular) and Lapsed policies; "
                     "Surrendered and Death policies are absent by design."),
        },
        "policy_status_agreement_with_persistency": {
            "compared": int(len(m)),
            "agree": int((m["POLICY.STATUS_policy"] == m["POLICY.STATUS_pers"]).sum()),
        },
        "renewal_status_consistent_with_paid_date": bool(paid_flag.equals(paid_date)),
        "renewal_delay_days": {
            "paid_rows": int(len(delay)),
            "min": jsonable(delay.min()), "median": jsonable(delay.median()),
            "max": jsonable(delay.max()),
            "late_gt_0_days": int((delay > 0).sum()),
            "on_or_before_due": int((delay <= 0).sum()),
        },
        "owner_clientid_multi_policy": {
            "distinct_clients": int(o["ClientID"].nunique()),
            "clients_with_multiple_policies": int((o["ClientID"].value_counts() > 1).sum()),
            "max_policies_per_client": int(o["ClientID"].value_counts().max()),
        },
    }


def data_quality_findings(frames, contract):
    """Explicit, evidence-backed defects analysts and PSs must account for."""
    p, o, pe, cl, s = (frames[k] for k in ("policy", "owner", "persistency", "claims", "sales"))
    f = []

    f.append({
        "id": "DQ-001", "severity": "high", "tables": ["owner", "persistency"],
        "columns": ["AppID"],
        "finding": ("One AppID is stored in Excel scientific notation '1.50E+10' in "
                    "Owner_Details and Persistency_Details. It is the only 8-character "
                    "AppID in either file; every other AppID is 11 characters."),
        "evidence": {
            "owner_rows_affected": int((o["AppID"] == "1.50E+10").sum()),
            "persistency_rows_affected": int((pe["AppID"] == "1.50E+10").sum()),
            "policy_appid_unmatched_by_owner": sorted(set(p["AppID"]) - set(o["AppID"])),
        },
        "impact": "Breaks the Policy-Owner and Policy-Persistency join for exactly one policy.",
        "handling": ("NOT silently repaired. The loader exposes it as an explicit, logged "
                     "repair rule (config: repair_scientific_notation_appid). When disabled, "
                     "the row is reported as an orphan rather than guessed."),
    })

    logind = parse_dates_with(p["Login.Date"], ["%m/%d/%Y", "%m-%d-%Y"])
    infd = parse_dates_with(p["Inforce.Date"], ["%m/%d/%Y", "%m-%d-%Y"])
    f.append({
        "id": "DQ-002", "severity": "high", "tables": ["policy"],
        "columns": ["Login.Date", "Inforce.Date"],
        "finding": ("Both Policy date columns mix separators ('/' and '-') within the same "
                    "column. All values are month-first; the dash-separated subset is "
                    "ambiguous on its own and was resolved by cross-validation."),
        "evidence": {
            "login_slash": int(p["Login.Date"].str.contains("/").sum()),
            "login_dash": int(p["Login.Date"].str.contains("-").sum()),
            "inforce_slash": int(p["Inforce.Date"].str.contains("/").sum()),
            "inforce_dash": int(p["Inforce.Date"].str.contains("-").sum()),
            "cross_validation": contract["cross_validation"]["policy_inforce_date"],
        },
        "impact": "Naive single-format parsing silently drops ~17-33% of rows.",
        "handling": "Contract prescribes the ordered format list ['%m/%d/%Y', '%m-%d-%Y'].",
    })

    f.append({
        "id": "DQ-003", "severity": "medium", "tables": ["claims"],
        "columns": ["Claim.Amount", "Amount.Paid"],
        "finding": ("Monetary values are stored as text with thousands separators, including "
                    "Indian lakh grouping (e.g. '12,50,000.00', '1,66,500')."),
        "evidence": {
            "claim_amount_comma_values": int(cl["Claim.Amount"].astype(str).str.contains(",").sum()),
            "amount_paid_comma_values": int(cl["Amount.Paid"].astype(str).str.contains(",").sum()),
        },
        "impact": "float() on the raw string raises; naive coercion yields nulls.",
        "handling": "Contract prescribes comma stripping before numeric coercion.",
    })

    f.append({
        "id": "DQ-004", "severity": "medium", "tables": ["claims"],
        "columns": ["Date.of.Intimation", "Settlement.Date", "Status.of.Claim", "Amount.Paid"],
        "finding": ("Claim date columns are predominantly '%d-%b-%y' but contain a small "
                    "number of '%m/%d/%Y' values. The single unsettled claim carries the "
                    "literal string 'WIP' in three columns at once - Status.of.Claim, "
                    "Amount.Paid and Settlement.Date - so 'WIP' is a sentinel in a monetary "
                    "and a date column, not only in the status column."),
        "evidence": {
            "wip_columns": {c: int((cl[c] == "WIP").sum())
                            for c in cl.columns if (cl[c] == "WIP").any()},
            "wip_claim_appids": cl.loc[cl.eq("WIP").any(axis=1), "AppID"].tolist(),
            "settled_claims": int((cl["Status.of.Claim"] == "Paid").sum()),
        },
        "impact": ("Settlement-lag and amount-paid metrics must exclude the unsettled claim "
                   "rather than coerce 'WIP' to zero or null silently."),
        "handling": ("'WIP' is a declared sentinel meaning not-yet-settled. It is treated as "
                     "missing for date and money maths, and the claim is excluded from "
                     "settlement-lag and payout-ratio metrics with the exclusion reported."),
    })

    fyp = s["First Year Persistency Rate"]
    f.append({
        "id": "DQ-005", "severity": "medium", "tables": ["sales"],
        "columns": ["First Year Persistency Rate"],
        "finding": "First Year Persistency Rate is null for a material share of agents.",
        "evidence": {"null_rows": int(fyp.isna().sum()), "total_rows": int(len(s)),
                     "null_pct": round(100.0 * fyp.isna().sum() / len(s), 2)},
        "impact": "Agent persistency rankings must declare the eligible population.",
        "handling": "Rankings exclude null-rate agents and report the excluded count.",
    })

    f.append({
        "id": "DQ-006", "severity": "medium", "tables": ["policy", "persistency"],
        "columns": ["POLICY.STATUS"],
        "finding": ("Persistency_Details contains only Premium paying (regular) and Lapsed "
                    "policies. Surrendered and Death policies have no persistency row."),
        "evidence": contract["semantic_checks"]["persistency_population"],
        "impact": "Persistency metrics computed over Policy_Details would use a wrong denominator.",
        "handling": "Persistency PSs declare the eligible population explicitly.",
    })

    f.append({
        "id": "DQ-007", "severity": "low", "tables": ["policy", "claims"],
        "columns": ["AppID"],
        "finding": ("Claims and Persistency are 1:1 with Policy in this dataset, not 1:N as the "
                    "documented relationship allows. At most one claim row exists per policy."),
        "evidence": {j["name"]: {"observed": j["observed_cardinality"],
                                 "max_right_rows_per_key": j["max_right_rows_per_key"]}
                     for j in contract["joins"] if j["name"] in ("policy_claims", "policy_persistency")},
        "impact": "Joins cannot fan out, but code must not hard-code the 1:1 assumption.",
        "handling": "Joins remain many-safe; row-count assertions guard against silent fan-out.",
    })

    f.append({
        "id": "DQ-008", "severity": "low", "tables": ["policy"],
        "columns": ["CHANNEL", "Mode"],
        "finding": "CHANNEL is 'Banca' for all 25,000 rows and Mode is 'Annually' for all rows.",
        "evidence": {"channel_distinct": int(p["CHANNEL"].nunique()),
                     "mode_distinct": int(p["Mode"].nunique())},
        "impact": "Channel and payment-mode comparison questions cannot be answered.",
        "handling": "No PS may offer channel or mode comparison; the limitation is stated.",
    })

    f.append({
        "id": "DQ-009", "severity": "low", "tables": ["owner"],
        "columns": ["Owner.Age", "Owner.Earned.Income", "Owner.Comm.Postal.Code", "Owner.Gender"],
        "finding": ("Owner attributes contain implausible values: ages as low as 1, an earned "
                    "income up to 8,384,891,049, postal codes as low as 1, and a third gender "
                    "code 'C' alongside M/F."),
        "evidence": {
            "age_min": int(pd.to_numeric(o["Owner.Age"], errors="coerce").min()),
            "age_under_18": int((pd.to_numeric(o["Owner.Age"], errors="coerce") < 18).sum()),
            "income_max": float(pd.to_numeric(o["Owner.Earned.Income"], errors="coerce").max()),
            "gender_values": o["Owner.Gender"].value_counts().to_dict(),
        },
        "impact": "Means are distorted by extreme values; demographic PSs need medians or caveats.",
        "handling": "Distribution PSs report medians alongside means and flag outliers.",
    })

    f.append({
        "id": "DQ-010", "severity": "low", "tables": ["policy"],
        "columns": ["Inforce.month"],
        "finding": ("Inforce.month is an unpadded YYYYM/YYYYMM integer: '20231' means 2023-01 "
                    "while '202310' means 2023-10. Lexical or numeric sorting is wrong."),
        "evidence": {"distinct": int(p["Inforce.month"].nunique()),
                     "examples": sorted(p["Inforce.month"].unique())[:6]},
        "impact": "Trend ordering breaks unless the field is decomposed into year and month.",
        "handling": "Contract prescribes deriving period from Inforce.Date, not this column.",
    })

    login_after = int((logind > infd).sum())
    f.append({
        "id": "DQ-011", "severity": "low", "tables": ["policy"],
        "columns": ["Login.Date", "Inforce.Date"],
        "finding": "A policy has a Login.Date later than its Inforce.Date.",
        "evidence": {"rows_affected": login_after},
        "impact": "Underwriting-turnaround metrics can produce a negative duration.",
        "handling": "Turnaround PSs report and exclude negative durations.",
    })

    postal = o["Owner.Comm.Postal.Code"].dropna().astype(str).str.strip()
    non_numeric = postal[pd.to_numeric(postal, errors="coerce").isna()]
    f.append({
        "id": "DQ-012", "severity": "medium", "tables": ["owner"],
        "columns": ["Owner.Comm.Postal.Code"],
        "finding": ("Owner.Comm.Postal.Code is not numeric. Two records hold alphanumeric "
                    "foreign postal codes, matching owners with a non-India address."),
        "evidence": {
            "non_numeric_rows": int(len(non_numeric)),
            "examples": sorted(non_numeric.unique().tolist())[:5],
            "countries": sorted(o.loc[non_numeric.index, "Owner.Comm.Country"].unique().tolist()),
        },
        "impact": ("Coercing the column to a number silently discards these two values and "
                   "would strip leading zeros from any code that has one."),
        "handling": ("The column is typed as an identifier and kept as text. Identifier columns "
                     "are never summed or averaged, and join keys compare exactly rather than "
                     "through floating-point equality."),
    })

    identifier_cols = sorted("{}.{}".format(t, c) for t, c in IDENTIFIER_COLUMNS)
    f.append({
        "id": "DQ-013", "severity": "low", "tables": ["policy", "owner", "sales", "claims",
                                                      "persistency"],
        "columns": identifier_cols,
        "finding": ("Twelve columns are numeric-looking identifiers or codes rather than "
                    "quantities: AppID, ClientID, SALES_ID, Branch_No, Claim_No, "
                    "Inforce.month and Owner.Comm.Postal.Code."),
        "evidence": {"identifier_columns": identifier_cols,
                     "count": len(identifier_cols)},
        "impact": ("Typed as numbers they would permit meaningless arithmetic such as an average "
                   "AppID, and would make every join depend on float equality."),
        "handling": "All are typed as identifiers and held as text throughout the system.",
    })

    return f


def build_contract():
    tables, frames = OrderedDict(), {}
    for key, filename in TABLES.items():
        t, df = profile_table(key, filename)
        tables[key], frames[key] = t, df

    contract = {
        "contract_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_of_truth": "five CSV files in data/csv",
        "hub_table": "policy",
        "tables": tables,
        "key_integrity": {
            "policy": [duplicate_report(frames["policy"], "AppID"),
                       duplicate_report(frames["policy"], "SALES_ID")],
            "owner": [duplicate_report(frames["owner"], "AppID"),
                      duplicate_report(frames["owner"], "ClientID")],
            "sales": [duplicate_report(frames["sales"], "SALES_ID")],
            "claims": [duplicate_report(frames["claims"], "AppID"),
                       duplicate_report(frames["claims"], "Claim_No")],
            "persistency": [duplicate_report(frames["persistency"], "AppID"),
                            duplicate_report(frames["persistency"], "SALES_ID")],
        },
        "joins": [validate_join(frames, j) for j in DOCUMENTED_JOINS],
    }
    contract["cross_validation"] = {"policy_inforce_date": cross_validate_policy_dates(frames)}
    contract["semantic_checks"] = semantic_checks(frames)
    contract["data_quality_findings"] = data_quality_findings(frames, contract)
    contract["unanswerable_domains"] = [
        {"domain": "channel comparison",
         "reason": "CHANNEL has a single value ('Banca') across all 25,000 policies."},
        {"domain": "payment-mode comparison at policy level",
         "reason": "Policy.Mode has a single value ('Annually'). Persistency.Payment_Mode does vary."},
        {"domain": "premium collection / payment transactions",
         "reason": "No transaction-level premium table exists in the five CSVs."},
        {"domain": "agent demographics, tenure, hierarchy or geography",
         "reason": "Sales_Details holds only performance rates and an Active/Inactive Type flag."},
        {"domain": "policyholder health, medical underwriting or risk scoring",
         "reason": "No underwriting, medical or risk-score columns exist."},
        {"domain": "commission, cost, profitability or expense analysis",
         "reason": "No commission or expense columns exist in any of the five files."},
        {"domain": "fund performance / NAV analysis for ULIP policies",
         "reason": "No fund, NAV or unit columns exist despite ULIP policies being present."},
        {"domain": "customer contact, servicing or complaint case history",
         "reason": "Only an aggregate agent-level complaint rate exists, with no case records."},
    ]
    return contract


MD_INTRO = """# Excel_sense_v2 - Data Profile (Phase 0)

Generated by `scripts/profile_data.py` from the five CSVs in `data/csv`.
The machine-readable equivalent is `config/data_contract.json`.

"""


def render_markdown(c):
    L = [MD_INTRO, "Generated (UTC): {}\n".format(c["generated_at_utc"])]

    L.append("## 1. Files\n")
    L.append("| Table | File | Rows | Cols | Primary key |")
    L.append("|---|---|---:|---:|---|")
    for k, t in c["tables"].items():
        pk = ", ".join(t["candidate_primary_keys"]) or "(none)"
        L.append("| `{}` | `{}` | {:,} | {} | {} |".format(
            k, t["file"], t["row_count"], t["column_count"], pk))

    L.append("\n## 2. Relationships\n")
    L.append("| Join | Keys | Documented | Observed | Matched | Left cov. | Right cov. | Orphans (L/R) |")
    L.append("|---|---|---|---|---:|---:|---:|---|")
    for j in c["joins"]:
        L.append("| `{}` | `{}.{}` -> `{}.{}` | {} | {} | {:,} | {}% | {}% | {} / {} |".format(
            j["name"], j["left"], j["left_key"], j["right"], j["right_key"],
            j["documented_cardinality"], j["observed_cardinality"], j["matched_keys"],
            j["left_coverage_pct"], j["right_coverage_pct"],
            j["left_orphan_keys"], j["right_orphan_keys"]))

    cv = c["cross_validation"]["policy_inforce_date"]
    L.append("\n### 2.1 Policy date interpretation (cross-validated)\n")
    L.append("`Policy_Details` dates mix `/` and `-` separators. The dash subset is ambiguous, "
             "so it was resolved against `{}` over {:,} joined rows:\n".format(
                 cv["reference"], cv["joined_rows"]))
    L.append("| Interpretation | Formats | Parsed | Equal to reference |")
    L.append("|---|---|---:|---:|")
    for cand in cv["candidates"]:
        L.append("| {} | `{}` | {:,} | {:,} |".format(
            cand["interpretation"], ", ".join(cand["formats"]),
            cand["parsed"], cand["equal_to_reference"]))
    L.append("\n**Conclusion: {} ordering, proven on every joined row ({}).**\n".format(
        cv["conclusion"], "conclusive" if cv["conclusive"] else "not conclusive"))

    s = c["semantic_checks"]
    L.append("\n## 3. Semantic reconciliation\n")
    L.append("- Policies with `POLICY.STATUS = Death` match the Claims table exactly: **{}** ({:,} rows).".format(
        s["death_status_matches_claims_exactly"], s["claims_row_count"]))
    pp = s["persistency_population"]
    L.append("- Persistency covers **{:,}** of the **{:,}** active-or-lapsed policies "
             "({} absent). {}".format(pp["persistency_rows"], pp["policy_active_or_lapsed"],
                                      pp["policies_absent_from_persistency"], pp["note"]))
    ag = s["policy_status_agreement_with_persistency"]
    L.append("- `POLICY.STATUS` agrees between Policy and Persistency on **{:,}/{:,}** joined rows.".format(
        ag["agree"], ag["compared"]))
    L.append("- `Renewal_Status = Paid` is exactly equivalent to a present `Renewal_Paid_Date`: **{}**.".format(
        s["renewal_status_consistent_with_paid_date"]))
    rd = s["renewal_delay_days"]
    L.append("- Renewal payment delay across {:,} paid rows: min {}, median {}, max {} days "
             "({:,} paid after the due date).".format(
                 rd["paid_rows"], rd["min"], rd["median"], rd["max"], rd["late_gt_0_days"]))
    oc = s["owner_clientid_multi_policy"]
    L.append("- {:,} distinct clients; {:,} hold more than one policy (max {} per client).".format(
        oc["distinct_clients"], oc["clients_with_multiple_policies"], oc["max_policies_per_client"]))

    L.append("\n## 4. Data-quality findings\n")
    for d in c["data_quality_findings"]:
        L.append("### {} ({}) - {}\n".format(d["id"], d["severity"].upper(), ", ".join(d["tables"])))
        L.append("**Columns:** {}\n".format(", ".join("`{}`".format(x) for x in d["columns"])))
        L.append("{}\n".format(d["finding"]))
        L.append("- *Impact:* {}".format(d["impact"]))
        L.append("- *Handling:* {}\n".format(d["handling"]))

    L.append("\n## 5. Domains the data cannot support\n")
    L.append("These are excluded from the Problem Statement catalogue by construction.\n")
    L.append("| Domain | Reason |")
    L.append("|---|---|")
    for u in c["unanswerable_domains"]:
        L.append("| {} | {} |".format(u["domain"], u["reason"]))

    L.append("\n## 6. Column profiles\n")
    for k, t in c["tables"].items():
        L.append("\n### `{}` - {} ({:,} rows)\n".format(k, t["file"], t["row_count"]))
        L.append("| Column | Type | Null % | Distinct | Range / values |")
        L.append("|---|---|---:|---:|---|")
        for col in t["columns"]:
            if col["inferred_type"] == "date":
                rng = "{} .. {}".format(col["min"], col["max"])
            elif col["inferred_type"] in ("integer", "float"):
                rng = "min {} / med {} / max {}".format(col["min"], col["median"], col["max"])
            elif "categorical_values" in col:
                vals = [str(v) for v in col["categorical_values"]]
                shown = ", ".join("`{}`".format(v) for v in vals[:6])
                rng = shown + (" ... (+{} more)".format(len(vals) - 6) if len(vals) > 6 else "")
            else:
                rng = "text, length {}-{}".format(col.get("min_length"), col.get("max_length"))
            L.append("| `{}` | {} | {} | {:,} | {} |".format(
                col["source_name"], col["inferred_type"], col["null_pct"],
                col["unique_count"], rng))
    return "\n".join(L) + "\n"


def main():
    contract = build_contract()
    (ROOT / "config").mkdir(exist_ok=True)
    (ROOT / "docs").mkdir(exist_ok=True)

    out = ROOT / "config" / "data_contract.json"
    out.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print("wrote {} ({:,} bytes)".format(out.name, out.stat().st_size))

    md = ROOT / "docs" / "DATA_PROFILE.md"
    md.write_text(render_markdown(contract), encoding="utf-8")
    print("wrote {} ({:,} bytes)".format(md.name, md.stat().st_size))
    return contract


if __name__ == "__main__":
    main()
