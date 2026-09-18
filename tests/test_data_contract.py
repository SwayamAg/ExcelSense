"""Phase 0 gate tests - the five CSVs load and their relationships hold.

These assert against the CSVs directly as well as against config/data_contract.json,
so the contract cannot silently drift away from the data it claims to describe.
"""
from __future__ import annotations

import pandas as pd
import pytest

from tests.conftest import EXPECTED_FILES

EXPECTED_ROWS = {
    "policy": 25000, "owner": 25000, "sales": 1956,
    "claims": 134, "persistency": 24542,
}
EXPECTED_COLS = {
    "policy": 16, "owner": 12, "sales": 10, "claims": 9, "persistency": 12,
}
PRIMARY_KEYS = {
    "policy": "AppID", "owner": "AppID", "sales": "SALES_ID",
    "claims": "AppID", "persistency": "AppID",
}


def parse_dates(series: pd.Series, formats) -> pd.Series:
    """Apply an ordered format list exactly as the contract prescribes."""
    s = series.dropna().astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index)
    for fmt in formats:
        todo = out.isna()
        if not todo.any():
            break
        got = pd.to_datetime(s[todo], format=fmt, errors="coerce")
        out.loc[got.dropna().index] = got.dropna()
    return out


def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


# --------------------------------------------------------------------------
# Files and schema
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key,filename", sorted(EXPECTED_FILES.items()))
def test_expected_file_exists(csv_dir, key, filename):
    path = csv_dir / filename
    assert path.exists(), "missing dataset file: {}".format(filename)
    assert path.stat().st_size > 0, "empty dataset file: {}".format(filename)


def test_only_the_five_datasets_are_present(csv_dir):
    found = {p.name for p in csv_dir.glob("*.csv")}
    assert found == set(EXPECTED_FILES.values()), (
        "data/csv must hold exactly the five source CSVs; found {}".format(sorted(found)))


@pytest.mark.parametrize("key", sorted(EXPECTED_FILES))
def test_all_five_files_load(raw_frames, key):
    df = raw_frames[key]
    assert not df.empty
    assert len(df.columns) == EXPECTED_COLS[key]


@pytest.mark.parametrize("key", sorted(EXPECTED_FILES))
def test_row_counts_are_stable(raw_frames, key):
    assert len(raw_frames[key]) == EXPECTED_ROWS[key]


@pytest.mark.parametrize("key", sorted(EXPECTED_FILES))
def test_contract_matches_actual_schema(contract, raw_frames, key):
    """The contract's column list must equal the file's column list, in order."""
    table = contract["tables"][key]
    assert table["source_columns"] == list(raw_frames[key].columns)
    assert table["row_count"] == len(raw_frames[key])
    assert table["column_count"] == len(raw_frames[key].columns)


def test_contract_is_readable_and_complete(contract):
    for field in ("contract_version", "generated_at_utc", "hub_table", "tables",
                  "joins", "key_integrity", "semantic_checks",
                  "data_quality_findings", "unanswerable_domains"):
        assert field in contract, "contract missing '{}'".format(field)
    assert contract["hub_table"] == "policy"
    assert set(contract["tables"]) == set(EXPECTED_FILES)


def test_normalized_column_names_are_unique_per_table(contract):
    for key, table in contract["tables"].items():
        norm = table["normalized_columns"]
        assert len(norm) == len(set(norm)), "normalized name collision in {}".format(key)


# --------------------------------------------------------------------------
# Keys, nulls and duplicates
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key,pk", sorted(PRIMARY_KEYS.items()))
def test_key_fields_exist(raw_frames, key, pk):
    assert pk in raw_frames[key].columns


@pytest.mark.parametrize("key,pk", sorted(PRIMARY_KEYS.items()))
def test_primary_keys_are_unique_and_populated(raw_frames, key, pk):
    s = raw_frames[key][pk]
    assert s.isna().sum() == 0, "{}.{} has nulls".format(key, pk)
    assert s.duplicated().sum() == 0, "{}.{} has duplicates".format(key, pk)


def test_documented_join_columns_are_present(raw_frames):
    assert "SALES_ID" in raw_frames["policy"].columns
    assert "SALES_ID" in raw_frames["sales"].columns
    assert "SALES_ID" in raw_frames["persistency"].columns
    assert "ClientID" in raw_frames["owner"].columns
    assert "Claim_No" in raw_frames["claims"].columns


def test_duplicate_and_null_checks_are_reported(contract):
    """Every key we rely on must have an integrity record, present or not."""
    integrity = contract["key_integrity"]
    assert set(integrity) == set(EXPECTED_FILES)
    for key, reports in integrity.items():
        assert reports, "no integrity report for {}".format(key)
        for r in reports:
            assert "column" in r and "present" in r
            if r["present"]:
                for field in ("non_null_values", "distinct_values",
                              "duplicated_rows", "max_rows_per_value"):
                    assert field in r


def test_clientid_is_not_unique_and_that_is_recorded(contract, raw_frames):
    """ClientID repeats: a client may hold several policies. Must not be used as a PK."""
    counts = raw_frames["owner"]["ClientID"].value_counts()
    assert counts.max() > 1
    recorded = contract["semantic_checks"]["owner_clientid_multi_policy"]
    assert recorded["max_policies_per_client"] == int(counts.max())
    assert recorded["clients_with_multiple_policies"] == int((counts > 1).sum())


# --------------------------------------------------------------------------
# Relationships
# --------------------------------------------------------------------------

def test_all_documented_joins_were_validated(contract):
    names = {j["name"] for j in contract["joins"]}
    assert names == {"policy_owner", "policy_claims", "policy_sales", "policy_persistency"}
    for j in contract["joins"]:
        assert j["validated"] is True, "join {} was not validated".format(j["name"])
        assert j["left_key_present"] and j["right_key_present"]


def test_join_coverage_is_measured_for_every_join(contract):
    for j in contract["joins"]:
        for field in ("matched_keys", "left_coverage_pct", "right_coverage_pct",
                      "left_orphan_keys", "right_orphan_keys", "observed_cardinality"):
            assert field in j, "join {} missing {}".format(j["name"], field)


def test_policy_sales_join_is_total(contract, raw_frames):
    """Every policy resolves to an agent and every agent has at least one policy."""
    pol, sales = set(raw_frames["policy"]["SALES_ID"]), set(raw_frames["sales"]["SALES_ID"])
    assert pol == sales
    j = next(x for x in contract["joins"] if x["name"] == "policy_sales")
    assert j["left_coverage_pct"] == 100.0 and j["right_coverage_pct"] == 100.0
    assert j["observed_cardinality"] == "N:1"


def test_policy_owner_join_covers_all_but_the_known_corrupted_id(raw_frames):
    pol, own = set(raw_frames["policy"]["AppID"]), set(raw_frames["owner"]["AppID"])
    assert pol - own == {"15011000000"}
    assert own - pol == {"1.50E+10"}


def test_every_claim_resolves_to_a_policy(raw_frames):
    assert set(raw_frames["claims"]["AppID"]) <= set(raw_frames["policy"]["AppID"])


def test_joins_do_not_fan_out(contract):
    """Claims and Persistency are 1:1 here; assert so a silent fan-out is caught."""
    for name in ("policy_claims", "policy_persistency", "policy_owner"):
        j = next(x for x in contract["joins"] if x["name"] == name)
        assert j["max_right_rows_per_key"] == 1, "{} fans out".format(name)


# --------------------------------------------------------------------------
# Parse contract - no silent data loss
# --------------------------------------------------------------------------

def test_every_date_column_parses_fully_under_the_contract(contract, raw_frames):
    """Applying the contract's formats must lose nothing except declared sentinels."""
    checked = 0
    for key, table in contract["tables"].items():
        for col in table["columns"]:
            if col["inferred_type"] != "date":
                continue
            checked += 1
            formats = [f["format"] for f in col["parse"]["formats"]]
            parsed = parse_dates(raw_frames[key][col["source_name"]], formats)
            unparsed = int(parsed.isna().sum())
            assert unparsed == col["parse"]["unparsed_count"], (
                "{}.{}: contract claims {} unparsed, got {}".format(
                    key, col["source_name"], col["parse"]["unparsed_count"], unparsed))
            if unparsed:
                leftovers = set(raw_frames[key][col["source_name"]].dropna()[parsed.isna()])
                assert leftovers <= {"WIP"}, (
                    "{}.{} has undeclared unparsable values {}".format(
                        key, col["source_name"], leftovers))
    assert checked == 8, "expected 8 date columns, profiled {}".format(checked)


def test_policy_dates_are_month_first_and_fully_parsed(raw_frames):
    """The mixed-separator Policy dates must parse at 100% under month-first rules."""
    for col in ("Login.Date", "Inforce.Date"):
        parsed = parse_dates(raw_frames["policy"][col], ["%m/%d/%Y", "%m-%d-%Y"])
        assert parsed.isna().sum() == 0, "{} did not fully parse".format(col)


def test_policy_date_reading_is_cross_validated(contract, raw_frames):
    """Month-first is proven, not assumed: it must match Persistency on every joined row."""
    cv = contract["cross_validation"]["policy_inforce_date"]
    assert cv["conclusion"] == "month-first" and cv["conclusive"] is True

    m = raw_frames["policy"][["AppID", "Inforce.Date"]].merge(
        raw_frames["persistency"][["AppID", "Inforce.Date"]],
        on="AppID", suffixes=("_pol", "_per"))
    truth = pd.to_datetime(m["Inforce.Date_per"], format="%m/%d/%Y", errors="coerce")
    got = parse_dates(m["Inforce.Date_pol"], ["%m/%d/%Y", "%m-%d-%Y"])
    assert (got == truth).all(), "month-first reading disagrees with Persistency"
    assert cv["joined_rows"] == len(m)


def test_monetary_columns_parse_despite_thousands_separators(raw_frames):
    """Indian lakh grouping must not produce nulls; only the declared sentinel may fail."""
    claims = raw_frames["claims"]
    for col in ("Claim.Amount", "Amount.Paid"):
        assert claims[col].astype(str).str.contains(",").any(), (
            "{} no longer contains grouped values - revisit DQ-003".format(col))
        failed = set(claims.loc[to_number(claims[col]).isna(), col].dropna())
        assert failed <= {"WIP"}, "{} lost rows on parse: {}".format(col, failed)


def test_wip_sentinel_is_confined_to_the_single_unsettled_claim(contract, raw_frames):
    """'WIP' marks not-yet-settled and must be declared everywhere it appears."""
    claims = raw_frames["claims"]
    wip_cols = {c for c in claims.columns if (claims[c] == "WIP").any()}
    assert wip_cols == {"Status.of.Claim", "Amount.Paid", "Settlement.Date"}
    assert len(claims.loc[claims.eq("WIP").any(axis=1)]) == 1

    dq = next(f for f in contract["data_quality_findings"] if f["id"] == "DQ-004")
    assert wip_cols <= set(dq["columns"]), (
        "DQ-004 does not declare every column carrying the WIP sentinel")
    assert set(dq["evidence"]["wip_columns"]) == wip_cols


def test_numeric_columns_declared_in_contract_parse_without_loss(contract, raw_frames):
    for key, table in contract["tables"].items():
        for col in table["columns"]:
            if col["inferred_type"] not in ("integer", "float"):
                continue
            series = raw_frames[key][col["source_name"]].dropna()
            unparsed = int(to_number(series).isna().sum())
            assert unparsed == col["parse"]["unparsed_count"], (
                "{}.{}: {} values failed numeric parse".format(key, col["source_name"], unparsed))


# --------------------------------------------------------------------------
# Semantic reconciliation
# --------------------------------------------------------------------------

def test_death_status_reconciles_with_claims(contract, raw_frames):
    policy = raw_frames["policy"]
    deaths = set(policy.loc[policy["POLICY.STATUS"] == "Death", "AppID"])
    assert deaths == set(raw_frames["claims"]["AppID"])
    assert contract["semantic_checks"]["death_status_matches_claims_exactly"] is True


def test_persistency_population_is_active_or_lapsed_only(contract, raw_frames):
    pers_status = set(raw_frames["persistency"]["POLICY.STATUS"])
    assert pers_status == {"Premium paying (regular)", "Lapsed"}
    recorded = contract["semantic_checks"]["persistency_population"]
    assert recorded["policies_absent_from_persistency"] == 1  # the corrupted AppID only


def test_policy_status_agrees_across_tables(contract, raw_frames):
    m = raw_frames["policy"][["AppID", "POLICY.STATUS"]].merge(
        raw_frames["persistency"][["AppID", "POLICY.STATUS"]],
        on="AppID", suffixes=("_p", "_e"))
    assert (m["POLICY.STATUS_p"] == m["POLICY.STATUS_e"]).all()
    rec = contract["semantic_checks"]["policy_status_agreement_with_persistency"]
    assert rec["agree"] == rec["compared"] == len(m)


def test_renewal_status_is_equivalent_to_a_paid_date(raw_frames):
    pers = raw_frames["persistency"]
    assert pers["Renewal_Status"].eq("Paid").equals(pers["Renewal_Paid_Date"].notna())


def test_claim_dates_are_chronologically_ordered(raw_frames):
    claims = raw_frames["claims"]
    death = parse_dates(claims["Date.of.Death"], ["%d-%b-%y", "%m/%d/%Y"])
    intim = parse_dates(claims["Date.of.Intimation"], ["%d-%b-%y", "%m/%d/%Y"])
    settle = parse_dates(claims["Settlement.Date"], ["%d-%b-%y", "%m/%d/%Y"])
    assert (intim >= death).all(), "a claim was intimated before the death"
    both = settle.notna() & intim.notna()
    assert (settle[both] >= intim[both]).all(), "a claim settled before intimation"


# --------------------------------------------------------------------------
# No unsupported assumptions
# --------------------------------------------------------------------------

def test_data_quality_findings_are_recorded_with_evidence(contract):
    findings = contract["data_quality_findings"]
    assert len(findings) >= 10
    ids = [f["id"] for f in findings]
    assert len(ids) == len(set(ids)), "duplicate data-quality finding ids"
    for f in findings:
        for field in ("id", "severity", "tables", "columns", "finding",
                      "evidence", "impact", "handling"):
            assert field in f and f[field] not in (None, "", [], {}), (
                "{} missing {}".format(f.get("id"), field))
        assert f["severity"] in ("high", "medium", "low")


def test_corrupted_appid_is_declared_and_not_silently_repaired(contract, raw_frames):
    dq = next(f for f in contract["data_quality_findings"] if f["id"] == "DQ-001")
    assert "1.50E+10" in dq["finding"]
    assert "NOT silently repaired" in dq["handling"]
    # The defect is still in the source data - the contract describes, never edits.
    assert (raw_frames["owner"]["AppID"] == "1.50E+10").sum() == 1


def test_single_valued_columns_are_declared_unanswerable(contract, raw_frames):
    """CHANNEL and Mode cannot support comparison; that must be stated, not discovered later."""
    assert raw_frames["policy"]["CHANNEL"].nunique() == 1
    assert raw_frames["policy"]["Mode"].nunique() == 1
    domains = " ".join(d["domain"] + " " + d["reason"] for d in contract["unanswerable_domains"])
    assert "CHANNEL" in domains and "Mode" in domains


def test_unanswerable_domains_have_reasons(contract):
    assert len(contract["unanswerable_domains"]) >= 5
    for d in contract["unanswerable_domains"]:
        assert d.get("domain") and d.get("reason")


def test_contract_declares_no_column_absent_from_the_files(contract, raw_frames):
    """Guards against the contract inventing a field the CSVs do not have."""
    for key, table in contract["tables"].items():
        actual = set(raw_frames[key].columns)
        for col in table["columns"]:
            assert col["source_name"] in actual, (
                "contract invents {}.{}".format(key, col["source_name"]))
