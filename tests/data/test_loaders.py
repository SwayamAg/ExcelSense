"""Phase 2 - loader tests. The central guarantee is no silent data loss."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import DataContract, load_dataset
from src.data.contract import ROW_ID, SOURCE_TABLE, TABLE_FILES
from src.data.loaders import CORRUPTED_APPID, CORRUPTED_APPID_TARGET, DataLoadError

EXPECTED_ROWS = {
    "policy": 25000, "owner": 25000, "sales": 1956,
    "claims": 134, "persistency": 24542,
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def test_all_five_tables_load(dataset):
    assert sorted(dataset.table_keys) == sorted(TABLE_FILES)


@pytest.mark.parametrize("key,rows", sorted(EXPECTED_ROWS.items()))
def test_row_counts_match_the_source_files(dataset, key, rows):
    assert len(dataset.table(key)) == rows
    assert dataset.report.rows_read[key] == rows


def test_loaded_columns_match_the_source_exactly(dataset, raw_frames):
    """Source column names are preserved, never renamed."""
    for key in EXPECTED_ROWS:
        assert dataset.table(key).columns == list(raw_frames[key].columns)


def test_missing_csv_directory_fails_loudly(tmp_path):
    with pytest.raises(DataLoadError, match="dataset file missing"):
        load_dataset(csv_dir=tmp_path)


def test_schema_drift_is_detected(tmp_path, raw_frames):
    """A CSV whose columns changed must fail the load, not load partially."""
    for key, name in TABLE_FILES.items():
        frame = raw_frames[key]
        if key == "policy":
            frame = frame.drop(columns=["APE"])
        frame.to_csv(tmp_path / name, index=False)
    with pytest.raises(DataLoadError, match="columns differ from the contract"):
        load_dataset(csv_dir=tmp_path)


# --------------------------------------------------------------------------
# No silent data loss
# --------------------------------------------------------------------------

def test_nothing_is_lost_silently(dataset):
    """Zero unparsed values: everything either parses or is a declared sentinel."""
    assert dataset.report.unparsed_total == 0, (
        "values failed to parse: {}".format(
            [i.as_dict() for i in dataset.report.issues if i.kind.startswith("unparsed")]))


def test_a_value_is_null_only_if_it_was_missing_or_a_sentinel(dataset, raw_frames):
    """The strongest loss check: compare null counts against the raw file."""
    for key, table in ((k, dataset.table(k)) for k in EXPECTED_ROWS):
        raw = raw_frames[key]
        for col in table.columns:
            raw_missing = raw[col].isna() | (raw[col].astype(str).str.strip() == "")
            sentinels = raw[col].astype(str).str.strip() == "WIP"
            allowed = int((raw_missing | sentinels).sum())
            actual = int(table.col(col).isna().sum())
            assert actual == allowed, (
                "{}.{}: {} nulls after load but only {} were missing or sentinel".format(
                    key, col, actual, allowed))


def test_declared_sentinels_are_reported(dataset):
    sentinels = {(i.table, i.column) for i in dataset.report.issues if i.kind == "sentinel"}
    assert sentinels == {
        ("claims", "Status.of.Claim"),
        ("claims", "Amount.Paid"),
        ("claims", "Settlement.Date"),
    }


def test_sentinel_becomes_missing_not_zero(dataset):
    """'WIP' must never be read as a payment of zero."""
    claims = dataset.table("claims")
    paid = claims.num("Amount.Paid")
    assert paid.isna().sum() == 1
    assert (paid == 0).sum() == 0, "a sentinel was coerced to zero"


def test_high_severity_findings_surface_as_warnings(dataset, contract):
    high = {f["id"] for f in contract["data_quality_findings"] if f["severity"] == "high"}
    text = " ".join(dataset.report.warnings)
    for finding_id in high:
        assert finding_id in text, "{} is high severity but never warned about".format(finding_id)


# --------------------------------------------------------------------------
# Typing
# --------------------------------------------------------------------------

def test_numeric_columns_are_numeric(dataset):
    assert dataset.table("policy").num("APE").dtype == np.float64
    assert dataset.table("policy").num("Sum.Assured").dtype == np.float64
    assert dataset.table("claims").num("Claim.Amount").dtype == np.float64


def test_date_columns_are_datetimes(dataset):
    for table, col in (("policy", "Inforce.Date"), ("policy", "Login.Date"),
                       ("claims", "Date.of.Death"), ("persistency", "Renewal_Due_Date")):
        assert pd.api.types.is_datetime64_any_dtype(dataset.table(table).date(col))


def test_identifiers_are_text_not_numbers(dataset):
    """An identifier is a label. Typing it as a number invites meaningless maths."""
    for table, col in (("policy", "AppID"), ("policy", "SALES_ID"), ("policy", "Branch_No"),
                       ("owner", "ClientID"), ("claims", "Claim_No"),
                       ("owner", "Owner.Comm.Postal.Code")):
        series = dataset.table(table).ident(col)
        assert series.dtype == object, "{}.{} is not text".format(table, col)
        with pytest.raises(TypeError, match="not numeric"):
            dataset.table(table).num(col)


def test_identifiers_keep_their_exact_source_text(dataset, raw_frames):
    for table, col in (("policy", "AppID"), ("owner", "ClientID"), ("sales", "SALES_ID")):
        loaded = dataset.table(table).ident(col)
        source = raw_frames[table][col].astype(str).str.strip()
        assert (loaded == source).all(), "{}.{} was altered on load".format(table, col)


def test_foreign_postal_codes_survive(dataset):
    """DQ-012: numeric coercion would have discarded these two values."""
    postal = dataset.table("owner").ident("Owner.Comm.Postal.Code").dropna()
    non_numeric = sorted(v for v in postal if not str(v).isdigit())
    assert non_numeric == ["L9T 6X4", "TW149RE"]


def test_typed_accessors_reject_the_wrong_type(dataset):
    policy = dataset.table("policy")
    with pytest.raises(TypeError, match="not a date"):
        policy.date("APE")
    with pytest.raises(TypeError, match="not numeric"):
        policy.num("Inforce.Date")
    with pytest.raises(TypeError, match="not an identifier"):
        policy.ident("APE")


# --------------------------------------------------------------------------
# Parse correctness (regression against known values)
# --------------------------------------------------------------------------

def test_mixed_separator_policy_dates_all_parse(dataset):
    """DQ-002: naive single-format parsing loses 17-33% of these rows."""
    policy = dataset.table("policy")
    assert policy.date("Inforce.Date").notna().sum() == 25000
    assert policy.date("Login.Date").notna().sum() == 25000


def test_policy_dates_are_read_month_first(dataset):
    """Cross-validated in Phase 0: Policy must agree with Persistency row for row."""
    policy, pers = dataset.table("policy"), dataset.table("persistency")
    merged = (policy.frame[["AppID", "Inforce.Date"]]
              .merge(pers.frame[["AppID", "Inforce.Date"]], on="AppID",
                     suffixes=("_policy", "_pers")))
    assert len(merged) == 24541
    assert (merged["Inforce.Date_policy"] == merged["Inforce.Date_pers"]).all()


def test_lakh_grouped_money_parses_correctly(dataset, raw_frames):
    """DQ-003: '12,50,000.00' must become 1250000.0, not fail or truncate."""
    claims = dataset.table("claims")
    raw = raw_frames["claims"]["Claim.Amount"].astype(str)
    grouped = raw.str.contains(",")
    assert grouped.any()
    for idx in raw[grouped].index[:5]:
        expected = float(raw.loc[idx].replace(",", ""))
        assert claims.num("Claim.Amount").loc[idx] == expected


def test_known_aggregate_values(dataset):
    """Regression anchors: these change only if the data or the parse rules change."""
    policy = dataset.table("policy")
    assert round(float(policy.num("APE").sum()), 2) == 2306202661.89
    assert round(float(policy.num("APE").mean()), 4) == 92248.1065
    assert float(policy.num("APE").median()) == 50000.0
    assert round(float(policy.num("Sum.Assured").sum()), 2) == 21026553569.89

    status = policy.col("POLICY.STATUS").value_counts().to_dict()
    assert status == {"Premium paying (regular)": 21805, "Lapsed": 2737,
                      "Surrendered": 324, "Death": 134}

    claims = dataset.table("claims")
    assert round(float(claims.num("Claim.Amount").sum()), 2) == 93030740.33
    assert round(float(claims.num("Amount.Paid").sum()), 2) == 92472739.22

    pers = dataset.table("persistency")
    assert int((pers.col("Renewal_Status") == "Paid").sum()) == 12439
    assert round(float(pers.num("Renewal_Premium_Amount").sum()), 2) == 2269575936.0

    sales = dataset.table("sales")
    assert int(sales.num("First Year Persistency Rate").isna().sum()) == 369


def test_date_ranges_are_as_profiled(dataset):
    policy = dataset.table("policy")
    assert policy.date("Inforce.Date").min() == pd.Timestamp("2023-01-08")
    assert policy.date("Inforce.Date").max() == pd.Timestamp("2025-12-31")


# --------------------------------------------------------------------------
# Duplicates, keys and traceability
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key,pk", [
    ("policy", "AppID"), ("owner", "AppID"), ("sales", "SALES_ID"),
    ("claims", "AppID"), ("persistency", "AppID"),
])
def test_primary_keys_are_unique_and_populated(dataset, key, pk):
    series = dataset.table(key).ident(pk)
    assert series.isna().sum() == 0
    assert series.duplicated().sum() == 0


def test_non_key_duplicates_are_preserved(dataset):
    """ClientID repeats by design; the loader must not deduplicate it."""
    clients = dataset.table("owner").ident("ClientID")
    assert clients.nunique() == 24757
    assert len(clients) == 25000
    assert clients.value_counts().max() == 3


def test_every_row_keeps_its_source_position(dataset):
    for key in EXPECTED_ROWS:
        table = dataset.table(key)
        ids = table.frame[ROW_ID]
        assert ids.tolist() == list(range(len(table))), "{} lost row ordering".format(key)
        assert (table.frame[SOURCE_TABLE] == key).all()


def test_raw_text_remains_available(dataset, raw_frames):
    """Typed values never replace the source text; both stay reachable."""
    policy = dataset.table("policy")
    assert policy.raw_col("APE").iloc[0] == raw_frames["policy"]["APE"].iloc[0]
    assert policy.raw_col("Inforce.Date").iloc[0] == raw_frames["policy"]["Inforce.Date"].iloc[0]


def test_column_access_is_normalized_without_renaming(dataset):
    """Several spellings reach one column, but the source name is unchanged."""
    policy = dataset.table("policy")
    expected = policy.col("POLICY.STATUS")
    for alias in ("policy_status", "POLICY.STATUS", "policy.status", "POLICY_STATUS"):
        assert policy.col(alias).equals(expected)
    assert "POLICY.STATUS" in policy.columns
    assert "policy_status" not in policy.columns


def test_unknown_column_raises_with_a_helpful_message(dataset):
    with pytest.raises(KeyError, match="no column matching"):
        dataset.table("policy").col("does_not_exist")


# --------------------------------------------------------------------------
# The DQ-001 repair is opt-in and never silent
# --------------------------------------------------------------------------

def test_corrupted_appid_is_not_repaired_by_default(dataset):
    owner = dataset.table("owner").ident("AppID")
    assert (owner == CORRUPTED_APPID).sum() == 1
    assert dataset.report.repairs == []


def test_default_load_leaves_exactly_one_orphan(dataset):
    policy = set(dataset.table("policy").ident("AppID"))
    owner = set(dataset.table("owner").ident("AppID"))
    assert policy - owner == {CORRUPTED_APPID_TARGET}
    assert owner - policy == {CORRUPTED_APPID}


def test_repair_is_opt_in_and_recorded():
    repaired = load_dataset(repair_corrupted_appid=True)
    owner = repaired.table("owner").ident("AppID")
    assert (owner == CORRUPTED_APPID).sum() == 0
    assert (owner == CORRUPTED_APPID_TARGET).sum() == 1

    assert len(repaired.report.repairs) == 2, "repair must be recorded for owner and persistency"
    for repair in repaired.report.repairs:
        assert repair["finding"] == "DQ-001"
        assert repair["basis"], "a repair must record why it was justified"
    assert any("DQ-001" in w for w in repaired.report.warnings)


def test_repair_closes_the_join(dataset):
    repaired = load_dataset(repair_corrupted_appid=True)
    policy = set(repaired.table("policy").ident("AppID"))
    for other in ("owner", "persistency"):
        keys = set(repaired.table(other).ident("AppID"))
        assert not keys - policy, "{} still has orphans after repair".format(other)


def test_repair_does_not_change_row_counts(dataset):
    repaired = load_dataset(repair_corrupted_appid=True)
    for key, rows in EXPECTED_ROWS.items():
        assert len(repaired.table(key)) == rows


# --------------------------------------------------------------------------
# Contract consistency
# --------------------------------------------------------------------------

def test_contract_version_is_exposed(dataset, contract):
    assert dataset.contract.version == contract["contract_version"]


def test_stale_contract_row_count_produces_a_warning(tmp_path, raw_frames, contract):
    """If the CSVs change under us, the load says so rather than quietly proceeding."""
    import copy
    import json

    for key, name in TABLE_FILES.items():
        frame = raw_frames[key]
        if key == "claims":
            frame = frame.head(100)
        frame.to_csv(tmp_path / name, index=False)

    payload = copy.deepcopy(contract)
    modified = DataContract(payload)
    result = load_dataset(contract=modified, csv_dir=tmp_path)
    assert any("contract may be stale" in w for w in result.report.warnings)
    assert len(result.table("claims")) == 100
