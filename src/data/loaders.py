"""Deterministic loading of the five CSVs under the Phase 0 parse contract.

Guarantees:
  - No silent data loss. Every value that fails to parse is counted and reported
    in the LoadReport; a value only becomes null if it was null, blank, or a
    declared sentinel.
  - Source semantics are preserved. Raw text stays available alongside typed
    values, and source column names are never renamed.
  - Every row keeps its source position, so any figure can be traced back.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .contract import (
    CSV_DIR,
    INTERNAL_COLUMNS,
    ROW_ID,
    SENTINEL_VALUES,
    SOURCE_TABLE,
    TABLE_FILES,
    ContractError,
    DataContract,
    TableContract,
)

NA_STRINGS = ["", "NA", "N/A", "null", "NULL", "nan"]

# The single Excel-corrupted AppID (DQ-001) and the policy it belongs to.
CORRUPTED_APPID = "1.50E+10"
CORRUPTED_APPID_TARGET = "15011000000"


class DataLoadError(Exception):
    """Raised when a dataset cannot be loaded in a way the contract guarantees."""


@dataclass
class ParseIssue:
    table: str
    column: str
    kind: str            # 'unparsed_date' | 'unparsed_numeric' | 'sentinel'
    count: int
    examples: List[str] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table, "column": self.column, "kind": self.kind,
            "count": self.count, "examples": self.examples, "note": self.note,
        }


@dataclass
class LoadReport:
    """Everything that happened during a load, so nothing is silent."""
    rows_read: Dict[str, int] = field(default_factory=dict)
    issues: List[ParseIssue] = field(default_factory=list)
    repairs: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def unparsed_total(self) -> int:
        return sum(i.count for i in self.issues if i.kind.startswith("unparsed"))

    def issues_for(self, table: str, column: Optional[str] = None) -> List[ParseIssue]:
        return [i for i in self.issues
                if i.table == table and (column is None or i.column == column)]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rows_read": dict(self.rows_read),
            "issues": [i.as_dict() for i in self.issues],
            "repairs": list(self.repairs),
            "warnings": list(self.warnings),
            "unparsed_total": self.unparsed_total,
        }


def _parse_dates(values: pd.Series, formats: List[str]) -> pd.Series:
    """Apply an ordered format list, as the contract prescribes."""
    out = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    remaining = values.notna()
    for fmt in formats:
        if not remaining.any():
            break
        got = pd.to_datetime(values[remaining], format=fmt, errors="coerce")
        matched = got.notna()
        if matched.any():
            out.loc[got.index[matched]] = got[matched]
            remaining.loc[got.index[matched]] = False
    return out


def _parse_numeric(values: pd.Series) -> pd.Series:
    """Numeric coercion tolerant of thousands separators, incl. lakh grouping."""
    cleaned = values.str.replace(",", "", regex=False).str.replace("%", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


class Table:
    """One loaded dataset: typed values, raw text, and its contract."""

    def __init__(self, contract: TableContract, typed: pd.DataFrame, raw: pd.DataFrame):
        self.contract = contract
        self.key = contract.key
        self._typed = typed
        self._raw = raw

    # -- access ------------------------------------------------------------

    @property
    def frame(self) -> pd.DataFrame:
        """Typed frame. Callers must not mutate it; use .copy() to derive."""
        return self._typed

    @property
    def raw(self) -> pd.DataFrame:
        """Original text exactly as read from the CSV."""
        return self._raw

    def __len__(self) -> int:
        return len(self._typed)

    @property
    def columns(self) -> List[str]:
        return [c for c in self._typed.columns if c not in INTERNAL_COLUMNS]

    def resolve(self, name: str) -> str:
        return self.contract.resolve(name)

    def col(self, name: str) -> pd.Series:
        """Typed column by any accepted spelling."""
        return self._typed[self.resolve(name)]

    def raw_col(self, name: str) -> pd.Series:
        return self._raw[self.resolve(name)]

    def num(self, name: str) -> pd.Series:
        col = self.resolve(name)
        kind = self.contract.inferred_type(col)
        if kind not in ("integer", "float"):
            hint = (" Identifiers are labels, not quantities; use .col() to read it."
                    if kind == "identifier" else "")
            raise TypeError("{}.{} is {}, not numeric.{}".format(self.key, col, kind, hint))
        return self._typed[col]

    def ident(self, name: str) -> pd.Series:
        """Identifier column, as exact text."""
        col = self.resolve(name)
        kind = self.contract.inferred_type(col)
        if kind != "identifier":
            raise TypeError("{}.{} is {}, not an identifier".format(self.key, col, kind))
        return self._typed[col]

    def date(self, name: str) -> pd.Series:
        col = self.resolve(name)
        kind = self.contract.inferred_type(col)
        if kind != "date":
            raise TypeError("{}.{} is {}, not a date".format(self.key, col, kind))
        return self._typed[col]

    def text(self, name: str) -> pd.Series:
        return self._typed[self.resolve(name)].astype("object")

    @property
    def primary_key(self) -> Optional[str]:
        return self.contract.primary_key

    def row_ids(self) -> pd.Series:
        return self._typed[ROW_ID]


class Dataset:
    """All five tables, loaded once and shared."""

    def __init__(self, contract: DataContract, tables: Dict[str, Table], report: LoadReport):
        self.contract = contract
        self._tables = tables
        self.report = report

    def __getitem__(self, key: str) -> Table:
        return self.table(key)

    def table(self, key: str) -> Table:
        if key not in self._tables:
            raise DataLoadError(
                "unknown table {!r}; loaded: {}".format(key, sorted(self._tables)))
        return self._tables[key]

    def frame(self, key: str) -> pd.DataFrame:
        return self.table(key).frame

    @property
    def table_keys(self) -> List[str]:
        return sorted(self._tables)

    @property
    def warnings(self) -> List[str]:
        return self.report.warnings


def _load_table(
    contract: DataContract,
    key: str,
    report: LoadReport,
    csv_dir: Path,
    repair_corrupted_appid: bool,
) -> Table:
    table_contract = contract.table(key)
    path = csv_dir / TABLE_FILES[key]
    if not path.exists():
        raise DataLoadError("dataset file missing: {}".format(path))

    raw = pd.read_csv(path, dtype=str, keep_default_na=True, na_values=NA_STRINGS)
    report.rows_read[key] = len(raw)

    expected = table_contract.source_columns
    if list(raw.columns) != expected:
        raise DataLoadError(
            "{}: columns differ from the contract.\n  file:     {}\n  contract: {}".format(
                key, list(raw.columns), expected))
    if len(raw) != table_contract.row_count:
        report.warnings.append(
            "{}: {} rows read but the contract recorded {}; the contract may be stale".format(
                key, len(raw), table_contract.row_count))

    typed = pd.DataFrame(index=raw.index)
    typed[ROW_ID] = raw.index.to_numpy()          # 0-based source row, header excluded
    typed[SOURCE_TABLE] = key

    for source_col in expected:
        series = raw[source_col]
        stripped = series.str.strip() if series.dtype == object else series
        blank = stripped.isna() | (stripped == "")
        present = ~blank

        # Declared sentinels mean "not applicable", not "zero" and not "unparsable".
        is_sentinel = present & stripped.isin(SENTINEL_VALUES)
        if is_sentinel.any():
            report.issues.append(ParseIssue(
                table=key, column=source_col, kind="sentinel",
                count=int(is_sentinel.sum()),
                examples=sorted(stripped[is_sentinel].unique().tolist())[:5],
                note=("declared sentinel meaning not-yet-available; treated as missing "
                      "for arithmetic, never as zero")))

        parseable = present & ~is_sentinel
        kind = table_contract.inferred_type(source_col)

        if kind == "date":
            values = stripped.where(parseable)
            out = _parse_dates(values, table_contract.date_formats(source_col))
            failed = parseable & out.isna()
        elif kind in ("integer", "float"):
            values = stripped.where(parseable)
            out = _parse_numeric(values)
            failed = parseable & out.isna()
        else:
            # 'string' and 'identifier' both stay as text. Identifiers additionally
            # promise exact comparison, so they are never coerced to a number.
            out = stripped.where(parseable)
            failed = pd.Series(False, index=raw.index)

        if failed.any():
            report.issues.append(ParseIssue(
                table=key, column=source_col,
                kind="unparsed_date" if kind == "date" else "unparsed_numeric",
                count=int(failed.sum()),
                examples=sorted(set(stripped[failed].tolist()))[:5],
                note="value present in the file but not parsable under the contract"))

        typed[source_col] = out

    if repair_corrupted_appid and "AppID" in typed.columns:
        mask = raw["AppID"].astype(str).str.strip() == CORRUPTED_APPID
        if mask.any():
            typed.loc[mask, "AppID"] = CORRUPTED_APPID_TARGET
            report.repairs.append({
                "finding": "DQ-001", "table": key, "column": "AppID",
                "rows_repaired": int(mask.sum()),
                "from": CORRUPTED_APPID, "to": CORRUPTED_APPID_TARGET,
                "basis": ("the only 8-character AppID in the file; the only policy AppID "
                          "unmatched in this table is 15011000000, and the corrupted value is "
                          "its Excel scientific-notation rendering"),
            })
            report.warnings.append(
                "{}: repaired {} corrupted AppID under DQ-001 (explicitly enabled)".format(
                    key, int(mask.sum())))

    return Table(table_contract, typed, raw)


def load_dataset(
    contract: Optional[DataContract] = None,
    csv_dir: Optional[Path] = None,
    repair_corrupted_appid: bool = False,
) -> Dataset:
    """Load all five CSVs under the contract.

    repair_corrupted_appid is off by default: the corrupted AppID (DQ-001) is
    reported as an orphan rather than guessed. Turning it on records the repair
    and its basis in the LoadReport, so it is never silent.
    """
    contract = contract or DataContract.load()
    csv_dir = csv_dir or CSV_DIR
    report = LoadReport()

    tables: Dict[str, Table] = {}
    for key in TABLE_FILES:
        tables[key] = _load_table(contract, key, report, csv_dir, repair_corrupted_appid)

    for finding in contract.data_quality_findings:
        if finding["severity"] == "high":
            report.warnings.append("{}: {}".format(finding["id"], finding["finding"]))

    return Dataset(contract, tables, report)
