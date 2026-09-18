"""Access to the Phase 0 data contract, and column-name resolution.

Source column names are never mutated. A normalized alias (snake_case) is offered
alongside the source name so callers can write `policy_status` or `POLICY.STATUS`
and reach the same column.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "config" / "data_contract.json"
CSV_DIR = ROOT / "data" / "csv"

TABLE_FILES = {
    "policy": "Policy_Details.csv",
    "owner": "Owner_Details.csv",
    "sales": "Sales_Details.csv",
    "claims": "Claims_Details.csv",
    "persistency": "Persistency_Details.csv",
}

# Documented join keys: name -> (left table, left key, right table, right key).
RELATIONSHIPS = {
    "policy_owner": ("policy", "AppID", "owner", "AppID"),
    "policy_claims": ("policy", "AppID", "claims", "AppID"),
    "policy_sales": ("policy", "SALES_ID", "sales", "SALES_ID"),
    "policy_persistency": ("policy", "AppID", "persistency", "AppID"),
    "persistency_sales": ("persistency", "SALES_ID", "sales", "SALES_ID"),
}

# Values that mean "not applicable yet" rather than a real measurement.
# 'WIP' marks the single unsettled claim in three of its columns (DQ-004).
SENTINEL_VALUES = {"WIP"}

# Internal bookkeeping columns. Prefixed so they cannot collide with a source name.
ROW_ID = "__row_id"
SOURCE_TABLE = "__source_table"
INTERNAL_COLUMNS = (ROW_ID, SOURCE_TABLE)


class ContractError(Exception):
    """Raised when the data contract is missing, stale or inconsistent with the CSVs."""


def normalize(name: str) -> str:
    """snake_case handle for a source column name."""
    s = re.sub(r"[.\s\-/]+", "_", name.strip())
    s = re.sub(r"[^0-9A-Za-z_]", "", s)
    return re.sub(r"_+", "_", s).strip("_").lower()


class TableContract:
    """The parse and schema contract for one table."""

    def __init__(self, spec: Dict[str, Any]):
        self.key: str = spec["table_key"]
        self.file: str = spec["file"]
        self.row_count: int = spec["row_count"]
        self.source_columns: List[str] = list(spec["source_columns"])
        self.primary_keys: List[str] = list(spec["candidate_primary_keys"])
        self.columns: Dict[str, Dict[str, Any]] = {
            c["source_name"]: c for c in spec["columns"]
        }
        self._alias: Dict[str, str] = {}
        for source in self.source_columns:
            self._alias[source] = source
            self._alias[source.lower()] = source
            self._alias[normalize(source)] = source

    @property
    def primary_key(self) -> Optional[str]:
        return self.primary_keys[0] if self.primary_keys else None

    def resolve(self, name: str) -> str:
        """Map any accepted spelling of a column to its exact source name."""
        if name in self._alias:
            return self._alias[name]
        lowered = name.lower()
        if lowered in self._alias:
            return self._alias[lowered]
        norm = normalize(name)
        if norm in self._alias:
            return self._alias[norm]
        raise KeyError(
            "{}: no column matching {!r}. Available: {}".format(
                self.key, name, ", ".join(sorted(self.source_columns))))

    def has_column(self, name: str) -> bool:
        try:
            self.resolve(name)
        except KeyError:
            return False
        return True

    def inferred_type(self, name: str) -> str:
        return self.columns[self.resolve(name)]["inferred_type"]

    def date_formats(self, name: str) -> List[str]:
        spec = self.columns[self.resolve(name)]
        if spec["inferred_type"] != "date":
            return []
        return [f["format"] for f in spec["parse"]["formats"]]

    def columns_of_type(self, kind: str) -> List[str]:
        return [c for c, spec in self.columns.items() if spec["inferred_type"] == kind]


class DataContract:
    """The whole Phase 0 contract."""

    def __init__(self, payload: Dict[str, Any]):
        self.raw = payload
        self.version: str = payload["contract_version"]
        self.generated_at: str = payload["generated_at_utc"]
        self.hub_table: str = payload["hub_table"]
        self.tables: Dict[str, TableContract] = {
            key: TableContract(spec) for key, spec in payload["tables"].items()
        }
        self.data_quality_findings: List[Dict[str, Any]] = payload["data_quality_findings"]
        self.unanswerable_domains: List[Dict[str, Any]] = payload["unanswerable_domains"]
        self.joins: List[Dict[str, Any]] = payload["joins"]
        self.semantic_checks: Dict[str, Any] = payload["semantic_checks"]

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "DataContract":
        path = path or CONTRACT_PATH
        if not path.exists():
            raise ContractError(
                "data contract not found at {}; run scripts/profile_data.py".format(path))
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def table(self, key: str) -> TableContract:
        if key not in self.tables:
            raise ContractError(
                "unknown table {!r}; known: {}".format(key, sorted(self.tables)))
        return self.tables[key]

    def finding(self, finding_id: str) -> Dict[str, Any]:
        for f in self.data_quality_findings:
            if f["id"] == finding_id:
                return f
        raise ContractError("no data-quality finding {!r}".format(finding_id))

    def resolve_qualified(self, ref: str) -> tuple:
        """Split and resolve a 'table.Column' reference."""
        if "." not in ref:
            raise ContractError("column reference {!r} must be table.Column".format(ref))
        table_key, column = ref.split(".", 1)
        table = self.table(table_key)
        return table_key, table.resolve(column)
