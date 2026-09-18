"""Deterministic query operations over the loaded datasets.

Every operation records what it did, so an answer can be traced back to the rows
and rules that produced it. Row counts are tracked through the whole chain; a step
that drops rows must say how many and why, which is what makes "no silent data
loss" checkable rather than aspirational.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .contract import INTERNAL_COLUMNS, RELATIONSHIPS, ROW_ID, SOURCE_TABLE
from .loaders import Dataset, Table

Scalar = Union[str, int, float, pd.Timestamp, None]

COMPARATORS: Dict[str, Callable[[pd.Series, Any], pd.Series]] = {
    "eq": lambda s, v: s == v,
    "ne": lambda s, v: s != v,
    "lt": lambda s, v: s < v,
    "lte": lambda s, v: s <= v,
    "gt": lambda s, v: s > v,
    "gte": lambda s, v: s >= v,
    "in": lambda s, v: s.isin(list(v)),
    "not_in": lambda s, v: ~s.isin(list(v)),
    "contains": lambda s, v: s.astype(str).str.contains(str(v), case=False, na=False),
    "is_null": lambda s, v: s.isna(),
    "not_null": lambda s, v: s.notna(),
    "between": lambda s, v: s.between(v[0], v[1]),
}

AGGREGATORS: Dict[str, Callable[[pd.Series], float]] = {
    "count": lambda s: int(s.notna().sum()),
    "count_all": lambda s: int(len(s)),
    "distinct": lambda s: int(s.nunique(dropna=True)),
    "sum": lambda s: float(s.sum()) if s.notna().any() else 0.0,
    "mean": lambda s: float(s.mean()) if s.notna().any() else None,
    "median": lambda s: float(s.median()) if s.notna().any() else None,
    "min": lambda s: s.min() if s.notna().any() else None,
    "max": lambda s: s.max() if s.notna().any() else None,
    "std": lambda s: float(s.std()) if s.notna().sum() > 1 else None,
}


class QueryError(Exception):
    """Raised when a query is not executable against the loaded data."""


@dataclass
class Step:
    """One recorded operation in a query chain."""
    op: str
    detail: Dict[str, Any]
    rows_in: int
    rows_out: int

    @property
    def rows_dropped(self) -> int:
        return self.rows_in - self.rows_out

    def as_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op, "detail": self.detail,
            "rows_in": self.rows_in, "rows_out": self.rows_out,
            "rows_dropped": self.rows_dropped,
        }


@dataclass
class Provenance:
    """The audit trail of a query."""
    tables_used: List[str] = field(default_factory=list)
    columns_used: List[str] = field(default_factory=list)
    filters: List[Dict[str, Any]] = field(default_factory=list)
    join_path: List[str] = field(default_factory=list)
    steps: List[Step] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def record(self, op: str, detail: Dict[str, Any], rows_in: int, rows_out: int) -> None:
        self.steps.append(Step(op=op, detail=detail, rows_in=rows_in, rows_out=rows_out))

    def use_table(self, key: str) -> None:
        if key not in self.tables_used:
            self.tables_used.append(key)

    def use_columns(self, table: str, columns: Sequence[str]) -> None:
        for c in columns:
            ref = "{}.{}".format(table, c)
            if ref not in self.columns_used:
                self.columns_used.append(ref)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tables_used": list(self.tables_used),
            "columns_used": list(self.columns_used),
            "filters": list(self.filters),
            "join_path": list(self.join_path),
            "steps": [s.as_dict() for s in self.steps],
            "notes": list(self.notes),
        }


class Query:
    """A chainable, fully-recorded query over one or more tables.

    Every method returns a new Query; nothing mutates the loaded data.
    """

    def __init__(self, dataset: Dataset, frame: pd.DataFrame,
                 provenance: Provenance, origin: str):
        self._ds = dataset
        self._df = frame
        self.provenance = provenance
        self._origin = origin

    # -- construction ------------------------------------------------------

    @classmethod
    def from_table(cls, dataset: Dataset, table_key: str) -> "Query":
        table = dataset.table(table_key)
        prov = Provenance()
        prov.use_table(table_key)
        prov.record("load", {"table": table_key}, len(table), len(table))
        return cls(dataset, table.frame.copy(), prov, table_key)

    def _next(self, frame: pd.DataFrame) -> "Query":
        return Query(self._ds, frame, self.provenance, self._origin)

    # -- inspection --------------------------------------------------------

    @property
    def frame(self) -> pd.DataFrame:
        return self._df

    def __len__(self) -> int:
        return len(self._df)

    @property
    def columns(self) -> List[str]:
        return [c for c in self._df.columns if c not in INTERNAL_COLUMNS]

    def resolve(self, name: str) -> str:
        """Resolve a column by source name, normalized alias, or table.Column."""
        if name in self._df.columns:
            return name
        if "." in name:
            head, rest = name.split(".", 1)
            if head in self._ds.table_keys:
                qualified = "{}_{}".format(head, rest)
                if qualified in self._df.columns:
                    return qualified
                if rest in self._df.columns:
                    return rest
        table = self._ds.table(self._origin).contract
        if table.has_column(name):
            resolved = table.resolve(name)
            if resolved in self._df.columns:
                return resolved
        lowered = {c.lower(): c for c in self._df.columns}
        if name.lower() in lowered:
            return lowered[name.lower()]
        raise QueryError(
            "no column {!r} in the current result. Available: {}".format(
                name, ", ".join(self.columns)))

    # -- row operations ----------------------------------------------------

    def filter(self, column: str, comparator: str, value: Any = None,
               keep_null: bool = False) -> "Query":
        """WHERE. Null handling is explicit: nulls are dropped unless keep_null."""
        if comparator not in COMPARATORS:
            raise QueryError("unknown comparator {!r}; known: {}".format(
                comparator, sorted(COMPARATORS)))
        col = self.resolve(column)
        series = self._df[col]
        mask = COMPARATORS[comparator](series, value)
        mask = mask.fillna(False).astype(bool)
        if keep_null and comparator not in ("is_null", "not_null"):
            mask = mask | series.isna()

        rows_in = len(self._df)
        out = self._df[mask]
        null_rows = int(series.isna().sum())
        detail = {"column": col, "comparator": comparator, "value": _jsonable(value),
                  "null_rows_in_column": null_rows, "nulls_kept": bool(keep_null)}
        self.provenance.filters.append(detail)
        self.provenance.use_columns(self._origin, [col])
        self.provenance.record("filter", detail, rows_in, len(out))
        return self._next(out)

    def filter_rows(self, mask: pd.Series, description: str) -> "Query":
        """WHERE with a caller-built mask; the description becomes the audit note."""
        mask = mask.reindex(self._df.index).astype("boolean").fillna(False).astype(bool)
        rows_in = len(self._df)
        out = self._df[mask]
        detail = {"description": description}
        self.provenance.filters.append(detail)
        self.provenance.record("filter_rows", detail, rows_in, len(out))
        return self._next(out)

    def distinct(self, columns: Optional[Sequence[str]] = None) -> "Query":
        cols = [self.resolve(c) for c in columns] if columns else self.columns
        rows_in = len(self._df)
        out = self._df.drop_duplicates(subset=cols)
        self.provenance.record("distinct", {"columns": cols}, rows_in, len(out))
        return self._next(out)

    def sort(self, columns: Union[str, Sequence[str]], ascending: bool = True,
             na_position: str = "last") -> "Query":
        cols = [self.resolve(c) for c in ([columns] if isinstance(columns, str) else columns)]
        out = self._df.sort_values(cols, ascending=ascending, na_position=na_position,
                                   kind="mergesort")  # stable, so ties keep source order
        self.provenance.record(
            "sort", {"columns": cols, "ascending": ascending, "na_position": na_position},
            len(self._df), len(out))
        return self._next(out)

    def limit(self, n: int) -> "Query":
        if n < 0:
            raise QueryError("limit must not be negative")
        rows_in = len(self._df)
        out = self._df.head(n)
        self.provenance.record("limit", {"n": n}, rows_in, len(out))
        return self._next(out)

    # -- column operations -------------------------------------------------

    def select(self, columns: Sequence[str], keep_internal: bool = True) -> "Query":
        cols = [self.resolve(c) for c in columns]
        keep = ([c for c in INTERNAL_COLUMNS if c in self._df.columns] if keep_internal else [])
        out = self._df[keep + [c for c in cols if c not in keep]]
        self.provenance.use_columns(self._origin, cols)
        self.provenance.record("select", {"columns": cols}, len(self._df), len(out))
        return self._next(out)

    def drop_columns(self, columns: Sequence[str], reason: str) -> "Query":
        """Remove columns from the result, recording why. Used to withhold personal data."""
        present = [c for c in columns if c in self._df.columns]
        out = self._df.drop(columns=present)
        detail = {"columns_dropped": present, "reason": reason}
        self.provenance.record("drop_columns", detail, len(self._df), len(out))
        return self._next(out)

    def derive(self, name: str, values: pd.Series, source: str) -> "Query":
        """Add a computed column, recording how it was computed."""
        if name in self._df.columns:
            raise QueryError("column {!r} already exists".format(name))
        out = self._df.copy()
        out[name] = values.reindex(out.index)
        detail = {"column": name, "source": source,
                  "null_count": int(out[name].isna().sum())}
        self.provenance.record("derive", detail, len(self._df), len(out))
        return self._next(out)

    def date_diff(self, name: str, end: str, start: str, unit: str = "days") -> "Query":
        """Whole-unit difference between two date columns. Negatives are kept."""
        if unit != "days":
            raise QueryError("only 'days' is supported; got {!r}".format(unit))
        end_col, start_col = self.resolve(end), self.resolve(start)
        delta = (self._df[end_col] - self._df[start_col]).dt.days
        negatives = int((delta < 0).sum())
        out = self.derive(name, delta, "{} - {} in {}".format(end_col, start_col, unit))
        if negatives:
            out.provenance.notes.append(
                "{}: {} rows have a negative duration; they are reported, not clipped".format(
                    name, negatives))
        return out

    def date_bucket(self, name: str, column: str, grain: str = "month") -> "Query":
        """Sortable period key derived from a parsed date."""
        col = self.resolve(column)
        series = self._df[col]
        if grain == "year":
            keys = series.dt.year.astype("Int64").astype("string")
        elif grain == "quarter":
            keys = (series.dt.year.astype("Int64").astype("string") + "-Q"
                    + series.dt.quarter.astype("Int64").astype("string"))
        elif grain == "month":
            keys = series.dt.strftime("%Y-%m")
        else:
            raise QueryError("unknown grain {!r}; use year, quarter or month".format(grain))
        keys = keys.where(series.notna())
        return self.derive(name, keys, "{} bucketed by {}".format(col, grain))

    def case_when(self, name: str, conditions: Sequence, default: Any = None) -> "Query":
        """CASE WHEN: an ordered list of (mask, value) pairs, first match wins."""
        result = pd.Series(default, index=self._df.index, dtype="object")
        assigned = pd.Series(False, index=self._df.index)
        described = []
        for mask, value in conditions:
            mask = mask.reindex(self._df.index).fillna(False).astype(bool)
            apply_to = mask & ~assigned
            result.loc[apply_to] = value
            assigned |= apply_to
            described.append({"value": _jsonable(value), "rows": int(apply_to.sum())})
        return self.derive(name, result,
                           "case_when with {} branches: {}".format(len(conditions), described))

    # -- joins -------------------------------------------------------------

    def join(self, relationship: str, how: str = "inner",
             columns: Optional[Sequence[str]] = None) -> "Query":
        """Join using a documented relationship, recording coverage and fan-out."""
        if relationship not in RELATIONSHIPS:
            raise QueryError("unknown relationship {!r}; known: {}".format(
                relationship, sorted(RELATIONSHIPS)))
        if how not in ("inner", "left", "anti"):
            raise QueryError("join type must be inner, left or anti; got {!r}".format(how))

        near_table, near_key, far_table_key, far_key = RELATIONSHIPS[relationship]

        # A relationship is undirected. If the query already starts from the table
        # the relationship names on the right, traverse it the other way: asking for
        # the largest claims must keep the 134 claims as the population, not expand
        # into the 25,000 policies.
        reversed_direction = self._origin == far_table_key and self._origin != near_table
        if reversed_direction:
            near_table, near_key, far_table_key, far_key = (
                far_table_key, far_key, near_table, near_key)

        left_key, right_table_key, right_key = near_key, far_table_key, far_key
        try:
            left_col = self.resolve(left_key)
        except QueryError:
            raise QueryError(
                "relationship {!r} needs column {!r}, which is not in the current "
                "result".format(relationship, left_key))

        right_table: Table = self._ds.table(right_table_key)
        right_df = right_table.frame
        if columns is not None:
            keep = [right_table.resolve(c) for c in columns]
            if right_key not in keep:
                keep = [right_key] + keep
            right_df = right_df[[ROW_ID] + keep]
        right_df = right_df.rename(columns={
            c: "{}_{}".format(right_table_key, c)
            for c in right_df.columns if c != right_key
        })
        right_on = right_key

        rows_in = len(self._df)
        left_keys = set(self._df[left_col].dropna())
        right_keys = set(right_df[right_on].dropna())
        matched = left_keys & right_keys
        fan_out = int(right_df[right_on].value_counts().max()) if len(right_df) else 0

        if how == "anti":
            out = self._df[~self._df[left_col].isin(right_keys)]
        else:
            out = self._df.merge(right_df, left_on=left_col, right_on=right_on,
                                 how="left" if how == "left" else "inner",
                                 suffixes=("", "_" + right_table_key))

        detail = {
            "relationship": relationship, "how": how,
            "direction": "reversed" if reversed_direction else "declared",
            "left_column": left_col, "right_table": right_table_key, "right_column": right_key,
            "left_distinct_keys": len(left_keys), "right_distinct_keys": len(right_keys),
            "matched_keys": len(matched),
            "left_keys_unmatched": len(left_keys - right_keys),
            "max_right_rows_per_key": fan_out,
        }
        if how == "inner" and len(left_keys - right_keys):
            detail["note"] = (
                "inner join dropped {} unmatched left keys; use how='left' to keep them".format(
                    len(left_keys - right_keys)))
        if fan_out > 1 and how != "anti":
            detail["fan_out_warning"] = (
                "the right side has up to {} rows per key, so left rows can multiply".format(
                    fan_out))

        self.provenance.use_table(right_table_key)
        self.provenance.join_path.append(
            "{}.{} = {}.{} ({})".format(self._origin, left_col, right_table_key, right_key, how))
        self.provenance.record("join", detail, rows_in, len(out))
        return self._next(out)

    # -- aggregation -------------------------------------------------------

    def aggregate(self, aggregations: Dict[str, tuple]) -> Dict[str, Any]:
        """Aggregate the whole result into a single row of named values."""
        result: Dict[str, Any] = {}
        for alias, (column, func) in aggregations.items():
            result[alias] = self._apply_agg(self._df, column, func)
        self.provenance.record(
            "aggregate",
            {"aggregations": {k: {"column": v[0], "func": v[1]}
                              for k, v in aggregations.items()}},
            len(self._df), 1)
        return result

    def group_by(self, by: Union[str, Sequence[str]],
                 aggregations: Dict[str, tuple],
                 dropna: bool = False) -> "GroupedResult":
        """GROUP BY. dropna=False keeps a null group so rows are never lost silently."""
        keys = [self.resolve(c) for c in ([by] if isinstance(by, str) else by)]
        rows_in = len(self._df)

        grouped = self._df.groupby(keys, dropna=dropna, observed=True, sort=False)
        result = self._aggregate_groups(grouped, keys, aggregations)
        covered = int(result["group_rows"].sum()) if len(result) else 0
        detail = {
            "keys": keys, "groups": len(result), "dropna": dropna,
            "rows_covered": covered, "rows_in": rows_in,
            "aggregations": {k: {"column": v[0], "func": v[1]}
                             for k, v in aggregations.items()},
        }
        if covered != rows_in:
            detail["rows_lost"] = rows_in - covered
        self.provenance.use_columns(self._origin, keys)
        self.provenance.record("group_by", detail, rows_in, len(result))
        return GroupedResult(result, keys, self.provenance, rows_in)

    def _aggregate_groups(self, grouped, keys: List[str],
                          aggregations: Dict[str, tuple]) -> pd.DataFrame:
        """Apply every aggregation across all groups at once.

        The obvious implementation iterates the groups and calls the aggregator on
        each chunk, which reads better but costs one pandas call per group per
        measure. On a 25,000-customer grouping with six measures that is 150,000
        calls, and it dominated the whole catalogue's runtime.

        pandas can do the same work in one pass per measure. The aggregator table
        stays the source of truth for *semantics* - the post-processing below
        reproduces each aggregator's contract exactly, and the regression baseline
        holds this to it.
        """
        frames = [grouped.size().rename("group_rows")]
        for alias, (column, func) in aggregations.items():
            if func not in AGGREGATORS:
                raise QueryError("unknown aggregate {!r}; known: {}".format(
                    func, sorted(AGGREGATORS)))
            if func == "count_all" or column is None:
                frames.append(grouped.size().rename(alias))
                continue
            col = self.resolve(column)
            series = grouped[col]
            if func == "count":
                # AGGREGATORS["count"]: non-null values only.
                frames.append(series.count().rename(alias))
            elif func == "distinct":
                frames.append(series.nunique(dropna=True).rename(alias))
            elif func == "sum":
                # AGGREGATORS["sum"]: 0.0 when every value is null, which is what
                # pandas returns with the default min_count=0.
                frames.append(series.sum().astype(float).rename(alias))
            elif func in ("mean", "median"):
                # None when every value is null; NaN becomes None in _jsonable.
                frames.append(getattr(series, func)().rename(alias))
            elif func == "std":
                # AGGREGATORS["std"]: None unless at least two non-null values.
                # pandas uses ddof=1 and returns NaN for a single observation,
                # which is the same rule.
                frames.append(series.std().rename(alias))
            elif func in ("min", "max"):
                frames.append(getattr(series, func)().rename(alias))
            else:  # pragma: no cover - guarded by the membership check above
                raise QueryError("aggregate {!r} has no grouped implementation".format(
                    func))

        combined = pd.concat(frames, axis=1)
        if combined.empty:
            return pd.DataFrame(columns=keys + ["group_rows"] + list(aggregations))

        combined = combined.reset_index()
        # A single key groups to a named index; more than one to a MultiIndex.
        # reset_index restores the key columns either way, but the names can come
        # back as level_0 when the column had no name.
        if list(combined.columns[:len(keys)]) != keys:
            combined.columns = keys + list(combined.columns[len(keys):])

        combined["group_rows"] = combined["group_rows"].astype(int)
        for alias, (column, func) in aggregations.items():
            if func == "count_all" or column is None or func in ("count", "distinct"):
                combined[alias] = combined[alias].astype(int)
        return combined

    def _apply_agg(self, frame: pd.DataFrame, column: Optional[str], func: str) -> Any:
        if func not in AGGREGATORS:
            raise QueryError("unknown aggregate {!r}; known: {}".format(
                func, sorted(AGGREGATORS)))
        if func == "count_all" or column is None:
            return int(len(frame))
        col = self.resolve(column)
        return _jsonable(AGGREGATORS[func](frame[col]))

    # -- output ------------------------------------------------------------

    def to_records(self, columns: Optional[Sequence[str]] = None,
                   limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Rows as plain dicts, each carrying its source row id for traceability."""
        df = self._df if columns is None else self._df[
            [c for c in INTERNAL_COLUMNS if c in self._df.columns]
            + [self.resolve(c) for c in columns]]
        if limit is not None:
            df = df.head(limit)
        return [{k: _jsonable(v) for k, v in row.items()} for row in df.to_dict("records")]


class GroupedResult:
    """The output of a group_by: a tidy frame plus ranking helpers."""

    def __init__(self, frame: pd.DataFrame, keys: List[str],
                 provenance: Provenance, rows_in: int):
        self._df = frame
        self.keys = keys
        self.provenance = provenance
        self.rows_in = rows_in

    @property
    def frame(self) -> pd.DataFrame:
        return self._df

    def __len__(self) -> int:
        return len(self._df)

    def having(self, column: str, comparator: str, value: Any) -> "GroupedResult":
        """HAVING: filter groups after aggregation."""
        if comparator not in COMPARATORS:
            raise QueryError("unknown comparator {!r}".format(comparator))
        rows_in = len(self._df)
        mask = COMPARATORS[comparator](self._df[column], value).fillna(False).astype(bool)
        out = self._df[mask]
        self.provenance.record(
            "having", {"column": column, "comparator": comparator, "value": _jsonable(value)},
            rows_in, len(out))
        return GroupedResult(out, self.keys, self.provenance, self.rows_in)

    def min_group_size(self, minimum: int, size_column: str = "group_rows") -> "GroupedResult":
        """Small-sample guard: drop groups too small to carry a rate, and say how many."""
        rows_in = len(self._df)
        keep = self._df[size_column] >= minimum
        excluded = self._df[~keep]
        out = self._df[keep]
        detail = {
            "minimum": minimum, "groups_excluded": int(len(excluded)),
            "rows_excluded": int(excluded[size_column].sum()) if len(excluded) else 0,
        }
        self.provenance.record("small_sample_guard", detail, rows_in, len(out))
        if len(excluded):
            self.provenance.notes.append(
                "{} groups below the {}-row threshold were excluded from ranking".format(
                    len(excluded), minimum))
        return GroupedResult(out, self.keys, self.provenance, self.rows_in)

    def rank(self, column: str, ascending: bool = False,
             rank_column: str = "rank") -> "GroupedResult":
        rows_in = len(self._df)
        out = self._df.sort_values(column, ascending=ascending, na_position="last",
                                   kind="mergesort").copy()
        out[rank_column] = range(1, len(out) + 1)
        self.provenance.record(
            "rank_entities", {"column": column, "ascending": ascending}, rows_in, len(out))
        return GroupedResult(out, self.keys, self.provenance, self.rows_in)

    def share_of_total(self, column: str, share_column: Optional[str] = None) -> "GroupedResult":
        """Each group's percentage share of the column total."""
        share_column = share_column or (column + "_share_pct")
        out = self._df.copy()
        total = out[column].sum()
        out[share_column] = (100.0 * out[column] / total).round(6) if total else np.nan
        self.provenance.record(
            "share_of_total", {"column": column, "total": _jsonable(total)},
            len(out), len(out))
        return GroupedResult(out, self.keys, self.provenance, self.rows_in)

    def top(self, n: int) -> "GroupedResult":
        rows_in = len(self._df)
        out = self._df.head(n)
        self.provenance.record("limit", {"n": n}, rows_in, len(out))
        return GroupedResult(out, self.keys, self.provenance, self.rows_in)

    def to_records(self) -> List[Dict[str, Any]]:
        return [{k: _jsonable(v) for k, v in row.items()}
                for row in self._df.to_dict("records")]


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and np.isnan(value):
        return None
    if value is pd.NaT:
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def query(dataset: Dataset, table_key: str) -> Query:
    """Entry point: start a query against one table."""
    return Query.from_table(dataset, table_key)
