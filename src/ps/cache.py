"""A result cache for deterministic executions.

Specification section 10: "cache safe deterministic results where appropriate".
The qualifier is the whole design. A cache that can return a figure the current
data no longer supports is worse than no cache at all, because the answer still
arrives with full provenance and reads as freshly computed.

Three properties make caching safe here:

  determinism   No analytical code reads the wall clock, a random source, or any
                state outside the loaded frames. The same PS with the same
                parameters over the same data can only produce the same evidence.
  fingerprint   The key includes a content fingerprint of the loaded data, plus
                the catalogue, contract and executor versions. Change any of
                them and every prior entry becomes unreachable rather than
                stale.
  isolation     Entries are deep-copied in and out. A caller that annotates an
                evidence package - and the summarizer does - cannot reach back
                into the cache and change what the next caller receives.

A hit is disclosed, not hidden: `provenance["cache"]` records that the figures
were computed earlier and when. The evidence's own `generated_at_utc` keeps the
time the computation actually happened, because that is what it means.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

CACHE_VERSION = "1.0.0"

# Entries are small (an evidence package, mostly tables capped at a few hundred
# rows) and the catalogue has 103 problem statements. This holds the whole
# catalogue several parameterisations deep without meaningful memory cost.
DEFAULT_MAX_ENTRIES = 512


def fingerprint_dataset(dataset) -> str:
    """A content fingerprint of the loaded data.

    Hashes what was actually computed on - the typed frames in memory - rather
    than the files on disk, so a differently-loaded dataset (the DQ-001 repair
    enabled, say) fingerprints differently even though the CSVs are identical.
    """
    import pandas as pd

    parts = ["contract={}".format(dataset.contract.version)]
    for key in sorted(dataset.table_keys):
        frame = dataset.frame(key)
        digest: str
        try:
            hashed = pd.util.hash_pandas_object(frame, index=False)
            digest = hashlib.sha256(hashed.values.tobytes()).hexdigest()[:16]
        except TypeError:
            # An unhashable cell type. Fall back to shape and dtypes, and say so
            # in the fingerprint rather than pretending to a content hash.
            digest = "shape:{}x{}:{}".format(
                len(frame), len(frame.columns),
                hashlib.sha256(
                    "|".join("{}={}".format(c, frame[c].dtype)
                             for c in frame.columns).encode()).hexdigest()[:16])
        parts.append("{}={}:{}".format(key, len(frame), digest))
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()[:24]


def _canonical(parameters: Optional[Dict[str, Any]]) -> str:
    """A stable text form of the parameters.

    Sorted keys, so argument order cannot produce two entries for one call.
    `default=str` keeps a date or a numpy scalar from breaking serialisation;
    two values that stringify alike are the same value for our purposes.
    """
    if not parameters:
        return "{}"
    return json.dumps(parameters, sort_keys=True, default=str)


class ResultCache:
    """An LRU cache of evidence packages, keyed on everything that can change one."""

    version = CACHE_VERSION

    def __init__(self, fingerprint: str, max_entries: int = DEFAULT_MAX_ENTRIES,
                 enabled: bool = True):
        self.fingerprint = fingerprint
        self.max_entries = max_entries
        self.enabled = enabled
        self._entries: "OrderedDict[Tuple[str, ...], Any]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.stores = 0

    @classmethod
    def for_dataset(cls, dataset, registry=None, executor_version: str = "",
                    max_entries: int = DEFAULT_MAX_ENTRIES,
                    enabled: bool = True) -> "ResultCache":
        """Build a cache bound to one dataset, catalogue and executor version."""
        parts = [fingerprint_dataset(dataset), CACHE_VERSION, executor_version]
        if registry is not None:
            parts.append(str(registry.version))
            parts.append(str(len(registry)))
        return cls(fingerprint=":".join(parts), max_entries=max_entries,
                   enabled=enabled)

    # -- keys --------------------------------------------------------------

    def key(self, problem_id: str, parameters: Optional[Dict[str, Any]] = None
            ) -> Tuple[str, ...]:
        """The cache key.

        The question is deliberately absent. It is display metadata - no
        analytical code reads it - so two phrasings of the same analysis are
        the same computation and should share one entry. The question is
        restored on the way out.
        """
        return (self.fingerprint, problem_id, _canonical(parameters))

    # -- access ------------------------------------------------------------

    def get(self, problem_id: str, parameters: Optional[Dict[str, Any]] = None,
            question: str = ""):
        if not self.enabled:
            return None
        key = self.key(problem_id, parameters)
        if key not in self._entries:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        evidence = copy.deepcopy(self._entries[key])
        if question:
            evidence.question = question
        evidence.provenance["cache"] = {
            "hit": True,
            "computed_at_utc": evidence.generated_at_utc,
            "fingerprint": self.fingerprint.split(":")[0],
        }
        return evidence

    def put(self, problem_id: str, parameters: Optional[Dict[str, Any]],
            evidence) -> None:
        if not self.enabled or evidence is None:
            return
        key = self.key(problem_id, parameters)
        stored = copy.deepcopy(evidence)
        # Never store a hit marker: the stored copy is the original computation.
        stored.provenance.pop("cache", None)
        self._entries[key] = stored
        self._entries.move_to_end(key)
        self.stores += 1
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self.evictions += 1

    def clear(self) -> None:
        self._entries.clear()

    # -- reporting ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return (self.hits / self.lookups) if self.lookups else 0.0

    def stats(self) -> Dict[str, Any]:
        return {
            "cache_version": CACHE_VERSION,
            "enabled": self.enabled,
            "entries": len(self._entries),
            "max_entries": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 4),
            "fingerprint": self.fingerprint.split(":")[0],
        }
