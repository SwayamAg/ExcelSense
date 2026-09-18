"""PS catalogue definitions, split by domain group."""
from __future__ import annotations

from . import (
    part1_portfolio,
    part2_lifecycle,
    part3_claims,
    part4_distribution,
    part5_customer,
    part6_executive,
)

MODULES = (
    part1_portfolio,
    part2_lifecycle,
    part3_claims,
    part4_distribution,
    part5_customer,
    part6_executive,
)


def all_definitions():
    out = []
    for mod in MODULES:
        out.extend(mod.defs())
    return out
