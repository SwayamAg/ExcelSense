"""Deterministic answer composition.

Writes the executive answer from the evidence package with no model involved. This
is the fallback when the LLM is unavailable, refuses, or returns something the
validator rejects.

It can be this capable because the hard part was already done: every figure was
computed, checked and labelled before any model was called. Composition only has to
choose which facts to lead with and put them in a sentence, and it follows the same
answer contract the specification sets out — direct answer, then the reason or
finding, then the supporting number.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from src.evidence.tables import humanize, infer_unit
from src.ps.evidence import EvidencePackage


def _format(value: Any, unit: str = "number") -> str:
    if value is None:
        return "not available"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if unit == "rate":
            return "{:,.2f}%".format(value)
        if unit == "money":
            return "{:,.0f}".format(value)
        if unit == "count":
            # Round rather than truncate: int() on 322518.71 prints 322,518, which is
            # not a figure the evidence contains.
            return "{:,.0f}".format(value)
        if unit == "days":
            return "{:,.0f} days".format(value)
        if float(value).is_integer():
            return "{:,}".format(int(value))
        return "{:,.2f}".format(value)
    return str(value)


def _format_key(key: str, value: Any) -> str:
    return _format(value, infer_unit(key, [value]))


# Terms a business reader expects to see in capitals. Lowercasing a column label
# to make it read as prose turns "APE" into "ape", which looks like a typo in a
# sentence a CEO is reading.
ACRONYMS = ("APE", "HHI", "ULIP", "ID", "PIVC", "RCU", "NPS", "GWP")


def _prose_case(text: str) -> str:
    """Lowercase a label for use mid-sentence, leaving acronyms alone."""
    words = []
    for word in text.split():
        stripped = word.strip(".,;:()").upper()
        words.append(word if stripped in ACRONYMS else word.lower())
    return " ".join(words)


def _readable_attribute(raw: str) -> str:
    """Turn a joined column name into the noun a reader expects."""
    for prefix in ("owner_", "sales_", "policy_", "persistency_", "claims_"):
        raw = raw.replace(prefix, "")
    return _prose_case(humanize(raw))


def _join(items) -> str:
    """Join a list the way it would be written, not the way it is iterated."""
    items = [i for i in items if i]
    if len(items) <= 1:
        return items[0] if items else ""
    if len(items) == 2:
        return "{} and {}".format(*items)
    return "{} and {}".format(", ".join(items[:-1]), items[-1])


def _count_noun(key: str) -> str:
    """The thing a count column counts.

    "claim_count" counts claims. Rendering it as "134 claim count" states the
    column name where the sentence needs a noun.
    """
    label = humanize(key)
    lowered = label.lower()
    for suffix in (" count", " total", " number"):
        if lowered.endswith(suffix):
            stem = label[: -len(suffix)].strip()
            if stem and not stem.lower().endswith("s"):
                # policy -> policies, not policys
                if re.search(r"[^aeiou]y$", stem, re.I):
                    stem = stem[:-1] + "ies"
                elif re.search(r"(ch|sh|s|x|z)$", stem, re.I):
                    stem += "es"
                else:
                    stem += "s"
            return _prose_case(stem)
    return _prose_case(label)


def _label_of(row: Dict[str, Any]) -> str:
    """The name of a row.

    A grouped row is labelled by its own value ("P91", "Lapsed") and must be left
    alone; a cohort or driver row is labelled by a column name, which is machinery
    and has to be turned back into prose.
    """
    if "attribute" in row and isinstance(row["attribute"], str):
        return _readable_attribute(row["attribute"])
    for key, value in row.items():
        if key.startswith("__") or key in ("rank", "group_rows"):
            continue
        if isinstance(value, str):
            return value
    for key, value in row.items():
        if not key.startswith("__") and value is not None:
            return str(value)
    return "the leading group"


def _headline_measure(evidence: EvidencePackage) -> Optional[str]:
    if evidence.ranking and evidence.ranking.get("measure"):
        return evidence.ranking["measure"]
    if not evidence.result_table:
        return None
    for key in evidence.result_table[0]:
        if key.endswith(("_rate", "_pct")):
            return key
    for key, value in evidence.result_table[0].items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) \
                and key not in ("rank", "group_rows"):
            return key
    return None


# ---------------------------------------------------------------------------
# Executive vocabulary
# ---------------------------------------------------------------------------

# What an identifier column names. "960773 leads" asks the reader to know that a
# six-digit number is an agent; "Agent 960773" does not.
ENTITY_NOUNS = {"AppID": "Policy", "SALES_ID": "Agent", "Branch_No": "Branch",
                "owner_ClientID": "Customer", "ClientID": "Customer"}

# Singular and plural for the thing a grouping column counts.
GROUP_NOUNS = {
    "Prod_No": ("product", "products"), "SALES_ID": ("agent", "agents"),
    "Branch_No": ("branch", "branches"), "AppID": ("policy", "policies"),
    "owner_ClientID": ("customer", "customers"), "ClientID": ("customer", "customers"),
    "owner_Owner.Comm.State": ("state", "states"),
    "owner_Owner.Comm.City.Name": ("city", "cities"),
    "owner_Owner.Comm.Country": ("country", "countries"),
}

# How a measure is spoken. Anything absent falls back to its humanized name.
MEASURE_PHRASES = {
    "policies": "policies", "policy_count": "policies", "claims": "claims",
    "renewals": "renewals", "pending": "pending renewals", "breaches": "breaches",
    "total_ape": "total APE", "avg_ape": "average APE", "median_ape": "median APE",
    "lapse_rate": "lapse rate", "renewal_paid_rate": "renewal payment rate",
    "first_year_persistency": "first-year persistency",
    "settlement_rate": "claim settlement rate", "share_pct": "share",
    "composite_score": "composite score", "inforce_rate": "in-force rate",
    "surrender_rate": "surrender rate", "total_sum_assured": "total sum assured",
    "median_days": "median time", "entity_share_pct": "share of customers",
    "premium_at_stake": "premium at stake", "exposure_ape": "APE exposed",
}

# Measures that are volumes: "the most policies", not "the highest policy count".
VOLUME_MEASURES = {"policies", "claims", "renewals", "pending", "breaches", "entities"}

# The two sides of a cohort comparison, as a reader names them.
COHORT_NAMES = {
    "lapsed": "lapsed policies", "inforce": "in-force policies",
    "surrendered": "surrendered policies", "active": "active agents",
    "inactive": "inactive agents", "named_agent": "this agent", "peers": "peers",
    "slower": "slower-settled claims", "faster": "faster-settled claims",
    "month_37": "37th-month renewals", "earlier_buckets": "earlier renewals",
}

# Below this many records a leading *rate* is flagged in the answer. A branch
# "worst" on 14 policies is not a finding to act on without knowing that.
SMALL_BASE = 30

_PLAN_PREFIXES = ("Canara HSBC Life Insurance ", "Canara HSBC ")
_UNIT_WORDS = {"y": ("year", "years"), "d": ("day", "days"), "m": ("month", "months")}


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _singular(noun: str) -> str:
    if noun.endswith("ies"):
        return noun[:-3] + "y"
    if noun.endswith(("ches", "shes", "sses", "xes")):
        return noun[:-2]
    return noun[:-1] if noun.endswith("s") else noun


def _measure_phrase(key: str) -> str:
    if key in MEASURE_PHRASES:
        return MEASURE_PHRASES[key]
    phrase = _prose_case(humanize(key))
    return re.sub(r"^avg\b", "average", phrase)


def _unit_of(key: str, value: Any) -> str:
    """The unit a figure is spoken in.

    Persistency is supplied as a percentage without "rate" in its name, so the
    column-name inference alone would print 99.30 rather than 99.30%.
    """
    unit = infer_unit(key, [value])
    if unit != "rate" and isinstance(value, (int, float)) and 0 <= value <= 100 \
            and re.search(r"persistency|rate$|_pct$", key, re.IGNORECASE):
        return "rate"
    return unit


def _humanize_label(label: Any) -> Any:
    """Speak a derived bucket label: "11_15y" is "11–15 years", "tier_1_highest"
    is "Tier 1 (highest)". Anything not in bucket form is returned untouched."""
    if not isinstance(label, str) or \
            not re.fullmatch(r"[A-Za-z0-9.]+(?:_[A-Za-z0-9.]+)+", label):
        return label
    parts = label.split("_")

    def with_unit(token: str) -> Tuple[str, str]:
        match = re.fullmatch(r"(\d+(?:\.\d+)?)([ydm])", token)
        if not match:
            return token, ""
        singular, plural = _UNIT_WORDS[match.group(2)]
        return match.group(1), singular if match.group(1) == "1" else plural

    def is_number(token: str) -> bool:
        return bool(re.fullmatch(r"\d+(?:\.\d+)?[kKL]?", token) or with_unit(token)[1])

    head = parts[0].lower()
    if len(parts) == 2 and head in ("upto", "over", "within", "under", "above", "below"):
        value, unit = with_unit(parts[1])
        return "{} {}{}".format("up to" if head == "upto" else head, value,
                                " " + unit if unit else "")
    if len(parts) == 2 and is_number(parts[0]) and is_number(parts[1]):
        value, unit = with_unit(parts[1])
        return "{}-{}{}".format(parts[0], value, " " + unit if unit else "")
    if head == "tier" and len(parts) >= 2:
        rest = " ".join(parts[2:])
        return "Tier {}{}".format(parts[1], " ({})".format(rest) if rest else "")
    if re.fullmatch(r"q\d", head) and len(parts) > 1:
        return "{} ({})".format(parts[0].upper(), " ".join(parts[1:]))
    return _cap(" ".join(p.upper() if p.upper() in ACRONYMS else p for p in parts))


def _short_plan_name(name: str) -> str:
    """The product name an executive uses, without the company and variant."""
    for prefix in _PLAN_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.split(" - ")[0].strip()


def _entity_label(row: Dict[str, Any]) -> str:
    """Name a row the way a business reader would.

    Identifier columns are checked in the row's own order, so a policy row that
    also carries its agent is named as the policy.
    """
    if row.get("Prod_No") and row.get("Plan.Name"):
        return "{} ({})".format(row["Prod_No"], _short_plan_name(str(row["Plan.Name"])))
    if row.get("Prod_No"):
        return str(row["Prod_No"])
    for key in row:
        if key in ENTITY_NOUNS and row.get(key) is not None:
            return "{} {}".format(ENTITY_NOUNS[key], row[key])
    city = row.get("owner_Owner.Comm.City.Name")
    state = row.get("owner_Owner.Comm.State")
    if city:
        return "{} ({})".format(str(city).title(), state) if state else str(city).title()
    country = row.get("owner_Owner.Comm.Country")
    if country:
        return str(country)
    return str(_humanize_label(_label_of(row)))


def _group_noun(rows: List[Dict[str, Any]]) -> Optional[Tuple[str, str]]:
    if not rows:
        return None
    for key in rows[0]:
        if key in GROUP_NOUNS:
            return GROUP_NOUNS[key]
    return None


def _base_count(row: Dict[str, Any]) -> Optional[Tuple[int, str]]:
    for key in ("policies", "claims", "renewals"):
        if isinstance(row.get(key), (int, float)):
            return int(row[key]), key
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _insufficient(evidence: EvidencePackage) -> str:
    reason = evidence.insufficient.reason if evidence.insufficient else "unknown"
    detail = evidence.insufficient.detail if evidence.insufficient else {}
    sentence = "This cannot be answered from the data: {}.".format(reason.rstrip("."))
    identifier = detail.get("identifier")
    if identifier:
        sentence += " No record carries the identifier {}.".format(identifier)
    required = detail.get("required_parameter")
    if required:
        sentence += " The analysis needs {} to be supplied.".format(
            humanize(str(required)).lower())
    return sentence


def _scalar_answer(evidence: EvidencePackage) -> str:
    """One headline figure, supported by the figure that gives it scale."""
    figures = [(k, v) for k, v in evidence.aggregates.items() if _is_number(v)]
    if not figures:
        return "The analysis completed but produced no figure to report."

    units = {k: _unit_of(k, v) for k, v in figures}
    counts = [(k, v) for k, v in figures if units[k] == "count"]

    # Several counts and nothing else: a statement of scale, listed.
    if len(counts) == len(figures) and len(figures) > 1:
        return _cap(_join(["{} {}".format(_format(v, "count"), _count_noun(k))
                           for k, v in figures])) + "."

    asks_how_many = re.match(r"\s*how many", evidence.problem_statement or "", re.I)
    rates = [(k, v) for k, v in figures if units[k] == "rate"]
    averages = [(k, v) for k, v in figures if k.startswith(("avg_", "mean_"))]
    ordered = (counts + rates + averages) if asks_how_many else (rates + averages + figures)
    lead_key, lead_value = ordered[0]
    lead_unit = units[lead_key]

    if lead_unit == "count":
        sentence = "There are {} {}".format(_format(lead_value, "count"),
                                           _count_noun(lead_key))
        total = next(((k, v) for k, v in figures
                      if k.startswith("total_") and units[k] != "count"), None)
        if total:
            sentence += ", with {} of {}".format(_measure_phrase(total[0]),
                                                 _format(total[1], units[total[0]]))
        return sentence + "."

    sentence = "{} is {}".format(_cap(_measure_phrase(lead_key)),
                                 _format(lead_value, lead_unit))

    # A rate over a part and a whole: "99.25%, or 133 of 134 claims".
    others = [(k, v) for k, v in counts if k != lead_key]
    if lead_unit == "rate" and len(others) >= 2:
        (_, part), (whole_key, whole) = sorted(others[:2], key=lambda kv: kv[1])
        if 0 < part <= whole:
            return "{}, or {} of {} {}.".format(
                sentence, _format(part, "count"), _format(whole, "count"),
                _count_noun(whole_key))

    extras = []
    median = next(((k, v) for k, v in figures if k.startswith("median_")), None)
    if median and lead_key.startswith(("avg_", "mean_")):
        extras.append("median {}".format(_format(median[1], units[median[0]])))
    if others:
        key, value = others[0]
        extras.append("across {} {}".format(_format(value, "count"), _count_noun(key)))
    if extras:
        sentence += " ({})".format("; ".join(extras))
    return sentence + "."


def _ranked_answer(evidence: EvidencePackage) -> str:
    """§8 rankings: the leader and its figure, then the comparison that gives it meaning."""
    rows = evidence.result_table
    if "outliers_flagged" in evidence.aggregates:
        return _outlier_answer(evidence)
    if rows and any(k.endswith("_ape_share_pct") for k in rows[0]) \
            and "entity_share_pct" in rows[0]:
        return _segment_answer(evidence)

    measure = _headline_measure(evidence)
    valued = [r for r in rows if measure and _is_number(r.get(measure))]
    if not valued:
        return "{} ranks first of the {} analysed.".format(
            _cap(_entity_label(rows[0])), len(rows)) if rows else _scalar_answer(evidence)

    ranking = evidence.ranking or {}
    ascending = bool(ranking.get("ascending"))
    declared = ranking.get("measure") == measure
    # "Highest" is only said of the row that is highest. Without a declared
    # ranking on this measure the rows are in the analysis's own order - band
    # order, say - so the leader is selected rather than assumed to be first.
    if declared:
        top = valued[0]
    else:
        top = (min if ascending else max)(valued, key=lambda r: r[measure])
    unit = _unit_of(measure, top[measure])
    label = _cap(_entity_label(top))
    group = _group_noun(rows)

    if len({r[measure] for r in valued}) == 1 and len(valued) > 1:
        noun = _measure_phrase(measure)
        value = top[measure]
        if measure in VOLUME_MEASURES and value == 1:
            noun = _singular(noun)
        return "All {} {} have {} {} each.".format(
            len(valued), group[1] if group else "groups", _format(value, unit), noun)

    if measure in VOLUME_MEASURES or measure.endswith("_count"):
        sentence = "{} has the {} {}: {}".format(
            label, "fewest" if ascending else "most", _measure_phrase(measure),
            _format(top[measure], "count"))
    elif re.search(r"at_stake|at_risk|exposure", measure):
        sentence = "{} has the {} {}: {}".format(
            label, "least" if ascending else "most", _measure_phrase(measure),
            _format(top[measure], "money"))
    elif measure == "share_pct":
        sentence = "{} is the largest group, at {} of the total".format(
            label, _format(top[measure], "rate"))
    else:
        sentence = "{} has the {} {} at {}".format(
            label, "lowest" if ascending else "highest", _measure_phrase(measure),
            _format(top[measure], unit))

    baseline = evidence.comparison_baseline
    delta = top.get("delta_vs_baseline")
    has_baseline = bool(declared and baseline and baseline.get("value") is not None
                        and _is_number(delta))
    if has_baseline:
        # A difference between two rates is in percentage points. Writing it as
        # "%" states a relative change the evidence does not contain.
        if unit == "rate":
            gap = "{} points".format(_format(abs(delta)))
        elif unit == "days":
            gap = "{} day{}".format(_format(abs(delta)), "" if abs(delta) == 1 else "s")
        else:
            gap = _format(abs(delta), unit)
        sentence += ", {} {} the portfolio's {}".format(
            gap, "above" if delta > 0 else "below", _format(baseline["value"], unit))
    elif measure != "share_pct" and _is_number(top.get("share_pct")):
        sentence += " ({} of the total)".format(_format(top["share_pct"], "rate"))

    base = _base_count(top)
    if unit == "rate" and base and base[0] < SMALL_BASE:
        noun = base[1] if base[0] != 1 else _singular(base[1])
        sentence += ", but on only {} {}".format(_format(base[0], "count"), noun)
    sentence += "."

    top_n = ranking.get("n")
    share = evidence.supporting_metrics.get("share_of_total_pct")
    if top_n and _is_number(share):
        sentence += " The top {} hold {} of the total.".format(top_n, _format(share, "rate"))

    ranked = ranking.get("entities_ranked") or len(rows)
    complete = not top_n and ranked <= len(rows)
    if complete and len(valued) > 2 and not has_baseline and measure != "share_pct":
        other = (max if ascending else min)(valued, key=lambda r: r[measure])
        tied = sum(1 for r in valued if r[measure] == other[measure]) > 1
        coded = re.fullmatch(r"\d{1,3}", _entity_label(other)) is not None
        if other is not top and not tied and not coded:
            sentence += " {} is {} at {}.".format(
                _cap(_entity_label(other)), "highest" if ascending else "lowest",
                _format(other[measure], unit))
    return sentence


def _outlier_answer(evidence: EvidencePackage) -> str:
    """How many were flagged, and which is most extreme. The raw score is not
    quoted: "the highest value at 27" means nothing without its definition."""
    flagged = evidence.aggregates.get("outliers_flagged")
    rows = evidence.result_table
    group = _group_noun(rows) or (_population_noun(evidence)[:-1], _population_noun(evidence))
    sentence = "{} {} are flagged as outliers".format(_format(flagged, "count"), group[1])
    if rows:
        sentence += "; the most extreme is {}".format(_entity_label(rows[0]))
    return sentence + "."


def _segment_answer(evidence: EvidencePackage) -> str:
    """A value segmentation, as the finding it exists to show: concentration of value."""
    top = evidence.result_table[0]
    value_key = next(k for k in top if k.endswith("_ape_share_pct"))
    return "{} holds {} of APE from {} of customers.".format(
        _cap(_entity_label(top)), _format(top[value_key], "rate"),
        _format(top["entity_share_pct"], "rate"))


def _comparison_answer(evidence: EvidencePackage) -> str:
    rows = evidence.result_table
    if any(_is_number(r.get("pct_delta_mean")) for r in rows):
        return _diagnostic_answer(evidence)
    measure = _headline_measure(evidence)
    if measure is None or len(rows) < 2:
        return _ranked_answer(evidence) if rows else _scalar_answer(evidence)
    parts = ["{} {}".format(_entity_label(r), _format(r[measure], _unit_of(measure, r[measure])))
             for r in rows[:4] if _is_number(r.get(measure))]
    if not parts:
        return _ranked_answer(evidence)
    return "{}: {}.".format(_cap(_measure_phrase(measure)), _join(parts))


def _timeseries_answer(evidence: EvidencePackage) -> str:
    rows = evidence.result_table
    if not rows:
        return _scalar_answer(evidence)
    period_key = next((k for k in ("period", "cohort_year") if k in rows[0]), None)
    numeric = [k for k, v in rows[0].items()
               if _is_number(v) and k not in ("rank", "group_rows")]
    if not period_key or not numeric:
        return _scalar_answer(evidence)

    # A series split by a second dimension (period x product): comparing the first
    # row with the last would compare two different products.
    split = next((k for k, v in rows[0].items()
                  if k != period_key and isinstance(v, str) and not k.startswith("__")), None)
    if split:
        latest = rows[-1][period_key]
        current = [r for r in rows if r.get(period_key) == latest]
        measure = numeric[0]
        top = max(current, key=lambda r: r.get(measure) or 0)
        return "In {}, {} had the most {}: {}.".format(
            latest, _humanize_label(top[split]), _measure_phrase(measure),
            _format(top[measure], _unit_of(measure, top[measure])))

    first, last = rows[0], rows[-1]
    change = evidence.supporting_metrics.get("change")
    # The measure the analysis reports a change for, identified by that change.
    measure = next((k for k in numeric if _is_number(change) and _is_number(first.get(k))
                    and _is_number(last.get(k))
                    and abs((last[k] - first[k]) - change) < 1e-6), None) \
        or _headline_measure(evidence) or numeric[0]
    unit = _unit_of(measure, first.get(measure))
    before, after = first.get(measure), last.get(measure)
    verb = "rose" if after > before else "fell" if after < before else "was unchanged"
    sentence = "{} {} from {} in {} to {} in {}".format(
        _cap(_measure_phrase(measure)), verb, _format(before, unit), first[period_key],
        _format(after, unit), last[period_key])
    pct = evidence.supporting_metrics.get("pct_change")
    if _is_number(pct) and verb != "was unchanged":
        sentence += " ({} {:.1f}%)".format("up" if pct > 0 else "down", abs(pct))
    return sentence + "."


def _distribution_answer(evidence: EvidencePackage) -> str:
    aggregates = evidence.aggregates
    rows = evidence.result_table
    largest = max(rows, key=lambda r: r.get("share_pct") or 0) if rows else None
    if "median" in aggregates and "mean" in aggregates:
        unit = ""
        if largest is not None:
            match = re.search(r"\d([ydm])$", str(_label_of(largest)))
            if match:
                unit = " " + _UNIT_WORDS[match.group(1)][1]
        sentence = "The median is {}{} (mean {})".format(
            _format(aggregates["median"]), unit, _format(aggregates["mean"]))
        if largest is not None and _is_number(largest.get("share_pct")):
            noun = _population_noun(evidence)
            if noun == "records":
                noun = next((k for k in ("policies", "claims", "renewals", "customers")
                             if k in largest), "records")
            sentence += ", and the largest band is {} with {:.1f}% of {}".format(
                _humanize_label(_label_of(largest)), largest["share_pct"], noun)
        return sentence + "."
    if largest is not None and _is_number(largest.get("share_pct")):
        return "{} is the largest group, at {}.".format(
            _cap(_entity_label(largest)), _format(largest["share_pct"], "rate"))
    return _scalar_answer(evidence)


def _record_answer(evidence: EvidencePackage) -> str:
    if not evidence.retrieved_records:
        return "No record was retrieved."
    record = evidence.retrieved_records[0]
    details = []
    if record.get("Plan.Name"):
        details.append(_short_plan_name(str(record["Plan.Name"])))
    if record.get("POLICY.STATUS"):
        details.append(str(record["POLICY.STATUS"]))
    if record.get("APE") is not None:
        details.append("APE {}".format(_format_key("APE", record["APE"])))
    if record.get("Sum.Assured") is not None:
        details.append("sum assured {}".format(_format_key("Sum.Assured", record["Sum.Assured"])))
    sentence = "Policy {}: {}.".format(record.get("AppID", ""), ", ".join(details)) \
        if details else "The record was found."
    absent = evidence.supporting_metrics.get("related_records_absent")
    if absent:
        names = {"claims": "claim", "persistency": "renewal"}
        sentence += " It has no {} record.".format(
            " or ".join(names.get(a, a) for a in absent))
    return sentence


def _leader_label(leader: Any) -> str:
    """Name a diagnostic's leader. Product codes are P plus two digits; SALES_ID is
    the only six-digit identifier a driver scan ranks in this dataset."""
    text = str(leader)
    if re.fullmatch(r"\d{6}", text):
        return "Agent " + text
    return text


def _diagnostic_answer(evidence: EvidencePackage) -> str:
    """§8 diagnostics: what differs, by how much, and that it is not a proven cause."""
    closing = " This is an observed difference, not a proven cause."
    rows = evidence.result_table

    # A leader against its peers: the leader's result, then where its book differs.
    drivers = [r for r in rows if _is_number(r.get("pct_delta"))
               and _is_number(r.get("leader")) and _is_number(r.get("peers"))]
    if drivers:
        top = drivers[0]
        ranking = evidence.ranking or {}
        who = _leader_label(ranking.get("leader")
                            or evidence.supporting_metrics.get("leader") or "The leader")
        attribute = str(top["attribute"])
        if " = " in attribute:
            what = "{} share".format(_humanize_label(attribute.split(" = ", 1)[1]))
            unit = "rate"
        else:
            what = _readable_attribute(attribute)
            unit = _unit_of(attribute, top["leader"])
        sentence = ""
        if ranking.get("measure") and _is_number(ranking.get("leader_value")):
            value_unit = _unit_of(ranking["measure"], ranking["leader_value"])
            sentence = "{} has the {} {} at {}. ".format(
                who, "lowest" if ranking.get("ascending") else "highest",
                _measure_phrase(ranking["measure"]),
                _format(ranking["leader_value"], value_unit))
        sentence += "Its biggest difference from peers is {}: {} vs {}.".format(
            what, _format(top["leader"], unit), _format(top["peers"], unit))
        return sentence + closing

    # Two cohorts compared attribute by attribute.
    cohorts = [r for r in rows if _is_number(r.get("pct_delta_mean"))]
    if not cohorts:
        return _scalar_answer(evidence) + closing
    top = cohorts[0]
    sides = [k[:-len("_mean")] for k, v in top.items()
             if k.endswith("_mean") and not k.startswith(("delta", "pct_delta"))
             and _is_number(v)]
    attribute = re.sub(r"^(?:owner|customer|client)\s+", "",
                       _readable_attribute(str(top["attribute"])))
    pct = top["pct_delta_mean"]
    if len(sides) < 2:
        return "The biggest difference is in {}, at {}.".format(
            attribute, _format(abs(pct), "rate")) + closing
    first, second = sides[0], sides[1]
    unit = _unit_of(str(top["attribute"]), top["{}_mean".format(first)])
    if unit not in ("rate", "money"):
        unit = "count"
    delta = top.get("delta_mean")
    if unit == "rate" and _is_number(delta):
        size = "{} points".format(_format(abs(delta)))
    else:
        size = _format(abs(pct), "rate")
    return ("{} is {} {} for {} than for {} ({} vs {}).".format(
        _cap(attribute), size, "lower" if pct < 0 else "higher",
        COHORT_NAMES.get(first, _humanize_label(first)),
        COHORT_NAMES.get(second, _humanize_label(second)),
        _format(top["{}_mean".format(first)], unit),
        _format(top["{}_mean".format(second)], unit))) + closing


def _executive_answer(evidence: EvidencePackage) -> str:
    """A briefing: one short line per section, each carrying its headline figure."""
    sections = evidence.result_table
    if not sections:
        return _scalar_answer(evidence)
    lines = []
    for section in sections:
        title = section.get("title", "Summary")
        figures = {k: v for k, v in (section.get("aggregates") or {}).items() if _is_number(v)}
        table = section.get("table") or []
        parts: List[str] = []
        if figures:
            counts = [(k, v) for k, v in figures.items()
                      if k.endswith("_count") or k in VOLUME_MEASURES][:1]
            rates = [(k, v) for k, v in figures.items() if _unit_of(k, v) == "rate"][:2]
            money = [(k, v) for k, v in figures.items()
                     if re.search(r"at_risk|at_stake|exposure", k)][:1]
            parts = ["{} {}".format(_format(v, "count"), _count_noun(k)) for k, v in counts]
            parts += ["{} {}".format(_format(v, "rate"), _measure_phrase(k)) for k, v in rates]
            parts += ["{} {}".format(_format(v, "money"), _measure_phrase(k)) for k, v in money]
            line = _join(parts)
        elif table:
            row = table[0]
            words = set(re.findall(r"[a-z]+", title.lower()))
            candidates = [k for k, v in row.items() if _is_number(v)
                          and (_unit_of(k, v) == "rate" or re.search(r"at_stake|at_risk|exposure", k))]
            money_first = bool(re.search(r"exposure|value|premium|stake|risk", title.lower()))
            key = (next((k for k in candidates if re.search(r"at_stake|at_risk|exposure", k)), None)
                   if money_first else None) \
                or next((k for k in candidates if words & set(k.split("_"))), None) \
                or next((k for k in candidates if re.search(r"at_stake|at_risk", k)), None) \
                or (candidates[0] if candidates else None)
            if key is None:
                continue
            unit = _unit_of(key, row[key])
            line = "{}, {} {}".format(_entity_label(row),
                                      _format(row[key], "money" if unit != "rate" else unit),
                                      _measure_phrase(key))
        else:
            continue
        if line:
            lines.append("{}: {}.".format(title, line))
        if len(lines) == 4:
            break
    return " ".join(lines) if lines else _scalar_answer(evidence)


def _concentration_answer(evidence: EvidencePackage) -> str:
    """How much of the total the largest holders account for.

    The groups are named for what they are. Premium concentrated in the top 1% of
    *products* and in the top 1% of *policies* are different findings.
    """
    aggregates = evidence.aggregates
    group = _group_noun(evidence.result_table)
    if group is None or ("claim" in (evidence.population or "").lower()
                         and group[1] == "policies"):
        plural = _population_noun(evidence)
        group = (_singular(plural), plural)

    shares = sorted((int(k.split("_")[1].rstrip("pct")), v)
                    for k, v in aggregates.items()
                    if k.startswith("top_") and k.endswith("pct_share") and _is_number(v))
    if shares:
        head = shares[0]
        sentence = "The largest {}% of {} hold {} of the total.".format(
            head[0], group[1], _format(head[1], "rate"))
        if len(shares) > 1:
            sentence += " The top {}% hold {}.".format(shares[-1][0],
                                                       _format(shares[-1][1], "rate"))
        return sentence

    points = sorted((int(k.split("_")[1]), v) for k, v in aggregates.items()
                    if k.startswith("top_") and k.endswith("_share_pct") and _is_number(v))
    if points:
        n, value = points[0]
        if n == 1:
            sentence = "The largest {} accounts for {} of the total.".format(
                group[0], _format(value, "rate"))
        else:
            sentence = "The largest {} {} account for {} of the total.".format(
                n, group[1], _format(value, "rate"))
        if len(points) > 1:
            sentence += " The top {} hold {}.".format(points[1][0],
                                                      _format(points[1][1], "rate"))
        return sentence
    return _scalar_answer(evidence)


def _population_noun(evidence: EvidencePackage) -> str:
    """The thing being counted, named the way a business reader would name it."""
    population = (evidence.population or "").lower()
    for noun in ("policies", "claims", "agents", "customers", "branches",
                 "products", "renewals"):
        if noun in population:
            return noun
    return "records"


def _table_answer(evidence: EvidencePackage) -> str:
    rows = evidence.result_table
    if not rows:
        return _scalar_answer(evidence)

    # Percentile shares mean this is a concentration analysis, whatever answer
    # type the PS declared.
    if any(k.startswith("top_") and "share" in k for k in evidence.aggregates):
        return _concentration_answer(evidence)
    if "check" in rows[0]:
        passed = evidence.aggregates.get("checks_passed")
        total = evidence.aggregates.get("checks_total")
        failed = [r["check"] for r in rows if not r.get("passed")]
        sentence = "{} of {} checks pass.".format(_format(passed), _format(total))
        if failed:
            sentence += " The exception is: {}.".format("; ".join(failed))
        return sentence
    if "domain" in rows[0]:
        return "{} analytical domains cannot be supported by this data, including {}.".format(
            _format(len(rows)), "; ".join(r["domain"] for r in rows[:3]))
    if "id" in rows[0] and "severity" in rows[0]:
        high = [r["id"] for r in rows if r.get("severity") == "high"]
        return "{} data-quality findings are recorded, {} of them high severity{}.".format(
            _format(len(rows)), _format(len(high)),
            " ({})".format(", ".join(high)) if high else "")
    return _ranked_answer(evidence)


COMPOSERS = {
    "scalar": _scalar_answer,
    "ranked_table": _ranked_answer,
    "comparison": _comparison_answer,
    "timeseries": _timeseries_answer,
    "distribution": _distribution_answer,
    "record": _record_answer,
    "diagnostic_narrative": _diagnostic_answer,
    "executive_summary": _executive_answer,
    "table": _table_answer,
}


def compose_answer(evidence: EvidencePackage) -> str:
    """The executive answer, composed from evidence alone."""
    if not evidence.is_sufficient:
        return _insufficient(evidence)

    composer = COMPOSERS.get(evidence.expected_answer_type, _table_answer)
    try:
        body = composer(evidence)
    except Exception:
        body = _scalar_answer(evidence)

    caveat = _leading_caveat(evidence)
    return "{} {}".format(body, caveat).strip() if caveat else body


def _leading_caveat(evidence: EvidencePackage) -> str:
    """A qualification appended to every answer type.

    None is appended any more. The one caveat that changes a decision - a leading
    rate resting on a small base - is written into the ranking sentence itself,
    where it qualifies the figure it applies to. A blanket "this rests on N
    records" was attached to lists of data-quality findings and single-record
    lookups, where it is simply false as a warning. Limitations remain in the
    evidence package, which every interface renders.
    """
    return ""


