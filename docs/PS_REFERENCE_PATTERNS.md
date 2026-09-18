# Excel_sense_v2 - Reference PS Pattern Digest

Distilled from the pre-project reference material before it was removed, per specification §3
("use the references as **design references**, not as specifications to copy blindly").

Sources analysed:

| Source | Content |
|---|---|
| `problem_statements.json` | 270 granular PSs, ~100 distinct `intent_type` values |
| `business_problem_statements.json` | 221 business PSs, 17 business domains, 2 analysis tiers |
| `PS.md` | 136 PSs across 60 problem types in 10 lettered groups, 243 headings |
| `retrieval_demo.json` | Worked retrieval traces: query → ranked PS candidates + confidence + evidence |

This digest preserves what was worth learning. It is **not** a catalogue and nothing here is
copied into the new one verbatim.

---

## 1. What the references got right (patterns adopted)

### 1.1 Two-tier catalogue
Granular, single-operation PSs (lookup, count, filter, average) sit alongside multi-step
business PSs (portfolio health, root cause, prescriptive action). The new catalogue keeps this
split via an `analysis_tier` field.

### 1.2 Compositional intent taxonomy
Intents are composed from primitives rather than enumerated flat:

```
{Filtering | GroupBy | Join | Multi-Join | Aggregation | Time-Series | Temporal |
 Ranking | Distribution | Conditional | Derived-Metric | Multi-Condition |
 Anomaly | Concentration | Window | Anti-Join}
   ×
{Count | Sum | Average | Min | Max | Ratio | Comparison | Trend | TopN | BottomN |
 Distinct | Bucketing | Seasonality | Diversity}
```

Roughly 100 distinct combinations appeared. The new catalogue reuses this grammar because it
maps cleanly onto deterministic execution primitives.

### 1.3 Operation vocabulary beyond SQL verbs
Alongside `GROUP BY`/`COUNT`/`WHERE`/`JOIN`, the business PSs used analytical macro-operations
that are genuinely useful as engine primitives:

`multi_metric_table` (112), `compare_to_baseline` (73), `portfolio_overview` (31),
`driver_scan` (25), `segment_profile` (23), `entity_shortlist` (22), `crosstab_metric` (20),
`composite_score` (19), `concentration` (15), `outliers` (11), `top_entity_concentration` (7)

### 1.4 Structural discipline from `PS.md`
Three habits worth keeping:
- **A join map per domain group** — states the exact join path before any PS uses it.
- **A "What this data cannot answer" section per group** — negative scope stated up front.
- **A test set per group** — each group ships its own validation queries.

### 1.5 Evidence-carrying retrieval
`retrieval_demo.json` shows routing returning *ranked candidates with confidence scores and
matched tables/columns*, not a single opaque answer. Adopted for Phase 5.

---

## 2. What the references got wrong (patterns rejected)

`retrieval_demo.json` is also a cautionary document. In a 4-query demo:

| Defect observed | Example |
|---|---|
| Malformed generated query | `SUBSTR(issuance_date, p.1, 4)` — arguments transposed |
| Answer that is not an answer | A trend question answered with the single scalar `2737` |
| Confidently wrong aggregate | `avg premiums_paid_count` for Lapsed returned `0.0` from a column that does not exist |
| Low-confidence match still surfaced | Top hit at `confidence 0.4866`, third at `0.3373`, all presented alike |
| Queries answered from absent data | "How do smokers compare…" routed to a PS despite no smoker column |

**Consequences for the new build:** the PS catalogue carries an explicit executable
specification rather than a generated query string; routing enforces a confidence floor and
refuses rather than degrades; every PS declares its expected answer shape so a scalar cannot
be returned for a trend question.

---

## 3. Schema reality check

Reference material references **73 distinct columns**. Against the five actual CSVs:

- **18 supported** — `POLICY.STATUS`, `Plan.Name`, `Sum.Assured`, `Claim.Amount`, `APE`,
  `CHANNEL`, `Owner.Earned.Income`, `Owner.Age`, `Status.of.Claim`, `SALES_ID`, `ClientID`,
  `Owner.Gender`, `Owner.Comm.State`, `Claim_No`, `AppID`, `Owner.Comm.City.Name`, `Prod_No`,
  and a payment-mode concept (present in Persistency, not Policy).
- **55 unsupported** — dropped entirely.

### 3.1 Heaviest unsupported concepts

| Concept | Uses | Why it cannot carry over |
|---|---:|---|
| `zone` | 44 | No zone column. Geography exists only as owner state/city/postal. |
| `total_premium_paid` | 39 | No cumulative premium field; only `APE` and `Renewal_Premium_Amount`. |
| `claim_type` | 32 | All 134 claims are death claims. No type dimension. |
| `retention_status` | 31 | Not a column; retention must be derived from `POLICY.STATUS`. |
| `is_smoker`, `bmi`, `medical_underwriting_flag` | 43 | No health or underwriting data at all. |
| `target_achievement_pct`, `nbp_achieved`, `policies_sold` | 23 | No agent target or production data. |
| `branch_name`, `branch_id` | 23 | Only an opaque `Branch_No` integer exists. |
| `surrender_value_paid`, `surrender_charge`, `surrender_reason` | 18 | Surrender is a status only, with no economics. |
| `rider_count`, `has_maturity_benefit` | 13 | No rider or benefit-structure columns. |
| `agent_incentive_amount` | 5 | No commission or incentive data. |

### 3.2 Reference domains that cannot carry over

| Reference domain | PSs | Blocker |
|---|---:|---|
| Geographic / Zone Analysis | 34 | No zone hierarchy (owner-level state/city analysis is a narrower substitute) |
| Sales Channel Performance | 10 | `CHANNEL` is `Banca` for all 25,000 rows |
| Profitability / Claims Economics | 10 | No cost, commission, expense or reserve data |

---

## 4. The gap the references never covered

**35 of the 59 actual columns are referenced by no supplied PS or metadata.** These are the
new catalogue's real subject matter:

| Table | Unreferenced columns | Analytical ground opened |
|---|---|---|
| `sales` | `Surrender Rate`, `Early Claim Rate`, `PIVC Number Mismatch Rate`, `PIVC Concern Raised Rate`, `RCU Rejections Rate`, `Free Look Cancellation Rate`, `Sales Related Complaint`, `First Year Persistency Rate`, `Type` | An entire agent quality-and-misconduct dimension: mis-selling signals, verification failures, complaint propensity, agent risk scoring |
| `persistency` | `Renewal_Due_Date`, `Renewal_Paid_Date`, `Renewal_Status`, `Renewal_Premium_Amount`, `Persistency_Bucket`, `Payment_Mode`, `PREMIUM.PAYING.TERM` | The full renewal lifecycle: due-vs-paid timing, payment delay, 13th/25th/37th-month persistency cohorts, payment-method behaviour |
| `claims` | `Date.of.Death`, `Cause.of.Death`, `Date.of.Intimation`, `Amount.Paid`, `Settlement.Date` | Cause-of-death mix, intimation lag, settlement turnaround, payout-vs-claimed variance, early-death exposure |
| `policy` | `Branch_No`, `Login.Date`, `Inforce.Date`, `Inforce.month`, `Trad.ULIP`, `Mode`, `PREMIUM.PAYING.TERM`, `Policy.Term` | Underwriting turnaround (login→inforce), traditional-vs-ULIP mix, term structure, branch concentration, monthly cohorts |
| `owner` | `Owner.Given.Name`, `Owner.Education`, `Owner.Comm.Country`, `Owner.Comm.Postal.Code`, `Owner.Occupation` | Education and occupation segmentation, NRI/overseas identification, postal concentration |

---

## 5. Question-phrasing patterns worth reusing

Observed across reference `example_queries`, and reused as the paraphrase space for Phase 5:

- **Direct:** "How many policies are lapsed?"
- **Executive:** "Give me a portfolio health snapshot."
- **Superlative:** "Which product has the most lapsed policies?" / "Who are my worst agents?"
- **Comparative:** "How does X compare to Y?" / "X vs Y"
- **Diagnostic why:** "Why is lapse rate rising?" / "What's driving surrenders?"
- **Diagnostic how:** "How is agent A outperforming peers?"
- **Quantitative:** "What is the average …?" / "What share of …?"
- **Trend:** "What is the trend in …?" / "How has … changed over time?"
- **Prescriptive:** "Where should I focus retention effort?"
- **Entity-scoped:** "Tell me about policy 15011000000."

Each new PS carries several such paraphrases so routing is trained against realistic phrasing
variety rather than a single canonical sentence.
