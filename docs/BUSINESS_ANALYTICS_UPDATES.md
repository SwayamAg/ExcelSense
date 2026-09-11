# Business Analytics Upgrade — Change Log & Architecture

> **What changed in one line:** the system stopped being a natural-language-to-SQL
> retriever and became an analytics engine — a business question now produces an
> *analytical plan* of several deterministic operations, executes them against a
> metric registry with declared denominators, and returns an evidence-backed
> business answer instead of a DataFrame.

**Status:** implemented, tested, running.
`streamlit run app.py` → http://localhost:8501

---

> ## ⚠️ This document describes an earlier dataset
>
> The figures quoted throughout — 149 claims, 40 surrenders, 440 retention
> contacts, 80 agents, `surrender_reason`, `status_change_date`, `chatbot.py` —
> come from the **dataset-v3** corpus this project no longer uses. They are kept
> because the *architecture* sections (the metric registry, multi-step plans, the
> 14 operations, the retrieval signals) remain accurate and are the reason those
> designs look the way they do.
>
> **For the current data contract, read [`../README.md`](../README.md) and
> `data/csv/Talk_to_Data_PoC_Documentation_Metadata_V4.docx`.** The current system
> runs on five tables: 25,000 policies, 25,000 owners, 24,542 renewal records,
> 1,956 agents, 134 claims. There is no surrender, retention or rider table.
>
> Grounding is enforced by `tests/validate_grounding.py`, not by this document.

---

## 1. Summary of changes

| | Before | After |
|---|---|---|
| Problem Statements | 270 (SQL template each) | 270 legacy **+ 146 business** = **416** |
| Analytical steps in the library | 0 | **464** |
| Question → query relationship | 1 question → 1 SQL/Cypher | 1 question → **N-step plan** (avg 3.35, max 6) |
| Derived metrics | frozen inside SQL strings | **39 registry metrics + 3 share metrics**, each with a declared numerator, denominator, population, unit and polarity |
| Analytical operations | COUNT / SUM / AVG / GROUP BY | **14 operations** incl. `driver_scan`, `composite_score`, `concentration`, `outliers`, `crosstab_metric`, `entity_shortlist`, `segment_profile` |
| Dimensions | ~6 hard-coded | **33** (incl. derived income / age / sum-assured / vintage / affordability bands) |
| Retrieval | TF-IDF cosine, hard 0.30 floor | **5-signal hybrid**: lexical + concept-expanded lexical + concept overlap + intent triggers + **operation affinity** |
| Answer | `fmt_ans(df)` or LLM prosing a frame | **Answer / Evidence / Analysis / Business implication / Recommendation / Caveat**, all generated from computed numbers |
| Out-of-scope questions | answered with a plausible-looking proxy | **refused with the specific missing data named** |
| Tests | 10 | **10 legacy (all still pass) + 71 new = 81** |
| Held-out business eval | none existed | **83 cases**, routing 100%, execution 100% |

---

## 2. Architecture changes

### 2.1 Before

```
NL question
  → regex entity resolution (IDs, 5 zones, 8 product names)
  → TF-IDF over 270 PS documents  (score < 0.30 → weak dynamic fallback)
  → semantic-compatibility filter
  → ONE sql_template  →  sql_to_cypher()  →  ONE DataFrame
  → fmt_ans(df)  or  LLM asked to narrate the frame
```

The blocking constraint was structural, not a tuning problem:
`plan_query()` returned `{"cypher": "<one string>"}`. There was no representation
for "step 2 depends on step 1", no place to put a denominator definition, and no
primitive for "compare this against the baseline".

### 2.2 After

```
                         NL business question
                                  │
              ┌───────────────────┴───────────────────┐
              │  business_engine.route()               │
              │  scope guard → style → retrieval       │
              └───────────────────┬───────────────────┘
                                  │
    ┌──────────────┬──────────────┼──────────────┬──────────────────┐
    ▼              ▼              ▼              ▼                  ▼
insufficient   simple_retrieval  business_analysis         adhoc_analysis
   _data       (legacy engine,   (business PS matched)     (analytical, but
 says what     UNCHANGED:                │                  no PS matched)
 is missing    graph + Cypher)           │                       │
                                         ▼                       ▼
                              ┌──────────────────────────────────────┐
                              │  analysis_planner                     │
                              │  • EntityResolver (data-driven vocab) │
                              │  • AnalyticalPlan = [AnalysisStep...] │
                              │  • ParameterBinder ($product, $zone,  │
                              │    $top_product ← step s1 output)     │
                              └───────────────────┬──────────────────┘
                                                  │  per-step failure isolation
                                                  ▼
                              ┌──────────────────────────────────────┐
                              │  analytics_core   (numerical truth)   │
                              │  DataContext → 5 fact frames          │
                              │  MetricSpec registry (denominators!)  │
                              │  14 operations                        │
                              └───────────────────┬──────────────────┘
                                                  ▼
                              ┌──────────────────────────────────────┐
                              │  answer_composer                      │
                              │  20 templates → 6-section answer      │
                              │  reliability + causation caveats      │
                              │  optional LLM = wording only          │
                              └──────────────────────────────────────┘
```

### 2.3 New modules

| File | Lines | Role |
|---|---:|---|
| `analytics_core.py` | 1,135 | **Numerical truth layer.** `DataContext` (5 fact frames), `MetricSpec` registry, 14 analytical operations. No LLM, no guessing. |
| `business_intents.py` | 423 | 16-intent taxonomy, 24-concept ontology (420 surface forms), query expansion, intent triggers. |
| `analysis_planner.py` | 441 | `AnalyticalPlan` / `AnalysisStep` / `StepResult`, data-driven `EntityResolver`, `ParameterBinder`, plan execution. |
| `business_retrieval.py` | 334 | 5-signal hybrid retriever over the combined 416-record library. |
| `answer_composer.py` | 1,757 | 20 answer templates → structured business narrative. Optional LLM polish. |
| `business_engine.py` | 348 | Orchestrator, 4-way routing, scope guard, capability introspection. |
| `business_dashboard.py` | 378 | Streamlit "Business Analyst" mode. |
| `build_business_ps*.py` (4 files) | ~2,300 | PS library generator + validator. |
| `build_business_eval.py` / `business_eval.py` | ~470 | Held-out eval set + scorer. |
| `test_business_analytics.py` | 608 | 71 deterministic regression tests. |

### 2.4 Modified files

| File | Change | Risk |
|---|---|---|
| `dashboard.py` | Added a sidebar **Mode** radio. `Business Analyst` renders the new UI and `st.stop()`s; `Query Pipeline Trace` is the original code, byte-identical below that point. Also replaced two hard-coded absolute paths with `__file__`-relative ones. | None — classic path untouched |
| *(nothing else)* | `graph_engine.py`, `query_workflow.py`, `chatbot.py`, `build_query_system.py`, `extend_dataset.py`, `regression_test.py`, `problem_statements.json` are **unmodified**. | — |

---

## 3. The metric registry — denominators are declared, not implied

This is the single most important correctness change. Every rate in the old
system had its denominator baked into an SQL string with no documentation.

```python
MetricSpec(
    "surrender_rate", "Surrender rate", base="policy", unit="%",
    definition="Surrendered policies divided by policies on products that "
               "actually carry a surrender value (has_surrender_value = Y). "
               "Term Life and Health Rider are excluded from the denominator "
               "because they cannot be surrendered.",
    numerator   = lambda d: d["is_surrendered"].sum(),
    denominator = lambda d: d["surrender_eligible"].sum(),
    population  = lambda d: d[d["surrender_eligible"]],
    higher_is_better=False, scale=100.0)
```

Consequences that show up in real answers:

* **`surrender_rate` for Term Life returns `n/a`, not `0%`.** Term Life has no
  surrender value; reporting 0% would imply it performs well on a metric that
  does not apply to it. Denominator = 0 → NaN → flagged `reliable = False`.
* **`lapse_rate` denominator = 918, not 1,014.** Matured (83) and Death Claim (13)
  policies ended by design; including them understates lapse by ~2.3pp.
  Both views are available (`lapse_rate` vs `lapse_rate_all_policies`).
* **`loss_ratio` and `risk_loss_ratio` are separate metrics.** 59 of 149 claims
  are Maturity and 28 are Surrender — contractual payouts, not underwriting
  losses. Term Life's headline loss ratio is **0.861x** but its risk-only ratio
  is also 0.861x (no maturity claims), whereas Annuity's 0.573x collapses to
  **0.041x** once maturities are stripped out. Answering "which products have
  bad economics" on the headline ratio alone gets Annuity wrong.
* **Money-denominated ratios are flagged by record count, not rupee count**
  (`denominator_is_count=False`), so a 3-policy group can't look "reliable"
  because its premium denominator happens to be large.

Registry contents: 39 metrics + 3 share metrics across volume/value, persistency,
surrender, claims & risk, retention, customer profile.

---

## 4. Multi-step analytical plans

A business PS carries an **executable plan**, not an SQL string:

```json
{
  "problem_id": "BPS-015",
  "problem_statement": "Which products should be prioritised for retention initiatives?",
  "analysis_tier": "multi_step_business",
  "analytical_steps": [
    {"id":"s1","op":"composite_score","params":{"dimension":"product","components":[
        {"metric":"lapse_rate","weight":0.35,"direction":"high"},
        {"metric":"surrender_rate","weight":0.15,"direction":"high"},
        {"metric":"policy_share","weight":0.25,"direction":"high"},
        {"metric":"avg_annual_premium","weight":0.25,"direction":"high"}]},
     "purpose":"Weight attrition by how much premium is at stake."},
    {"id":"s2","op":"compare_to_baseline","params":{"dimension":"product","metric":"lapse_rate"}},
    {"id":"s3","op":"multi_metric_table","params":{"dimension":"product",
        "metrics":["retention_contacts","retention_success_rate"]}},
    {"id":"s4","op":"driver_scan","params":{"metric":"lapse_rate",
        "focus_filters":{"product":"$top_product"},"min_n":12,"top_k":6}}
  ],
  "parameters": {"top_product": {"source":"step:s1","field":"product_name"}}
}
```

`$top_product` in step 4 is **bound from step 1's output at runtime** — that is
the step-chaining the old architecture had no way to express. Binding priority:
1. an entity the user named ("Why is lapse high in **Annuity**?")
2. a value produced by an earlier step
3. the PS's declared default

**Three analysis tiers, visible in the response object and in the UI:**

| Tier | Meaning | Count in library |
|---|---|---:|
| `simple_retrieval` | one query is enough → legacy engine | (270 legacy PS) |
| `analytical` | aggregation + derived metric + baseline comparison | 56 |
| `multi_step_business` | several chained operations | 90 |

A failing step is isolated: it is recorded in the answer's
**"Not answerable from this dataset"** section and the remaining steps still run.

---

## 5. The 14 analytical operations

| Operation | What it adds beyond GROUP BY |
|---|---|
| `metric_by_dimension` / `multi_metric_table` | N registry metrics × any dimension, each on its own denominator, with `_n` and `_reliable` columns |
| `overall_metric` | portfolio baseline for any metric, with its definition |
| `compare_to_baseline` | rate, gap to baseline (`diff_vs_baseline`), relative `lift`, reliability |
| `rank` | reliability-aware ordering (won't rank a 3-policy group first) |
| `composite_score` | **weighted, declared prioritisation frameworks** — min-max normalised per component, orientation explicit, per-component contributions returned as `score_*` columns so the ranking is arguable |
| `driver_scan` | **root-cause primitive** — scans up to 17 dimensions inside a focus population and ranks segments by deviation from that population's own average |
| `concentration` | Herfindahl-Hirschman Index, effective-N, top-N share |
| `top_entity_concentration` | single-record exposure (top 10 policies = 11.2% of sum assured) |
| `outliers` | IQR fence and z-score, with the fence printed |
| `crosstab_metric` | two-way metric grids, cells below `min_n` **dropped rather than ranked** |
| `portfolio_overview` | the standing KPI block with every denominator |
| `entity_shortlist` | named, ranked, actionable lists of policies / customers / agents |
| `segment_profile` | focus segment vs whole portfolio on ~11 metrics side by side |

---

## 6. Problem Statement library — 146 new business records

`business_problem_statements.json` (validated at build time against the metric,
dimension and operation registries — a PS that names something the engine can't
execute fails the build).

| Business domain | PS | Example |
|---|---:|---|
| Product Performance & Strategy | 12 | "Which products have unfavourable claims economics?" |
| Customer Segmentation & Value | 12 | "What characteristics distinguish customers with poor persistency?" |
| Lapse / Persistency Analysis | 11 | "What should management investigate to improve persistency?" |
| Sales Channel Performance | 10 | "Is any channel producing high acquisition but weak persistency?" |
| Surrender Analysis | 10 | "Are high-value policies disproportionately represented among surrenders?" |
| Retention Strategy | 10 | "Which customers are the best candidates for retention intervention?" |
| Claims & Risk | 10 | "Are there signs of adverse-selection-like patterns in the available data?" |
| Profitability / Claims Economics | 10 | "Which products appear economically attractive versus unattractive?" |
| Agent / Sales Performance | 10 | "Which agents outperform targets but carry concerning portfolio quality?" |
| Comparative / Benchmarking | 10 | "Compare the major sales channels across volume, value, lapse, surrender and retention." |
| Root-Cause / Diagnostic | 10 | "Why is a specific product performing poorly?" |
| Geographic / Zone Analysis | 9 | "Are there meaningful differences in product/channel mix across zones?" |
| Executive / Portfolio Health | 8 | "What are the biggest risks currently sitting in the portfolio?" |
| Anomaly / Concentration | 8 | "Where are the concentration risks the board should know about?" |
| Actionable Targeting | 6 | "Which live policies most resemble the profile of policies that lapsed?" |
| **Total** | **146** | **464 analytical steps, 601 example queries** |

**Schema — additive, backward-compatible.** Every business record still carries
`problem_id`, `problem_statement`, `intent_type`, `example_queries`,
`relevant_tables`, `relevant_columns`, `required_operations`, `sql_template`
(null), `example_answers`, so any existing consumer can read them. New fields:
`business_intent`, `business_domain`, `business_objective`, `analysis_tier`,
`required_metrics`, `relationships`, `concepts`, `analytical_steps`,
`expected_answer_type`, `answer_template`, `parameters`, `caveat`.

IDs are namespaced `BPS-###` — **no collision with the legacy `PS-###` records**
(asserted by a regression test).

---

## 7. Retrieval changes

The legacy retriever's failure mode was structural: "where should management
focus retention effort?" shares no vocabulary with "What is the lapse rate by
product?", so it fell below the 0.30 floor into a weak dynamic-Cypher fallback.

Five signals, combined with fixed weights:

| Signal | Weight | What it contributes |
|---|---:|---|
| lexical | 0.25 | TF-IDF cosine, word 1–2 grams |
| expanded | 0.25 | same, over a **concept-expanded** query ("focus retention" gains `retention lapse surrender attention priority…`) |
| concept | 0.19 | blended coverage/Jaccard over the 24-concept ontology |
| intent | 0.10 | trigger-phrase score for the PS's business intent |
| **operation** | **0.21** | **does the phrasing imply machinery the PS's plan actually contains** |

The operation signal is what separates PS that share a topic but need different
analysis: *"what is driving claims risk"* → `driver_scan`; *"do a handful of
claims account for most of the payout"* → `concentration` +
`top_entity_concentration`. Both are claims questions; only one needs an HHI.

`question_style()` classifies lookup / business / neutral from lookup markers,
business markers, implied operations and concept count. Instance lookups (a
named `CUST…`/`POL…`/`AGT…`) are routed to the legacy graph engine unchanged.

**Retrieval quality on the 292 in-library example queries:**
hit@1 = 0.818, hit@3 = 0.949, same-domain@1 = 0.887.

---

## 8. Business answer format

Every analytical answer is composed deterministically from computed numbers:

```
Answer                 one-line executive conclusion
Evidence               key numbers, each with its denominator / n
Analysis               what was compared, on what definition, at what sample size
Business implication   why it matters in portfolio terms
Recommendation         what to do or investigate next
Caveat                 correlation-vs-causation, sample size, missing data
```

20 templates (`executive_assessment`, `risk_register`, `priority_ranking`,
`strength_weakness`, `leaderboard`, `comparison`, `grid`, `economics`,
`driver_analysis`, `diagnostic`, `concentration`, `anomaly`, `shortlist`,
`action_plan`, `risk_segmentation`, `distribution`, `kpi_block`,
`segment_contrast`, `retention_priority`, `retention_shortlist`).

Three guardrails are automatic:

1. **Reliability caveat** — any group below its metric's `min_n` triggers a
   caveat naming the affected metric; the conclusion is drawn from reliable rows.
2. **Causation caveat** — every `diagnostic` and `driver_analysis` answer states
   that the dataset has no reason codes and supports association only.
3. **Materiality correction** — a top-ranked group holding <15% of the portfolio
   gets an explicit "ranks first on severity but is only X% of the book, so the
   absolute number recovered will be small; **Y** is the largest block" note.

**The LLM is optional and cannot change a number.** `answer_composer.polish()`
is handed the already-composed answer and instructed to rewrite wording only.
The deterministic answer is always retained and always displayed.

---

## 9. Real output — `"Which products should management prioritize for retention?"`

Route `business_analysis` · PS `BPS-015` · tier `multi_step_business` · 4 steps · 32 ms

> **Answer:** **Annuity** ranks first for attention, scoring 0.658 on a framework
> weighting lapse rate (35%), surrender rate (15%), share of portfolio policies
> (25%), average annual premium (25%).
>
> **Evidence** — Annuity (rank 1): lapse rate 39.13%; surrender rate 3.39%;
> 5.82% of policies; average annual premium Rs 497,255. · Term Life (rank 2,
> 0.361): lapse 25.39%, 26.04% of policies. · Against the portfolio baseline
> Annuity is 14.5 percentage points above the book average of 24.62%. ·
> Retention effort saves 45.95% of contacted Child Plan cases but only 12.50%
> of Annuity cases.
>
> **Business implication** — Annuity ranks first on severity but is only 5.8% of
> the portfolio, so the absolute number of policies recovered will be small.
> Term Life is the largest block at 26.0% and is where the same percentage
> improvement would be worth the most. A high lapse rate *and* a low save rate
> together mean effort is being spent without recovering the policy.
>
> **Recommendation** — Target the intervention rather than the whole product:
> within Annuity the worst segment is policy vintage 2–5 years at 53.33%
> (14.2pp above Annuity's own average, n=15). Re-run the ranking with different
> weights if management's priorities differ — the framework is explicit.
>
> **Caveat** — Some groups fell below the minimum sample size for retention
> success rate and surrender rate; those rows are flagged `reliable = False`
> and were not used to draw the conclusion.

---

## 10. New business questions that now work

All ten acceptance-criterion questions route to `business_analysis`, execute
every planned step, and return a six-section answer:

| Question | PS | Steps | Operations used |
|---|---|---:|---|
| What are the biggest risks in our insurance portfolio? | BPS-002 | 6 | overview, compare_to_baseline ×2, concentration, top_entity_concentration, outliers |
| Which products should management prioritize for retention? | BPS-015 | 4 | composite_score, compare_to_baseline, multi_metric_table, driver_scan |
| Which sales channel gives us the best balance of growth and persistency? | BPS-021 | 3 | multi_metric_table ×2, composite_score |
| Which customer segments are most at risk of lapsing? | BPS-032 | 5 | compare_to_baseline ×4, driver_scan |
| Why is Whole Life performing poorly? | BPS-131 | 6 | segment_profile, compare_to_baseline ×2, driver_scan, multi_metric_table ×2 |
| Which products have unfavorable claims economics? | BPS-013 | 3 | multi_metric_table, compare_to_baseline, crosstab_metric |
| Which customers should the retention team prioritize? | BPS-066 | 4 | driver_scan, compare_to_baseline, entity_shortlist, multi_metric_table |
| Are there unusual concentrations or patterns management should investigate? | BPS-113 | 6 | concentration, top_entity_concentration ×2, outliers ×3 |
| Compare the major sales channels across volume, value, lapse, surrender, retention | BPS-121 | 3 | multi_metric_table ×2, compare_to_baseline |
| Give me a management-level assessment of the current portfolio | BPS-001 | 5 | overview, multi_metric_table ×2, compare_to_baseline, concentration |

Plus the honest-refusal path:

| Question | Response |
|---|---|
| "How do we compare with our competitors on market share?" | `insufficient_data` — names competitor/market-share data as missing |
| "What is our expense ratio by product?" | `insufficient_data` — names expense data as missing |
| "What is our solvency ratio?" | `insufficient_data` — names solvency/capital/reserving data as missing |

And the preserved lookup path:

| Question | Response |
|---|---|
| "How many customers are smokers?" | `simple_retrieval` → legacy graph engine, unchanged |
| "Show me the details of agent AGT0101" | `simple_retrieval` → Cypher on `AGT0101`, unchanged |

---

## 11. Test results — before vs after

### Legacy regression suite (`regression_test.py`)

| | Before changes | After changes |
|---|---|---|
| `python3 -m pytest regression_test.py` | **10 passed** | **10 passed** |

No legacy test was modified. Backward compatibility is additionally asserted by
5 new tests (legacy PS file unmodified and still 270 records, no ID collisions,
combined library preserves every legacy ID, business records keep the legacy
schema keys, `QueryWorkflowSystem` still plans and executes).

### New deterministic suite (`test_business_analytics.py`) — 71 tests

| Group | Tests | What it pins down |
|---|---:|---|
| `TestDataContext` | 8 | No row invented or lost in any join; claim rollups reconcile to the raw CSV; risk-claim amounts exclude maturity/surrender |
| `TestMetricDenominators` | 11 | Each rate's denominator recomputed independently from the CSVs; surrender rate excludes ineligible products; shares sum to 100 |
| `TestZeroAndMissingData` | 6 | Zero denominator → NaN not a crash; empty filter → empty frame; unknown metric/dimension → `InsufficientDataError`; small groups flagged |
| `TestRankingAndScoring` | 7 | Sort direction; `direction: high/low` inverts correctly; component contributions sum to the total; score bounded [0,1]; `diff_vs_baseline` arithmetic |
| `TestAnalyticalOperations` | 15 | driver_scan never rescans the filtered dimension and uses the focus baseline; HHI bounds; crosstab cell matches a hand-computed rate; shortlist sorting and filtering |
| `TestPlanConstruction` | 9 | Every PS references only executable ops/metrics; multi-step PS really have >1 step; entity overrides default; **step-output binding**; a failing step doesn't abort the plan |
| `TestEngineRouting` | 6 | Business/lookup/instance/out-of-scope routing; all 9 flagship questions execute every step |
| `TestAnswerStructure` | 6 | Six sections present; causation caveat on diagnostics; expense caveat on economics; **headline number equals the recomputed metric** |
| `TestBackwardCompatibility` | 5 | (above) |

```
$ python3 -m pytest regression_test.py test_business_analytics.py -q
81 passed in 2.22s
```

**Mutation check.** Deliberately breaking `lapse_rate`'s denominator (so it
counts Matured and Death-Claim policies) makes
`test_lapse_rate_denominator_excludes_terminal_by_design` fail with
`1014 != 918` — confirming the assertions bite rather than pass vacuously.

### Held-out business evaluation (`business_eval.py`) — 83 cases

Every question is phrased differently from any `example_queries` entry, so this
is out-of-sample. Nothing here scores generated prose.

| Score | Result | Meaning |
|---|---|---|
| `routing_ok` | **83/83 · 1.000** | business / adhoc / lookup / out-of-scope all routed correctly |
| `execution_ok` | **74/74 · 1.000** | zero failed steps across 248 executed operations |
| `multi_step_ok` | **71/71 · 1.000** | every question requiring multi-step analysis got one |
| `structure_ok` | **73/74 · 0.987** | six-section answer produced |
| `tables_ok` | 69/74 · 0.932 | expected tables touched |
| `metrics_ok` | 65/74 · 0.878 | expected metrics computed |
| `tier_ok` | 64/74 · 0.865 | analysis tier at least what the case required |
| `grounding_ok` | 15/18 · 0.833 | answer names the expected entity |
| `intent_family_ok` | 60/74 · 0.811 | matched PS in the expected or an adjacent intent |
| `operations_ok` | 59/74 · 0.797 | expected analytical operations present in the plan |
| `intent_ok` (strict) | 37/74 · 0.500 | matched PS carried the *exact* expected intent label |

Average **3.35 analytical steps per business question**; 248 operations executed
across the run.

**On the strict `intent_ok` = 0.50.** The taxonomy overlaps by design: *"why are
people cashing out?"* is legitimately `SURRENDER_ANALYSIS` **or** `ROOT_CAUSE`;
*"do savings products hold up better than protection?"* is
`PRODUCT_PERFORMANCE` **or** `COMPARATIVE`. `intent_family_ok` (0.811) accepts a
documented adjacent intent. Both numbers are reported rather than only the
flattering one — but note that `operations_ok`, `metrics_ok` and `execution_ok`
measure whether the *right analysis actually ran*, which matters more than which
of two overlapping labels was attached to it.

### Legacy retrieval evaluation

`evaluation_summary.json` (Recall@1 0.467 / @3 0.633 / @5 0.733 on the original
30 held-out queries) is **untouched** — that pipeline was not modified.

---

## 12. Known limitations

**Dataset limits (not fixable in code — the system says so instead of guessing):**

* **No expense, commission, reserve or investment-return data.** Everything
  labelled "economics" is claims-to-premium only. No answer in this system is a
  profitability statement, and every economics answer says so.
* **Small absolute counts.** 40 surrenders, 149 claims (of which only 62 are risk
  claims), 21 early claims, 440 retention contacts. Product- and segment-level
  cuts of these are single- and double-digit. `min_n` thresholds and
  `reliable` flags surface this rather than hiding it.
* **80 agents servicing 5–33 policies each.** Individual agent rates move
  10–20 percentage points on one policy. Agent outlier flags are review prompts,
  explicitly not performance verdicts.
* **No lapse/surrender reason codes** beyond the 5-value `surrender_reason`
  field, no time-to-event data, no declined-risk or application-stage data. The
  system therefore cannot support causal claims and never makes them. Adverse
  selection in particular is explicitly declared unanswerable.
* **Vintage comparisons are exposure-biased.** Older cohorts have had longer to
  lapse and are the only ones that can produce maturity claims. Flagged as a
  caveat wherever a vintage cut appears.
* **Retention outreach was not randomly assigned.** Channel/offer effectiveness
  differences may reflect case selection, not treatment effect. Caveated.

**Engine limits:**

* **Strict intent accuracy is 0.50** on held-out phrasing (0.81 accepting
  adjacent intents). A handful of near-duplicate PS compete for the same query.
* **No confounding control.** `driver_scan` and `crosstab_metric` can show that a
  gap persists inside sub-groups but cannot decompose mix vs execution effects.
  There is no regression, no standardisation, no significance test.
* **Retrieval is lexical + ontological, not dense.** No `sentence-transformers`
  dependency was added (offline/reproducibility). Genuinely novel phrasing that
  misses the 420-term ontology falls to the ad-hoc analytical plan — which is
  still a baseline-anchored comparison, not a failure, but is generic.
* **Shortlists are rankings, not models.** `entity_shortlist` ranks by observed
  characteristics with declared weights. It is not a calibrated propensity score
  and the answers say so.
* **The ad-hoc planner is shallow** — 3 fixed steps (baseline, compare, support).
* **The `chatbot.py` ML stack** (RandomForest lapse predictor, IsolationForest,
  clustering) is untouched and not wired into the business engine.

---

## 13. Recommended next improvements

1. **Standardised / adjusted rates.** Add a `standardise_by` parameter to
   `compare_to_baseline` so a channel's lapse rate can be reported holding
   product mix constant. This is the single highest-value addition — it converts
   "Online lapses more" into "Online lapses more *after* adjusting for what it
   sells", which is the question management actually asks next.
2. **Significance and confidence intervals.** Wilson intervals on every rate;
   suppress or grey any gap whose interval crosses the baseline. The reliability
   flag is a blunt proxy for this.
3. **Time-to-lapse from `status_change_date`.** The field exists (386 non-null)
   and would support a proper survival/hazard view, replacing the current
   exposure-biased vintage cuts.
4. **Dense retrieval as a sixth signal.** `all-MiniLM-L6-v2` embeddings blended
   with the current five would close most of the remaining intent gap without
   replacing the interpretable signals.
5. **PS de-duplication pass.** Merge the ~12 near-duplicate business PS competing
   for the same queries; this alone should lift strict `intent_ok` materially.
6. **Plan critic step.** A final deterministic check — "did any step return an
   unreliable-only result that the narrative nonetheless leaned on?" — reported
   alongside the answer.
7. **Wire `chatbot.py`'s lapse model in as a metric** (`predicted_lapse_risk`)
   so shortlists can rank on a calibrated score instead of value-at-stake, with
   the model's AUC published next to it.
8. **Write back the analytical result** so `example_answers` on business PS carry
   validated ground truth the way the legacy PS do.

---

## 14. How to run

```bash
cd "/Users/aarushiarora/Desktop/hsbc canara/dataset-v3"

# App (Business Analyst mode is the default; classic trace is a sidebar toggle)
streamlit run dashboard.py            # → http://localhost:8501

# Tests
python3 -m pytest regression_test.py test_business_analytics.py -v

# Held-out business evaluation
python3 business_eval.py --verbose

# Rebuild the business PS library (validates against the metric registry)
python3 build_business_ps_all.py
python3 build_business_eval.py

# Ask from the CLI
python3 business_engine.py "Why is Whole Life performing poorly?"
```

Programmatic use:

```python
import business_engine as be
eng = be.get_engine()
r = eng.ask("Which products should management prioritize for retention?")

r.route              # 'business_analysis'
r.tier               # 'multi_step_business'
r.plan.describe()    # the analytical plan, before results
r.step_results       # per-step StepResult (op, params, result frame, ok/error)
r.answer.to_dict()   # six sections + tables + metric definitions
print(r.answer.to_markdown())
```

Optional LLM narrative (wording only, numbers unchanged):

```bash
export GROQ_API_KEY="gsk_..."     # or XAI_API_KEY="xai-..."
```
