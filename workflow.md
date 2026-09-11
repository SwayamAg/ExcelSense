# Workflow — Input to Output

> **Scope:** every stage a question passes through, from the text box to the
> rendered answer and back into the conversation state.
> **Grounding:** the five production CSVs in [`data/csv/`](data/csv/), described by
> `Talk_to_Data_PoC_Documentation_Metadata_V4.docx`. Nothing else is a source.

The single rule that shapes the whole design:

> **No language model ever produces a number.** Every figure in every answer is
> computed by `analytics_core` from the CSVs. The local model only decides *what
> to compute*, and its output is validated against the registries before it is
> trusted.

---

## 1. The ten stages

```
 [1] Question              chat_dashboard.render()
      │                    duplicate guard -> claim the question
      ▼
 [2] Resolution            conversation.FollowUpResolver
      │                    turn 1: rules · turn 2+: local model -> JSON
      │                    validate -> calibrate confidence -> ambiguity check
      ▼
 [3] Proof panel           exec_views.render_proof()
      │                    "How I read your question" (no numbers)
      ▼
 [4] Routing               7 branches, first match wins
      │
      ├─ refusal ─────────── out of scope / forward-looking
      ├─ clarification ───── confidence < 0.75 or data-backed ambiguity
      ├─ agent_improvement ─ exec_views.agent_book_contrast()
      ├─ agent_risk ──────── composite_score -> Methods 1 & 2
      ├─ agent_composition ─ one agent's own book
      ├─ entity_comparison ─ one agent vs peer cohort
      └─ engine ──────────── contextualise -> BusinessAnalyticsEngine.ask()
      │
      ▼
 [5] Computation           analytics_core: MetricSpec + 12 operations
      │                    cohort gating, min_n, degenerate-dimension guard
      ▼
 [6] Composition           answer_composer (engine route only)
      ▼
 [7] Render                headline -> evidence -> toggles
      ▼
 [8] Verification          V1 denominators · V3 decomposition · V4 source rows
      ▼
 [9] Persist               turn_state + render spec -> chat_history.json
      ▼
[10] Replay                every prior turn re-rendered on the next run
```

---

## 2. Stage detail

### [1] Question

`st.chat_input` in [`ui/chat_dashboard.py`](ui/chat_dashboard.py).

Streamlit re-runs the entire script on every interaction, and some paths
re-deliver the last input. The question is **claimed** in `session_state` before
any work begins:

```python
claimed = st.session_state.get("chat_last_processed")
if claimed is not None and claimed.strip() == question.strip():
    return                      # already answered
st.session_state["chat_last_processed"] = question
```

Without this, one question produced several turns and shifted what the next
follow-up inherited.

### [2] Resolution — the only stateful stage

`BusinessAnalyticsEngine.ask(question)` takes **one string and has no memory**.
"Why is *their* persistency low?" has no referent. [`src/engine/conversation.py`](src/engine/conversation.py)
supplies that layer and nothing else.

**Turn 1 — rules only.** No prior state to resolve, and skipping the model keeps
the opening question fast.

**Turn 2+ — local model, then validation.** The model receives the question, the
previous turn's state, and the allowed vocabulary (62 metrics, ~40 dimensions,
14 operations). It returns JSON:

```json
{ "entity_type": "agent", "entity_id": null, "inherit_entity": true,
  "intent": "comparison", "metric": "...", "dimension": "...", "op": "...",
  "interpretation": "<one sentence, no digits>", "ambiguity": null,
  "confidence": 0.0 }
```

Four guards sit between that JSON and the engine:

| Guard | What it stops |
|:---|:---|
| **Registry validation** | Any metric/dimension/op not in the registries is dropped and the turn marked degraded. Invalid JSON falls back to rules. |
| **Entity-type override** | A small model returns "Customer 800812" for what is an agent. When the id matches the one in focus, the *previous turn's* type wins — that came from data. |
| **Contrast detection** | "the **other** agents", "everyone else" points *away* from the focus entity. Inheritance is blocked, overriding the model's own `inherit_entity`. |
| **Confidence calibration** | Small models report `1.0` almost unconditionally, which would make the threshold dead code. Model confidence is a ceiling; deterministic penalties subtract for a metric or dimension nothing in the question or prior turn supports. |

**Ambiguity is checked against the data, not the model.** `check_dimension_ambiguity`
tests whether a breakdown that reads as singular actually is. It scans the
dimensions the *question* named as well as the one the model chose:

> Agent 800812 spans **9 branch values** across 45 policies; the largest holds 17.

So "their branch" is caught regardless of what the model picked.

### [3] Proof panel

`exec_views.render_proof()` shows how the question was read — pronoun → entity,
metric, dimension, confidence, and any ambiguity. The model-written
`interpretation` contains **no digits** (enforced: a sentence with digits is
discarded). A model error therefore shows as a visibly wrong interpretation the
user can correct, never as a wrong figure they can't detect.

### [4] Routing

Seven branches, first match wins, in this order:

| # | Branch | Trigger | Produces |
|---:|:---|:---|:---|
| 1 | `refusal` | forward-looking wording | scope answer + what *can* be shown |
| 2 | `clarification` | confidence < 0.75, or data-backed ambiguity | the question, restated + options |
| 3 | `agent_improvement` | agent subject + improve verb | `agent_book_contrast` |
| 4 | `agent_risk` | agent cue + superlative | Methods 1 & 2 + V4 |
| 5 | `agent_composition` | agent in focus + "sell/book/mix" | that agent's own breakdown |
| 6 | `entity_comparison` | agent in focus | 9-measure scorecard vs peers |
| 7 | `engine` | everything else | contextualise → planner |

**Why branches 3–6 exist.** The PS planner has no way to scope to one
`SALES_ID`, and it ranks on one metric. Left to it:
- *"how to improve persistency of sales persons"* returned a ranking of **plans**
- *"which plans do they sell most"* returned **persistency rate**, portfolio-wide

Both are category errors — answering a different question confidently.

**Refusal needs corroboration.** `intent: forecast` from the model is *not*
enough; the wording must support it, and explicit prescriptive phrasing ("how to
improve") wins outright. A 0.5B model labelled a prescriptive question as a
forecast, turning an answerable question into a refusal.

**Query rewriting** (branch 7 only). The stateless engine gets a bare "why is
that happening?" and retrieves against a question with no subject — it answered
about *surrender* rate when the prior turn was about *lapse*. `_contextualise()`
splices the previous turn's metrics and focus back in, and the rewritten question
is shown as *"Read as: …"*. It does **not** inject a metric when the question
names its own measure ("most", "how many", "share").

### [5] Computation — where every number comes from

[`src/analytics/analytics_core.py`](src/analytics/analytics_core.py).

**`MetricSpec`** declares numerator, denominator, population, unit and direction
per metric — so a wrong denominator is impossible rather than unlikely. `compute()`
returns the value **and** its numerator, denominator, `n` and `reliable` flag.

**Three refusal guards:**

- `RETIRED_METRICS` — a metric registered against a field this dataset lacks
  explains itself instead of raising `KeyError`. The registry subclasses `dict`
  so every existing call site gets the better error untouched.
- `RETIRED_DIMENSIONS` — `smoker` and `marital_status` were aliased onto
  **`gender`**; a breakdown "by smoker" was silently answered by gender.
- `DEGENERATE_DIMENSIONS` — `CHANNEL` is `Banca` for all 25,000 policies. A
  channel comparison returns one row that reads as a leaderboard, so
  `check_dimension()` refuses and says why.

**Cohort gating before scoring.** `composite_score` min-max normalises across
whatever cohort it is handed, so 1,485 one- and two-policy agents would rescale
every score around noise. `build_agent_cohort()` gates first, and the exclusions
are printed beside the chart:

> ranked over 471 agents · 1,485 excluded (book < 20) · 369 excluded (no rate on file)

### [6] Composition

`answer_composer` turns executed steps into headline, evidence, analysis,
implication, recommendation, caveats and data gaps. Every sentence is generated
from figures `analytics_core` already computed; the composer never estimates and
never asserts causation. Reliability and sample-size caveats attach automatically.

### [7] Render — executive shape

```
┌ HEADLINE ──── one sentence, the decision
├ EVIDENCE ──── 3 lines visible, rest behind "N more"
├ TOGGLES ───── score decomposition · quadrant · source rows · caveats
└ EXPORT ────── CSV per table
```

Every number carries **value + denominator + anchor**:

```
Persistency  78.70% (67 policies)  ·  peer median 83.00%  ▼ 4.30 pts  ⚠
```

### [8] Verification

| Level | What it shows | Status |
|:---|:---|:---|
| **V1** | inline denominators — `60.2% (27 of 45)` | live |
| **V3** | score decomposition: raw → normalised → weight → contribution | live |
| **V4** | the actual source rows + CSV | live |
| **V6** | `query_trace.step_table()` — op, params, source tables | built, not wired |
| **V5** | stated vs recomputed reconciliation | not built |

V3 discloses that min-max scaling is **relative to the ranked cohort**, so scores
move when the cohort changes even though no underlying data did.

V4 states what the rows do and don't prove: `First Year Persistency Rate` is
declared **once per agent** in `Sales_Details` and copied onto each policy row.
`n` counts policies, but the evidence is one stated figure — so the rows verify
book size, product mix and status, **not the rate itself**.

### [9] Persist

One record per turn, in the **same shape the classic dashboard writes** — `query`,
`route`, `headline`, `evidence`, `analysis`, `implication`, `recommendation`,
`caveats`, `unavailable`, `llm_narrative`, `legacy_output`, `tables` — plus the
transcript fields:

```json
{ "turn_id": 2, "parent_turn": 1,
  "resolved_from": "inherited: Agent 800812 from turn 1",
  "resolution_source": "llm", "confidence": 0.45,
  "ambiguity": "...", "degraded": false,
  "turn_state": { "focus_entity": {...}, "focus_metrics": [...],
                  "active_filters": {...}, "cohort": {...}, "ops_run": [...] },
  "render_kind": "agent_risk", "render_params": {"focus_id": "800812"} }
```

`resolved_from` is what lets the user see *why* the system thought "their" meant
800812 — and correct it when wrong.

**Two files, same format, different lifecycles:**
`chat_history.json` is the open thread; `chat_archive.json` holds past threads.
"Start a new conversation" archives the current one; the sidebar lists archives,
and clicking one restores both the transcript and the `turn_state` a follow-up
needs.

### [10] Replay

Streamlit re-runs the script on every interaction, so anything not re-emitted
disappears. `_replay_turn()` re-renders every prior turn from its recorded
`render_kind` + `render_params`, producing a real transcript:

```
Q1 → A1 + toggles
Q2 → A2 + tables
Q3 → …
```

Charts are **recomputed** from the stored focus rather than serialised — the fact
frames are cached and the computation is deterministic — which keeps the history
file small. Narrative answers replay from recorded text, since re-asking the
engine would be slow and could drift.

---

## 3. Worked example

**Turn 1** — *"tell me about the sales person having lowest persistency and highest claim rate"*

Rules (turn 1) → no entity → `agent_risk`. Cohort gated to 471; `composite_score`
on persistency (inverted) + early-claim.

> No single agent is both. Ranked on the two together, **Agent 800812** is the
> clearest case: 45 policies, persistency 60.2% against a peer median of 83.0%,
> early-claim rate 3.55%.

Toggles: score decomposition · quadrant · 45 source rows.
State: `focus_entity=agent 800812`, `focus_metrics=[persistency, early_claim]`.

**Turn 2** — *"how do they compare to others in their branch?"*

Model resolves; confidence damped `1.00 → 0.45`; entity inherited; data check
finds 9 branch values → **clarification, not an answer**:

> "their branch" presumes something the records do not support. Agent 800812
> spans 9 branch values across 45 policies; the largest holds 17 — below the 20
> needed to be reliable.

**Turn 3** — *"which plans do they sell most?"*

Entity still in focus, composition cue → `agent_composition`. No metric injected
(the question says "most"):

> Agent 800812 sells mostly **Flexi Edge - Flexi Savings** — 25 of 67 policies (37%).

---

## 4. What the pipeline will not do

- **Compute a number in a language model.** Structurally impossible: the model
  emits names, never values.
- **Break down by a single-valued column.** `CHANNEL`, `Mode` — refused with the reason.
- **Answer "by smoker", "by rider", "by surrender reason".** Those fields do not
  exist; the aliases that faked them are removed.
- **Predict.** No forward-looking data exists.
- **Claim cause.** No intervention is recorded anywhere — no training, activity,
  coaching or contact history. Agent findings are associations and say so.

---

## 5. Running it

```bash
streamlit run app.py          # Conversation (default) | Classic dashboard

python -m unittest discover -s tests -p "*.py"      # 40 tests
python -m unittest tests.validate_grounding         # schema/metric/dimension/PS gates
python builders/migrate_ps_to_v4_schema.py          # dry run: PS that no longer resolve
```

### Known state

| | |
|:---|---:|
| Tests passing | 40 / 40 |
| PS answerable | 445 / 491 (46 tagged `unsupported_reason`) |
| Ranked agent cohort | 471 of 1,956 |
| Local model | `qwen2.5:0.5b` — 7B+ recommended; the 0.5B needed extra guards |

### Open

- `driver_scan` sorts by **rate extremity, not contribution** — for "why"
  questions that ranks the most extreme segment, not the one moving the number.
- Dict-returning ops (`portfolio_overview`, `segment_profile`,
  `top_entity_concentration`) are dropped at the composer boundary — **17.6% of
  plan steps**, including the KPI block.
- `Sales_Details."Surrender Rate"` disagrees with a recomputation from
  `Policy_Details` for **250 of 1,956 agents**; needs someone who knows the
  source system. Agent 800812 is one of them.
