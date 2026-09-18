# Talk-to-Data Life Insurance PoC --- Phasewise Development Specification

## 0. Purpose

Build an evidence-backed, executive-focused Talk-to-Data chatbot over
exactly five life-insurance CSV datasets:

-   `Policy_Details.csv`
-   `Owner_Details.csv`
-   `Sales_Details.csv`
-   `Claims_Details.csv`
-   `Persistency_Details.csv`

The five datasets are the source of truth. `Policy_Details.csv` is the
central policy/application table. The documented join keys are:

-   Policy ↔ Owner: `AppID`
-   Policy ↔ Claims: `AppID`
-   Policy ↔ Sales: `SALES_ID`
-   Policy ↔ Persistency: `AppID`

The system must demonstrate accurate schema interpretation, relationship
handling, data-quality awareness, and explainable insights.

## 1. Core Architectural Rule

**Do NOT train, fine-tune, or otherwise train an LLM on the CSV data.**

The application itself must contain the deterministic intelligence:

``` text
User question
    ↓
Intent / Problem Statement identification
    ↓
Hardcoded Problem Statement / analytical recipe
    ↓
Deterministic retrieval + joins + filtering + aggregation
    ↓
Hardcoded business rules / formulas
    ↓
Evidence + calculations + comparisons
    ↓
Optional table / graph evidence
    ↓
LLM API
    ↓
Short executive explanation
```

The LLM is used only for reasoning over supplied evidence and concise
natural-language summarization. It must never be treated as the source
of factual data.

If evidence is insufficient, the system must say so instead of guessing.

------------------------------------------------------------------------

# 2. Development Principles

1.  **Dataset-first:** inspect the actual five CSVs before implementing
    assumptions.
2.  **Problem-statement-first:** create and approve the new PS catalogue
    before building the full chatbot.
3.  **Deterministic retrieval:** facts must come from CSV data through
    explicit rules.
4.  **Deterministic calculations:** formulas are implemented explicitly;
    the LLM does not perform authoritative calculations.
5.  **Evidence-first answers:** every substantive answer should be
    traceable to source records and/or calculated evidence.
6.  **Explainable reasoning:** distinguish observed facts, calculated
    findings, comparisons, and unsupported hypotheses.
7.  **No hallucination:** never invent missing fields, relationships,
    causes, or business facts.
8.  **Scope control:** do not create PSs that require data absent from
    the five CSVs.
9.  **Executive UX:** final answers must be short, crisp, direct, and
    business-friendly.
10. **Phase gates:** do not proceed to the next phase until the current
    phase's tests pass.
11. **Agent-owned status:** the coding agent must maintain the project
    checklist/status; the user should not manually update it.
12. **Regression protection:** every completed phase must retain its
    tests and pass all previous tests.

------------------------------------------------------------------------

# 3. Reference Material for Problem Statement Development

Use the supplied reference files as **design references**, not as
specifications to copy blindly:

-   `problem_statements.json`
-   `business_problem_statements.json`
-   `retrieval_demo.json`
-   `PS.md`

The references demonstrate both granular query capabilities and broad
business-analysis capabilities, including counts, filtering,
aggregation, ranking, comparisons, trends, portfolio health, risk,
lifecycle, financial analysis, customer analysis, distribution, product
analysis, integrity/compliance, and business strategy.

The existing `PS.md` contains 136 PSs across 60 problem types. Use its
breadth and analytical sophistication as a benchmark, but regenerate the
catalogue around the actual five-CSV schema.

## Critical PS-generation rule

Create a **new** Problem Statement catalogue.

Do NOT:

-   copy the old PSs and rename them;
-   retain concepts simply because they appear in the reference files;
-   invent columns or tables;
-   assume undocumented relationships;
-   preserve business concepts that cannot be answered from the five
    CSVs.

Instead:

``` text
Reference PSs
    ↓
Identify reusable analytical patterns
    ↓
Inspect actual five-CSV schema
    ↓
Map supported concepts
    ↓
Remove unsupported concepts
    ↓
Adapt / combine / split / create new PSs
    ↓
Validate each PS against actual data
    ↓
Generate new PS catalogue
```

The new catalogue should have **broad domain coverage and roughly
comparable richness/scale** to the references, while being genuinely
grounded in the five datasets.

------------------------------------------------------------------------

# 4. Required Problem Statement Capabilities

The new PS catalogue should cover, where the data supports them:

-   Direct lookup
-   Count
-   Filter
-   Group-by
-   Aggregation
-   Average / minimum / maximum
-   Ratio / percentage
-   Ranking / Top-N / Bottom-N
-   Comparison
-   Difference vs baseline
-   Trend / time series
-   Cohort analysis
-   Distribution analysis
-   Concentration analysis
-   Outlier / anomaly analysis
-   Multi-table joins
-   Reconciliation / integrity checks
-   Derived metrics
-   Formula-based calculations
-   Policy-level diagnostics
-   Claim analysis
-   Persistency analysis
-   Lapse analysis
-   Renewal analysis
-   Agent / sales performance
-   Product / plan analysis
-   Customer / owner analysis
-   Geographic / channel analysis
-   Portfolio-level analysis
-   Executive assessment
-   Evidence-backed "why" analysis
-   Evidence-backed "how" analysis
-   "Who performs best/worst" analysis
-   Peer comparison
-   Performance-driver analysis
-   Data-supported root-cause-style analysis

Only include a capability when it can be executed from the actual
schema.

------------------------------------------------------------------------

# 5. Required PS JSON Structure

Each PS should contain, where applicable:

``` json
{
  "problem_id": "BPS-001",
  "problem_statement": "...",
  "intent_type": "...",
  "business_intent": "...",
  "business_domain": "...",
  "business_objective": "...",
  "analysis_tier": "...",
  "example_queries": [],
  "relevant_tables": [],
  "relevant_columns": [],
  "required_operations": [],
  "required_metrics": [],
  "relationships": [],
  "concepts": [],
  "analytical_steps": [],
  "expected_answer_type": "...",
  "answer_template": "...",
  "parameters": {},
  "caveat": "...",
  "evidence_requirements": {},
  "visualization_recommended": false,
  "grounded_in": "Talk_to_Data_PoC_Documentation_Metadata_V4"
}
```

The exact schema may be extended if useful, but unnecessary fields must
not be invented merely for complexity.

------------------------------------------------------------------------

# 6. "Why" and "How" Analysis Rules

For diagnostic questions, the PS must specify the analytical path needed
to establish the answer.

Example:

> Why does Agent A have the highest persistency?

The system should be able to:

1.  Identify the KPI and eligible population.
2.  Calculate/rank agents.
3.  Identify the top performer.
4.  Establish a peer benchmark.
5.  Compare relevant observable metrics.
6.  Identify statistically/business-relevant differences supported by
    the data.
7.  Retrieve supporting records/aggregates.
8.  State only evidence-supported explanations.
9.  Flag any causal explanation that cannot be established from the
    data.

Never convert correlation or observed difference into proven causation
without supporting evidence.

------------------------------------------------------------------------

# 7. Evidence Contract

Every analytical execution should produce an internal evidence package
containing, as applicable:

``` text
question
detected_problem_id
tables_used
columns_used
filters
join_path
retrieved_records_or_aggregates
calculations
formulas_applied
business_rules_applied
comparison_baseline
ranking
supporting_metrics
data_quality_notes
limitations
```

The evidence package is the factual contract passed to the LLM.

The LLM must not receive an instruction to independently "search" or
invent missing facts.

------------------------------------------------------------------------

# 8. Executive Answer Contract

Default response structure:

``` text
Direct answer.
Key reason / finding.
Supporting number(s) or comparison.
```

For analytical questions:

``` text
Conclusion
Why / observable driver
Evidence
```

For rankings:

``` text
Leader + KPI
Peer comparison
Key evidence
```

For diagnostics:

``` text
What happened
Why the data supports it
Evidence
What cannot be concluded
```

Keep responses concise unless the user explicitly asks for detail.

Where appropriate, provide a compact evidence table or graph in addition
to the text.

------------------------------------------------------------------------

# 9. Phasewise Development Plan

## PHASE 0 --- Project Reconnaissance and Dataset Contract

### Objective

Understand the actual five CSVs and establish the canonical schema.

### Agent tasks

-   Inspect all five CSVs.
-   Validate filenames.
-   Record row/column counts.
-   Inspect data types.
-   Identify primary/important keys.
-   Identify nulls and duplicate keys.
-   Validate documented join relationships.
-   Detect actual categorical values.
-   Identify numeric/date fields.
-   Identify columns present in the actual files but absent from the
    metadata.
-   Identify metadata claims that require data-quality caution.
-   Produce a canonical machine-readable schema/data contract.

### Deliverables

-   `config/data_contract.json`
-   `docs/DATA_PROFILE.md`
-   `tests/test_data_contract.py`

### Tests

-   All five files load.
-   Expected files exist.
-   Schema is readable.
-   Key fields exist.
-   Join coverage is measured.
-   Duplicate/null checks are reported.
-   No unsupported assumptions are silently introduced.

### Gate

**PASS only when the five CSVs can be loaded and their relationships are
validated.**

------------------------------------------------------------------------

## PHASE 1 --- Problem Statement Generation

### Objective

Generate the new domain-specific PS catalogue before building the
application.

### Agent tasks

-   Read the reference PS JSONs and `PS.md`.
-   Extract analytical patterns and domain coverage.
-   Compare them against the canonical five-CSV schema.
-   Generate a new PS catalogue.
-   Ensure broad coverage without copying unsupported concepts.
-   Ensure every PS has a feasible retrieval/analysis path.
-   Add "why/how" diagnostic PSs where evidence can support them.
-   Add ranking, comparison, trend, anomaly, concentration and
    multi-step PSs where feasible.
-   Mark visualization recommendations.
-   Add caveats/limitations where required.

### Deliverables

-   `config/problem_statements.json`
-   `docs/PROBLEM_STATEMENT_CATALOGUE.md`
-   `tests/test_problem_statements.py`

### Tests

Every PS must pass:

-   table existence validation;
-   column existence validation;
-   relationship validation;
-   operation validation;
-   metric validation;
-   required-field validation;
-   unsupported-domain check;
-   duplicate problem-ID check;
-   example-query coverage check;
-   analytical-step consistency check.

### Additional quality checks

The agent must identify:

-   PSs that are duplicates or near-duplicates;
-   PSs that are too vague;
-   PSs requiring unavailable data;
-   PSs claiming causality without evidence;
-   PSs whose formulas are undefined;
-   PSs whose requested visualization cannot be generated.

### Gate

**Do not begin chatbot implementation until this catalogue passes
validation.**

------------------------------------------------------------------------

## PHASE 2 --- Deterministic Data Layer

### Objective

Build the reliable data access layer.

### Agent tasks

-   Load CSVs safely.
-   Normalize column access without changing source semantics.
-   Implement typed field access.
-   Implement filtering.
-   Implement grouping.
-   Implement aggregation.
-   Implement sorting/ranking.
-   Implement date operations.
-   Implement joins using documented keys.
-   Preserve source record identifiers.
-   Add data-quality warnings.

### Deliverables

-   `src/data/`
-   `src/data/loaders.*`
-   `src/data/query_engine.*`
-   `tests/data/`

### Tests

-   Unit tests for every operation.
-   Join tests.
-   Null handling tests.
-   Duplicate handling tests.
-   Numeric/date parsing tests.
-   Regression tests using known dataset values.

### Gate

**All data-layer tests pass with zero silent data-loss errors.**

------------------------------------------------------------------------

## PHASE 3 --- Formula and Business Rule Engine

### Objective

Make all authoritative calculations deterministic.

### Agent tasks

-   Create a registry for formulas.
-   Create a registry for business rules.
-   Map PSs to formulas/rules.
-   Implement explicit rounding/denominator/null semantics.
-   Keep formulas versioned and inspectable.
-   Never delegate authoritative calculations to the LLM.

### Deliverables

-   `config/formulas.json`
-   `config/business_rules.json`
-   `src/logic/formula_engine.*`
-   `src/logic/rule_engine.*`
-   `tests/logic/`

### Tests

-   Formula unit tests.
-   Edge-case tests.
-   Zero-denominator tests.
-   Null tests.
-   Boundary tests.
-   Cross-check calculations against manually verified examples.

### Gate

**Every formula used by an active PS has a deterministic implementation
and passing tests.**

------------------------------------------------------------------------

## PHASE 4 --- Problem Statement Execution Engine

### Objective

Turn a selected PS into an executable analytical workflow.

### Agent tasks

-   Build PS registry.
-   Match PS → required data → operations → formulas → evidence.
-   Execute analytical steps in declared order.
-   Support multi-table analysis.
-   Support ranking and peer comparison.
-   Support baseline comparison.
-   Support diagnostic analysis.
-   Produce structured evidence.
-   Fail safely when required evidence is unavailable.

### Deliverables

-   `src/ps/`
-   `src/analysis/`
-   `tests/ps/`
-   `tests/analysis/`

### Tests

For every PS category:

-   direct lookup;
-   aggregation;
-   ratio;
-   ranking;
-   comparison;
-   time series;
-   multi-table;
-   diagnostic;
-   anomaly;
-   executive analysis.

At least one executable test per PS, with expanded coverage for
high-value/multi-step PSs.

### Gate

**The agent must demonstrate that a PS produces reproducible evidence
from the CSVs without LLM assistance.**

------------------------------------------------------------------------

## PHASE 5 --- Query Understanding and PS Routing

### Objective

Map natural-language questions to the correct PS.

### Agent tasks

-   Build query normalization.
-   Build intent/PS matching.
-   Support aliases and business terminology.
-   Support entity extraction:
    -   AppID
    -   ClientID
    -   SALES_ID
    -   product/plan
    -   channel
    -   branch
    -   dates
    -   statuses
-   Handle ambiguity.
-   Support multiple candidate PSs.
-   Add confidence thresholds.
-   Refuse unsupported questions rather than forcing a bad PS match.

### Deliverables

-   `src/router/`
-   `tests/router/`

### Tests

Create a query suite containing:

-   direct questions;
-   paraphrases;
-   executive language;
-   "why" questions;
-   "how" questions;
-   ranking questions;
-   follow-up questions where supported;
-   ambiguous questions;
-   unsupported/out-of-domain questions.

### Gate

**Routing accuracy must meet the project-defined acceptance threshold,
with no unsafe fallback to an unrelated PS.**

------------------------------------------------------------------------

## PHASE 6 --- Evidence and Visualization Layer

### Objective

Make answers auditable and visually understandable.

### Agent tasks

-   Build evidence tables.
-   Preserve source references/record IDs.
-   Build compact analytical tables.
-   Generate graphs when useful.
-   Select visualization type based on PS metadata.
-   Ensure graphs are derived from calculated evidence, never invented.
-   Provide graph-ready structured data.

### Deliverables

-   `src/evidence/`
-   `src/visualization/`
-   `tests/evidence/`
-   `tests/visualization/`

### Tests

-   Evidence matches source data.
-   Calculated values match formula engine.
-   Graph values equal evidence values.
-   No graph is generated from unsupported data.
-   Small-sample warnings are preserved.

### Gate

**Every supported analytical visualization must be reproducible from the
evidence package.**

------------------------------------------------------------------------

## PHASE 7 --- LLM Reasoning and Executive Summarization

### Objective

Integrate the connected LLM API only as the final
reasoning/summarization layer.

### Agent tasks

-   Define strict LLM prompt contract.
-   Pass only question + structured evidence + relevant business
    context.
-   Instruct LLM not to invent facts.
-   Instruct LLM to distinguish:
    -   observed;
    -   calculated;
    -   inferred;
    -   unsupported.
-   Keep answers concise.
-   Preserve numbers exactly.
-   Do not allow LLM-generated formulas to override deterministic
    calculations.
-   Add fallback behavior for LLM failures.

### Deliverables

-   `src/llm/`
-   `config/llm_prompts.*`
-   `tests/llm/`

### Tests

-   Evidence fidelity.
-   Number fidelity.
-   No unsupported claims.
-   No fabricated records.
-   No formula substitution.
-   Concision.
-   Correct handling of insufficient evidence.
-   API timeout/error fallback.

### Gate

**The same evidence must produce factually consistent answers even if
the wording changes.**

------------------------------------------------------------------------

## PHASE 8 --- Conversational Chatbot

### Objective

Combine routing, analysis, evidence, visualization and LLM summarization
into one user-facing chatbot.

### Agent tasks

Implement:

-   question input;
-   PS routing;
-   deterministic execution;
-   evidence generation;
-   LLM summarization;
-   table output;
-   graph output;
-   error handling;
-   unsupported-question handling;
-   conversation context where safe and useful.

### Deliverables

-   `src/api/` or application backend
-   `src/ui/`
-   integration tests
-   end-to-end tests

### Tests

End-to-end scenarios:

1.  Policy lookup.
2.  Claim lookup.
3.  Lapse diagnosis.
4.  Persistency lookup.
5.  Agent ranking.
6.  Agent peer comparison.
7.  "Why is this agent performing better?"
8.  Product comparison.
9.  Channel comparison.
10. Customer-level question.
11. Multi-table question.
12. Formula-based question.
13. Unsupported question.
14. Insufficient-evidence question.

### Gate

**All critical end-to-end scenarios pass.**

------------------------------------------------------------------------

## PHASE 9 --- Evidence, Security and Data-Governance Hardening

### Objective

Make the PoC trustworthy and safe for executive use.

### Agent tasks

-   Prevent arbitrary file access.
-   Restrict analysis to the five datasets.
-   Protect sensitive owner/customer fields.
-   Avoid exposing unnecessary personal data.
-   Add logging without leaking sensitive values.
-   Validate all query parameters.
-   Add prompt-injection defenses around user input.
-   Ensure user text cannot alter system rules or formulas.
-   Ensure the LLM cannot override deterministic evidence.

### Tests

-   Out-of-scope dataset access.
-   Prompt injection.
-   Formula manipulation attempts.
-   SQL/code injection if applicable.
-   Sensitive-field exposure.
-   Malformed query.
-   Missing data.
-   Large/expensive query.
-   LLM malicious instruction handling.

### Gate

**No critical grounding, data-isolation, or rule-override
vulnerabilities remain.**

------------------------------------------------------------------------

## PHASE 10 --- Performance and Regression Testing

### Objective

Ensure the complete system remains reliable as the PS catalogue grows.

### Agent tasks

-   Build a permanent regression suite.
-   Benchmark representative queries.
-   Measure retrieval/execution latency.
-   Measure LLM latency separately.
-   Cache safe deterministic results where appropriate.
-   Detect PS regressions.
-   Detect schema regressions.
-   Detect formula regressions.

### Deliverables

-   `tests/regression/`
-   `tests/benchmark/`
-   `docs/TEST_REPORT.md`

### Gate

**All regression tests pass after every subsequent code change.**

------------------------------------------------------------------------

## PHASE 11 --- Deployment Readiness

### Objective

Prepare the PoC for deployment.

### Agent tasks

-   Environment configuration.
-   Secrets management.
-   Production configuration.
-   Health endpoint.
-   Startup validation.
-   Dataset validation on startup.
-   Logging.
-   Error monitoring hooks.
-   Docker/container configuration if applicable.
-   Deployment documentation.

### Deliverables

-   deployment configuration
-   `Dockerfile` if applicable
-   `.env.example`
-   `docs/DEPLOYMENT.md`
-   production test suite

### Tests

-   Clean-environment startup.
-   Missing-secret handling.
-   Missing-CSV handling.
-   Schema mismatch handling.
-   LLM API failure.
-   Health check.
-   Full end-to-end smoke test.

### Gate

**A clean environment can start the application and successfully execute
the critical test suite.**

------------------------------------------------------------------------

# 10. Agent-Owned Project Checklist

**IMPORTANT:** This checklist must be maintained and updated by the
coding agent only.

The user should not manually mark phases complete.

The agent must update:

`PROJECT_STATUS.md`

after every meaningful phase, test run, or architectural change.

Use this format:

``` markdown
# Project Status

Last updated: <timestamp>
Current phase: <phase>
Overall status: IN_PROGRESS | BLOCKED | READY_FOR_REVIEW | COMPLETE

## Phase Checklist

- [ ] Phase 0 — Dataset Contract
- [ ] Phase 1 — Problem Statement Catalogue
- [ ] Phase 2 — Deterministic Data Layer
- [ ] Phase 3 — Formula & Business Rule Engine
- [ ] Phase 4 — PS Execution Engine
- [ ] Phase 5 — Query Understanding & Routing
- [ ] Phase 6 — Evidence & Visualization
- [ ] Phase 7 — LLM Reasoning & Summarization
- [ ] Phase 8 — Conversational Chatbot
- [ ] Phase 9 — Security & Grounding Hardening
- [ ] Phase 10 — Performance & Regression
- [ ] Phase 11 — Deployment Readiness

## Current Phase

### Completed
- ...

### In Progress
- ...

### Blocked
- ...

### Tests
- Passed: ...
- Failed: ...
- Skipped: ...

### Known Limitations
- ...

### Next Action
- ...
```

A phase may be checked only when:

1.  Its implementation is complete.
2.  Its required tests pass.
3.  Its deliverables exist.
4.  No critical blocker remains.
5.  The agent has recorded the result in `PROJECT_STATUS.md`.

------------------------------------------------------------------------

# 11. Feature Checklist

The agent must additionally maintain a feature checklist in
`PROJECT_STATUS.md`:

## Data

-   [ ] Five CSV ingestion
-   [ ] Schema validation
-   [ ] Data profiling
-   [ ] Null handling
-   [ ] Duplicate handling
-   [ ] Join validation
-   [ ] Data-quality warnings

## Problem Statements

-   [ ] New PS catalogue generated
-   [ ] Reference PS patterns studied
-   [ ] Unsupported PSs removed
-   [ ] PSs mapped to actual columns
-   [ ] PSs mapped to actual relationships
-   [ ] Direct lookup PSs
-   [ ] Calculation PSs
-   [ ] Ranking PSs
-   [ ] Comparison PSs
-   [ ] Trend PSs
-   [ ] Diagnostic/Why PSs
-   [ ] How/driver PSs
-   [ ] Multi-step business PSs
-   [ ] Evidence requirements
-   [ ] Visualization metadata

## Deterministic Engine

-   [ ] Filtering
-   [ ] Grouping
-   [ ] Aggregation
-   [ ] Ranking
-   [ ] Joins
-   [ ] Date analysis
-   [ ] Ratios
-   [ ] Baselines
-   [ ] Comparisons
-   [ ] Outliers
-   [ ] Concentration
-   [ ] Reconciliation
-   [ ] Formula engine
-   [ ] Business rule engine

## Grounding

-   [ ] Evidence package
-   [ ] Source record traceability
-   [ ] No unsupported claims
-   [ ] Insufficient-evidence response
-   [ ] No LLM calculation authority
-   [ ] No LLM data-source authority
-   [ ] No hallucinated fields
-   [ ] No hallucinated records

## LLM

-   [ ] API integration
-   [ ] Evidence-only prompt
-   [ ] Executive summarization
-   [ ] Number fidelity
-   [ ] Causal-claim safeguards
-   [ ] Error fallback
-   [ ] Token/latency control

## Analytics / Visualization

-   [ ] Evidence tables
-   [ ] Ranking charts
-   [ ] Comparison charts
-   [ ] Trend charts
-   [ ] Distribution charts
-   [ ] Diagnostic comparison visuals
-   [ ] Graph data matches evidence

## Chatbot

-   [ ] Natural-language input
-   [ ] PS routing
-   [ ] Entity extraction
-   [ ] Ambiguity handling
-   [ ] Unsupported-question handling
-   [ ] Multi-table questions
-   [ ] Follow-up questions
-   [ ] Executive response formatting

## Quality

-   [ ] Unit tests
-   [ ] Integration tests
-   [ ] PS execution tests
-   [ ] Routing tests
-   [ ] LLM grounding tests
-   [ ] Visualization tests
-   [ ] End-to-end tests
-   [ ] Regression tests
-   [ ] Performance tests
-   [ ] Security tests

## Deployment

-   [ ] Configuration management
-   [ ] Secrets management
-   [ ] Startup validation
-   [ ] Health check
-   [ ] Deployment documentation
-   [ ] Clean-environment test
-   [ ] Production smoke test

------------------------------------------------------------------------

# 12. Mandatory Agent Behavior

The coding agent must:

### Before coding

-   Inspect the repository.
-   Inspect all five CSVs.
-   Inspect the supplied reference PS files.
-   Establish the canonical schema.
-   Generate the new PS catalogue.
-   Test the catalogue.
-   Only then begin application development.

### During coding

After each phase:

``` text
Implement
→ Test
→ Fix failures
→ Re-test
→ Update PROJECT_STATUS.md
→ Proceed only if gate passes
```

### If a requirement cannot be implemented

Do NOT silently approximate it.

Record:

``` text
BLOCKED:
Reason:
Affected phase:
Affected PS:
Available alternatives:
```

### If the dataset does not support a requested question

The system must return a grounded limitation rather than inventing an
answer.

------------------------------------------------------------------------

# 13. Final Acceptance Criteria

The project is complete only when all of the following are true:

-   [ ] Five CSVs are correctly loaded and validated.
-   [ ] Actual schema and joins are documented.
-   [ ] New PS catalogue is generated from the references but is not a
    copy of them.
-   [ ] PS catalogue is demonstrably grounded in the five datasets.
-   [ ] Unsupported domains are excluded.
-   [ ] PSs can express What, Who, Why, How and How-much questions where
    data supports them.
-   [ ] Deterministic retrieval works.
-   [ ] Deterministic formulas work.
-   [ ] Business rules work.
-   [ ] Multi-table analysis works.
-   [ ] Rankings and comparisons work.
-   [ ] Evidence is preserved.
-   [ ] Graphs/tables are derived from evidence.
-   [ ] LLM receives evidence rather than raw authority over the
    dataset.
-   [ ] LLM cannot override deterministic calculations.
-   [ ] Unsupported conclusions are explicitly identified.
-   [ ] Executive answers are concise.
-   [ ] End-to-end tests pass.
-   [ ] Regression tests pass.
-   [ ] Security/grounding tests pass.
-   [ ] Deployment smoke test passes.
-   [ ] `PROJECT_STATUS.md` accurately reflects the final state.

# 14. Definition of Done

The PoC is considered successful when an executive can ask a
natural-language question about the five life-insurance datasets and
receive:

``` text
A concise answer
+
The relevant calculated evidence
+
A supporting table/graph when useful
+
A transparent limitation when the data cannot establish the answer
```

while all factual conclusions remain grounded in the five CSVs and their
explicitly defined rules/formulas.

**Core principle:**

> The application determines the facts.\
> The rule/formula engine determines the calculations.\
> The evidence layer proves the answer.\
> The LLM explains the answer.
