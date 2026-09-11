# System Documentation: Third Draft (5-Table Production Architecture)

## 1. System Architecture

Third Draft provides an enterprise-ready intelligence and graph reasoning pipeline built strictly on the **5 Canara HSBC Life Insurance CSV tables**:
1. `Policy_Details.csv`
2. `Owner_Details.csv`
3. `Persistency_Details.csv`
4. `Sales_Details.csv`
5. `Claims_Details.csv`

---

## 2. Directory Layout & Module Responsibilities

```
ThIrD_dRAfT/
├── app.py                     # Unified Streamlit entry point
├── README.md                  # Project overview & running instructions
├── requirements.txt           # Python dependencies
├── data/
│   ├── csv/                   # The 5 production CSVs + Metadata V4 docx
│   ├── history/               # Persistent query history (query_history.json)
│   └── problem_statements/    # 491 active PS catalog (270 lookup + 221 multi-step)
├── docs/                      # Data dictionary & relational model documentation
├── src/
│   ├── analytics/             # Deterministic analytics, fact tables, graph engine
│   ├── engine/                # Business engine coordinator & query workflow
│   ├── retrieval/             # 5-signal hybrid PS retriever & semantic intents
│   └── llm/                   # Local Ollama client (qwen2.5:14b inference)
├── ui/                        # Streamlit dashboard & UI components
├── builders/                  # PS normalization & data indexing scripts
└── tests/                     # Deterministic unit & regression test suite
```

---

## 3. Query Execution Lifecycle

1. **Entity Extraction**: `query_workflow.py` scans user queries for `AppID`, `ClientID`, `SALES_ID`, `Claim_No`, `Plan.Name`, `State`, etc.
2. **PS Retrieval**: `business_retrieval.py` searches the 491 PS library using lexical, expanded, concept, intent, and operational signal weights.
3. **Analytical Planning**: `analysis_planner.py` constructs sequential deterministic operations.
4. **Execution**: `analytics_core.py` runs vectorized Pandas computations over the 5 fact tables.
5. **Presentation & LLM Polish**: `answer_composer.py` structures the evidence, and `ollama_client.py` synthesizes executive insights locally.
6. **Local Caching**: `business_dashboard.py` saves output into `data/history/query_history.json`.
