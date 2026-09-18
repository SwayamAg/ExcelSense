# Excel_sense_v2 — Deployment Guide

This document details the deployment, configuration, containerization, and operations of the **Excel_sense_v2** life insurance analytics platform powered by Groq and deterministic computation.

---

## 1. Overview & Architecture

Excel_sense_v2 operates as an evidence-first, deterministic analytical system:
- **Data Ingestion**: Reads 5 life insurance CSV files into memory with strict typing and data-quality contracts.
- **Analytical Recipes**: 103 problem statements across 20 domains, applying 18 formulas and 15 business rules.
- **Evidence Generation**: Computes exact statistics, counts, baselines, and distributions deterministically without LLM assistance.
- **Summarization & Reasoning**: **Groq API** (`llama-3.3-70b-versatile`) summarizes the computed evidence with ultra-low latency; if no API key is provided, the built-in deterministic composer generates complete executive prose.
- **Grounding Validation**: Validates answers against the computed evidence and prevents PII leakage.

---

## 2. Prerequisites

- **Python**: Version 3.10 or higher.
- **Memory**: Minimum 512 MB RAM (the full 5-dataset portfolio loads in ~500 ms and consumes ~120 MB RAM).
- **Disk**: ~200 MB for application code, dependencies, and datasets.
- **Container Runtime** (Optional): Docker 20.10+ / Podman.
- **LLM API Key** (Optional): `GROQ_API_KEY` for Groq fast inference (or deterministic offline mode).

---

## 3. Configuration & Environment Variables

Create a `.env` file in the project root based on `.env.example`:

```bash
cp .env.example .env
```

### Configuration Parameters

| Variable | Description | Default | Required? |
|---|---|---|---|
| `GROQ_API_KEY` | Groq API key for fast Llama 3.3 70B summarization | `""` | No (falls back to deterministic composer) |
| `GROQ_MODEL` | Groq model identifier | `llama-3.3-70b-versatile` | No |
| `TALK_TO_DATA_DISABLE_LLM` | Force deterministic mode even if key exists (`1` or `0`) | `0` | No |
| `STREAMLIT_SERVER_PORT` | Port for the Streamlit web dashboard | `8501` | No |
| `STREAMLIT_SERVER_ADDRESS` | Network interface address to bind | `0.0.0.0` | No |
| `STREAMLIT_SERVER_HEADLESS` | Run Streamlit in headless server mode | `true` | No |
| `TALK_TO_DATA_DIR` | Path to the directory containing the 5 CSVs | `data/csv` | No |
| `TALK_TO_DATA_CACHE_ENABLED`| Enable query result caching (`1` or `0`) | `1` | No |
| `LOG_LEVEL` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `INFO` | No |

---

## 4. Local Deployment

### 4.1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4.2. Configure API Key

Edit `.env` and add your Groq API key:

```bash
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 4.3. Verify System Health

Run the pre-flight health check to validate datasets, catalogues, rules, router, and LLM readiness:

```bash
python -m src.api.health
```

Expected output:
```text
✅ System Status: HEALTHY (45.2 ms)
--------------------------------------------------
  ✅ Dataset      policy=ready, owner=ready, sales=ready, claims=ready, persistency=ready
  ✅ Contract     tables_count=5, status=valid
  ✅ Catalogue    ps_count=103, formulas_count=18, rules_count=15, status=ready
  ✅ Router       sample_route_outcome=routed, sample_matched_ps=PS-035, status=ready
  ✅ Llm          provider=groq, model=llama-3.3-70b-versatile, is_available=True, mode=live_llm
```

### 4.4. Run Web Interface

```bash
streamlit run app.py
```
Access the application at `http://localhost:8501`.

### 4.5. Run CLI Interface

Execute a one-shot query:
```bash
python app.py "What is the portfolio lapse rate?"
```

Or start an interactive terminal session:
```bash
python app.py
```

---

## 5. Docker Deployment

### 5.1. Build the Docker Image

```bash
docker build -t excel-sense-v2:latest .
```

### 5.2. Run the Container

**Standard Mode (with Groq API Key):**
```bash
docker run -d \
  --name excel-sense \
  -p 8501:8501 \
  -e GROQ_API_KEY="your-groq-api-key" \
  excel-sense-v2:latest
```

**Air-Gapped / Fully Offline Deterministic Mode:**
```bash
docker run -d \
  --name excel-sense-offline \
  -p 8501:8501 \
  -e TALK_TO_DATA_DISABLE_LLM=1 \
  excel-sense-v2:latest
```

**Mounting Custom Datasets:**
```bash
docker run -d \
  --name excel-sense-custom \
  -p 8501:8501 \
  -v /path/to/my/csvs:/app/data/csv:ro \
  excel-sense-v2:latest
```

### 5.3. Docker Compose Example

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  analytics-ui:
    image: excel-sense-v2:latest
    build: .
    container_name: excel_sense_v2
    ports:
      - "8501:8501"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY:-}
      - GROQ_MODEL=${GROQ_MODEL:-llama-3.3-70b-versatile}
      - TALK_TO_DATA_DISABLE_LLM=${TALK_TO_DATA_DISABLE_LLM:-0}
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
      - STREAMLIT_SERVER_HEADLESS=true
    volumes:
      - ./data/csv:/app/data/csv:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-m", "src.api.health", "--quiet"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s
```

Run with:
```bash
docker compose up -d
```

---

## 6. Security, Compliance & Production Hardening

1. **Non-Root Execution**: The Docker container executes under dedicated user `appuser` (UID 1000).
2. **Zero Training / Zero Data Ingestion into LLMs**: Datasets are never sent to external LLM providers for embedding or fine-tuning. Only computed aggregate evidence is sent for text formatting.
3. **Personal Data Withholding**: Owner personal names and unmasked identifiers are blocked at the data access layer by business rule `BR-006` and stripped before output.
4. **Prompt Injection Defense**: Input questions undergo multi-tier screening before routing; adversarial instructions are blocked with 0.07 ms latency.
5. **Auditing**: Turns are audited with hashed session keys and query metadata in compliance with enterprise audit requirements.

---

## 7. Operational Runbook & Troubleshooting

### Issue: Container health check is failing
- Run `docker exec -it <container_id> python -m src.api.health` to inspect individual component diagnostics.
- Ensure all five CSV files exist in `data/csv/` with non-zero size.

### Issue: LLM API latency or timeout
- The system automatically catches API errors and falls back to the deterministic composer without user disruption.
- To eliminate external network dependencies entirely, set `TALK_TO_DATA_DISABLE_LLM=1`.

### Issue: Memory consumption
- The application uses vectorized pandas operations and fingerprinted caches. Memory usage is bounded and remains under 200 MB for typical concurrent sessions.
