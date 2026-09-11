# Streamlit UI & Layout Specification — Conversation Mode

This document details the complete user interface layout, visual hierarchy, components, and interactive flow of **Conversation Mode** (`ui/chat_dashboard.py`) in ExcelSense.

---

## 1. Page Configuration & Global Theme

* **Page Title**: `Insurance Analytics — Conversation`
* **Page Icon**: `💬`
* **Layout**: `wide` (spans full viewport width)
* **Initial Sidebar State**: `expanded`
* **Base Theme**: Dark mode (`#0e1117` background, `#161b22` cards, `#e6edf3` typography, `#58a6ff` primary accents).

---

## 2. Global CSS Design Tokens & Component Styling

| CSS Class | Target Element | Visual Appearance | Purpose |
|---|---|---|---|
| `.ct-turn-q` | User Question Bubble | Background `#0f1b2d`, border `#1f6feb55`, **left border 4px `#58a6ff`**, rounded corners (`10px`), font-weight `600`. | Distinct user question card in the conversation feed. |
| `.ct-headline` | Standard Answer Headline | Background `#161b22`, border `#30363d`, rounded corners (`10px`), padding `16px 20px`, text color `#e6edf3`. | Executive takeaway and primary analytical finding. |
| `.ct-clarify` | Clarification Alert | Background `#1d1804`, border `#d2992255`, **left border 4px `#d29922` (Gold)**, text `#e3b341`. | When query is ambiguous or below confidence threshold. |
| `.ct-refuse` | Out-of-Domain / Refusal Alert | Background `#1d0d0d`, border `#f8514955`, **left border 4px `#f85149` (Red)**, text `#ff7b72`. | Out-of-domain requests, non-life insurance, or dataset gaps. |
| `.ct-metric` | Evidence / Analysis Bullet | Font-size `0.92rem`, text `#c9d1d9`, margin `2px 0`. | Structured analytical evidence points. |
| `.ct-metric b` | Metric Highlight | Text color `#58a6ff` (electric blue). | Highlights key numbers, percentages, and entity names. |

---

## 3. Sidebar Layout & Controls

The sidebar manages mode switching, LLM follow-up resolving, active chat session controls, and resumable past conversations.

```
┌──────────────────────────────────────────────┐
│  Mode Selector (Radio: Conversation/Classic) │
├──────────────────────────────────────────────┤
│  ### Conversation                            │
│  [x] Use local model to resolve follow-ups   │
│  (Caption: Local model status / Groq Cloud)  │
├──────────────────────────────────────────────┤
│  (Caption: "X turns in this conversation")    │
│  [ ➕ New Chat ]     [ 🗑️ Clear Chat ]      │
├──────────────────────────────────────────────┤
│  ### Past conversations        [ Clear All ] │
│  [ 💬 Which product has the hi... (4 turns) ] │
│  [ 💬 Agent 800812 risk deep-d... (2 turns) ] │
│  [ 💬 Western zone claims anal... (3 turns) ] │
├──────────────────────────────────────────────┤
│  [ ⚠️ Wipe All Chat History ]                │
└──────────────────────────────────────────────┘
```

### Sidebar Components:

1. **Follow-Up Resolver Toggle**:
   * Checkbox: `"Use local model to resolve follow-ups"`
   * Dynamic caption: displays active model (e.g. `Local model: qwen2.5:14b` or `Groq Cloud / Rule-based follow-ups`).
2. **Active Chat Actions (2-Column Layout)**:
   * **`➕ New Chat` Button**: Archives the current conversation thread to disk and initializes a clean conversation.
   * **`🗑️ Clear Chat` Button**: Resets the current chat thread immediately without saving to archive.
3. **Past Conversations (Resumable Archive List)**:
   * Header with **`Clear All`** button to wipe all archived sessions.
   * List of up to 10 recent threads showing the first query as the button label.
   * **Click to Resume**: Clicking any archived thread restores its full history, transcript, and inherited turn context into the active window.
4. **Danger Zone**:
   * **`⚠️ Wipe All Chat History` Button**: Full purge of both active session memory and all archived conversation files.

---

## 4. Main Stage Layout & Turn Structure

The main stage is structured chronologically. Every user question and assistant response forms a **Turn**.

```
┌────────────────────────────────────────────────────────────────────────────┐
│  ## Insurance Analytics — Conversation                                      │
│  Caption: "Every figure is computed by the analytics engine..."            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 👤 "Which products have the highest 13th-month lapse rate?"          │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 📌 Headline: Canara HSBC Guaranteed Fortune Plan leads on lapse...   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  ┌─────────────────────────────────┬────────────────────────────────────┐  │
│  │ 📊 Evidence                     │ 🔍 Analysis                        │  │
│  │ • 53.37% (0.0 pts above baseline)│ • Lapse rate defined as Lapsed/Tot  │ │
│  │ • Spread between best & worst   │ • Supporting metric: 6,942 policies│  │
│  ├─────────────────────────────────┼────────────────────────────────────┤  │
│  │ 💼 Business Implication         │ 💡 Recommendation                  │  │
│  │ • Differences of this size are  │ • Focus retention outreach before  │  │
│  │   worth prioritizing on large n │   13th-month renewal window.       │  │
│  └─────────────────────────────────┴────────────────────────────────────┘  │
│                                                                            │
│  ▼ 🧪 Analytical Caveats (Expander)                                         │
│  ▼ 📁 Underlying Data Tables (Expander: 3 rows)                            │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 👤 "Why does this product suffer from high lapse?"                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  ▼ 🔎 How I read your question (Proof / Interpretation Panel)               │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 📌 Headline: Driver decomposition for Guaranteed Fortune Plan...     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│  ...                                                                       │
├────────────────────────────────────────────────────────────────────────────┤
│  [ 💬 Ask a question, or follow up on the answer above...                ] │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Visual Turn Card Templates

### A. Standard Analytical Card (2-Column Grid)
* **Top Headline**: Full-width `.ct-headline` summarizing the verified finding.
* **Left Column**:
  * **Evidence Block**: Up to 3 bullet points with bold electric blue numbers + expandable `+X more`.
  * **Business Implication Block**: Actionable business significance grounded on sample size ($n$).
* **Right Column**:
  * **Analysis Block**: Definition of metric, calculation mechanics, and distribution bounds.
  * **Recommendation Block**: Prescriptive next steps for portfolio managers.
* **Expanders**:
  * `Analytical caveats`: Non-causality notes, sample size warnings.
  * `Not answerable from this dataset`: Explicit gaps identified during execution.
  * `Underlying Data Tables (N rows)`: Interactive Streamlit dataframes.

---

### B. Agent Risk Visual Dashboard Card
Triggered when queries focus on salesperson persistency vs. claim risk (e.g. *"Which agents have high sales volume but severe persistency risk?"*):

1. **Method 1 — Ranked Horizontal Bar Chart**:
   * Renders the top-6 highest combined risk agents.
   * Red highlight on the single highest-risk agent; blue bars for peer comparisons.
2. **Method 2 — 2×2 Interactive Quadrant Scatter Plot (Plotly)**:
   * **X-Axis**: Persistency Benchmark ($0\% - 100\%$).
   * **Y-Axis**: Early-Claim Rate ($0\% - 10\%$).
   * **Four Color-Coded Quadrants**:
     * 🔴 *Critical Attention* (Low Persistency + High Early Claims)
     * 🟡 *Persistency Watch* (Low Persistency + Normal Claims)
     * 🟠 *Claims Watch* (Normal Persistency + High Early Claims)
     * 🟢 *Healthy Book* (High Persistency + Low Claims)
3. **Agent Drilldown Expander (`Rows behind Agent <ID>`)**:
   * Interactive table showing individual policies, product mix, and early lapse flags for the focused agent.

---

### C. "How I Read Your Question" Proof Panel
For Turn 2 and later follow-ups, an expandable interpretation card displays before the answer:

| Field | Example Display | Meaning |
|---|---|---|
| **Subject Reference** | `"this product"` $\rightarrow$ `Canara HSBC Guaranteed Fortune Plan (carried from turn 1)` | Entity carried forward from conversation context |
| **Metric** | `13th month persistency rate` | Target business metric resolved |
| **Broken Down By** | `Sales Channel` | Dimension for segmentation |
| **Analysis Operation** | `driver_scan` / `crosstab_metric` / `outliers` | Analytical machine executed |
| **Confidence** | `0.94` | Resolution certainty score |

---

### D. System Guardrail & Refusal Cards

* **Greeting / Identity Card**:
  * Displays friendly welcome banner: `👋 Welcome to ExcelSense — Enterprise Life Insurance Analytics`.
  * 5 domain capability bullets (Policies, Persistency, Claims, Customers, Agents).
  * 4 clickable example starter questions.
* **Out-of-Domain Refusal Card (`.ct-refuse`)**:
  * Displays `⛔ Out of Domain Request`.
  * Explains domain boundaries (restricting queries to Life Insurance dataset).
* **Clarification Card (`.ct-clarify`)**:
  * Displays amber card asking user to clarify intent when confidence is below threshold.

---

## 6. Input & Interaction Flow

1. **Bottom Chat Input (`st.chat_input`)**:
   * Pinned to bottom of the screen.
   * Placeholder: `"Ask a question, or follow up on the answer above"`.
2. **Automatic Input Normalization**:
   * Strips outer quotes (`"..."`, `'...'`), extra whitespace, and trailing punctuation.
3. **De-duplication Protection**:
   * Prevents accidental double-execution on widget reruns using `chat_last_processed` session state.
4. **Deterministic Turn Replay**:
   * Every interaction re-renders previous turns from stored JSON state so conversation history stays persistent across page refreshes.
