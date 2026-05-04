# Financial Knowledge Graph

A Python project that demonstrates **knowledge graphs in Generative AI** by combining three
data sources — a Neo4j graph database, a Snowflake data warehouse, and unstructured compliance
documents — into a Claude AI agent that answers financial compliance questions.

---

## What this project demonstrates

| Capability | Technology | What it enables |
|---|---|---|
| Knowledge graph storage | Neo4j | Customer → Account → Transaction → Advisor relationships |
| Graph pattern detection | Neo4j Cypher | Circular flows, blast radius, multi-hop paths, risk clusters |
| Ontology-grounded reasoning | Semantic layer + LLM | Maps raw data to business concepts (HRC, HVT, Circular Movement) |
| Time-series analytics | Snowflake | Volume trends, Q4 anomaly spikes, country-level exposure |
| Unstructured document RAG | Neo4j vector index + sentence-transformers | Policy rules, SAR/CTR thresholds, EDD procedures |
| Hybrid query routing | LLM classifier (Anthropic or OpenAI) | Auto-routes questions to graph, documents, or both |
| AI Agent with tool use | LangGraph + LangChain (provider-agnostic) | Autonomous multi-source reasoning with ontology-annotated results |
| Interactive graph visualisation | pyvis | Network diagrams — red HRC nodes, cycle detection, blast radius maps |
| Guided investigation story mode | Streamlit | 5-step scripted Alice Morgan case for live presentations |
| One-click setup | `setup_all.py` | Full pipeline from raw data to running UI in one command |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                         │
│                                                                             │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────────┐   │
│  │   Neo4j Graph   │   │   Snowflake DW   │   │  Compliance Documents   │   │
│  │                 │   │                 │   │                         │   │
│  │ Customer nodes  │   │ CUSTOMERS       │   │ aml_policy.pdf          │   │
│  │ Account nodes   │   │ ACCOUNTS        │   │ customer_risk_          │   │
│  │ Transaction     │   │ TRANSACTIONS    │   │   framework.pdf         │   │
│  │ Advisor nodes   │   │ ADVISORS        │   │ advisor_compliance_     │   │
│  │ DocumentChunk   │   │ MONTHLY_TX_     │   │   guide.pdf             │   │
│  │ (vector index)  │   │   SUMMARY       │   │                         │   │
│  └────────┬────────┘   └───────┬─────────┘   └───────────┬─────────────┘   │
│           │                   │                           │                 │
│    WHO is connected?    HOW MUCH / WHEN?           WHAT do rules say?       │
│    Circular flows       Volume trends              SAR thresholds           │
│    Blast radius         Risk score history         EDD procedures           │
│    Network patterns     Country exposure           Advisor limits           │
└───────────┬───────────────────┴───────────────────────────┘─────────────────┘
            │                   │                           │
            ▼                   ▼                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              LangGraph ReAct Agent  (provider-agnostic)                     │
│                                                                             │
│   Model: Claude (Anthropic) or GPT-4o (OpenAI) — switchable in sidebar     │
│   User question ──► Agent reasons ──► calls tool(s) ──► synthesises answer │
│                                                                             │
│   Tools:  query_graph (Cypher)  ·  query_analytics (SQL)  ·  search_docs   │
└─────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI  (localhost:8501)                       │
│                                                                             │
│  Tab 0: Investigation   ← scripted 5-step Alice Morgan case (present mode)  │
│  Tab 1: AI Agent        ← flagship demo — full reasoning trace + viz        │
│  Tab 2: Explore Concepts← predefined ontology queries + network graph       │
│  Tab 3: Ask a Question  ← hybrid router: graph + docs                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Features in Detail

### Knowledge Graph Ontology

The semantic layer (`docs/ontology_semantic_layer.md`) defines 8 business concepts that
live above the raw data. The ontology flows through the entire stack:

| Where | How it is used |
|---|---|
| `docs/ontology_semantic_layer.md` | Authoritative definition — thresholds, rules, patterns |
| `scripts/generate_data.py` | Synthetic data embeds all 8 fraud scenarios so queries return results |
| `scripts/run_queries.py` | Each function implements one ontology concept as a Cypher query |
| `ui/graph_rag.py` — `ONTOLOGY_CONTEXT` | Injected as LLM system prompt so the model knows what HRC means |
| `ui/agent.py` — `_AGENT_SYSTEM_BASE` | Ontology definitions baked into agent system prompt |
| `ui/agent.py` — `_annotate_ontology()` | Every graph result row is tagged with ontology flags before Claude sees it |

This means the agent reasons over *labeled semantic concepts*, not anonymous numbers.
A `risk_score` of `0.92` arrives as `HIGH_RISK_CUSTOMER (risk_score > 0.8 → EDD required)`.

---

### Graph Visualisation

Implemented in `ui/graph_viz.py` using [pyvis](https://pyvis.readthedocs.io/).

- **Explore Concepts tab**: each concept button triggers a companion `VIZ_QUERIES` Cypher
  that returns Node/Relationship objects (not scalars). pyvis renders an interactive
  network with physics simulation.
- **AI Agent tab**: after every `query_graph` tool call, entity IDs are extracted from the
  result rows and a local subgraph is fetched and rendered.
- **Investigation tab**: each Neo4j step renders its subgraph inline.

All graphs use the same ontology-driven colour scheme so the audience never needs to
relearn the legend.

---

### AI Agent — LangGraph + Provider-Agnostic Model

`ui/agent.py` implements a **LangGraph ReAct agent** using `create_react_agent` from
`langgraph.prebuilt`. The model backend is fully configurable — switch between Claude
and GPT-4o from the UI sidebar with no code changes.

**How provider switching works:**

```python
# ui/agent.py — build_llm()
if provider == "Anthropic":
    return ChatAnthropic(model=model, api_key=api_key)   # langchain-anthropic
if provider == "OpenAI":
    return ChatOpenAI(model=model, api_key=api_key)      # langchain-openai
```

Both return a `BaseChatModel` — LangGraph calls the same tool-use loop regardless of
which provider is chosen. Adding a new provider (Bedrock, Azure OpenAI, Gemini) is one
`elif` and one `langchain-*` package.

**LangGraph agent loop:**

```
User question
    │
    ▼
LangGraph create_react_agent (LLM + tools + ontology system prompt)
    │
    ▼
LLM decides which tool(s) to call
    │
    ├── query_graph(cypher, reason)        → run_cypher() → _annotate_ontology() → ontology-tagged rows
    ├── query_analytics(sql, reason)       → Snowflake DictCursor → list[dict]
    └── search_documents(question)         → semantic_search() → DocumentChunk passages
    │
    ▼
Tool results (AIMessage / ToolMessage pairs) returned to LangGraph
    │
    ▼
LLM synthesises final answer using ontology language
    │
    ▼
_parse_output() converts LangGraph messages → AgentStep list + answer string
```

**Tools are LangChain `@tool` functions** built by a factory (`make_tools()`) that
injects `neo4j_driver`, `sf_conn`, and `embedder` via closure — no global state.
`Annotated` type hints on each parameter auto-generate the JSON tool schema the model
receives.

**Ontology grounding:** The full `ONTOLOGY_CONTEXT` string is injected as the agent's
system prompt via the `prompt=` parameter of `create_react_agent`. Every graph result
row is annotated with semantic flags by `_annotate_ontology()` before the LLM sees it,
so the model reasons over labeled concepts, not raw numbers.

---

### Hybrid RAG Pipeline

`ui/graph_rag.py` implements three retrieval modes:

- **`run_graph_rag()`** — generates Cypher from the question (LLM), executes it, then
  explains the results in plain English using the ontology as grounding context.
- **`semantic_search()`** — embeds the question with `all-MiniLM-L6-v2` (384-dim),
  queries the `document_chunks` Neo4j vector index via `db.index.vector.queryNodes()`.
- **`run_hybrid_rag()`** — runs both, combines results, and produces a single explanation
  that cites both sources.
- **`route_query()`** — LLM classifier returns `"structured"` / `"unstructured"` / `"hybrid"`.

Documents are stored as `DocumentChunk` nodes in Neo4j alongside the graph data —
the vector index lives in the same database as the knowledge graph.

---

### Synthetic Data — 8 Embedded Fraud Scenarios

`scripts/generate_data.py` generates 20 customers, 23 accounts, 30 transactions, and 5
advisors. Eight scenarios are deliberately embedded so every demo query returns interesting
results:

| Scenario | What it creates |
|---|---|
| Circular 3-hop ring | A008 → A009 → A010 → A008 |
| Structuring / smurfing | A011 receives $28k, fans out to 3 payments below $10k |
| Risk contagion | Low-risk C006/C007 transact with HRC C001/C002 |
| Hub account | A012: 3 inbound + 2 outbound connections |
| Risk cluster | C001 (0.95) ↔ C002 (0.91) ↔ C003 (0.87) — all interconnected |
| Advisor concentration | John Miller manages 4 HRC clients (100% of portfolio) |
| Blast radius | C001 (0.95) → 3 destination accounts in one hop |
| HRC Q4 volume spike | High-risk customers show 2.5×–6× volume increase in Oct–Dec 2025 (Snowflake only) |

---

### Project File Map

```
scripts/
  setup_all.py          ← one-click pipeline runner with connection checks
  generate_data.py      ← synthetic CSV data with 8 fraud scenarios
  create_schema.py      ← Neo4j uniqueness constraints
  load_to_neo4j.py      ← loads nodes and relationships via MERGE
  generate_pdfs.py      ← creates 3 compliance PDFs using reportlab
  ingest_pdfs.py        ← chunks, embeds, stores as DocumentChunk nodes
  load_to_snowflake.py  ← loads CSVs + generates MONTHLY_TX_SUMMARY time-series
  run_queries.py        ← runs the 4 ontology concept queries directly

ui/
  app.py                ← Streamlit UI (4 tabs)
  graph_rag.py          ← GraphRAG, hybrid routing, semantic search, ONTOLOGY_CONTEXT
  agent.py              ← LangGraph ReAct agent (provider-agnostic), build_llm(), make_tools(), ontology annotation
  graph_viz.py          ← pyvis network visualisation, VIZ_QUERIES, colour legend

docs/
  ontology_semantic_layer.md  ← authoritative semantic layer definition
  graph_model.md              ← graph schema documentation

cypher/
  schema.cypher         ← uniqueness constraints
  queries.cypher        ← all demo Cypher queries in one file

data/
  customers.csv / accounts.csv / transactions.csv / advisors.csv
  pdfs/                 ← aml_policy.pdf, customer_risk_framework.pdf, advisor_compliance_guide.pdf
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | `python --version` to check |
| Docker + Docker Compose | Runs Neo4j locally |
| Anthropic **or** OpenAI API key | Agent works with either — select provider in the sidebar |
| Snowflake account (optional) | Free 30-day trial at [snowflake.com/try](https://signup.snowflake.com/) — needed for AI Agent analytics tool |

---

## End-to-End Setup

### Quick path (recommended)

After completing Steps 1–4 (clone, install, configure `.env`, start Docker), run the full
pipeline in one command:

```bash
python scripts/setup_all.py
```

This runs all six pipeline steps in order — generate data, create schema, load Neo4j,
generate PDFs, ingest PDFs, load Snowflake — with colour-coded progress output and
connection pre-checks. Then:

```bash
streamlit run ui/app.py
```

To validate connections without loading data:

```bash
python scripts/setup_all.py --check-only
```

To skip Snowflake (if you don't have an account):

```bash
python scripts/setup_all.py --skip-snowflake
```

---

### Manual steps (if you prefer step-by-step control)

### Step 1 — Clone and create virtual environment

```bash
git clone <repo-url>
cd knowledge-graph-kickstart

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First run downloads `all-MiniLM-L6-v2` (~90 MB) via `sentence-transformers`
> when `ingest_pdfs.py` is run. Subsequent runs use the local cache.

### Step 3 — Configure environment

```bash
# Windows:
copy .env.example .env
# macOS/Linux:
cp .env.example .env
```

Edit `.env` and fill in your values:

```env
# Neo4j (local Docker — no changes needed unless you customised the password)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# LLM — fill in at least one
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Snowflake (optional — enables the analytics tool in the AI Agent)
# Account format: <locator>.<region>  e.g. abc12345.us-east-1
SNOWFLAKE_ACCOUNT=
SNOWFLAKE_USER=
SNOWFLAKE_PASSWORD=
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DATABASE=FINANCIAL_KG
SNOWFLAKE_SCHEMA=PUBLIC
```

### Step 4 — Start Neo4j

```bash
docker compose up -d
```

Wait ~15 seconds for Neo4j to be ready. Browser available at
**http://localhost:7474** (user: `neo4j`, password: `password`).

### Step 5 — Generate synthetic data and load Neo4j

```bash
# Generate CSVs with 8 embedded fraud scenarios
python scripts/generate_data.py

# Create Neo4j uniqueness constraints
python scripts/create_schema.py

# Load all nodes and relationships into Neo4j
python scripts/load_to_neo4j.py
```

Expected output from `generate_data.py`:
```
Generated 20 customers, 23 accounts, 30 transactions, 20 advisor-customer links.

Embedded scenarios:
  Q1 + Q4  Circular 3-hop ring      A008 → A009 → A010 → A008
  Q2       Structuring / smurfing   A011 receives $28k, fans out to 3 smaller payments
  Q3       Risk contagion           C006 (0.22) & C007 (0.18) transact with C001/C002 (>0.8)
  Q5       Hub account              A012: 3 inbound + 2 outbound connections
  Q6       Risk cluster             C001 (0.95) ↔ C002 (0.91) via T001
  Q7       Advisor portfolio risk   John Miller: 4/4 clients high-risk (100%)
  Q8       Blast radius             C001 (0.95) → 3 destination accounts
```

### Step 6 — Generate compliance PDFs and ingest into Neo4j vector index

```bash
# Create 3 compliance PDFs in data/pdfs/
python scripts/generate_pdfs.py

# Chunk, embed, and store as DocumentChunk nodes in Neo4j
python scripts/ingest_pdfs.py
```

Documents created:
- `data/pdfs/aml_policy.pdf` — SAR/CTR thresholds, structuring rules, EDD requirements
- `data/pdfs/customer_risk_framework.pdf` — risk score methodology, blast radius, hub accounts
- `data/pdfs/advisor_compliance_guide.pdf` — portfolio concentration limits, HRC obligations

### Step 7 — (Optional) Load Snowflake for AI Agent analytics

> Skip this step if you do not have a Snowflake account. The AI Agent and UI work without
> it — the analytics tool will return a "not connected" message and the agent will use
> Neo4j and documents instead.

Fill in the `SNOWFLAKE_*` vars in `.env`, then:

```bash
python scripts/load_to_snowflake.py
```

This creates `FINANCIAL_KG.PUBLIC` with 5 tables. The key table is `MONTHLY_TX_SUMMARY` —
12 months of synthetic per-customer transaction volume data, with a realistic Q4 spike for
high-risk customers that is invisible in Neo4j's snapshot data.

```
Expected output:
  Loaded   20 rows → CUSTOMERS
  Loaded   23 rows → ACCOUNTS
  Loaded   30 rows → TRANSACTIONS
  Loaded   20 rows → ADVISORS
  Loaded  240 rows → MONTHLY_TX_SUMMARY (12 months × 20 customers)
```

### Step 8 — Launch the UI

```bash
streamlit run ui/app.py
```

Opens at **http://localhost:8501**.

### Step 9 — Run tests

```bash
pytest
```

---

## UI Tabs

### Tab 0 — 🕵️ Investigation (Guided Story Mode)

A scripted 5-step investigation of **Alice Morgan (C001)** — designed for live presentations
to audiences without deep technical backgrounds. Each step is a button; results build up
sequentially so a presenter can pause and explain before moving on.

| Step | Data Source | What it shows |
|---|---|---|
| 1 — Identify the Subject | Neo4j | Alice's risk score (0.95), accounts, advisor as a network diagram |
| 2 — Follow the Money | Neo4j | All her transactions; HIGH_VALUE_TXN nodes highlighted orange |
| 3 — Check for Circular Flows | Neo4j | The A001→A002→…→A001 cycle visualised as a visible loop |
| 4 — Volume Trend Analysis | Snowflake | Line chart showing a 6× outgoing volume spike in December 2025 |
| 5 — Compliance Obligations | Documents | Policy excerpts: EDD threshold, SAR obligation, advisor escalation rules |

Every step shows:
- **The data** (table or chart)
- **The network graph** (for Neo4j steps) — coloured by ontology concept
- **"What this means"** — plain-English insight for the audience
- **"Why the graph"** — one sentence on what the graph uniquely contributes vs. SQL

When all 5 steps are complete, a summary banner states the conclusion:
> *No single data source could have produced this — it took the knowledge graph, the data warehouse, and the compliance documents together.*

Controls: **Run each step individually** (for a paced presentation) or **Run all steps** button for a rapid full demo. **Reset investigation** clears all results.

---

### Tab 1 — 🤖 AI Agent (Flagship Demo)

A **LangGraph ReAct agent** backed by a configurable LLM (Claude or GPT-4o — select in
the sidebar) that autonomously decides what to query, in what order, and synthesises a
grounded answer. The UI shows the complete reasoning trace — every tool call with its
Cypher or SQL input, the raw results, and the ontology concepts that were triggered.

**Three tools:**

| Tool | Data source | Used for |
|---|---|---|
| `query_graph` | Neo4j | WHO is connected, circular flows, blast radius, multi-hop paths |
| `query_analytics` | Snowflake | HOW MUCH, WHEN — volume trends, time-series, aggregations |
| `search_documents` | Neo4j vector index | WHAT rules say — SAR thresholds, EDD criteria, advisor limits |

**Ontology grounding:** Every graph result is automatically annotated with ontology flags
before being returned to the agent:
- `HIGH_RISK_CUSTOMER (risk_score > 0.8 → EDD required)`
- `HIGH_VALUE_TXN (amount > $10,000 → CTR filing threshold)`
- `CIRCULAR_MOVEMENT (A→B→A cycle detected → fraud indicator)`
- `ADVISOR_RISK_EXPOSURE (N HIGH_RISK_CUSTOMER(s) in portfolio)`

These flags appear in the UI under each graph result table so the audience can see the
ontology translating raw numbers into compliance obligations.

**Network visualisation:** After every `query_graph` tool call, an interactive pyvis
network diagram appears inline showing the entities involved, coloured by ontology concept.

**Try the SAR question first:**
> *"Should we file a SAR for Alice Morgan?"*

Expected tool sequence:
1. `search_documents` → retrieves SAR trigger rules from `aml_policy.pdf`
2. `query_graph` → finds Alice (risk 0.95, `HIGH_RISK_CUSTOMER`) and her circular ring
3. `query_analytics` → confirms Dec-2025 outgoing volume is 6× January baseline

This question **cannot be answered by any single source** — that is the point of the demo.

---

### Tab 2 — 🔍 Explore Concepts (Predefined Graph Queries)

Six predefined graph queries — one per ontology concept — with LLM explanations grounded
in the semantic layer. Each query is selected by clicking a button; results appear
immediately in two columns (data + explanation) followed by a full-width interactive
network graph.

| Concept | What it detects | Key ontology rule |
|---|---|---|
| High-Risk Customers | Customers with risk_score > 0.8 | `HIGH_RISK_CUSTOMER` → EDD required |
| High-Value Transactions | Transactions > $10,000 | `HIGH_VALUE_TXN` → CTR threshold |
| Circular Money Movements | A→B→A transaction cycles | `CIRCULAR_MOVEMENT` → fraud indicator |
| Advisor Risk Exposure | Advisors managing HRC clients | `ADVISOR_RISK_EXPOSURE` → portfolio review |
| Multi-Hop Customer Exposure | Customers within 2 hops of HRC | Risk contagion path |
| High-Risk Network | Full HRC subgraph with transactions | Risk cluster map |

**No API key needed** to retrieve data and see the graph visualisation.
API key needed only for the LLM explanation column.

**Graph visualisation colour legend:**
- 🔴 Red — `HIGH_RISK_CUSTOMER` (risk > 0.8)
- 🟠 Orange — `HIGH_VALUE_TXN` (amount > $10,000)
- 🟢 Green — normal Transaction
- 🔵 Blue — normal Customer
- ⚫ Grey — Account
- 🟣 Purple — Advisor

Hover over any node to see full details: name, risk score, country, ontology flag with
its compliance implication. Drag nodes to rearrange; scroll to zoom.

---

### Tab 3 — 💬 Ask a Question (Hybrid RAG)

Free-form question input with automatic routing. The selected LLM (Claude or GPT-4o)
classifies the question and decides which data source to use:

| Route | Trigger | Sources queried |
|---|---|---|
| `structured` | Questions about live data — customers, accounts, transactions | Neo4j only |
| `unstructured` | Questions about rules, thresholds, definitions, procedures | Documents only |
| `hybrid` | Compliance questions that need both data and policy context | Neo4j + Documents |

The system then runs the appropriate retrieval, generates or uses a provided Cypher query,
and produces an LLM explanation grounded in the ontology context.

You can optionally provide your own Cypher query — leave blank for auto-generation.

---

## Graph Model

```
(Advisor)-[:MANAGES]->(Customer)-[:OWNS]->(Account)-[:SENT]->(Transaction)-[:TO]->(Account)
```

| Node | Key Properties |
|---|---|
| Customer | id, name, risk_score (0–1), country |
| Account | id, type (SAVINGS \| CHECKING \| BROKERAGE \| INVESTMENT) |
| Transaction | id, amount (USD), timestamp |
| Advisor | id, name |
| DocumentChunk | id, text, source, page, embedding (384-dim vector) |

---

## Snowflake Schema

| Table | Contents |
|---|---|
| `CUSTOMERS` | customer_id, name, risk_score, country |
| `ACCOUNTS` | account_id, customer_id, account_type |
| `TRANSACTIONS` | txn_id, from_account, to_account, amount, txn_timestamp |
| `ADVISORS` | advisor_id, name, customer_id |
| `MONTHLY_TX_SUMMARY` | month, customer_id, customer_name, risk_score, country, total_sent, total_received, txn_count_sent, txn_count_received, unique_counterparties |

---

## Semantic Layer

| Business Concept | Definition |
|---|---|
| High Risk Customer (HRC) | `risk_score >= 0.80` |
| Critical Risk Customer | `risk_score >= 0.90` — immediate CCO escalation |
| High Value Transaction | `amount > $10,000` |
| Structuring / Smurfing | Large inflow (>$10k) → multiple outflows (avg <$5k) |
| Circular Money Movement | Closed transaction ring: A→B→C→A |
| Hub Account | Receives from 2+ sources AND sends to 2+ destinations |
| Risk Contagion | Low-risk customer (<0.30) transacts with HRC (>0.80) |
| Advisor Risk Exposure | Advisor manages one or more HRC clients |

---

## AI Agent Sample Questions

### All three tools (Neo4j + Snowflake + Documents)

**"Should we file a SAR for Alice Morgan?"**
The agent checks the policy rules, maps Alice's network, and checks her volume trend.
All three SAR triggers are satisfied.

**"Which advisors violate portfolio concentration rules?"**
Finds John Miller (100% HRC clients) and checks the policy threshold for CCO escalation.

**"Are any accounts showing structuring patterns that require SAR review?"**
Detects A011's large inflow / multiple small outflow pattern, cross-references the $10k/$5k
thresholds from the AML policy.

---

### Neo4j + Documents

**"Are the circular ring accounts compliant with our risk framework?"**
Detects A008→A009→A010→A008, then checks whether the participants' risk scores comply
with the policy rule that ring members must be elevated to ≥ 0.50.

**"Which low-risk customers are exposed to risk contagion?"**
Maps Fiona Green (0.22) → Alice Morgan (0.95) and George Hall (0.18) → Bob Carter (0.91),
then cites the 30-day review and auto-escalation rules from the framework.

---

### Snowflake only

**"Show the monthly transaction volume trend for high-risk customers"**
Demonstrates why Snowflake is necessary: the Q4 spike is invisible from Neo4j's snapshot.

**"Compare total transaction volumes by country"**
Pure SQL aggregation across the MONTHLY_TX_SUMMARY table.

---

### Neo4j only

**"What is the blast radius of our riskiest customer?"**
Traverses every account and customer reachable within one hop from Alice Morgan (0.95).
Pure graph traversal — not expressible in SQL without recursive CTEs.

**"Which accounts form 3-hop circular transaction rings?"**

```cypher
MATCH (a1:Account)-[:SENT]->(:Transaction)-[:TO]->(a2:Account)
      -[:SENT]->(:Transaction)-[:TO]->(a3:Account)
      -[:SENT]->(:Transaction)-[:TO]->(a1)
WHERE a1 <> a2 AND a2 <> a3
RETURN DISTINCT a1.id AS account_a, a2.id AS account_b, a3.id AS account_c
```

---

### Documents only

**"What is the SAR filing threshold under our AML policy?"**
$5,000 for known violations, $25,000 when the subject cannot be identified. 30-day deadline.

**"What training must advisors managing HRC clients complete?"**
Annual AML Certification (≥80%), HRC Management module (4h), Network Risk workshop (2h).

**"What does structuring mean and what triggers a SAR?"**
Large inflow > $10k followed by multiple outflows averaging below $5k = prima facie
structuring pattern requiring immediate SAR review.

---

## Ask a Question Tab — Hybrid RAG Queries

Paste into **Tab 3** after running `ingest_pdfs.py`.

| Question | Expected route | What it tests |
|---|---|---|
| Does Alice Morgan require EDD? | 🔀 Hybrid | Risk score (graph) + EDD threshold (docs) |
| What actions for John Miller's portfolio? | 🔀 Hybrid | Portfolio data (graph) + concentration rules (docs) |
| What is the structuring definition? | 📄 Documents | Pure policy lookup |
| Which customers have risk > 0.8? | 📊 Graph | Pure graph query |
| How should the circular ring be handled? | 🔀 Hybrid | Ring detection (graph) + remediation rules (docs) |

---

## Knowledge Graph & Ontology Queries

These six queries demonstrate what knowledge graphs uniquely enable — each one maps a
business concept from the ontology to a graph pattern that SQL cannot express cleanly.
Paste the Cypher into the Neo4j Browser at **http://localhost:7474** or into the
**Ask a Question** tab with the Cypher field filled in.

The ontology concepts referenced below are defined in `docs/ontology_semantic_layer.md`
and in the **Architecture & Data Sources** expander inside the UI.

---

### 1. Advisor indirect exposure through circular flows
**Ontology concept applied:** Circular Money Movement + Advisor Risk Exposure

*"Which advisors are indirectly exposed to the circular money ring, even if their own clients look clean on paper?"*

This crosses 4 relationship types in one pattern:
`Advisor → MANAGES → Customer → OWNS → Account → (in circular ring)`

```cypher
MATCH (a1:Account)-[:SENT]->(:Transaction)-[:TO]->(a2:Account)
      -[:SENT]->(:Transaction)-[:TO]->(a3:Account)
      -[:SENT]->(:Transaction)-[:TO]->(a1)
WITH collect(DISTINCT a1) + collect(DISTINCT a2) + collect(DISTINCT a3) AS ring_accounts
UNWIND ring_accounts AS ring_acct
MATCH (owner:Customer)-[:OWNS]->(ring_acct)
MATCH (adv:Advisor)-[:MANAGES]->(owner)
RETURN adv.name                        AS advisor,
       collect(DISTINCT owner.name)    AS exposed_clients,
       count(DISTINCT owner)           AS exposure_count,
       collect(DISTINCT ring_acct.id)  AS ring_accounts_touched
ORDER BY exposure_count DESC
```

**Why SQL can't do this cleanly:** The ring detection requires a recursive CTE with a
cycle guard. Add 2 more joins for OWNS and MANAGES and the query becomes a 50-line
maintenance burden. Cypher expresses the entire pattern in 3 lines.

**Expected result:** Sarah Wilson (manages C008/Hannah Park who owns ring node A008)
and Michael Chen (manages C009/Ivan Cruz and C010/Julia Stone who own A009 and A010).

---

### 2. Full risk cluster — connected subgraph of HRC nodes
**Ontology concept applied:** High Risk Customer + Risk Cluster

*"Map every high-risk customer and every transaction that directly connects them — the complete risk cluster as a network."*

```cypher
MATCH (c1:Customer)-[:OWNS]->(a1:Account)-[:SENT]->(t:Transaction)
      -[:TO]->(a2:Account)<-[:OWNS]-(c2:Customer)
WHERE c1.risk_score > 0.8
  AND c2.risk_score > 0.8
  AND c1 <> c2
RETURN c1.name                    AS from_customer,
       round(c1.risk_score, 2)    AS from_risk,
       a1.id                      AS from_account,
       t.id                       AS transaction,
       round(t.amount, 2)         AS amount,
       a2.id                      AS to_account,
       c2.name                    AS to_customer,
       round(c2.risk_score, 2)    AS to_risk
ORDER BY amount DESC
```

**Why SQL can't do this cleanly:** SQL returns rows, not subgraph structure. You get the
edge list but have to reconstruct the cluster in application code. A graph database
returns the subgraph natively and can compute cluster properties (density, diameter)
in the same query.

**Expected result:** Alice Morgan (0.95) ↔ Bob Carter (0.91) via T001 ($32k),
Alice ↔ Charlie Ross (0.87) via T002 ($15.5k), Alice ↔ Diana Blake (0.85) via T003 ($18.2k).

---

### 3. Variable-length contamination path from the smurfing account
**Ontology concept applied:** Structuring / Smurfing + Multi-Hop Exposure

*"What accounts and customers are reachable from the smurfing account A011 within 2 transaction hops?"*

```cypher
MATCH path = (start:Account {id: 'A011'})-[:SENT|TO*1..4]-(reached:Account)
WHERE start <> reached
WITH DISTINCT reached,
     length(path)                                              AS hops,
     [n IN nodes(path) WHERE n:Account | n.id]                AS path_accounts
MATCH (owner:Customer)-[:OWNS]->(reached)
RETURN reached.id                AS account,
       owner.name                AS owner,
       round(owner.risk_score,2) AS risk_score,
       hops,
       path_accounts
ORDER BY hops, risk_score DESC
```

**Why SQL can't do this cleanly:** `*1..4` is a single Cypher operator. The SQL
equivalent requires 4 separate self-joins with `UNION ALL` and a `NOT IN` anti-join
at each level to suppress already-visited nodes — and it still does not generalise to
arbitrary depth.

**Expected result:** Direct neighbours include A005 (Ethan Hunt, 0.82), A007 (George Hall, 0.18),
A012 (Laura Kim, 0.44). Two hops out reaches A006, A003, and more.

---

### 4. Intersection of ontology patterns — customers in multiple risk categories
**Ontology concept applied:** High Risk Customer ∩ Circular Money Movement ∩ Blast Radius

*"Which customers are both high-risk AND appear in at least one other ontology pattern: circular flow or blast radius?"*

```cypher
// Collect HRC nodes
MATCH (c:Customer) WHERE c.risk_score > 0.8
WITH collect(c) AS hrc_nodes

// Pattern A: owners of circular ring accounts
MATCH (a1:Account)-[:SENT]->(:Transaction)-[:TO]->(a2:Account)
      -[:SENT]->(:Transaction)-[:TO]->(a1)
MATCH (ring_owner:Customer)-[:OWNS]->(a1)
WITH hrc_nodes, collect(DISTINCT ring_owner) AS in_ring

// Pattern B: within blast radius of the highest-risk customer
MATCH (top:Customer) WITH top ORDER BY top.risk_score DESC LIMIT 1
MATCH (top)-[:OWNS]->(:Account)-[:SENT]->(:Transaction)
      -[:TO]->(:Account)<-[:OWNS]-(neighbor:Customer)
WITH hrc_nodes, in_ring, collect(DISTINCT neighbor) AS in_blast_radius

// Return HRC customers that match at least one additional pattern
UNWIND hrc_nodes AS c
WITH c, in_ring, in_blast_radius,
     c IN in_ring         AS in_circular_flow,
     c IN in_blast_radius AS in_blast_radius_flag
WHERE in_circular_flow OR in_blast_radius_flag
RETURN c.name                AS customer,
       round(c.risk_score,2) AS risk_score,
       in_circular_flow,
       in_blast_radius_flag
ORDER BY risk_score DESC
```

**Why SQL can't do this cleanly:** Each pattern is a separate subquery. Combining them
requires 3 CTEs and a multi-way JOIN. The graph query computes each pattern as a node
set and intersects them directly — aligning with how the ontology defines risk as the
presence of multiple named patterns.

**Expected result:** Alice Morgan appears in blast_radius (she IS the source), Bob Carter
appears in the blast radius. The ring owners (Hannah Park, Ivan Cruz, Julia Stone) appear
in circular_flow.

---

### 5. Shortest money path between two flagged accounts
**Ontology concept applied:** Multi-Hop Exposure (shortest path variant)

*"What is the shortest transaction chain connecting Alice Morgan's account to the smurfing account — how many hops and through which intermediaries?"*

```cypher
MATCH (alice:Customer {name: 'Alice Morgan'})-[:OWNS]->(src:Account),
      (kevin:Customer {name: 'Kevin White'})-[:OWNS]->(dst:Account)
MATCH path = shortestPath((src)-[:SENT|TO*]-(dst))
RETURN [n IN nodes(path) |
        CASE WHEN n:Account     THEN 'Account:'     + n.id
             WHEN n:Transaction THEN 'Transaction:' + n.id
             ELSE n.id END]  AS path_nodes,
       length(path)          AS hops
```

**Why SQL can't do this cleanly:** `shortestPath()` is a built-in graph algorithm.
SQL has no path primitive — finding the shortest path requires iteratively expanding
CTEs one level at a time and stopping when the target is found, which is both complex
to write and expensive to run on large datasets.

**Expected result:** A001 → T001 → A002 → T009 → A011 (4 hops through Bob Carter's account
and the smurfing inflow transaction).

---

### 6. Composite advisor risk score using graph-computed network exposure
**Ontology concept applied:** Advisor Risk Exposure (extended with network neighbourhood)

*"Rank advisors by composite risk: 60% direct client risk + 40% average risk of their clients' transaction counterparties."*

This score cannot be computed from a flat table — it requires traversing one hop beyond
each client to measure the risk of who those clients transact with.

```cypher
MATCH (adv:Advisor)-[:MANAGES]->(c:Customer)-[:OWNS]->(a:Account)
OPTIONAL MATCH (a)-[:SENT]->(:Transaction)-[:TO]->(a2:Account)<-[:OWNS]-(peer:Customer)
WHERE peer <> c
WITH adv, c, collect(DISTINCT peer.risk_score) AS peer_scores
WITH adv,
     count(DISTINCT c)                                      AS total_clients,
     sum(CASE WHEN c.risk_score > 0.8 THEN 1 ELSE 0 END)   AS direct_hrc_count,
     round(avg(c.risk_score), 3)                            AS avg_direct_risk,
     round(avg(
       CASE WHEN size(peer_scores) > 0
            THEN reduce(s = 0.0, x IN peer_scores | s + x) / size(peer_scores)
            ELSE c.risk_score END
     ), 3)                                                  AS avg_network_risk
RETURN adv.name                                                          AS advisor,
       total_clients,
       direct_hrc_count,
       avg_direct_risk,
       avg_network_risk,
       round(avg_direct_risk * 0.6 + avg_network_risk * 0.4, 3)         AS composite_risk_score
ORDER BY composite_risk_score DESC
```

**Why SQL can't do this cleanly:** The `avg_network_risk` column requires a subgraph
expansion — for each client, find their transaction counterparties, collect those risk
scores, and average them. In SQL this is a correlated subquery inside an aggregation:
a pattern that most SQL optimisers handle poorly at scale.

**Expected result:** John Miller ranks highest — his 4 direct HRC clients (avg ~0.90)
also transact heavily with other HRC nodes, compounding his composite score well above
any other advisor.

---

## AI Agent Demo Queries — Ontology-Driven Multi-Source Reasoning

These eight questions are designed to demonstrate the **full value of the architecture**:
each requires the agent to call at least two tools, apply ontology concept labels to raw
data, and synthesise an answer no single system could produce alone.

Paste any question into the **AI Agent** tab. Watch the reasoning trace: which tools were
called, in what order, which ontology flags were triggered, and how the agent wove all
sources into one grounded answer.

---

### Q1 — SAR trigger: does Alice's Q4 spike breach policy?

> *"Alice Morgan's transaction volume spiked in Q4 2025. Does this breach AML policy SAR thresholds,
> and which of her direct counterparties should be co-flagged on the same SAR?"*

**Why it needs all three sources:**
- **Documents** → what is the SAR filing threshold and which transactions qualify?
- **Snowflake** → confirm the Q4 spike: exact months, exact volumes against the threshold
- **Graph** → traverse `Account → Transaction → Account → Customer` to identify counterparties
  who must appear on the same SAR (impossible in SQL without 3 self-joins)

**Expected tool sequence:** `search_documents` → `query_analytics` → `query_graph`

**Ontology flags triggered:** `HIGH_RISK_CUSTOMER` (Alice, risk 0.95) · `HIGH_VALUE_TXN`
(transactions > $10k) · `ADVISOR_RISK_EXPOSURE` (her advisor co-implicated)

**What the ontology adds:** The graph query returns raw rows; the `HIGH_RISK_CUSTOMER` flag
tells the agent that Alice already has elevated EDD status, so the SAR recommendation is
stronger than it would be for a borderline-risk customer with the same volume spike.

---

### Q2 — Advisor liability: concentration breach + blast radius

> *"Which advisors have violated the portfolio concentration limit defined in the compliance guide,
> and if their highest-risk client is investigated, what is the full blast radius?"*

**Why it needs all three sources:**
- **Documents** → what is the portfolio concentration limit? (advisor_compliance_guide.pdf)
- **Graph** → Advisor → MANAGES → Customer traversal to count HRC clients per advisor;
  then `shortestPath` / `*1..N` to expand the blast radius of the worst client
- **Snowflake** → validate with actual transaction counts, not just risk scores

**Expected tool sequence:** `search_documents` → `query_graph` → `query_analytics`

**Ontology flags triggered:** `ADVISOR_RISK_EXPOSURE` · `HIGH_RISK_CUSTOMER`

**What the ontology adds:** The agent knows from the `ADVISOR_RISK_EXPOSURE` flag that
the threshold is "manages ≥ 1 HRC client" — it therefore interprets the blast radius
expansion as a compliance liability, not just a network curiosity.

---

### Q3 — Seasonal structuring: when did the ring peak and how far did it propagate?

> *"The structuring ring involving accounts A009, A010, and A011 — which month in 2025 showed
> the highest combined transaction volume, and what is the full contamination path from A011
> within two hops?"*

**Why it needs all three sources:**
- **Documents** → definition of structuring and the $5,000 / $10,000 thresholds
- **Snowflake** → monthly summary for customers owning A009/A010/A011 to find the peak month
- **Graph** → `(A011)-[:SENT|TO*1..4]-(reached:Account)` to map the 2-hop propagation network

**Expected tool sequence:** `search_documents` → `query_analytics` → `query_graph`

**Ontology flags triggered:** `CIRCULAR_MOVEMENT` · `HIGH_VALUE_TXN`

**What the ontology adds:** The `CIRCULAR_MOVEMENT` flag on the ring-account rows signals
to the agent that the propagation path is not normal payment flow — it is a layering network,
which triggers a different set of remediation recommendations from the policy docs.

---

### Q4 — EDD gap analysis: who qualifies but has no flag yet?

> *"According to the customer risk framework, which customers currently qualify for Enhanced Due
> Diligence? Of those, which have shown anomalous transaction growth in the past 3 months of 2025?"*

**Why it needs all three sources:**
- **Documents** → exact EDD eligibility criteria (risk_score threshold, country flags,
  transaction patterns) from customer_risk_framework.pdf
- **Graph** → retrieve all customers with `risk_score > 0.8` (the ontology-defined HRC threshold)
- **Snowflake** → compare Oct / Nov / Dec 2025 volumes for each HRC against their Jan–Sep baseline

**Expected tool sequence:** `search_documents` → `query_graph` → `query_analytics`

**Ontology flags triggered:** `HIGH_RISK_CUSTOMER` on every returned customer row

**What the ontology adds:** The `HIGH_RISK_CUSTOMER` flag directly maps to the EDD obligation —
the agent does not need to re-derive the rule; it reads the flag and immediately frames its
answer in terms of compliance action required, not just data observation.

---

### Q5 — Risk cluster SAR: does the connected subgraph collectively cross the threshold?

> *"Identify all high-risk customers who transact directly with each other. Does their combined
> Q4 2025 outflow collectively cross the SAR filing threshold? What does AML policy say about
> aggregate suspicious activity?"*

**Why it needs all three sources:**
- **Graph** → connected HRC subgraph: `(c1)-[:OWNS]->()...()<-[:OWNS]-(c2)` where both
  `c1.risk_score > 0.8` and `c2.risk_score > 0.8` — impossible in SQL without recursive CTEs
- **Snowflake** → sum `TOTAL_SENT` for those specific customers in Q4 2025
- **Documents** → does AML policy recognise aggregate activity as a SAR trigger?

**Expected tool sequence:** `query_graph` → `query_analytics` → `search_documents`

**Ontology flags triggered:** `HIGH_RISK_CUSTOMER` · `HIGH_VALUE_TXN`

**What the ontology adds:** The cluster query returns rows flagged as `HIGH_RISK_CUSTOMER`.
The agent uses those flags to select the right Snowflake customers (no manual ID list needed)
and to frame the SAR question correctly — individual threshold vs. aggregate threshold.

---

### Q6 — Cross-border exposure: which circular flows cross jurisdictions?

> *"Are any accounts in the circular money ring owned by customers from different countries?
> What does AML policy say about cross-border layering, and what is the total cross-border
> volume in 2025?"*

**Why it needs all three sources:**
- **Graph** → circular ring detection (`A→B→A`) then `OWNS` traversal to get customer
  countries — combining two ontology patterns in one query
- **Documents** → cross-border layering rules, FATF reporting obligations
- **Snowflake** → `GROUP BY COUNTRY` on `MONTHLY_TX_SUMMARY` for the ring owners

**Expected tool sequence:** `query_graph` → `search_documents` → `query_analytics`

**Ontology flags triggered:** `CIRCULAR_MOVEMENT` · `HIGH_RISK_CUSTOMER` (if ring owners are HRC)

**What the ontology adds:** The `CIRCULAR_MOVEMENT` flag tells the agent this is not a normal
international payment — it is a cross-border layering pattern, triggering a different policy
section (FATF guidance) than a plain international wire.

---

### Q7 — Advisor intelligence brief: full risk profile from all sources

> *"Give me a complete risk intelligence brief on advisor John Miller: how many high-risk clients,
> what are their transaction trends over 2025, are there any circular flows in his portfolio,
> and what does the compliance guide say his obligations are?"*

**Why it needs all three sources:**
- **Graph** → `(Advisor {name:'John Miller'})-[:MANAGES]->(c:Customer)` then HRC count,
  circular ring membership check for each client
- **Snowflake** → monthly trend for all of John's clients across 2025 (12 months × N clients)
- **Documents** → what obligations apply to an advisor with this level of HRC exposure?

**Expected tool sequence:** `search_documents` → `query_graph` → `query_analytics`

**Ontology flags triggered:** `ADVISOR_RISK_EXPOSURE` · `HIGH_RISK_CUSTOMER` ·
`CIRCULAR_MOVEMENT` (if any client in ring)

**What the ontology adds:** Every flag the agent receives carries its compliance implication
in parentheses — `ADVISOR_RISK_EXPOSURE (3 HIGH_RISK_CUSTOMER(s) in portfolio)`. The agent
synthesises this into actionable obligations: EDD for each client, portfolio review trigger,
potential supervisory escalation.

---

### Q8 — Full investigation bootstrap: unknown customer, all signals

> *"I need to investigate customer C005. Tell me everything: their risk classification,
> every account and transaction in the graph, their 2025 transaction trend, which compliance
> obligations apply, and whether any of their counterparties are also flagged."*

**Why it needs all three sources:**
- **Graph** → full ego-network: all accounts, transactions, counterparties, advisor, risk score;
  ontology flags on every row
- **Snowflake** → 12-month volume trend for C005, identify which months spiked
- **Documents** → which policies apply given the customer's risk score and transaction patterns

**Expected tool sequence:** `query_graph` → `query_analytics` → `search_documents`

**Ontology flags triggered:** Depends on C005's data — potentially all four concepts

**What the ontology adds:** This is the purest demonstration of ontology-grounded reasoning.
The agent does not present raw rows — every data point is interpreted through the semantic
layer. A `risk_score` of 0.91 is not reported as a number; it is reported as
`HIGH_RISK_CUSTOMER → EDD required → SAR-eligible activity threshold applies`.
That translation from data to compliance meaning *is* the knowledge graph.

---

## Why Graph + Warehouse + Documents vs. Any One Alone

| Capability | SQL/Snowflake | Neo4j | Documents |
|---|---|---|---|
| Volume trends, aggregations | ✅ Native | ❌ Awkward | ❌ Not data |
| 3-hop circular flow detection | ❌ Recursive CTEs | ✅ One Cypher pattern | ❌ Not data |
| SAR filing threshold | ❌ Not in DB | ❌ Not in DB | ✅ AML policy |
| Blast radius of a node | ❌ 4+ self-joins | ✅ Path traversal | ❌ Not data |
| Variable-length path traversal | ❌ Exponential complexity | ✅ `*1..N` syntax | ❌ Not data |
| Compliance rules and procedures | ❌ Not in DB | ❌ Not in DB | ✅ Policy docs |

The agent is the glue that calls the right tool for each part of the question and
synthesises a single grounded answer — the knowledge graph architecture in practice.
