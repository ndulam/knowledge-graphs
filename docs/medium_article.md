# Why Your AI Agent Needs a Knowledge Graph: A Hands-On Demo with Neo4j, Snowflake, and Claude

*How combining a graph database, a data warehouse, and compliance documents produces answers that no single source could — and how ontology is the glue that makes AI agents reason like experts.*

---

## The Problem with One-Dimensional AI

Ask an AI agent "Should we file a SAR for this customer?" and something interesting happens.

If the agent only has access to a SQL database, it can tell you the customer's risk score is 0.95. That is a number. It means nothing without context.

If the agent only has access to policy documents, it can tell you that a SAR must be filed when "suspicious activity involves funds of $5,000 or more." That is a rule. It means nothing without the data.

If the agent only has access to transaction records, it can tell you the customer sent $32,000 last month. That is a fact. It means nothing without knowing who received it or whether the pattern repeats.

The answer — *yes, file the SAR, here's why, here's who else to include, here's the policy section that applies* — requires all three sources to be combined, interpreted, and reasoned over simultaneously.

That is the problem this project solves.

---

## What We Built

**Financial Knowledge Graph** is an open-source Python demo that connects three data sources to a Claude AI agent:

- **Neo4j** — a graph database storing customers, accounts, transactions, and advisors as a connected network
- **Snowflake** — a data warehouse with 12 months of per-customer transaction time-series
- **Compliance PDFs** — three policy documents chunked, embedded, and stored in Neo4j's vector index

The agent uses Claude's tool-use API to autonomously decide which source to query, executes the query, receives **ontology-annotated results**, and synthesises a grounded compliance answer.

The Streamlit UI includes four tabs, the most presentation-friendly being a scripted 5-step investigation of a suspicious customer — built specifically to walk a non-technical audience through the "aha moment" of why knowledge graphs change what AI agents can reason about.

---

## The Key Insight: Ontology as a Reasoning Bridge

Before diving into the architecture, there is one concept worth understanding clearly: **ontology**.

In this project, the ontology is a lightweight semantic layer — not RDF/OWL, just a documented set of business concepts with precise definitions:

| Concept | Definition |
|---|---|
| High Risk Customer (HRC) | `risk_score > 0.8` — triggers EDD obligations |
| High Value Transaction (HVT) | `amount > $10,000` — CTR filing threshold |
| Circular Money Movement | Account A sends to B, B sends back to A — fraud indicator |
| Advisor Risk Exposure | Advisor manages one or more HRC clients |

These concepts do three things in the system:

**1. They shape the data.** The synthetic data generator deliberately embeds all four patterns — a circular ring (A008→A009→A010→A008), high-value transactions, HRC customers, advisor concentration — so every query returns meaningful results.

**2. They ground the LLM.** Every Claude call includes this ontology as system context. The model does not infer that a `risk_score` of 0.92 is significant — it is told: *"A Customer where risk_score > 0.8 is a High Risk Customer. This triggers Enhanced Due Diligence obligations."*

**3. They annotate tool results.** Before the agent receives graph results, a Python function tags every row:

```python
def _annotate_ontology(row: dict) -> dict:
    flags = []
    if float(row.get("risk_score", 0)) > 0.8:
        flags.append("HIGH_RISK_CUSTOMER (risk_score > 0.8 -> EDD required)")
    if float(row.get("amount", 0)) > 10_000:
        flags.append("HIGH_VALUE_TXN (amount > $10,000 -> CTR filing threshold)")
    if "account_a" in row and "account_b" in row:
        flags.append("CIRCULAR_MOVEMENT (A->B->A cycle -> fraud indicator)")
    if row.get("_ontology_flags"):
        row["_ontology_flags"] = flags
    return row
```

The agent does not receive a row with `risk_score: 0.92`. It receives a row with `risk_score: 0.92` **and** `_ontology_flags: ["HIGH_RISK_CUSTOMER (risk_score > 0.8 -> EDD required)"]`.

This is the difference between data and meaning. The ontology is the translation layer between raw numbers and compliance obligations.

---

## Why a Knowledge Graph — and Not Just SQL?

This is the question every graph database demo needs to answer directly.

SQL is excellent at answering *what* questions: what is Alice's risk score, what transactions did she send, what is the total volume by country.

Graph databases answer *who* and *how* questions at any depth: who is Alice connected to, how many hops does it take to reach a flagged account, is there a cycle anywhere in the transaction network?

Here is the clearest example. This Cypher query detects circular money flows:

```cypher
MATCH (a1:Account)-[:SENT]->(:Transaction)-[:TO]->(a2:Account)
      -[:SENT]->(:Transaction)-[:TO]->(a1)
RETURN DISTINCT a1.id AS account_a, a2.id AS account_b
```

Three lines. In SQL, the equivalent requires a recursive CTE with a visited-node guard to prevent infinite loops — typically 40+ lines — and it still does not generalise to 3-hop or 4-hop cycles without rewriting.

The more important point for AI applications: **SQL returns rows. A graph returns a subgraph.** When you visualise the result, you do not see a table of `account_a, account_b` pairs. You see a network diagram with a visible cycle — accounts pointing back to themselves through transactions. That visualisation is itself the explanation.

The demo uses [pyvis](https://pyvis.readthedocs.io/) to render these networks interactively in Streamlit, with nodes coloured by ontology concept:

- 🔴 Red — HIGH_RISK_CUSTOMER
- 🟠 Orange — HIGH_VALUE_TXN
- 🟢 Green — normal transaction
- 🟣 Purple — Advisor

When you click "Circular Money Movements", you see a loop. When you click "Advisor Risk Exposure", you see purple nodes fanning out to clusters of red. No table communicates this as immediately.

---

## The Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Neo4j           Snowflake         Compliance Documents      │
│  WHO/HOW?        HOW MUCH/WHEN?    WHAT DO RULES SAY?        │
│  Graph patterns  Time-series       AML policy PDFs           │
│  Vector index    Aggregations      EDD procedures            │
└──────────┬────────────┬────────────────────┬────────────────┘
           │            │                    │
           ▼            ▼                    ▼
┌──────────────────────────────────────────────────────────────┐
│  Claude AI Agent  (tool_use API + ontology system prompt)     │
│                                                              │
│  query_graph(cypher) → annotate with ontology flags          │
│  query_analytics(sql) → Snowflake DictCursor                 │
│  search_documents(question) → Neo4j vector index             │
└──────────────────────────────────────────────────────────────┘
           │
           ▼
  Streamlit UI — 4 tabs
  Investigation | AI Agent | Explore Concepts | Ask a Question
```

The agent loop is straightforward Claude tool use:

1. User submits a question
2. Claude decides which tool(s) to call and generates the query (Cypher or SQL)
3. The tool executes against the real data source
4. Results are annotated with ontology flags and returned to Claude
5. Claude calls more tools if needed (up to 10 turns)
6. Claude synthesises a final answer using ontology language, not raw numbers

---

## The Demo: Investigating Alice Morgan

The **Investigation tab** walks through a real compliance scenario in five steps, designed to be run live in a presentation.

**Step 1 — Identify the Subject**

```cypher
MATCH (c:Customer {id:'C001'})-[:OWNS]->(a:Account)
OPTIONAL MATCH (adv:Advisor)-[:MANAGES]->(c)
RETURN c.name, c.risk_score, c.country, collect(a.id), adv.name
```

*What we learn:* Alice Morgan, risk score 0.95, managed by John Miller. The network graph shows her accounts and advisor. The ontology flag fires: **HIGH_RISK_CUSTOMER → EDD required**.

**Step 2 — Follow the Money**

```cypher
MATCH (c:Customer {id:'C001'})-[:OWNS]->(a:Account)-[:SENT]->(t:Transaction)
RETURN a.id, t.id, t.amount, t.timestamp ORDER BY t.amount DESC
```

*What we learn:* Multiple transactions above $10,000. The graph shows orange HIGH_VALUE_TXN nodes. Each one is at or above the CTR filing threshold.

**Step 3 — Check for Circular Flows**

```cypher
MATCH (c:Customer {id:'C001'})-[:OWNS]->(a1:Account)
      -[:SENT]->(:Transaction)-[:TO]->(a2:Account)
      -[:SENT]->(:Transaction)-[:TO]->(a1)
MATCH (owner:Customer)-[:OWNS]->(a2)
RETURN a1.id, a2.id, owner.name, owner.risk_score
```

*What we learn:* Alice's account A001 loops back through Bob Carter (risk 0.91). **CIRCULAR_MOVEMENT detected** — a known layering pattern. Both parties are HIGH_RISK_CUSTOMER. This is the pattern that SQL cannot find without a recursive CTE.

**Step 4 — Volume Trend Analysis (Snowflake)**

```sql
SELECT MONTH, TOTAL_SENT, TOTAL_RECEIVED
FROM MONTHLY_TX_SUMMARY
WHERE CUSTOMER_ID = 'C001'
ORDER BY MONTH
```

*What we learn:* A Snowflake line chart shows Alice's outgoing volume at a stable baseline from January through September 2025, then spiking **6× in December**. This time-series signal exists *only in Snowflake* — Neo4j stores individual transactions, not monthly aggregates.

**Step 5 — Compliance Obligations (Documents)**

The vector search retrieves the exact policy text from `aml_policy.pdf`:
> *"A SAR must be filed for transactions of $5,000 or more when the institution knows, suspects, or has reason to suspect the transaction involves funds from illegal activity..."*

*What we learn:* Three obligations are confirmed: EDD (risk score > 0.80), SAR filing (circular flow + Q4 spike), advisor escalation (John Miller's portfolio concentration).

**The conclusion:** No single data source could have produced this answer. The graph found the structural pattern. Snowflake found the temporal pattern. The documents provided the legal framework. The ontology connected all three into a compliance decision.

---

## Key Technical Concepts You Can Explore

**Graph RAG** — Retrieval-Augmented Generation using a graph database. Instead of embedding documents into vectors, you query a knowledge graph with Cypher, then pass both the query and results to the LLM with ontology context. The LLM explains what the graph *means*, not just what it *contains*.

**Hybrid RAG** — The system auto-routes questions to the right source:
- *"Which customers have risk > 0.8?"* → Graph (live data query)
- *"What is the SAR filing threshold?"* → Documents (policy lookup)
- *"Does Alice Morgan require EDD?"* → Hybrid (data + policy combined)

The router is itself a Claude call: classify this question as `structured`, `unstructured`, or `hybrid`.

**Semantic layer / Ontology** — The bridge between raw data and business meaning. In production systems, this is where compliance rules, regulatory thresholds, and risk definitions live — separate from both the data and the UI, injected into every LLM call as grounding context.

**Vector index in Neo4j** — The compliance PDFs are chunked (120 words, 20-word overlap), embedded with `sentence-transformers/all-MiniLM-L6-v2`, and stored as `DocumentChunk` nodes in the same Neo4j database as the graph. The vector index (`db.index.vector.queryNodes`) coexists with the graph index — you can ask a graph question and a semantic question in the same database session.

---

## Running It Locally in Under 10 Minutes

**Prerequisites:** Python 3.11+, Docker, an Anthropic or OpenAI API key.

```bash
# 1. Clone and set up
git clone https://github.com/ndulam/knowledge-graphs.git
cd knowledge-graphs
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

```bash
# 2. Configure environment
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY or OPENAI_API_KEY
```

```bash
# 3. Start Neo4j (Docker)
docker compose up -d
# Wait ~15 seconds. Neo4j Browser at http://localhost:7474
```

```bash
# 4. Run the full pipeline (one command)
python scripts/setup_all.py
# Generates data → loads Neo4j → creates PDFs → ingests vector index
# Add --skip-snowflake if you don't have a Snowflake account
```

```bash
# 5. Launch the UI
streamlit run ui/app.py
# Opens at http://localhost:8501
```

**Start here:** Open the **Investigation tab** and click "Run all steps" to see the full Alice Morgan case in under 30 seconds. Then open the **AI Agent tab** and ask:

> *"Should we file a SAR for Alice Morgan?"*

Watch the reasoning trace: the agent calls `search_documents` first to learn the SAR rules, then `query_graph` to find the circular flow and risk score, then `query_analytics` (if Snowflake is configured) to confirm the volume spike. The final answer cites specific amounts, policy sections, and ontology concepts.

---

## What This Means for Production AI Systems

The patterns in this demo apply directly to real AI agent architectures:

**Separate your data sources by question type.** A knowledge graph answers structural questions. A data warehouse answers aggregate questions. Documents answer policy questions. Trying to put all three into one system (a vector database, a SQL table, a single LLM context) creates a system that handles none of them well.

**Make the ontology explicit.** The biggest failure mode in enterprise AI is a model that treats `risk_score: 0.92` as just a number. The ontology converts data into meaning before the LLM ever sees it. This is not a prompt engineering trick — it is a data architecture decision.

**Show the reasoning trace.** The value of the AI agent in this demo is not just the final answer — it is the visible chain of: what data was retrieved, from which source, with which query, annotated with which ontology concepts, leading to which conclusion. Auditability is a first-class requirement in compliance use cases.

**Graphs scale where SQL cannot.** As your data model grows from 4 hops to 6 to N, Cypher's `*1..N` operator handles it in one pattern. SQL's equivalent grows as a combinatorial explosion of JOINs. For AI agents that need to traverse relationships at query time (not at ETL time), graph databases are not optional.

---

## Repository

The full project is on GitHub: [github.com/ndulam/knowledge-graphs](https://github.com/ndulam/knowledge-graphs)

It includes:
- Synthetic data generator with 8 embedded fraud scenarios
- Three compliance PDFs generated with `reportlab`
- Complete Streamlit UI with all 4 tabs
- Ontology definition in `docs/ontology_semantic_layer.md`
- 14 demo Cypher queries with SQL comparisons in the README
- One-click setup script with connection health checks

---

*If you found this useful, the repo is open — contributions, issues, and questions welcome.*

*Tags: #KnowledgeGraphs #GenerativeAI #Neo4j #Snowflake #Claude #RAG #AML #Compliance #Python #MachineLearning*
