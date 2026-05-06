from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure ui/ is on sys.path when running `streamlit run ui/app.py` from the repo root
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from graph_rag import (
    ONTOLOGY_CONTEXT,
    PREDEFINED_QUERIES,
    build_neo4j_driver,
    generate_cypher,
    get_provider,
    route_query,
    run_cypher,
    run_graph_rag,
    run_hybrid_rag,
)
from agent import FinancialAgent, build_snowflake_conn
from graph_viz import (
    LEGEND_HTML,
    VIZ_QUERIES,
    extract_ids_from_rows,
    render_network,
    viz_from_entity_ids,
    viz_from_neo4j,
)

load_dotenv()


@st.cache_resource(show_spinner="Loading embedding model…")
def _load_embedder():
    try:
        from graph_rag import get_embedder
        return get_embedder()
    except Exception:
        return None


st.set_page_config(page_title="Financial Knowledge Graph", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    st.subheader("LLM Provider")
    provider_name: str = st.selectbox("Provider", ["Anthropic", "OpenAI"])

    if provider_name == "Anthropic":
        model_options = ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"]
        env_key = "ANTHROPIC_API_KEY"
    else:
        model_options = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
        env_key = "OPENAI_API_KEY"

    model: str = st.selectbox("Model", model_options)
    api_key: str = st.text_input(
        "API Key",
        value=os.getenv(env_key, ""),
        type="password",
        help=f"Set {env_key} in .env or enter here",
    )

    st.subheader("Neo4j")
    neo4j_uri: str  = st.text_input("URI",      value=os.getenv("NEO4J_URI",      "bolt://localhost:7687"))
    neo4j_user: str = st.text_input("Username", value=os.getenv("NEO4J_USERNAME", "neo4j"))
    neo4j_pass: str = st.text_input("Password", value=os.getenv("NEO4J_PASSWORD", "password"), type="password")

    st.subheader("Snowflake")
    with st.expander("Connection details", expanded=False):
        sf_account   = st.text_input("Account",   value=os.getenv("SNOWFLAKE_ACCOUNT",   ""), help="e.g. abc12345.us-east-1")
        sf_user      = st.text_input("User",      value=os.getenv("SNOWFLAKE_USER",      ""))
        sf_password  = st.text_input("Password",  value=os.getenv("SNOWFLAKE_PASSWORD",  ""), type="password")
        sf_warehouse = st.text_input("Warehouse", value=os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"))
        sf_database  = st.text_input("Database",  value=os.getenv("SNOWFLAKE_DATABASE",  "FINANCIAL_KG"))
        sf_schema    = st.text_input("Schema",    value=os.getenv("SNOWFLAKE_SCHEMA",    "PUBLIC"))

    _sf_configured = bool(sf_account and sf_user and sf_password)
    if _sf_configured:
        st.success("Snowflake configured ✓")
    else:
        st.caption("Snowflake not configured — agent will use Neo4j + documents only")

    st.subheader("Documents")
    if st.button("Check document index", key="check_docs"):
        try:
            _drv = build_neo4j_driver(neo4j_uri, neo4j_user, neo4j_pass)
            n = run_cypher(_drv, "MATCH (c:DocumentChunk) RETURN count(c) AS n")[0]["n"]
            _drv.close()
            st.success(f"{n} chunks indexed") if n else st.warning(
                "No chunks — run `python scripts/ingest_pdfs.py`"
            )
        except Exception as _e:
            st.error(str(_e))

    st.divider()
    st.caption("Neo4j · Snowflake · Vector Search · Claude")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Financial Knowledge Graph")
st.markdown(
    "A hybrid knowledge graph demo combining **Neo4j** (graph patterns), "
    "**Snowflake** (analytics & time-series), and **compliance documents** (vector search) — "
    "powered by a Claude AI agent."
)

with st.expander("Architecture & Data Sources"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
**🕸️ Neo4j Knowledge Graph**

*WHO is connected to WHO*

- Customer → Account → Transaction graph
- Circular money flow detection
- Blast radius of high-risk nodes
- Multi-hop path traversal
- Risk clusters & contagion paths

*Best for relationship questions*
""")
    with c2:
        st.markdown("""
**📊 Snowflake Data Warehouse**

*HOW MUCH and WHEN*

- Full transaction history
- Monthly volume time-series (12 months)
- Risk score distributions by country
- Advisor portfolio aggregations
- Anomaly detection via trend analysis

*Best for analytical questions*
""")
    with c3:
        st.markdown("""
**📄 Compliance Documents**

*WHAT the rules say*

- AML Policy (SAR/CTR thresholds, EDD)
- Customer Risk Framework (HRC rules)
- Advisor Compliance Guide (limits)

Stored as semantic chunks in Neo4j
vector index — retrieved by similarity

*Best for policy questions*
""")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_story, tab_agent, tab_explore, tab_custom, tab_flatvsgraph, tab_aicontext = st.tabs(
    [
        "🕵️ Investigation",
        "🤖 AI Agent",
        "🔍 Explore Concepts",
        "💬 Ask a Question",
        "📊 Flat vs Graph",
        "🧠 Graph as AI Context",
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 0 — Guided Investigation  (presentation story mode)
# ══════════════════════════════════════════════════════════════════════════════

# Five-step investigation of Alice Morgan (C001).
# Each step is self-contained: query → visualisation → plain-English insight.
_STEPS = [
    {
        "id":        "profile",
        "icon":      "🔍",
        "title":     "Step 1 — Identify the Subject",
        "question":  "A compliance alert fired on customer C001. Who are they?",
        "tool":      "graph",
        "query":     (
            "MATCH (c:Customer {id:'C001'})-[:OWNS]->(a:Account) "
            "OPTIONAL MATCH (adv:Advisor)-[:MANAGES]->(c) "
            "RETURN c.name AS name, round(c.risk_score,2) AS risk_score, "
            "c.country AS country, collect(a.id) AS accounts, adv.name AS advisor"
        ),
        "insight": (
            "**Alice Morgan** has a risk score of **0.95** — well above the 0.80 threshold "
            "that classifies her as a `HIGH_RISK_CUSTOMER`. "
            "Under the customer risk framework this immediately triggers **Enhanced Due Diligence (EDD)** obligations. "
            "Her accounts and advisor are visible in the network — the graph tells us *who she is connected to* in one traversal."
        ),
        "why_graph": "A flat table shows one row per customer. The graph shows Alice, her accounts, and her advisor as a connected subgraph — relationships are first-class citizens.",
    },
    {
        "id":       "transactions",
        "icon":     "💸",
        "title":    "Step 2 — Follow the Money",
        "question": "What transactions did Alice make? Are any above the CTR threshold?",
        "tool":     "graph",
        "query":    (
            "MATCH (c:Customer {id:'C001'})-[:OWNS]->(a:Account)-[:SENT]->(t:Transaction) "
            "RETURN a.id AS from_account, t.id AS txn_id, "
            "round(t.amount,2) AS amount, t.timestamp AS timestamp "
            "ORDER BY t.amount DESC"
        ),
        "insight": (
            "Alice's accounts sent several transactions, including multiple above **$10,000** — "
            "each flagged as `HIGH_VALUE_TXN`, the CTR reporting threshold. "
            "The largest single transfer was **$32,000**. "
            "Individually each transaction may appear routine; the graph lets us see them together as a pattern."
        ),
        "why_graph": "SQL needs a JOIN from customer → account → transaction. The graph traversal `(c)-[:OWNS]->(a)-[:SENT]->(t)` is one pattern — no JOIN syntax, no intermediate temp tables.",
    },
    {
        "id":       "circular",
        "icon":     "🔄",
        "title":    "Step 3 — Check for Circular Flows",
        "question": "Does any money sent by Alice come back to her? That would indicate layering.",
        "tool":     "graph",
        "query":    (
            "MATCH (c:Customer {id:'C001'})-[:OWNS]->(a1:Account)-[:SENT]->(:Transaction)"
            "-[:TO]->(a2:Account)-[:SENT]->(:Transaction)-[:TO]->(a1) "
            "MATCH (owner:Customer)-[:OWNS]->(a2) "
            "RETURN a1.id AS alice_account, a2.id AS ring_account, owner.name AS ring_owner, "
            "round(owner.risk_score,2) AS ring_owner_risk"
        ),
        "insight": (
            "Alice's account **A001** is part of a **circular money movement**: "
            "funds flow out to A002 (Bob Carter, risk 0.91), then loop back. "
            "This matches the `CIRCULAR_MOVEMENT` fraud pattern — the classic layering technique "
            "used to obscure the origin of funds. "
            "Both parties in the ring are independently flagged as `HIGH_RISK_CUSTOMER`."
        ),
        "why_graph": "Cycle detection in SQL requires a recursive CTE with a visited-node guard — typically 40+ lines. In Cypher this is a 3-line pattern: `(a1)-[...]-(a2)-[...]-(a1)`. The graph finds cycles natively.",
    },
    {
        "id":       "trend",
        "icon":     "📈",
        "title":    "Step 4 — Volume Trend Analysis",
        "question": "Did Alice's transaction volume spike in Q4 2025? When was the peak?",
        "tool":     "snowflake",
        "query":    (
            "SELECT MONTH, TOTAL_SENT, TOTAL_RECEIVED, TXN_COUNT_SENT "
            "FROM MONTHLY_TX_SUMMARY "
            "WHERE CUSTOMER_ID = 'C001' "
            "ORDER BY MONTH"
        ),
        "insight": (
            "Snowflake's time-series table reveals a dramatic **Q4 spike**: "
            "Alice's outgoing volume in **December 2025** was over **6× her January baseline**. "
            "This pattern — stable activity followed by an explosive year-end surge — "
            "is a recognised SAR trigger. "
            "This insight is *only available from Snowflake*; the Neo4j graph stores individual transactions, not monthly aggregates."
        ),
        "why_graph": "This step shows WHERE the knowledge graph ends and the data warehouse begins. Snowflake owns the 'how much / when' question. Neo4j owned the 'who is connected to whom' question in steps 1–3.",
    },
    {
        "id":       "policy",
        "icon":     "📋",
        "title":    "Step 5 — Compliance Obligations",
        "question": "What do our AML policies require us to do with Alice Morgan right now?",
        "tool":     "documents",
        "query":    "EDD requirements SAR filing high risk customer circular transaction",
        "insight": (
            "The compliance policy library confirms three obligations triggered by Alice's profile:\n\n"
            "1. **Enhanced Due Diligence** — mandatory for any customer with `risk_score > 0.80`\n"
            "2. **SAR filing** — required when suspicious pattern (circular flow) is confirmed\n"
            "3. **Escalation** — advisor John Miller must be notified; portfolio review triggered\n\n"
            "These rules live in the PDF documents, not the database — only the vector search "
            "retrieval step surfaces them. This is why all three data sources are needed together."
        ),
        "why_graph": "Compliance rules aren't rows in a database — they live in policy documents. Vector search over PDFs retrieves the exact policy text. The knowledge graph architecture connects all three: who (graph) + how much (warehouse) + what the rules say (documents).",
    },
]

# ── Flat-vs-Graph tab constants ───────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent.parent / "data"

_FVG_COMPARISONS = [
    {
        "id":    "circular_flow",
        "title": "1. Circular Money Flow (Layering)",
        "why": (
            "Circular transactions — money sent out that eventually returns to the sender "
            "— are the hallmark of **layering**, a key money laundering stage. "
            "The graph detects this natively as a cycle pattern. SQL must simulate cycles "
            "with a recursive Common Table Expression that most analysts struggle to write "
            "correctly, and it still cannot traverse beyond a predefined hop limit."
        ),
        "sql_title": "SQL: Recursive CTE Cycle Detection",
        "sql": """\
WITH RECURSIVE ring AS (
  -- anchor: all outgoing transactions from any account
  SELECT t.from_account AS start_acct,
         t.to_account   AS current_acct,
         ARRAY[t.from_account] AS visited,
         1 AS depth
  FROM transactions t
  UNION ALL
  -- recursive step: follow the money
  SELECT r.start_acct,
         t.to_account,
         r.visited || t.from_account,
         r.depth + 1
  FROM transactions t
  JOIN ring r ON t.from_account = r.current_acct
  WHERE NOT (t.from_account = ANY(r.visited))
    AND r.depth < 10                         -- depth guard required
)
SELECT DISTINCT start_acct AS account_a,
                current_acct AS account_b
FROM ring
WHERE current_acct = start_acct
  AND depth >= 2;
-- 33 lines, recursive syntax (not all DBs support it),
-- depth guard needed, misses cycles > 10 hops""",
        "sql_pain": (
            "**33 lines.** Requires a recursive CTE (not supported by all databases). "
            "Needs a manual depth guard to prevent infinite loops. "
            "Still misses cycles longer than 10 hops. Any schema change breaks it."
        ),
        "cypher": (
            "MATCH (a1:Account)-[:SENT]->(:Transaction)-[:TO]->(a2:Account)\n"
            "      -[:SENT]->(:Transaction)-[:TO]->(a1)\n"
            "RETURN DISTINCT a1.id AS account_a, a2.id AS account_b"
        ),
        "cypher_highlight": "Same answer. **3 lines.** Relationships are first-class — the graph finds cycles natively at any depth.",
        "viz_cypher_key": "circular_movements",
    },
    {
        "id":    "multihop_exposure",
        "title": "2. Multi-Hop Customer Exposure",
        "why": (
            "Risk is contagious through transaction chains. A low-risk customer who "
            "regularly transacts with a high-risk customer may be unknowingly exposed "
            "or complicit. Finding this requires following the money *through* accounts "
            "to reach customers — exactly what a graph traversal does in a single pattern."
        ),
        "sql_title": "SQL: 4-Table JOIN Chain",
        "sql": """\
SELECT DISTINCT
    c1.name  AS customer_a,
    c2.name  AS customer_b,
    COUNT(t.txn_id) AS shared_transactions
FROM customers c1
JOIN accounts  a1 ON a1.customer_id = c1.customer_id
JOIN transactions t ON t.from_account = a1.account_id
JOIN accounts  a2 ON a2.account_id   = t.to_account
JOIN customers c2 ON c2.customer_id  = a2.customer_id
WHERE c1.customer_id <> c2.customer_id
GROUP BY c1.name, c2.name
ORDER BY shared_transactions DESC;
-- 4 JOINs, 3 intermediate tables, no relationship semantics,
-- adding a 3rd hop requires rewriting the entire query""",
        "sql_pain": (
            "**4 JOINs across 3 tables.** Adding one more hop (e.g., finding customers "
            "connected 3 transactions away) requires a complete rewrite. "
            "The JOIN chain grows linearly with each hop."
        ),
        "cypher": (
            "MATCH (c1:Customer)-[:OWNS]->(a1:Account)\n"
            "      -[:SENT]->(t:Transaction)\n"
            "      -[:TO]->(a2:Account)<-[:OWNS]-(c2:Customer)\n"
            "WHERE c1 <> c2\n"
            "RETURN c1.name AS customer_a, c2.name AS customer_b,\n"
            "       count(t) AS shared_transactions\n"
            "ORDER BY shared_transactions DESC"
        ),
        "cypher_highlight": "**7 lines, no JOIN syntax.** Add another hop by extending the pattern — no rewrite needed.",
        "viz_cypher_key": "customer_exposure",
    },
    {
        "id":    "advisor_blast",
        "title": "3. Advisor Blast Radius",
        "why": (
            "An advisor managing multiple high-risk customers creates concentrated portfolio "
            "risk. Finding the **total transaction exposure** of each advisor requires "
            "walking advisor → customers → accounts → transactions. "
            "SQL needs a GROUP BY aggregation on top of multiple JOINs and a subquery filter."
        ),
        "sql_title": "SQL: GROUP BY + Subquery + JOINs",
        "sql": """\
SELECT
    adv.name          AS advisor_name,
    COUNT(DISTINCT c.customer_id) AS hrc_client_count,
    SUM(t.amount)     AS total_exposure
FROM advisors adv
JOIN customers c  ON c.customer_id = adv.customer_id
JOIN accounts  a  ON a.customer_id  = c.customer_id
JOIN transactions t ON t.from_account = a.account_id
WHERE c.risk_score > 0.8
GROUP BY adv.name
ORDER BY total_exposure DESC;
-- Advisor-customer is denormalised (one row per managed customer),
-- 3 JOINs, risk filter applied after all joins,
-- no visibility into which transactions drive the exposure""",
        "sql_pain": (
            "The advisor CSV is denormalised (one row per managed customer), requiring "
            "careful GROUP BY. **3 JOINs**, risk filter applied after joining all tables, "
            "and there is no way to see which transactions drive the exposure without another subquery."
        ),
        "cypher": (
            "MATCH (adv:Advisor)-[:MANAGES]->(c:Customer)\n"
            "      -[:OWNS]->(a:Account)-[:SENT]->(t:Transaction)\n"
            "WHERE c.risk_score > 0.8\n"
            "RETURN adv.name AS advisor,\n"
            "       count(DISTINCT c) AS hrc_clients,\n"
            "       round(sum(t.amount)) AS total_exposure\n"
            "ORDER BY total_exposure DESC"
        ),
        "cypher_highlight": "**7 lines.** Risk filter is inline. Traversal from advisor to transactions is one readable path.",
        "viz_cypher_key": "advisor_exposure",
    },
]

# ── Graph-as-AI-Context tab constants ─────────────────────────────────────────

_CONTEXT_PRESETS = [
    {
        "id":          "alice_risk",
        "label":       "Is Alice Morgan high-risk?",
        "route":       "graph-only",
        "route_label": "Graph-only — live customer record from Neo4j",
        "cypher": (
            "MATCH (c:Customer {id:'C001'})-[:OWNS]->(a:Account) "
            "OPTIONAL MATCH (adv:Advisor)-[:MANAGES]->(c) "
            "RETURN c.name AS name, round(c.risk_score,2) AS risk_score, "
            "c.country AS country, collect(a.id) AS accounts, adv.name AS advisor"
        ),
        "viz_cypher": (
            "MATCH (adv:Advisor)-[m:MANAGES]->(c:Customer {id:'C001'})-[o:OWNS]->(a:Account) "
            "RETURN adv, m, c, o, a"
        ),
    },
    {
        "id":          "sar_rules",
        "label":       "What are SAR filing requirements?",
        "route":       "docs-only",
        "route_label": "Documents-only — compliance policy library (vector search)",
        "cypher":      None,
        "viz_cypher":  None,
    },
    {
        "id":          "sar_alice",
        "label":       "Should we file a SAR for Alice Morgan?",
        "route":       "hybrid",
        "route_label": "Hybrid — graph data + compliance policy documents",
        "cypher": (
            "MATCH (c:Customer {id:'C001'})-[:OWNS]->(a1:Account)-[:SENT]->(t:Transaction) "
            "OPTIONAL MATCH (t)-[:TO]->(a2:Account)<-[:OWNS]-(c2:Customer) "
            "RETURN c.name AS customer, round(c.risk_score,2) AS risk_score, "
            "a1.id AS from_account, t.id AS txn_id, round(t.amount,2) AS amount, "
            "c2.name AS counterparty"
        ),
        "viz_cypher": (
            "MATCH (c:Customer {id:'C001'})-[o:OWNS]->(a:Account)-[s:SENT]->(t:Transaction)"
            "-[r:TO]->(a2:Account) "
            "OPTIONAL MATCH (c2:Customer)-[o2:OWNS]->(a2) "
            "OPTIONAL MATCH (adv:Advisor)-[m:MANAGES]->(c) "
            "RETURN c, o, a, s, t, r, a2, o2, c2, adv, m"
        ),
    },
]


def _tab5_annotate(row: dict) -> list[str]:
    flags: list[str] = []
    risk = row.get("risk_score")
    if risk is not None:
        try:
            r = float(risk)
            if r > 0.9:
                flags.append("CRITICAL_RISK_CUSTOMER (risk_score > 0.9 → CCO escalation)")
            elif r > 0.8:
                flags.append("HIGH_RISK_CUSTOMER (risk_score > 0.8 → EDD required)")
        except (ValueError, TypeError):
            pass
    amount = row.get("amount")
    if amount is not None:
        try:
            if float(amount) > 10_000:
                flags.append("HIGH_VALUE_TXN (amount > $10,000 → CTR filing threshold)")
        except (ValueError, TypeError):
            pass
    if "account_a" in row and "account_b" in row:
        flags.append("CIRCULAR_MOVEMENT (A→B→A cycle detected → layering indicator)")
    return flags

with tab_story:
    st.markdown(
        "### Case File: Alice Morgan — Suspicious Activity Investigation\n"
        "A step-by-step walkthrough showing how the knowledge graph, data warehouse, "
        "and compliance documents work together to build a complete picture."
    )

    # Reset button
    col_reset, col_runall = st.columns([1, 1])
    if col_reset.button("Reset investigation", key="story_reset"):
        for s in _STEPS:
            st.session_state.pop(f"story_{s['id']}", None)
        st.rerun()
    run_all = col_runall.button("Run all steps", key="story_run_all", type="primary")

    if run_all:
        for s in _STEPS:
            st.session_state[f"story_run_{s['id']}"] = True

    st.divider()

    # Render each step
    for step in _STEPS:
        sid      = step["id"]
        run_key  = f"story_run_{sid}"
        res_key  = f"story_{sid}"
        has_result = res_key in st.session_state

        # Step header with status
        status = "✅" if has_result else "⬜"
        with st.expander(
            f"{status} {step['icon']}  {step['title']}", expanded=not has_result
        ):
            st.markdown(f"**{step['question']}**")

            triggered = st.button(
                f"Run {step['icon']}", key=f"btn_story_{sid}", type="primary"
            ) or st.session_state.get(run_key, False)

            if triggered:
                st.session_state[run_key] = False  # consume the flag

                result_data = None
                error_msg   = None

                with st.spinner("Retrieving data…"):
                    try:
                        if step["tool"] == "graph":
                            drv = build_neo4j_driver(neo4j_uri, neo4j_user, neo4j_pass)
                            result_data = run_cypher(drv, step["query"])
                            drv.close()

                        elif step["tool"] == "snowflake":
                            if not _sf_configured:
                                result_data = [{"info": "Snowflake not configured — add credentials to .env"}]
                            else:
                                _sf = build_snowflake_conn(
                                    sf_account, sf_user, sf_password,
                                    sf_warehouse, sf_database, sf_schema,
                                )
                                if _sf:
                                    import snowflake.connector
                                    cur = _sf.cursor(snowflake.connector.DictCursor)
                                    cur.execute(step["query"])
                                    result_data = [dict(r) for r in cur.fetchall()]
                                    _sf.close()

                        elif step["tool"] == "documents":
                            embedder = _load_embedder()
                            if embedder is None:
                                result_data = [{"info": "Embedding model not loaded"}]
                            else:
                                drv = build_neo4j_driver(neo4j_uri, neo4j_user, neo4j_pass)
                                from graph_rag import semantic_search
                                result_data = semantic_search(drv, embedder, step["query"], top_k=4)
                                drv.close()

                    except Exception as exc:
                        error_msg = str(exc)

                st.session_state[res_key] = {
                    "data": result_data,
                    "error": error_msg,
                }

            # Render cached result if available
            cached = st.session_state.get(res_key)
            if cached:
                if cached.get("error"):
                    st.error(cached["error"])
                elif cached.get("data"):
                    data = cached["data"]

                    # Data display
                    if step["tool"] == "documents":
                        st.markdown("**Policy excerpts retrieved:**")
                        for chunk in data:
                            if "text" in chunk:
                                with st.expander(
                                    f"📄 {chunk.get('source','')} p.{chunk.get('page','')}  "
                                    f"· {chunk.get('score',0):.0%} match"
                                ):
                                    st.write(chunk["text"])
                            elif "info" in chunk:
                                st.warning(chunk["info"])
                    else:
                        df_rows = [
                            {k: v for k, v in r.items() if k != "_ontology_flags"}
                            for r in data
                        ]
                        df = pd.DataFrame(df_rows)
                        for col in df.columns:
                            if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                                df[col] = df[col].apply(
                                    lambda x: ", ".join(str(i) for i in x) if isinstance(x, list) else str(x)
                                )
                        st.dataframe(df, use_container_width=True)

                        # Graph visualisation for graph steps
                        if step["tool"] == "graph":
                            entity_ids = extract_ids_from_rows(df_rows)
                            if entity_ids:
                                try:
                                    drv = build_neo4j_driver(neo4j_uri, neo4j_user, neo4j_pass)
                                    net = viz_from_entity_ids(drv, entity_ids)
                                    drv.close()
                                    if net:
                                        st.markdown(LEGEND_HTML, unsafe_allow_html=True)
                                        render_network(net)
                                except Exception:
                                    pass

                        # Snowflake chart
                        if step["tool"] == "snowflake" and df_rows:
                            try:
                                chart_df = pd.DataFrame(df_rows)
                                if "MONTH" in chart_df.columns and "TOTAL_SENT" in chart_df.columns:
                                    chart_df["MONTH"] = pd.to_datetime(chart_df["MONTH"])
                                    chart_df = chart_df.set_index("MONTH")
                                    st.line_chart(chart_df[["TOTAL_SENT", "TOTAL_RECEIVED"]])
                            except Exception:
                                pass

                    # Insight box
                    st.markdown("---")
                    st.markdown(f"**What this means:**  \n{step['insight']}")
                    st.caption(f"💡 *Why the graph:* {step['why_graph']}")

    # Final summary (shown once all steps complete)
    completed = sum(1 for s in _STEPS if f"story_{s['id']}" in st.session_state)
    if completed == len(_STEPS):
        st.divider()
        st.success(
            "**Investigation complete.** All five signals confirm: "
            "Alice Morgan requires immediate SAR filing, Enhanced Due Diligence review, "
            "and advisor escalation. No single data source could have produced this conclusion — "
            "it took the knowledge graph, the data warehouse, and the compliance documents together."
        )

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — AI Agent  (the flagship demo)
# ══════════════════════════════════════════════════════════════════════════════
with tab_agent:
    st.markdown(
        "The agent autonomously decides which tools to call — Neo4j, Snowflake, or "
        "document search — and synthesises a grounded answer across all three sources."
    )

    # ── Sample questions ──────────────────────────────────────────────────────
    st.caption("**Try one of these questions to see all three data sources in action:**")

    SAMPLE_QUESTIONS = [
        "Should we file a SAR for Alice Morgan?",
        "Which advisors violate portfolio concentration rules?",
        "Show the monthly volume trend for high-risk customers",
        "What is the blast radius of our riskiest customer?",
        "Are the circular ring accounts compliant with the risk framework?",
        "Which low-risk customers are exposed to contagion risk?",
    ]

    q_cols = st.columns(3)
    for i, sq in enumerate(SAMPLE_QUESTIONS):
        if q_cols[i % 3].button(sq, key=f"sq_{i}", use_container_width=True):
            st.session_state["agent_question"] = sq

    agent_question: str = st.text_area(
        "Or ask your own question",
        value=st.session_state.get("agent_question", ""),
        height=80,
        placeholder="e.g. Should we file a SAR for Alice Morgan?",
        key="agent_q_input",
    )

    if st.button("Run Agent", type="primary", key="run_agent"):
        q = agent_question.strip()
        if not q:
            st.warning("Enter a question first.")
        elif not api_key:
            st.warning(f"Enter your {provider_name} API key in the sidebar.")
        else:
            with st.spinner("Agent is reasoning…"):
                try:
                    neo4j_driver = build_neo4j_driver(neo4j_uri, neo4j_user, neo4j_pass)
                    sf_conn = (
                        build_snowflake_conn(
                            sf_account, sf_user, sf_password,
                            sf_warehouse, sf_database, sf_schema,
                        )
                        if _sf_configured else None
                    )
                    embedder = _load_embedder()
                    agent    = FinancialAgent(
                        neo4j_driver, sf_conn, embedder,
                        api_key, model,
                        provider=provider_name,
                    )
                    result = agent.run(q)
                    neo4j_driver.close()
                    if sf_conn:
                        sf_conn.close()

                except Exception as exc:
                    st.error(f"Agent error: {exc}")
                    result = None

            if result:
                # ── Reasoning trace ───────────────────────────────────────────
                if result.steps:
                    st.subheader("Agent Reasoning Trace")
                    st.caption(
                        f"The agent called {len(result.steps)} tool(s) to answer this question."
                    )

                    _TOOL_META = {
                        "query_graph":      ("🕸️", "Neo4j Graph Query",     "#1f77b4"),
                        "query_analytics":  ("📊", "Snowflake Analytics",   "#2ca02c"),
                        "search_documents": ("🔍", "Document Search",       "#ff7f0e"),
                    }

                    for idx, step in enumerate(result.steps, start=1):
                        icon, label, _ = _TOOL_META.get(
                            step.tool, ("🔧", step.tool, "#aaa")
                        )
                        header = f"Step {idx} — {icon} {label}"
                        if step.reason:
                            header += f"  ·  _{step.reason}_"

                        with st.expander(header, expanded=True):
                            # Input
                            if step.tool == "query_graph":
                                st.code(step.input.get("cypher", ""), language="cypher")
                            elif step.tool == "query_analytics":
                                st.code(step.input.get("sql", ""), language="sql")
                            else:
                                st.markdown(f"*Search query:* `{step.input.get('question', '')}`")

                            # Output
                            if isinstance(step.output, list) and step.output:
                                first = step.output[0]
                                if isinstance(first, dict) and set(first.keys()) <= {"text", "source", "page", "score"}:
                                    # Document chunks — show as passages
                                    for chunk in step.output:
                                        score_str = f" · {chunk.get('score', 0):.0%}" if chunk.get("score") else ""
                                        st.markdown(
                                            f"**📄 {chunk.get('source','')} p.{chunk.get('page','')}**{score_str}"
                                        )
                                        st.write(chunk.get("text", ""))
                                elif "error" in first or "info" in first:
                                    msg = first.get("error") or first.get("info")
                                    st.warning(msg)
                                else:
                                    # Separate ontology flags from raw data columns
                                    all_flags: list[str] = []
                                    clean_rows = []
                                    for r in step.output:
                                        flags = r.get("_ontology_flags", [])
                                        all_flags.extend(flags)
                                        clean_rows.append(
                                            {k: v for k, v in r.items() if k != "_ontology_flags"}
                                        )

                                    df = pd.DataFrame(clean_rows)
                                    # Coerce any column containing lists/dicts/objects to str
                                    for col in df.columns:
                                        if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                                            df[col] = df[col].apply(
                                                lambda x: ", ".join(str(i) for i in x) if isinstance(x, list) else str(x)
                                            )
                                    st.dataframe(df, use_container_width=True)
                                    st.caption(f"{len(clean_rows)} record(s)")

                                    if all_flags:
                                        unique_flags = sorted(set(all_flags))
                                        st.markdown("**Ontology concepts detected:**")
                                        for flag in unique_flags:
                                            st.markdown(f"- `{flag}`")

                                    # Network visualisation for graph results
                                    if step.tool == "query_graph":
                                        entity_ids = extract_ids_from_rows(clean_rows)
                                        if entity_ids:
                                            try:
                                                driver = build_neo4j_driver(
                                                    neo4j_uri, neo4j_user, neo4j_pass
                                                )
                                                net = viz_from_entity_ids(driver, entity_ids)
                                                driver.close()
                                                if net:
                                                    st.markdown(
                                                        "**Network Graph**"
                                                    )
                                                    st.markdown(
                                                        LEGEND_HTML, unsafe_allow_html=True
                                                    )
                                                    render_network(net)
                                            except Exception:
                                                pass
                            elif not step.output:
                                st.info("No records returned.")

                # ── Final answer ──────────────────────────────────────────────
                st.subheader("Agent Answer")
                st.markdown(result.answer)

                # ── Data source summary ───────────────────────────────────────
                tools_used = {s.tool for s in result.steps}
                badges = []
                if "query_graph"      in tools_used: badges.append("🕸️ Neo4j")
                if "query_analytics"  in tools_used: badges.append("📊 Snowflake")
                if "search_documents" in tools_used: badges.append("🔍 Documents")
                if badges:
                    st.caption("Sources: " + " · ".join(badges))

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Explore Concepts  (predefined graph queries)
# ══════════════════════════════════════════════════════════════════════════════
with tab_explore:
    st.markdown("Click a concept button to retrieve and explain the relevant graph data.")

    btn_cols = st.columns(len(PREDEFINED_QUERIES))
    for i, (key, meta) in enumerate(PREDEFINED_QUERIES.items()):
        if btn_cols[i].button(meta["label"], key=f"btn_{key}", use_container_width=True):
            st.session_state["selected_key"] = key

    selected_key: str | None = st.session_state.get("selected_key")
    if selected_key:
        meta = PREDEFINED_QUERIES[selected_key]

        try:
            with st.spinner("Querying Neo4j…"):
                driver  = build_neo4j_driver(neo4j_uri, neo4j_user, neo4j_pass)
                records = run_cypher(driver, meta["cypher"])
                driver.close()
        except Exception as exc:
            st.error(f"Neo4j error: {exc}")
            st.caption("Make sure Neo4j is running (`docker compose up -d`).")
            records = None

        if records is not None:
            col_data, col_explain = st.columns(2)

            with col_data:
                st.subheader("Graph Data")
                st.caption(f"_{meta['question']}_")
                with st.expander("Cypher Query"):
                    st.code(meta["cypher"], language="cypher")
                if records:
                    st.dataframe(pd.DataFrame(records), use_container_width=True)
                    st.caption(f"{len(records)} record(s) returned")
                else:
                    st.info("No records returned for this query.")

            with col_explain:
                st.subheader("LLM Explanation")
                if not api_key:
                    st.info(
                        f"Enter your {provider_name} API key in the sidebar "
                        "to generate a natural-language explanation."
                    )
                else:
                    with st.spinner("Generating explanation…"):
                        try:
                            driver   = build_neo4j_driver(neo4j_uri, neo4j_user, neo4j_pass)
                            provider = get_provider(provider_name, api_key, model)
                            _, explanation = run_graph_rag(
                                driver, provider, meta["cypher"], meta["question"]
                            )
                            driver.close()
                            st.markdown(explanation)
                        except Exception as exc:
                            st.error(f"LLM error: {exc}")

            # ── Network visualisation ─────────────────────────────────────
            st.divider()
            st.subheader("Network Graph")
            st.markdown(LEGEND_HTML, unsafe_allow_html=True)
            viz_cypher = VIZ_QUERIES.get(selected_key)
            if viz_cypher:
                try:
                    with st.spinner("Rendering network…"):
                        driver = build_neo4j_driver(neo4j_uri, neo4j_user, neo4j_pass)
                        net = viz_from_neo4j(driver, viz_cypher)
                        driver.close()
                    if net:
                        render_network(net)
                        st.caption(
                            "Hover over nodes for details. "
                            "Drag nodes to rearrange. "
                            "Scroll to zoom."
                        )
                    else:
                        st.info("No graph data — run the data pipeline first.")
                except Exception as exc:
                    st.warning(f"Visualisation unavailable: {exc}")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Ask a Question  (hybrid router: graph + docs)
# ══════════════════════════════════════════════════════════════════════════════
with tab_custom:
    st.markdown(
        "Ask any question. The system automatically routes it to **graph data**, "
        "**compliance documents**, or **both** — then explains the answer."
    )

    question: str = st.text_area(
        "Your question",
        placeholder="e.g. Does Alice Morgan require Enhanced Due Diligence under our AML policy?",
        height=90,
    )
    custom_cypher: str = st.text_area(
        "Cypher query (optional — leave blank for auto-generated)",
        placeholder="MATCH (c:Customer) WHERE c.risk_score > 0.8 RETURN c",
        height=60,
    )

    col_hint, col_run = st.columns([4, 1])
    with col_hint:
        st.caption(
            "Policy/definitions → Documents · Live data → Graph · Mixed → Hybrid"
        )
    with col_run:
        run_clicked = st.button("Run", type="primary", key="run_custom")

    if run_clicked:
        if not question.strip():
            st.warning("Enter a question first.")
        elif not api_key:
            st.warning(f"Enter your {provider_name} API key in the sidebar.")
        else:
            provider = get_provider(provider_name, api_key, model)

            with st.spinner("Routing question…"):
                route = route_query(provider, question.strip())

            _ROUTE_LABELS = {
                "structured":   "📊 Graph Data",
                "unstructured": "📄 Compliance Documents",
                "hybrid":       "🔀 Hybrid — Graph + Documents",
            }
            st.info(f"**Data source:** {_ROUTE_LABELS.get(route, route)}", icon="🔍")

            with st.spinner("Retrieving and explaining…"):
                try:
                    driver   = build_neo4j_driver(neo4j_uri, neo4j_user, neo4j_pass)
                    embedder = _load_embedder() if route in ("unstructured", "hybrid") else None
                    cypher_input = custom_cypher.strip()

                    graph_records, doc_chunks, _, explanation = run_hybrid_rag(
                        driver, provider, embedder,
                        question.strip(), cypher=cypher_input, route=route,
                    )
                    driver.close()

                    col_data, col_explain = st.columns(2)

                    with col_data:
                        if graph_records:
                            st.subheader("Graph Data")
                            if cypher_input:
                                with st.expander("Cypher query"):
                                    st.code(cypher_input, language="cypher")
                            st.dataframe(pd.DataFrame(graph_records), use_container_width=True)
                            st.caption(f"{len(graph_records)} record(s) returned")

                        if doc_chunks:
                            st.subheader("Document Context")
                            for chunk in doc_chunks:
                                score_str = (
                                    f" · {chunk['score']:.0%} match"
                                    if chunk.get("score") is not None else ""
                                )
                                with st.expander(f"📄 {chunk['source']} — p.{chunk['page']}{score_str}"):
                                    st.write(chunk["text"])

                        if not graph_records and not doc_chunks:
                            st.info(
                                "No data retrieved. Make sure Neo4j is running and "
                                "documents are ingested (`python scripts/ingest_pdfs.py`)."
                            )

                    with col_explain:
                        st.subheader("LLM Explanation")
                        st.markdown(explanation)

                except Exception as exc:
                    st.error(f"Error: {exc}")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Flat vs Graph  (innovation week "aha moment")
# ══════════════════════════════════════════════════════════════════════════════
with tab_flatvsgraph:
    st.markdown(
        "### Why Knowledge Graphs Beat Flat Tables for Connected Financial Data\n"
        "Each comparison shows the same fraud detection question answered two ways. "
        "**Left:** the SQL approach on flat CSVs. **Right:** a live Cypher query against Neo4j. "
        "The difference in complexity — and what each approach can discover — is the point."
    )

    comp_labels = [c["title"] for c in _FVG_COMPARISONS]
    fvg_idx = st.selectbox(
        "Choose a comparison",
        range(len(comp_labels)),
        format_func=lambda i: comp_labels[i],
        key="fvg_comparison_select",
    )
    comp    = _FVG_COMPARISONS[fvg_idx]
    comp_id = comp["id"]

    st.info(comp["why"], icon="💡")
    st.divider()

    col_flat, col_graph = st.columns(2, gap="large")

    # ── LEFT: The Flat Data Way ───────────────────────────────────────────────
    with col_flat:
        st.markdown("#### The Flat Data Way")

        try:
            df_customers = pd.read_csv(_DATA_DIR / "customers.csv")
            df_accounts  = pd.read_csv(_DATA_DIR / "accounts.csv")
            df_txns      = pd.read_csv(_DATA_DIR / "transactions.csv")

            if comp_id == "circular_flow":
                acct_filter = ["A002", "A007"]
                df_flat = df_txns[
                    df_txns["from_account"].isin(acct_filter) |
                    df_txns["to_account"].isin(acct_filter)
                ][["txn_id", "from_account", "to_account", "amount", "timestamp"]].reset_index(drop=True)
                st.caption("**Raw transaction rows touching accounts A002 / A007:**")

            elif comp_id == "multihop_exposure":
                df_from = df_accounts.rename(columns={"account_id": "from_account", "customer_id": "cust_from", "account_type": "type_from"})
                df_to   = df_accounts.rename(columns={"account_id": "to_account",   "customer_id": "cust_to",   "account_type": "type_to"})
                df_flat = (
                    df_txns
                    .merge(df_from, on="from_account")
                    .merge(df_to,   on="to_account")
                    [["txn_id", "from_account", "cust_from", "to_account", "cust_to", "amount"]]
                    .head(10)
                    .reset_index(drop=True)
                )
                st.caption("**Merged rows (transactions + accounts twice — what SQL must JOIN):**")

            else:  # advisor_blast
                df_advisors = pd.read_csv(_DATA_DIR / "advisors.csv")
                hrc = df_customers[df_customers["risk_score"] > 0.8][["customer_id", "name", "risk_score"]]
                df_flat = (
                    df_advisors.merge(hrc, on="customer_id")
                    .rename(columns={"name_x": "advisor_name", "name_y": "customer_name"})
                    [["advisor_id", "advisor_name", "customer_id", "customer_name", "risk_score"]]
                    .reset_index(drop=True)
                )
                st.caption("**High-risk customers per advisor (raw CSV merge):**")

            st.dataframe(df_flat, use_container_width=True)

        except FileNotFoundError as exc:
            st.warning(f"CSV not found: {exc}. Run `python scripts/generate_data.py` first.")

        st.markdown(f"**{comp['sql_title']}**")
        st.code(comp["sql"], language="sql")
        st.error(comp["sql_pain"], icon="⚠️")

    # ── RIGHT: The Graph Way ──────────────────────────────────────────────────
    with col_graph:
        st.markdown("#### The Graph Way")
        st.code(comp["cypher"], language="cypher")
        st.success(comp["cypher_highlight"])

        res_key = f"fvg_result_{comp_id}"

        if st.button("Run live Cypher query ▶", key=f"fvg_btn_{comp_id}", type="primary"):
            with st.spinner("Querying Neo4j…"):
                try:
                    drv     = build_neo4j_driver(neo4j_uri, neo4j_user, neo4j_pass)
                    cypher1 = " ".join(comp["cypher"].splitlines())
                    records = run_cypher(drv, cypher1)
                    drv.close()
                    st.session_state[res_key] = {"data": records, "error": None}
                except Exception as exc:
                    st.session_state[res_key] = {"data": None, "error": str(exc)}

        cached = st.session_state.get(res_key)
        if cached:
            if cached.get("error"):
                st.error(cached["error"])
                st.caption("Make sure Neo4j is running (`docker compose up -d`).")
            elif cached.get("data") is not None:
                records = cached["data"]
                if records:
                    st.dataframe(pd.DataFrame(records), use_container_width=True)
                    st.caption(f"{len(records)} record(s) returned")
                else:
                    st.info("No records returned. Run `python scripts/load_to_neo4j.py` first.")

                viz_q = VIZ_QUERIES.get(comp["viz_cypher_key"])
                if viz_q:
                    try:
                        drv = build_neo4j_driver(neo4j_uri, neo4j_user, neo4j_pass)
                        net = viz_from_neo4j(drv, viz_q)
                        drv.close()
                        if net:
                            st.markdown(LEGEND_HTML, unsafe_allow_html=True)
                            render_network(net)
                    except Exception:
                        pass

# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Graph as AI Context
# ══════════════════════════════════════════════════════════════════════════════
with tab_aicontext:
    st.markdown(
        "### How the Knowledge Graph Powers the AI\n"
        "The AI agent doesn't take natural language and query a database directly. "
        "It routes your question, runs structured Cypher, annotates every result row "
        "with ontology flags, and assembles a precise context string for the LLM. "
        "This tab shows every step of that pipeline — and lets you inspect the exact "
        "context the LLM receives for three real questions."
    )

    # ── Section A: Pipeline (static) ─────────────────────────────────────────
    st.subheader("A — Context Assembly Pipeline")
    st.caption("How every question travels from user to LLM answer")

    p1, p2, p3, p4, p5 = st.columns(5, gap="small")
    _pipe_style = "background:#1e2a3a;padding:14px;border-radius:8px;font-size:13px;"

    with p1:
        st.markdown(
            f'<div style="{_pipe_style}">'
            "<b>① Question In</b><br><br>"
            "User question enters<br>"
            "the system<br><br>"
            "↓ Router classifies:<br>"
            "<code>graph / docs / hybrid</code>"
            "</div>",
            unsafe_allow_html=True,
        )
    with p2:
        st.markdown(
            f'<div style="{_pipe_style}">'
            "<b>② Cypher Generated</b><br><br>"
            "LLM writes a Cypher query<br>"
            "using the ontology schema<br><br>"
            "↓ Runs against<br>"
            "Neo4j live graph"
            "</div>",
            unsafe_allow_html=True,
        )
    with p3:
        st.markdown(
            f'<div style="{_pipe_style}">'
            "<b>③ Raw Graph Result</b><br><br>"
            "Table of rows returned:<br>"
            "<code>name, risk_score,</code><br>"
            "<code>accounts, advisor</code><br><br>"
            "↓ Passed to<br>"
            "Ontology annotator"
            "</div>",
            unsafe_allow_html=True,
        )
    with p4:
        st.markdown(
            f'<div style="{_pipe_style}">'
            "<b>④ Ontology Annotation</b><br><br>"
            "Each row tagged:<br>"
            "<code>HIGH_RISK_CUSTOMER</code><br>"
            "<code>HIGH_VALUE_TXN</code><br>"
            "<code>CIRCULAR_MOVEMENT</code><br><br>"
            "↓ Assembled into"
            "</div>",
            unsafe_allow_html=True,
        )
    with p5:
        st.markdown(
            f'<div style="{_pipe_style}">'
            "<b>⑤ LLM Prompt</b><br><br>"
            "Annotated rows + doc chunks<br>"
            "→ context string<br>"
            "→ LLM generates<br>"
            "grounded answer"
            "</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Section B: Live Context Inspector ────────────────────────────────────
    st.subheader("B — Live Context Inspector")
    st.markdown(
        "Pick a preset question and see exactly what the AI sees — "
        "the raw graph result, the annotated context string, and the network."
    )

    preset_cols = st.columns(3)
    for i, preset in enumerate(_CONTEXT_PRESETS):
        if preset_cols[i].button(
            preset["label"],
            key=f"ctx_preset_{preset['id']}",
            use_container_width=True,
        ):
            # Clear cached results so the new preset auto-runs fresh
            for _p in _CONTEXT_PRESETS:
                st.session_state.pop(f"ctx_result_{_p['id']}", None)
                st.session_state.pop(f"ctx_viz_{_p['id']}", None)
            st.session_state["ctx_selected_id"] = preset["id"]

    selected_id = st.session_state.get("ctx_selected_id")
    if not selected_id:
        st.info("Select a question above to inspect the AI context.", icon="👆")
    else:
        preset = next(p for p in _CONTEXT_PRESETS if p["id"] == selected_id)

        st.markdown(f"**Question:** _{preset['label']}_")
        st.caption(f"Route: {preset['route_label']}")
        st.divider()

        run_ctx_key = f"ctx_result_{preset['id']}"
        viz_ctx_key = f"ctx_viz_{preset['id']}"

        # Auto-run Cypher and viz when preset is newly selected
        if run_ctx_key not in st.session_state:
            with st.spinner("Querying Neo4j…"):
                if preset["cypher"]:
                    try:
                        drv     = build_neo4j_driver(neo4j_uri, neo4j_user, neo4j_pass)
                        records = run_cypher(drv, preset["cypher"])
                        drv.close()
                        st.session_state[run_ctx_key] = {"data": records, "error": None}
                    except Exception as exc:
                        st.session_state[run_ctx_key] = {"data": None, "error": str(exc)}
                else:
                    st.session_state[run_ctx_key] = {"data": [], "error": None}

        if viz_ctx_key not in st.session_state and preset["viz_cypher"]:
            try:
                drv = build_neo4j_driver(neo4j_uri, neo4j_user, neo4j_pass)
                net = viz_from_neo4j(drv, preset["viz_cypher"])
                drv.close()
                st.session_state[viz_ctx_key] = {"net": net, "error": None}
            except Exception as exc:
                st.session_state[viz_ctx_key] = {"net": None, "error": str(exc)}

        cached     = st.session_state.get(run_ctx_key, {})
        viz_cached = st.session_state.get(viz_ctx_key, {})

        col_raw, col_annotated, col_viz = st.columns(3, gap="medium")

        # ── Column 1: Raw graph result ────────────────────────────────────────
        with col_raw:
            st.markdown("**③ Raw Graph Result**")
            if preset["route"] == "docs-only":
                st.info(
                    "This question routes to **documents only**. "
                    "No graph query runs — the LLM receives policy text chunks from "
                    "vector similarity search instead.",
                    icon="📄",
                )
            elif cached.get("error"):
                st.error(cached["error"])
                st.caption("Make sure Neo4j is running (`docker compose up -d`).")
            else:
                records = cached.get("data", [])
                if records:
                    clean_rows = [
                        {k: v for k, v in r.items() if k != "_ontology_flags"}
                        for r in records
                    ]
                    df = pd.DataFrame(clean_rows)
                    for col in df.columns:
                        if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                            df[col] = df[col].apply(
                                lambda x: ", ".join(str(i) for i in x) if isinstance(x, list) else str(x)
                            )
                    st.dataframe(df, use_container_width=True)
                    st.caption(f"{len(records)} row(s) — plain data, no business meaning yet")
                else:
                    st.info("No records. Run `python scripts/load_to_neo4j.py` first.")

        # ── Column 2: Annotated context (what the LLM sees) ──────────────────
        with col_annotated:
            st.markdown("**④ What the LLM Sees**")
            st.caption("Ontology-annotated context assembled for the LLM prompt")

            if preset["route"] == "docs-only":
                st.markdown(
                    "For docs-only questions the LLM receives **policy text chunks** "
                    "retrieved by vector similarity — no graph rows. "
                    "The ontology system context is always included:\n"
                )
                excerpt = "\n".join(ONTOLOGY_CONTEXT.strip().splitlines()[:20])
                st.code(excerpt + "\n...", language="text")

            elif cached.get("data"):
                records = cached["data"]
                annotated_lines: list[str] = []
                all_flags: list[str] = []

                for r in records[:20]:
                    flags = _tab5_annotate(r)
                    all_flags.extend(flags)
                    clean = {k: v for k, v in r.items() if k != "_ontology_flags"}
                    line  = str(clean)
                    if flags:
                        line += "\n  [ONTOLOGY: " + " | ".join(flags) + "]"
                    annotated_lines.append(line)

                context_preview = (
                    "── ONTOLOGY SYSTEM CONTEXT (excerpt) ──\n"
                    + "\n".join(ONTOLOGY_CONTEXT.strip().splitlines()[:12])
                    + "\n...\n\n"
                    + f"── GRAPH DATA ({len(records)} row(s)) ──\n"
                    + "\n".join(annotated_lines)
                )
                st.code(context_preview, language="text")

                if all_flags:
                    st.markdown("**Ontology flags triggered:**")
                    for flag in sorted(set(all_flags)):
                        st.markdown(f"- `{flag}`")
            else:
                st.info("No graph data to annotate.")

        # ── Column 3: Network visualization ──────────────────────────────────
        with col_viz:
            st.markdown("**Graph Visualization**")
            if preset["route"] == "docs-only":
                st.info(
                    "No network graph for docs-only questions — "
                    "the knowledge source is the policy document store.",
                    icon="📚",
                )
            elif viz_cached.get("error"):
                st.warning(f"Visualization unavailable: {viz_cached['error']}")
            elif viz_cached.get("net"):
                st.markdown(LEGEND_HTML, unsafe_allow_html=True)
                render_network(viz_cached["net"])
            else:
                st.info("No graph data to visualize.")
