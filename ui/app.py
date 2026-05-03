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
tab_agent, tab_explore, tab_custom = st.tabs(
    ["🤖 AI Agent", "🔍 Explore Concepts", "💬 Ask a Question"]
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
                        neo4j_driver, sf_conn, embedder, api_key, model
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
