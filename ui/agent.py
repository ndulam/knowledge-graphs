"""
Financial Knowledge Graph AI Agent — LangGraph implementation
=============================================================
Provider-agnostic agent backed by LangGraph's create_react_agent.
Supports any LangChain-compatible model (Claude, OpenAI, etc.)
via a unified BaseChatModel interface.

Three tools:
  query_graph        — Neo4j Cypher  (WHO / HOW, network patterns)
  query_analytics    — Snowflake SQL  (HOW MUCH / WHEN, time-series)
  search_documents   — Vector search  (WHAT rules say, compliance docs)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Annotated, Any

from graph_rag import ONTOLOGY_CONTEXT, EmbeddingProvider, run_cypher, semantic_search

# ── System prompt ─────────────────────────────────────────────────────────────

_AGENT_SYSTEM_BASE = """\
You are an AI compliance analyst for a financial institution with access to three
specialised data tools. Always retrieve facts before drawing conclusions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KNOWLEDGE GRAPH ONTOLOGY  (use these definitions when reasoning)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ontology}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOL 1 — query_graph  (Neo4j Knowledge Graph)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use for: relationship traversal, circular money flows, blast radius,
multi-hop paths, risk clusters, hub accounts, network patterns.
Results are pre-annotated with ontology flags — use them when reasoning.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOL 2 — query_analytics  (Snowflake Data Warehouse)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use for: transaction volumes, time-series trends, aggregations,
risk score distributions, country-level exposure, advisor stats.
Tables (uppercase): CUSTOMERS, ACCOUNTS, TRANSACTIONS, ADVISORS,
MONTHLY_TX_SUMMARY(MONTH, CUSTOMER_ID, CUSTOMER_NAME, RISK_SCORE,
  COUNTRY, TOTAL_SENT, TOTAL_RECEIVED, TXN_COUNT_SENT,
  TXN_COUNT_RECEIVED, UNIQUE_COUNTERPARTIES)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOL 3 — search_documents  (Compliance Policy Library)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use for: AML policy rules, risk score thresholds, SAR/CTR filing
requirements, EDD procedures, advisor portfolio limits, definitions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
APPROACH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. For compliance questions call search_documents FIRST to retrieve
   exact policy thresholds, then use graph/analytics to check data.
2. Apply ontology definitions when interpreting graph results —
   risk_score > 0.8 is HIGH_RISK_CUSTOMER that triggers EDD;
   A->B->A is CIRCULAR_MOVEMENT, a known fraud indicator.
3. For questions needing both trends and network patterns, call
   query_analytics for time-series then query_graph for structure.
4. Always cite specific values, ontology labels, and policy thresholds.
5. Note any unavailable tool and proceed with the data you have.\
"""

_SNOWFLAKE_UNAVAILABLE = """

NOTE: Snowflake (query_analytics) is not connected in this session.
Do NOT call query_analytics. Use Neo4j and documents only.\
"""


def _system_prompt(sf_available: bool) -> str:
    base = _AGENT_SYSTEM_BASE.format(ontology=ONTOLOGY_CONTEXT)
    return base if sf_available else base + _SNOWFLAKE_UNAVAILABLE


# ── Ontology annotation ───────────────────────────────────────────────────────

def _annotate_ontology(row: dict) -> dict:
    """Tag each graph result row with ontology concept labels."""
    flags: list[str] = []

    risk = row.get("risk_score")
    if risk is not None and float(risk) > 0.8:
        flags.append("HIGH_RISK_CUSTOMER (risk_score > 0.8 -> EDD required)")
    if risk is not None and float(risk) > 0.9:
        flags.append("CRITICAL_RISK_CUSTOMER (risk_score > 0.9 -> immediate CCO escalation)")

    amount = row.get("amount")
    if amount is not None and float(amount) > 10_000:
        flags.append("HIGH_VALUE_TXN (amount > $10,000 -> CTR filing threshold)")

    if "account_a" in row and "account_b" in row:
        flags.append("CIRCULAR_MOVEMENT (A->B->A cycle detected -> layering indicator)")

    hrc = row.get("high_risk_count")
    if hrc is not None and int(hrc) > 0:
        flags.append(f"ADVISOR_RISK_EXPOSURE ({hrc} HIGH_RISK_CUSTOMER(s) in portfolio)")

    # Structuring / smurfing
    total_inflow = row.get("total_inflow") or row.get("total_in")
    avg_outflow  = row.get("avg_outflow") or row.get("avg_out")
    outflow_count = row.get("outflow_count")
    if (total_inflow is not None and float(total_inflow) > 10_000
            and avg_outflow is not None and float(avg_outflow) < 5_000
            and outflow_count is not None and int(outflow_count) >= 3):
        flags.append("STRUCTURING (large inflow + multiple small outflows -> CTR avoidance)")

    # Layering chain
    if "hop_1" in row and "hop_4" in row:
        flags.append("LAYERING (3-hop transaction chain with declining amounts -> money trail obfuscation)")

    # Velocity anomaly
    burst = row.get("burst_count")
    if burst is not None and int(burst) >= 3:
        flags.append(f"VELOCITY_ANOMALY ({burst} transactions in one day -> burst-firing pattern)")

    # Geographic concentration risk
    from_country = row.get("from_country")
    to_country   = row.get("to_country")
    if from_country and to_country and from_country != to_country:
        txn_count = row.get("txn_count", 0)
        if int(txn_count) > 0:
            flags.append(f"GEOGRAPHIC_RISK ({from_country}→{to_country} cross-border HRC flow)")

    # Money mule
    forward_pct = row.get("forward_pct")
    if forward_pct is not None and float(forward_pct) >= 85:
        flags.append(f"MONEY_MULE ({forward_pct}% of received funds forwarded -> pass-through conduit)")

    # Advisor network contagion
    contagion = row.get("contagion_count") or (
        len(row["hrc_counterparties"]) if isinstance(row.get("hrc_counterparties"), list) else None
    )
    if contagion is not None and int(contagion) > 0:
        flags.append(f"ADVISOR_NETWORK_CONTAGION ({contagion} HRC counterparty contact(s) -> indirect portfolio contamination)")

    if flags:
        row = dict(row)
        row["_ontology_flags"] = flags
    return row


# ── LLM factory ──────────────────────────────────────────────────────────────

def build_llm(provider: str, api_key: str, model: str):
    """Return a LangChain BaseChatModel for the chosen provider."""
    if provider == "Anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, api_key=api_key, max_tokens=4096)
    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=api_key)
    raise ValueError(f"Unknown provider '{provider}'. Supported: Anthropic, OpenAI")


# ── Tool factory (closures capture runtime dependencies) ──────────────────────

def make_tools(
    neo4j_driver: Any,
    sf_conn: Any,
    embedder: EmbeddingProvider | None,
    sf_available: bool,
) -> list:
    """Build LangChain tool objects with injected runtime dependencies."""
    from langchain_core.tools import tool

    @tool
    def query_graph(
        cypher: Annotated[str, "Valid Cypher query to execute against Neo4j"],
        reason: Annotated[str, "One sentence explaining what this query checks"],
    ) -> str:
        """Run a Cypher query against the Neo4j knowledge graph.
        Best for relationship traversal, cycle detection, blast radius,
        multi-hop paths, and network-level risk patterns.
        """
        try:
            rows = run_cypher(neo4j_driver, cypher)
            annotated = [_annotate_ontology(r) for r in rows]
            return json.dumps(annotated, default=str)
        except Exception as exc:
            return json.dumps([{"error": str(exc)}])

    @tool
    def query_analytics(
        sql: Annotated[str, "Valid Snowflake SQL query"],
        reason: Annotated[str, "One sentence explaining what this query checks"],
    ) -> str:
        """Run a SQL query against Snowflake.
        Best for transaction volumes, monthly time-series, aggregations,
        risk score distributions, and trend analysis.
        """
        if not sf_available or sf_conn is None:
            return json.dumps([{"info": "Snowflake not connected — set SNOWFLAKE_* vars in .env"}])
        try:
            import snowflake.connector
            cur = sf_conn.cursor(snowflake.connector.DictCursor)
            cur.execute(sql)
            return json.dumps([dict(r) for r in cur.fetchall()], default=str)
        except Exception as exc:
            return json.dumps([{"error": str(exc)}])

    @tool
    def search_documents(
        question: Annotated[str, "The policy topic or rule to look up"],
    ) -> str:
        """Semantic search over compliance policy documents.
        Best for AML policy rules, thresholds, SAR/CTR requirements,
        EDD procedures, and advisor portfolio concentration limits.
        """
        if embedder is None:
            return json.dumps([{"info": "Embedding model not loaded — install sentence-transformers"}])
        results = semantic_search(neo4j_driver, embedder, question, top_k=4)
        return json.dumps(results, default=str)

    tools = [query_graph, search_documents]
    if sf_available:
        tools.insert(1, query_analytics)
    return tools


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class AgentStep:
    tool:   str
    reason: str
    input:  dict
    output: Any


@dataclass
class AgentResult:
    steps:  list[AgentStep] = field(default_factory=list)
    answer: str = ""


# ── Agent ─────────────────────────────────────────────────────────────────────

class FinancialAgent:
    """
    LangGraph ReAct agent for financial compliance analysis.
    Provider-agnostic: pass provider='Anthropic' or provider='OpenAI'.
    """

    def __init__(
        self,
        neo4j_driver: Any,
        sf_conn: Any,
        embedder: EmbeddingProvider | None,
        api_key: str,
        model: str,
        provider: str = "Anthropic",
    ) -> None:
        self._neo4j    = neo4j_driver
        self._sf       = sf_conn
        self._embedder = embedder
        self._sf_ok    = sf_conn is not None
        self._provider = provider
        self._model    = model
        self._api_key  = api_key

    def run(self, question: str) -> AgentResult:
        from langgraph.prebuilt import create_react_agent
        from langchain_core.messages import HumanMessage

        llm     = build_llm(self._provider, self._api_key, self._model)
        tools   = make_tools(self._neo4j, self._sf, self._embedder, self._sf_ok)
        prompt  = _system_prompt(sf_available=self._sf_ok)

        graph   = create_react_agent(llm, tools, prompt=prompt)
        output  = graph.invoke({"messages": [HumanMessage(content=question)]})

        steps, answer = self._parse_output(output)
        return AgentResult(steps=steps, answer=answer)

    def _parse_output(self, output: dict) -> tuple[list[AgentStep], str]:
        """Convert LangGraph message list into AgentStep objects and final answer."""
        from langchain_core.messages import AIMessage, ToolMessage

        messages = output.get("messages", [])
        steps: list[AgentStep] = []
        answer  = ""
        pending: dict[str, dict] = {}  # tool_call_id -> tool_call info

        for msg in messages:
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        pending[tc["id"]] = tc
                elif msg.content:
                    answer = (
                        msg.content
                        if isinstance(msg.content, str)
                        else "".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in msg.content
                        )
                    )

            elif isinstance(msg, ToolMessage):
                tc = pending.pop(msg.tool_call_id, None)
                if tc:
                    try:
                        output_data = json.loads(msg.content)
                    except (json.JSONDecodeError, TypeError):
                        output_data = msg.content
                    steps.append(AgentStep(
                        tool=tc["name"],
                        reason=tc["args"].get("reason", ""),
                        input=tc["args"],
                        output=output_data,
                    ))

        return steps, answer


# ── Snowflake connection helper ───────────────────────────────────────────────

def build_snowflake_conn(
    account:   str,
    user:      str,
    password:  str,
    warehouse: str = "COMPUTE_WH",
    database:  str = "FINANCIAL_KG",
    schema:    str = "PUBLIC",
) -> Any:
    """Return a live Snowflake connection, or None if any credential is empty."""
    if not account.strip() or not user.strip() or not password.strip():
        return None
    import snowflake.connector
    return snowflake.connector.connect(
        account=account, user=user, password=password,
        warehouse=warehouse, database=database, schema=schema,
    )
