"""
Financial Knowledge Graph AI Agent
===================================
A tool-calling agent (Claude) with three tools:

  query_graph        — Neo4j Cypher  (WHO is connected to WHO, network patterns)
  query_analytics    — Snowflake SQL  (HOW MUCH, WHEN, trends over time)
  search_documents   — Vector search  (WHAT the compliance rules say)

The agent autonomously decides which tools to call, in what order, and synthesises
a grounded answer from all retrieved sources — demonstrating the value of combining
a knowledge graph, a data warehouse, and unstructured document retrieval.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

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
Questions answered: "WHO sent money to WHO?", "Is there a cycle?",
"What is reachable from this customer within N hops?"
Results are pre-annotated with ontology flags (e.g. HIGH_RISK_CUSTOMER,
HIGH_VALUE_TXN, CIRCULAR_MOVEMENT) — use these flags when reasoning.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOL 2 — query_analytics  (Snowflake Data Warehouse)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use for: transaction volumes, time-series trends, aggregations,
risk score distributions, country-level exposure, advisor stats.
Questions answered: "How much?", "Which month showed the spike?",
"What is the average risk score by country?"

Tables (all column names uppercase):
  CUSTOMERS(CUSTOMER_ID, NAME, RISK_SCORE, COUNTRY)
  ACCOUNTS(ACCOUNT_ID, CUSTOMER_ID, ACCOUNT_TYPE)
  TRANSACTIONS(TXN_ID, FROM_ACCOUNT, TO_ACCOUNT, AMOUNT, TXN_TIMESTAMP)
  ADVISORS(ADVISOR_ID, NAME, CUSTOMER_ID)
  MONTHLY_TX_SUMMARY(MONTH DATE, CUSTOMER_ID, CUSTOMER_NAME, RISK_SCORE,
      COUNTRY, TOTAL_SENT, TOTAL_RECEIVED, TXN_COUNT_SENT,
      TXN_COUNT_RECEIVED, UNIQUE_COUNTERPARTIES)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOL 3 — search_documents  (Compliance Policy Library)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use for: AML policy rules, risk score thresholds, SAR/CTR filing
requirements, EDD procedures, advisor portfolio limits, definitions.
Documents: aml_policy.pdf, customer_risk_framework.pdf,
           advisor_compliance_guide.pdf

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
APPROACH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. For compliance questions, call search_documents FIRST to retrieve the
   exact policy thresholds, then call graph/analytics tools to check
   whether those thresholds are breached in the actual data.
2. Apply the ontology definitions above when interpreting graph results —
   a risk_score > 0.8 is not just a number, it is a HIGH_RISK_CUSTOMER
   that triggers EDD obligations; an A->B->A cycle is CIRCULAR_MOVEMENT,
   a known fraud indicator.
3. For questions needing both historical trends and network patterns,
   call query_analytics for the time-series, then query_graph for the
   structural explanation of WHY the spike occurred (which entities,
   which relationships).
4. Always cite specific values: amounts, risk scores, customer names,
   ontology concept labels, policy thresholds, and month-over-month changes.
5. If a tool returns an error or is unavailable, note it and proceed
   with the data you have.\
"""

_SNOWFLAKE_UNAVAILABLE_NOTE = """

NOTE: Snowflake (query_analytics) is not connected in this session.
Do NOT call query_analytics. Answer using Neo4j and document search only.\
"""


def build_agent_system(sf_available: bool) -> str:
    base = _AGENT_SYSTEM_BASE.format(ontology=ONTOLOGY_CONTEXT)
    if sf_available:
        return base
    return base + _SNOWFLAKE_UNAVAILABLE_NOTE


# ── Tool schemas (Claude tool_use API) ────────────────────────────────────────

TOOLS: list[dict] = [
    {
        "name": "query_graph",
        "description": (
            "Run a Cypher query against the Neo4j knowledge graph. "
            "Best for relationship traversal, cycle detection, blast radius analysis, "
            "multi-hop paths, and network-level risk patterns."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cypher": {
                    "type": "string",
                    "description": "Valid Cypher query to execute against Neo4j.",
                },
                "reason": {
                    "type": "string",
                    "description": "One sentence explaining what this query checks.",
                },
            },
            "required": ["cypher", "reason"],
        },
    },
    {
        "name": "query_analytics",
        "description": (
            "Run a SQL query against Snowflake. "
            "Best for transaction volumes, monthly time-series, aggregations, "
            "risk score distributions, and trend analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "Valid Snowflake SQL query.",
                },
                "reason": {
                    "type": "string",
                    "description": "One sentence explaining what this query checks.",
                },
            },
            "required": ["sql", "reason"],
        },
    },
    {
        "name": "search_documents",
        "description": (
            "Semantic search over compliance policy documents. "
            "Best for AML policy rules, thresholds, SAR/CTR requirements, "
            "EDD procedures, and advisor portfolio concentration limits."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The policy topic or rule to look up.",
                },
            },
            "required": ["question"],
        },
    },
]


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class AgentStep:
    tool: str
    reason: str
    input: dict
    output: Any


@dataclass
class AgentResult:
    steps: list[AgentStep] = field(default_factory=list)
    answer: str = ""


# ── Ontology annotation ───────────────────────────────────────────────────────

def _annotate_ontology(row: dict) -> dict:
    """
    Tag each graph result row with ontology concept labels so the agent
    reasons over semantic meaning, not raw numbers.
    """
    flags: list[str] = []

    risk = row.get("risk_score")
    if risk is not None and float(risk) > 0.8:
        flags.append("HIGH_RISK_CUSTOMER (risk_score > 0.8 -> EDD required)")

    amount = row.get("amount")
    if amount is not None and float(amount) > 10_000:
        flags.append("HIGH_VALUE_TXN (amount > $10,000 -> CTR filing threshold)")

    # Circular movement: query returns account_a / account_b pair
    if "account_a" in row and "account_b" in row:
        flags.append("CIRCULAR_MOVEMENT (A->B->A cycle detected -> fraud indicator)")

    # Advisor exposure: query returns high_risk_count
    hrc = row.get("high_risk_count")
    if hrc is not None and int(hrc) > 0:
        flags.append(
            f"ADVISOR_RISK_EXPOSURE ({hrc} HIGH_RISK_CUSTOMER(s) in portfolio)"
        )

    if flags:
        row = dict(row)
        row["_ontology_flags"] = flags
    return row


# ── Agent ─────────────────────────────────────────────────────────────────────

class FinancialAgent:
    """Tool-calling Claude agent for financial compliance analysis."""

    MAX_TURNS = 10

    def __init__(
        self,
        neo4j_driver,
        sf_conn,                      # snowflake.connector connection or None
        embedder: EmbeddingProvider | None,
        api_key: str,
        model: str = "claude-opus-4-7",
    ) -> None:
        from anthropic import Anthropic
        self._client  = Anthropic(api_key=api_key)
        self._model   = model
        self._neo4j   = neo4j_driver
        self._sf      = sf_conn
        self._embedder = embedder

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, question: str) -> AgentResult:
        result   = AgentResult()
        sf_ok    = self._sf is not None
        system   = build_agent_system(sf_available=sf_ok)
        messages = [{"role": "user", "content": question}]

        for _ in range(self.MAX_TURNS):
            response = self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=system,
                tools=TOOLS,
                messages=messages,
            )

            tool_calls = [b for b in response.content if b.type == "tool_use"]

            # Agent finished — collect the text response
            if not tool_calls:
                result.answer = next(
                    (b.text for b in response.content if b.type == "text"), ""
                )
                break

            # Execute all tool calls in this turn
            tool_results = []
            for tc in tool_calls:
                output = self._execute(tc.name, tc.input)
                result.steps.append(
                    AgentStep(
                        tool=tc.name,
                        reason=tc.input.get("reason", ""),
                        input=tc.input,
                        output=output,
                    )
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": json.dumps(output, default=str),
                    }
                )

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        return result

    # ── Tool implementations ──────────────────────────────────────────────────

    def _execute(self, name: str, inputs: dict) -> Any:
        if name == "query_graph":
            return self._run_graph(inputs["cypher"])
        if name == "query_analytics":
            return self._run_snowflake(inputs["sql"])
        if name == "search_documents":
            return self._run_docs(inputs["question"])
        return {"error": f"Unknown tool: {name}"}

    def _run_graph(self, cypher: str) -> list[dict]:
        try:
            rows = run_cypher(self._neo4j, cypher)
            return [_annotate_ontology(row) for row in rows]
        except Exception as exc:
            return [{"error": str(exc)}]

    def _run_snowflake(self, sql: str) -> list[dict]:
        if self._sf is None:
            return [{"info": "Snowflake not connected — set SNOWFLAKE_* vars in .env"}]
        try:
            import snowflake.connector
            cur = self._sf.cursor(snowflake.connector.DictCursor)
            cur.execute(sql)
            return [dict(r) for r in cur.fetchall()]
        except Exception as exc:
            return [{"error": str(exc)}]

    def _run_docs(self, question: str) -> list[dict]:
        if self._embedder is None:
            return [{"info": "Embedding model not loaded — install sentence-transformers"}]
        return semantic_search(self._neo4j, self._embedder, question, top_k=4)


# ── Snowflake connection helper (used by app.py) ──────────────────────────────

def build_snowflake_conn(
    account: str,
    user: str,
    password: str,
    warehouse: str = "COMPUTE_WH",
    database: str = "FINANCIAL_KG",
    schema: str = "PUBLIC",
):
    """Return a Snowflake connection, or None if credentials are empty."""
    if not account.strip() or not user.strip() or not password.strip():
        return None
    import snowflake.connector
    return snowflake.connector.connect(
        account=account,
        user=user,
        password=password,
        warehouse=warehouse,
        database=database,
        schema=schema,
    )
