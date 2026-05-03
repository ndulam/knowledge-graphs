from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

ONTOLOGY_CONTEXT = """You are an expert analyst for a financial knowledge graph used for risk and fraud detection.

GRAPH STRUCTURE
Nodes:
- Customer (id, name, risk_score 0-1, country)
- Account (id, type: SAVINGS | CHECKING | BROKERAGE | INVESTMENT)
- Transaction (id, amount USD, timestamp)
- Advisor (id, name)

Relationships:
- (:Customer)-[:OWNS]->(:Account)
- (:Account)-[:SENT]->(:Transaction)
- (:Transaction)-[:TO]->(:Account)
- (:Advisor)-[:MANAGES]->(:Customer)

SEMANTIC ONTOLOGY — business concepts encoded in the graph
- High Risk Customer    : Customer where risk_score > 0.8
- High Value Transaction: Transaction where amount > $10,000
- Circular Money Movement: Account A sends funds to Account B, which sends them back to Account A — a fraud indicator
- Advisor Risk Exposure : An advisor managing one or more high-risk customers, representing concentrated portfolio risk

YOUR TASK
Given a business question, the Cypher query executed against the graph, and its results, explain what
the data reveals in plain business language. Reference the ontology concepts where relevant.
Highlight any risk signals or fraud patterns. Be concise and direct."""

PREDEFINED_QUERIES: dict[str, dict[str, str]] = {
    "high_risk_customers": {
        "label": "High-Risk Customers",
        "cypher": (
            "MATCH (c:Customer) WHERE c.risk_score > 0.8 "
            "RETURN c.id AS id, c.name AS name, c.risk_score AS risk_score, c.country AS country "
            "ORDER BY c.risk_score DESC"
        ),
        "question": "Which customers are classified as high-risk and what does their profile look like?",
    },
    "high_value_transactions": {
        "label": "High-Value Transactions",
        "cypher": (
            "MATCH (a1:Account)-[:SENT]->(t:Transaction)-[:TO]->(a2:Account) "
            "WHERE t.amount > 10000 "
            "RETURN t.id AS transaction_id, t.amount AS amount, t.timestamp AS timestamp, "
            "a1.id AS from_account, a2.id AS to_account "
            "ORDER BY t.amount DESC"
        ),
        "question": "What high-value transactions occurred and what do they indicate about risk?",
    },
    "circular_movements": {
        "label": "Circular Money Movements",
        "cypher": (
            "MATCH (a1:Account)-[:SENT]->(:Transaction)-[:TO]->(a2:Account)"
            "-[:SENT]->(:Transaction)-[:TO]->(a1) "
            "RETURN DISTINCT a1.id AS account_a, a2.id AS account_b"
        ),
        "question": "Are there any circular money movements that could indicate layering or fraud?",
    },
    "advisor_exposure": {
        "label": "Advisor Risk Exposure",
        "cypher": (
            "MATCH (adv:Advisor)-[:MANAGES]->(c:Customer) WHERE c.risk_score > 0.8 "
            "RETURN adv.id AS advisor_id, adv.name AS advisor_name, "
            "collect(c.name) AS high_risk_customers, count(c) AS high_risk_count "
            "ORDER BY high_risk_count DESC"
        ),
        "question": "Which advisors have the most exposure to high-risk customers?",
    },
    "customer_exposure": {
        "label": "Multi-Hop Exposure",
        "cypher": (
            "MATCH (c1:Customer)-[:OWNS]->(a1:Account)-[:SENT]->(t:Transaction)"
            "-[:TO]->(a2:Account)<-[:OWNS]-(c2:Customer) "
            "WHERE c1 <> c2 "
            "RETURN DISTINCT c1.name AS customer_a, c2.name AS customer_b, "
            "count(t) AS shared_transactions "
            "ORDER BY shared_transactions DESC"
        ),
        "question": "Which customers are connected through shared transaction flows?",
    },
}

_CYPHER_GENERATION_SUFFIX = (
    "\n\nYour task now is different: given a question, write a single Cypher query "
    "to retrieve the relevant data from the graph described above. "
    "Return ONLY the Cypher query — no explanation, no markdown code fences."
)


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> str: ...


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-opus-4-7") -> None:
        from anthropic import Anthropic
        self._client = Anthropic(api_key=api_key)
        self._model = model

    def complete(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content


def get_provider(provider_name: str, api_key: str, model: str) -> LLMProvider:
    if provider_name == "Anthropic":
        return AnthropicProvider(api_key=api_key, model=model)
    if provider_name == "OpenAI":
        return OpenAIProvider(api_key=api_key, model=model)
    raise ValueError(f"Unknown provider: {provider_name}")


def build_neo4j_driver(uri: str, username: str, password: str):
    return GraphDatabase.driver(uri, auth=(username, password))


def _serialize(value):
    """Recursively convert Neo4j driver objects to plain Python so pyarrow can display them."""
    from neo4j.graph import Node, Relationship, Path
    if isinstance(value, Path):
        parts = []
        for node in value.nodes:
            parts.append(node.get("id") or node.element_id)
        return " -> ".join(str(p) for p in parts)
    if isinstance(value, Node):
        return dict(value)
    if isinstance(value, Relationship):
        return {"type": value.type, **dict(value)}
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    return value


def run_cypher(driver, cypher: str) -> list[dict]:
    with driver.session() as session:
        result = session.run(cypher)
        return [{k: _serialize(v) for k, v in dict(record).items()} for record in result]


def generate_cypher(provider: LLMProvider, question: str) -> str:
    raw = provider.complete(
        system=ONTOLOGY_CONTEXT + _CYPHER_GENERATION_SUFFIX,
        user=f"Question: {question}",
    )
    # Strip markdown code fences if the model wraps the query
    lines = [ln for ln in raw.strip().splitlines() if not ln.strip().startswith("```")]
    return "\n".join(lines).strip()


def run_graph_rag(
    driver,
    provider: LLMProvider,
    cypher: str,
    question: str,
) -> tuple[list[dict], str]:
    records = run_cypher(driver, cypher)

    if not records:
        graph_summary = "The query returned no results."
    else:
        rows = "\n".join(str(r) for r in records[:50])
        graph_summary = f"Query returned {len(records)} result(s):\n{rows}"

    user_prompt = (
        f"Question: {question}\n\n"
        f"Cypher query executed:\n{cypher}\n\n"
        f"Graph results:\n{graph_summary}\n\n"
        "Explain what these results mean from a financial risk and fraud perspective, "
        "referencing the ontology concepts where relevant."
    )

    explanation = provider.complete(system=ONTOLOGY_CONTEXT, user=user_prompt)
    return records, explanation


# ── Embedding provider ────────────────────────────────────────────────────────

class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

    @property
    @abstractmethod
    def dimensions(self) -> int: ...


class SentenceTransformerEmbedder(EmbeddingProvider):
    _MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self._MODEL_NAME)

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [e.tolist() for e in self._model.encode(texts)]

    @property
    def dimensions(self) -> int:
        return 384


def get_embedder() -> EmbeddingProvider:
    return SentenceTransformerEmbedder()


# ── Semantic search ───────────────────────────────────────────────────────────

def semantic_search(
    driver,
    embedder: EmbeddingProvider,
    question: str,
    top_k: int = 5,
) -> list[dict]:
    """Query the Neo4j vector index and return the top-k document chunks."""
    try:
        embedding = embedder.embed(question)
        with driver.session() as session:
            result = session.run(
                """
                CALL db.index.vector.queryNodes('document_chunks', $k, $embedding)
                YIELD node, score
                RETURN node.text AS text, node.source AS source,
                       node.page AS page, score
                ORDER BY score DESC
                """,
                k=top_k,
                embedding=embedding,
            )
            return [dict(r) for r in result]
    except Exception:
        return []


# ── Query router ──────────────────────────────────────────────────────────────

_ROUTING_SYSTEM = """\
You are a query router for a hybrid financial intelligence system that has two data sources:
  1. A graph database with live customer records, account data, transaction history, risk scores,
     and advisor-client relationships.
  2. Compliance document library containing AML policies, risk assessment frameworks, and
     advisor guidelines.

Classify the user's question into exactly one category:

structured   – Fully answerable from the live graph database. Questions about specific named
               customers, accounts, transactions, amounts, risk scores, circular flows, or
               advisor-client network patterns.
               Examples: "List high-risk customers", "Find circular transactions",
               "Which advisor manages the most HRC clients?"

unstructured – Requires policy, procedure, threshold definitions, or regulatory rules from
               the document library. No live data needed.
               Examples: "What is the SAR filing threshold?", "Define structuring",
               "What training must advisors complete?"

hybrid       – Needs BOTH live graph data AND policy/procedure context to give a complete answer.
               Examples: "Does Alice Morgan require EDD under our AML policy?",
               "Which advisors violate portfolio concentration limits?",
               "How should we handle the circular ring per our risk framework?"

Reply with exactly one word: structured, unstructured, or hybrid.\
"""


def route_query(
    provider: LLMProvider,
    question: str,
) -> Literal["structured", "unstructured", "hybrid"]:
    raw = provider.complete(system=_ROUTING_SYSTEM, user=f"Question: {question}").strip().lower()
    if raw in ("structured", "unstructured", "hybrid"):
        return raw  # type: ignore[return-value]
    # Fallback: pick the first matching word in the response
    for word in ("hybrid", "unstructured", "structured"):
        if word in raw:
            return word  # type: ignore[return-value]
    return "hybrid"


# ── Hybrid RAG ────────────────────────────────────────────────────────────────

_HYBRID_SYSTEM = """\
You are an expert financial compliance analyst with access to both a live graph database
and a library of compliance policy documents.

GRAPH STRUCTURE (live data)
- Customer nodes: id, name, risk_score (0-1), country
- Account nodes: id, type (SAVINGS | CHECKING | BROKERAGE | INVESTMENT)
- Transaction nodes: id, amount (USD), timestamp
- Advisor nodes: id, name
- Relationships: Customer-[:OWNS]->Account-[:SENT]->Transaction-[:TO]->Account,
  Advisor-[:MANAGES]->Customer

SEMANTIC ONTOLOGY
- High Risk Customer (HRC): risk_score >= 0.80
- Critical Risk Customer: risk_score >= 0.90 (immediate CCO escalation required)
- High Value Transaction: amount > $10,000
- Structuring / Smurfing: large inflow > $10k followed by multiple outflows avg < $5k
- Circular Money Movement: accounts forming a closed transaction ring (A→B→C→A)
- Hub Account: receives from 2+ sources AND sends to 2+ destinations
- Risk Contagion: low-risk customer (<0.30) transacting with high-risk counterpart (>0.80)

YOUR TASK
Given the user's question, the retrieved graph data, and the retrieved policy passages,
provide a clear, business-oriented answer that:
1. Cites specific data from the graph (names, amounts, risk scores) where available.
2. References the relevant policy rules or thresholds where applicable.
3. Highlights any compliance gaps, required actions, or escalation needs.
Be direct and concise. Avoid generic statements — ground every claim in the provided data.\
"""


def run_hybrid_rag(
    driver,
    provider: LLMProvider,
    embedder: EmbeddingProvider | None,
    question: str,
    cypher: str = "",
    route: Literal["structured", "unstructured", "hybrid"] = "hybrid",
) -> tuple[list[dict], list[dict], str, str]:
    """
    Returns (graph_records, doc_chunks, route_used, explanation).
    Pass route="" to skip routing (uses the supplied value directly).
    """
    graph_records: list[dict] = []
    doc_chunks: list[dict] = []

    # Structured retrieval
    if route in ("structured", "hybrid"):
        if not cypher:
            cypher = generate_cypher(provider, question)
        try:
            graph_records = run_cypher(driver, cypher)
        except Exception:
            graph_records = []

    # Unstructured retrieval
    if route in ("unstructured", "hybrid") and embedder is not None:
        doc_chunks = semantic_search(driver, embedder, question, top_k=5)

    # Build combined prompt
    graph_section = ""
    if graph_records:
        rows = "\n".join(str(r) for r in graph_records[:30])
        graph_section = f"\n\n## GRAPH DATA ({len(graph_records)} records)\n{rows}"

    doc_section = ""
    if doc_chunks:
        passages = "\n\n---\n".join(
            f"[{c['source']} p.{c['page']} | relevance {c.get('score', 0):.2f}]\n{c['text']}"
            for c in doc_chunks
        )
        doc_section = f"\n\n## POLICY / FRAMEWORK CONTEXT\n{passages}"

    user_prompt = (
        f"Question: {question}\n"
        f"Data sources used: {route}"
        f"{graph_section}"
        f"{doc_section}"
        "\n\nProvide your analysis based on the data and context above."
    )

    explanation = provider.complete(system=_HYBRID_SYSTEM, user=user_prompt)
    return graph_records, doc_chunks, route, explanation
