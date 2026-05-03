"""
Graph visualization using pyvis.
Renders Neo4j subgraphs as interactive network diagrams in Streamlit.

Colour legend (ontology-driven):
  Red    = HIGH_RISK_CUSTOMER  (risk_score > 0.8)
  Orange = HIGH_VALUE_TXN      (amount > $10,000)
  Green  = normal Transaction
  Blue   = normal Customer
  Grey   = Account
  Purple = Advisor
"""
from __future__ import annotations

import os
import re
import tempfile
from typing import Any

import streamlit.components.v1 as components

# ── Colour palette ────────────────────────────────────────────────────────────
_C = {
    "Customer":        "#4A90D9",   # blue   — normal risk
    "Customer_HRC":    "#E74C3C",   # red    — HIGH_RISK_CUSTOMER (>0.8)
    "Account":         "#7F8C8D",   # grey
    "Transaction":     "#27AE60",   # green  — normal amount
    "Transaction_HVT": "#E67E22",   # orange — HIGH_VALUE_TXN (>$10k)
    "Advisor":         "#9B59B6",   # purple
    "edge":            "#BDC3C7",
}

# ── Visualization queries — return Node/Relationship objects ──────────────────
# One companion query per predefined concept (key must match PREDEFINED_QUERIES)
VIZ_QUERIES: dict[str, str] = {
    "high_risk_customers": """
        MATCH (adv:Advisor)-[m:MANAGES]->(c:Customer)-[o:OWNS]->(a:Account)
        WHERE c.risk_score > 0.8
        RETURN adv, m, c, o, a
    """,
    "high_value_transactions": """
        MATCH (a1:Account)-[s:SENT]->(t:Transaction)-[to_r:TO]->(a2:Account)
        WHERE t.amount > 10000
        OPTIONAL MATCH (c1:Customer)-[o1:OWNS]->(a1)
        OPTIONAL MATCH (c2:Customer)-[o2:OWNS]->(a2)
        RETURN c1, o1, a1, s, t, to_r, a2, o2, c2
        LIMIT 30
    """,
    "circular_movements": """
        MATCH (a1:Account)-[s1:SENT]->(t1:Transaction)-[r1:TO]->(a2:Account)
              -[s2:SENT]->(t2:Transaction)-[r2:TO]->(a1)
        OPTIONAL MATCH (c1:Customer)-[o1:OWNS]->(a1)
        OPTIONAL MATCH (c2:Customer)-[o2:OWNS]->(a2)
        RETURN c1, o1, a1, s1, t1, r1, a2, s2, t2, r2, c2, o2
    """,
    "advisor_exposure": """
        MATCH (adv:Advisor)-[m:MANAGES]->(c:Customer)-[o:OWNS]->(a:Account)
        WHERE c.risk_score > 0.8
        RETURN adv, m, c, o, a
    """,
    "customer_exposure": """
        MATCH (c1:Customer)-[o1:OWNS]->(a1:Account)-[s:SENT]->(t:Transaction)
              -[to_r:TO]->(a2:Account)<-[o2:OWNS]-(c2:Customer)
        RETURN c1, o1, a1, s, t, to_r, a2, o2, c2
        LIMIT 40
    """,
    "high_risk_network": """
        MATCH (c:Customer)-[o:OWNS]->(a:Account)
        WHERE c.risk_score > 0.8
        OPTIONAL MATCH (a)-[s:SENT]->(t:Transaction)-[to_r:TO]->(a2:Account)
        OPTIONAL MATCH (adv:Advisor)-[m:MANAGES]->(c)
        RETURN c, o, a, s, t, to_r, a2, adv, m
        LIMIT 60
    """,
}

# ── Node styling ──────────────────────────────────────────────────────────────

def _node_style(node: Any) -> tuple[str, str, str]:
    """Return (display_label, hex_color, hover_tooltip) for a Neo4j Node."""
    label = list(node.labels)[0] if node.labels else "Node"

    if label == "Customer":
        risk  = float(node.get("risk_score", 0))
        name  = node.get("name", node.element_id)
        color = _C["Customer_HRC"] if risk > 0.8 else _C["Customer"]
        tip   = f"👤 {name}\nRisk: {risk:.2f} | {node.get('country', '')}"
        tip  += "\n⚠ HIGH_RISK_CUSTOMER — EDD required" if risk > 0.8 else ""
        return name, color, tip

    if label == "Account":
        aid  = node.get("id", node.element_id)
        return aid, _C["Account"], f"🏦 Account: {aid}\nType: {node.get('type','')}"

    if label == "Transaction":
        amount = float(node.get("amount", 0))
        tid    = node.get("id", node.element_id)
        color  = _C["Transaction_HVT"] if amount > 10_000 else _C["Transaction"]
        tip    = f"💸 Txn: {tid}\nAmount: ${amount:,.0f}"
        tip   += "\n⚠ HIGH_VALUE_TXN — CTR threshold" if amount > 10_000 else ""
        return f"${amount:,.0f}", color, tip

    if label == "Advisor":
        name = node.get("name", node.element_id)
        return name, _C["Advisor"], f"🧑‍💼 Advisor: {name}"

    return str(node.element_id), "#AAA", str(dict(node))


# ── Network builder ───────────────────────────────────────────────────────────

def _make_net() -> Any:
    from pyvis.network import Network
    net = Network(
        height="500px",
        width="100%",
        bgcolor="#16213e",
        font_color="#ECF0F1",
        directed=True,
    )
    net.set_options("""{
        "nodes": {
            "borderWidth": 2,
            "font": {"size": 13, "bold": {"mod": "bold"}},
            "shadow": {"enabled": true}
        },
        "edges": {
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.7}},
            "color": {"inherit": false},
            "font": {"size": 10, "align": "middle"},
            "smooth": {"type": "curvedCW", "roundness": 0.25},
            "shadow": {"enabled": true}
        },
        "physics": {
            "stabilization": {"iterations": 200},
            "barnesHut": {
                "gravitationalConstant": -9000,
                "centralGravity": 0.3,
                "springLength": 120
            }
        },
        "interaction": {"hover": true, "tooltipDelay": 100}
    }""")
    return net


def _add_node(net: Any, node: Any, seen: set) -> None:
    nid = str(node.element_id)
    if nid in seen:
        return
    seen.add(nid)
    label, color, title = _node_style(node)
    net.add_node(nid, label=label, color=color, title=title, size=24)


def _walk(value: Any, net: Any, seen: set) -> None:
    """Recursively process any Neo4j value type."""
    try:
        from neo4j.graph import Node, Relationship, Path
    except ImportError:
        return

    if value is None:
        return
    if isinstance(value, Node):
        _add_node(net, value, seen)
    elif isinstance(value, Relationship):
        # Ensure both endpoint nodes are added
        _add_node(net, value.start_node, seen)
        _add_node(net, value.end_node, seen)
        net.add_edge(
            str(value.start_node.element_id),
            str(value.end_node.element_id),
            label=value.type,
            color=_C["edge"],
            width=2,
        )
    elif isinstance(value, Path):
        for n in value.nodes:
            _add_node(net, n, seen)
        for r in value.relationships:
            _walk(r, net, seen)
    elif isinstance(value, list):
        for item in value:
            _walk(item, net, seen)


def viz_from_neo4j(driver: Any, viz_cypher: str) -> Any | None:
    """
    Run viz_cypher against Neo4j and build a pyvis Network.
    The query must SELECT Node/Relationship columns (not scalars).
    Returns None if no nodes were found.
    """
    net  = _make_net()
    seen: set = set()

    with driver.session() as session:
        result = session.run(viz_cypher)
        for record in result:
            for value in record.values():
                _walk(value, net, seen)

    return net if seen else None


def viz_from_entity_ids(driver: Any, entity_ids: list[str]) -> Any | None:
    """
    Build a local subgraph for a list of node .id values.
    Used by the Agent tab to visualise ad-hoc query results.
    """
    if not entity_ids:
        return None

    cypher = """
    MATCH (n) WHERE n.id IN $ids
    OPTIONAL MATCH (n)-[r]-(m)
    WHERE m.id IN $ids
       OR (n:Account AND (n)-[:SENT|TO]-(m))
       OR (n:Customer AND (n)-[:OWNS]-(m))
       OR (n:Advisor  AND (n)-[:MANAGES]-(m))
    RETURN n, r, m LIMIT 80
    """
    net  = _make_net()
    seen: set = set()

    try:
        from neo4j.graph import Node, Relationship
        with driver.session() as session:
            for record in session.run(cypher, ids=entity_ids):
                n, r, m = record["n"], record["r"], record["m"]
                if n: _add_node(net, n, seen)
                if m: _add_node(net, m, seen)
                if r and n and m:
                    _add_node(net, r.start_node, seen)
                    _add_node(net, r.end_node, seen)
                    net.add_edge(
                        str(r.start_node.element_id),
                        str(r.end_node.element_id),
                        label=r.type,
                        color=_C["edge"],
                        width=2,
                    )
    except Exception:
        return None

    return net if seen else None


def extract_ids_from_rows(rows: list[dict]) -> list[str]:
    """Pull node ID values (A001, C005, T012, ADV003 …) out of agent result rows."""
    pattern = re.compile(r"^(ADV|[ACT])\d+$", re.IGNORECASE)
    ids: set[str] = set()
    for row in rows:
        for v in row.values():
            if isinstance(v, str) and pattern.match(v):
                ids.add(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, str) and pattern.match(item):
                        ids.add(item)
    return list(ids)


# ── Streamlit rendering ───────────────────────────────────────────────────────

LEGEND_HTML = """
<div style="display:flex; gap:18px; flex-wrap:wrap; margin-bottom:8px; font-size:13px;">
  <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;
        background:#E74C3C;margin-right:5px;vertical-align:middle;"></span>
        HIGH_RISK_CUSTOMER (risk&gt;0.8)</span>
  <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;
        background:#4A90D9;margin-right:5px;vertical-align:middle;"></span>
        Customer (normal)</span>
  <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;
        background:#E67E22;margin-right:5px;vertical-align:middle;"></span>
        HIGH_VALUE_TXN (&gt;$10k)</span>
  <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;
        background:#27AE60;margin-right:5px;vertical-align:middle;"></span>
        Transaction (normal)</span>
  <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;
        background:#7F8C8D;margin-right:5px;vertical-align:middle;"></span>
        Account</span>
  <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;
        background:#9B59B6;margin-right:5px;vertical-align:middle;"></span>
        Advisor</span>
</div>
"""


def render_network(net: Any) -> None:
    """Render a pyvis Network as an interactive HTML component in Streamlit."""
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".html", mode="w", encoding="utf-8"
    ) as f:
        fname = f.name
    net.save_graph(fname)
    html = open(fname, encoding="utf-8").read()
    os.unlink(fname)
    components.html(html, height=520, scrolling=False)
