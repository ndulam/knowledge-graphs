# Lightweight Ontology and Semantic Layer

## Lightweight ontology

This project does not implement full RDF/OWL ontology in v1. Instead, it uses a practical lightweight ontology.

## Concepts

| Concept | Meaning |
|---|---|
| Customer | Person or entity that owns financial accounts |
| Account | Financial container used for transactions |
| Transaction | Money movement from one account to another |
| Advisor | Person responsible for managing customer relationships |

## Semantic layer examples

| Semantic Term | Graph Logic |
|---|---|
| High Risk Customer | Customer where risk_score > 0.8 — triggers Enhanced Due Diligence (EDD) |
| High Value Transaction | Transaction where amount > $10,000 — CTR filing threshold |
| Circular Money Movement | Account A sends to Account B which sends back to Account A — layering indicator |
| Advisor Risk Exposure | Advisor managing one or more high-risk customers — concentrated portfolio risk |
| Structuring (Smurfing) | Account receives large inflow (>$10k) then fans out to ≥3 outflows each averaging <$5k — deliberate CTR avoidance |
| Layering | Transaction chain of 3+ hops through different accounts with declining amounts — obscures money trail |
| Velocity Anomaly | Account fires 3+ outgoing transactions within the same calendar day — burst-firing pattern |
| Geographic Concentration Risk | High-risk customer (risk >0.70) transacting cross-border across multiple countries — jurisdiction exposure |
| Money Mule | Account forwards ≥85% of received funds onward — pass-through conduit for illicit flows |
| Advisor Network Contagion | Advisor's low-risk client (≤0.80) directly transacting with a HIGH_RISK_CUSTOMER — indirect portfolio contamination |

## Rules

- A customer can own one or more accounts.
- An account belongs to a customer.
- A transaction has a sending account and receiving account.
- An advisor can manage multiple customers.
- A High Risk Customer (risk_score > 0.8) requires Enhanced Due Diligence procedures.
- A Critical Risk Customer (risk_score > 0.9) requires immediate CCO escalation.
- Any transaction above $10,000 must be evaluated for CTR filing requirements.
- Circular movements (A→B→A or A→B→C→A) are known layering patterns.
- Structuring (smurfing) is illegal regardless of the individual transaction amounts.

## How this connects to Knowledge Graphs

A knowledge graph combines:

1. Connected data
2. Business meaning
3. Queryable relationships
4. Reusable context for applications and AI systems
