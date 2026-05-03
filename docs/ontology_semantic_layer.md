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

## Rules

- A customer can own one or more accounts.
- An account belongs to a customer.
- A transaction has a sending account and receiving account.
- An advisor can manage multiple customers.

## Semantic layer examples

| Semantic Term | Graph Logic |
|---|---|
| High Risk Customer | Customer where risk_score > 0.8 |
| High Value Transaction | Transaction where amount > 10000 |
| Circular Movement | Account sends to another account and receives money back |
| Advisor Risk Exposure | Advisor manages high-risk customers |

## How this connects to Knowledge Graphs

A knowledge graph combines:

1. Connected data
2. Business meaning
3. Queryable relationships
4. Reusable context for applications and AI systems
