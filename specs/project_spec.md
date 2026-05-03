# Financial Knowledge Graph Project Spec

## Purpose
Teach graph databases, semantic modeling, and knowledge graph construction using a realistic financial services mini project.

## Business use case
Financial institutions need to understand relationships between customers, accounts, advisors, and transactions. Traditional tables can store the data, but graph databases make relationship-based analysis easier.

## Example questions
- Which customers are high risk?
- Which accounts are involved in circular money movement?
- Which advisors manage high-risk customers?
- Which customers are indirectly connected through transaction flows?

## Data entities

### Customer
Fields:
- customer_id
- name
- risk_score
- country

### Account
Fields:
- account_id
- customer_id
- account_type

### Transaction
Fields:
- txn_id
- from_account
- to_account
- amount
- timestamp

### Advisor
Fields:
- advisor_id
- name
- customer_id

## Knowledge graph mapping

| CSV/Table | Graph Node | Key |
|---|---|---|
| customers.csv | Customer | customer_id |
| accounts.csv | Account | account_id |
| transactions.csv | Transaction | txn_id |
| advisors.csv | Advisor | advisor_id |

| Source | Relationship | Target |
|---|---|---|
| Customer | OWNS | Account |
| Account | SENT | Transaction |
| Transaction | TO | Account |
| Advisor | MANAGES | Customer |

## Semantic layer concepts

| Business Concept | Graph Representation |
|---|---|
| High Risk Customer | Customer where risk_score > 0.8 |
| High Value Transaction | Transaction where amount > 10000 |
| Circular Money Movement | Account A sends to B, and B sends back to A |
| Advisor Exposure | Advisor manages one or more high-risk customers |

## Out of scope for v1
- Real-time streaming
- Full RDF/OWL ontology
- Production security hardening
- Enterprise data governance integration
- LLM/Graph RAG integration
