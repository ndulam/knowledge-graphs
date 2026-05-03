# Claude Code Instructions: Financial Knowledge Graph

You are building a GitHub-ready Python project named `financial-knowledge-graph`.

## Goal
Convert simple financial data lake-style CSV files into a Neo4j Knowledge Graph for risk and fraud analysis.

## Core requirements
1. Use Python 3.11+.
2. Use Neo4j Python driver.
3. Use pandas for CSV loading.
4. Use python-dotenv for environment config.
5. Include Docker Compose for local Neo4j.
6. Include reusable scripts for:
   - generating synthetic financial data
   - creating Neo4j constraints
   - loading CSV data into Neo4j
   - running demo fraud/risk queries
7. Include tests for data generation and basic Cypher query construction.
8. Include clean README with setup and demo steps.

## Project structure to create

financial-knowledge-graph/
├── README.md
├── requirements.txt
├── .env.example
├── docker-compose.yml
├── data/
│   ├── customers.csv
│   ├── accounts.csv
│   ├── transactions.csv
│   └── advisors.csv
├── scripts/
│   ├── generate_data.py
│   ├── create_schema.py
│   ├── load_to_neo4j.py
│   └── run_queries.py
├── cypher/
│   ├── schema.cypher
│   └── queries.cypher
├── tests/
│   ├── test_generate_data.py
│   └── test_project_files.py
└── docs/
    ├── graph_model.md
    └── ontology_semantic_layer.md

## Graph model

Nodes:
- Customer
- Account
- Transaction
- Advisor

Relationships:
- (:Customer)-[:OWNS]->(:Account)
- (:Account)-[:SENT]->(:Transaction)
- (:Transaction)-[:TO]->(:Account)
- (:Advisor)-[:MANAGES]->(:Customer)

## Sample fraud/risk patterns
1. High-risk customers: risk_score > 0.8
2. High-value transactions: amount > 10000
3. Circular money movement: account A sends to B and B sends back to A
4. Advisor exposure: advisors managing multiple high-risk customers
5. Multi-hop customer exposure through transactions

## Code quality expectations
- Modular functions
- Clear variable names
- Comments only where useful
- No hardcoded credentials
- Idempotent Neo4j writes using MERGE
- Meaningful error handling
- README commands must work from repo root

## Do not
- Do not use real financial data.
- Do not include secrets.
- Do not over-engineer with RDF/OWL in v1.
- Do not require paid cloud services.
