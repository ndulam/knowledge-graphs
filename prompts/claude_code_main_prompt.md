Build a GitHub-ready Python project called `financial-knowledge-graph`.

The project should demonstrate how to convert financial data lake data into a Neo4j Knowledge Graph for risk and fraud analysis.

Please create the full repo with:
- README.md
- requirements.txt
- .env.example
- docker-compose.yml
- data CSV files
- scripts for data generation, schema creation, loading Neo4j, and running queries
- cypher schema and query files
- tests
- docs explaining graph model, lightweight ontology, and semantic layer

Use this graph model:

Nodes:
- Customer(id, name, risk_score, country)
- Account(id, type)
- Transaction(id, amount, timestamp)
- Advisor(id, name)

Relationships:
- Customer OWNS Account
- Account SENT Transaction
- Transaction TO Account
- Advisor MANAGES Customer

Include demo queries for:
- high-risk customers
- high-value transactions
- circular money movement
- advisor exposure
- customer-to-customer transaction exposure

Use Python 3.11+, Neo4j Python driver, pandas, python-dotenv, and pytest.

Make the project runnable locally using Docker Compose for Neo4j.

Acceptance criteria:
1. `docker compose up -d` starts Neo4j.
2. `python scripts/generate_data.py` creates sample CSVs.
3. `python scripts/create_schema.py` creates constraints.
4. `python scripts/load_to_neo4j.py` loads data idempotently.
5. `python scripts/run_queries.py` prints useful demo outputs.
6. `pytest` passes.
7. README explains setup, architecture, graph model, and use cases.
