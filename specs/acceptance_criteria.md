# Acceptance Criteria

## Local setup
- User can clone the repo.
- User can copy `.env.example` to `.env`.
- User can start Neo4j using Docker Compose.

## Data generation
- Running `python scripts/generate_data.py` creates four CSV files in `data/`.
- CSVs include customers, accounts, transactions, and advisors.

## Schema creation
- Running `python scripts/create_schema.py` creates uniqueness constraints in Neo4j.
- Script is idempotent.

## Data loading
- Running `python scripts/load_to_neo4j.py` loads nodes and relationships into Neo4j.
- Re-running the script should not duplicate nodes or relationships.

## Demo queries
- Running `python scripts/run_queries.py` prints:
  - high-risk customers
  - high-value transactions
  - circular transaction paths
  - advisor high-risk exposure

## Tests
- `pytest` passes.
- Tests validate data generation output files and required project files.

## Documentation
README includes:
- project overview
- architecture
- setup steps
- graph model
- example queries
- next steps for Graph RAG
