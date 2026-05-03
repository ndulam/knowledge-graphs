from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

REQUIRED_FILES = [
    "requirements.txt",
    ".env.example",
    "docker-compose.yml",
    "README.md",
    "scripts/generate_data.py",
    "scripts/create_schema.py",
    "scripts/load_to_neo4j.py",
    "scripts/run_queries.py",
    "cypher/schema.cypher",
    "cypher/queries.cypher",
    "data/customers.csv",
    "data/accounts.csv",
    "data/transactions.csv",
    "data/advisors.csv",
    "docs/graph_model.md",
    "docs/ontology_semantic_layer.md",
]


def test_required_files_exist():
    missing = [f for f in REQUIRED_FILES if not (PROJECT_ROOT / f).exists()]
    assert not missing, f"Missing required files:\n" + "\n".join(f"  {f}" for f in missing)
