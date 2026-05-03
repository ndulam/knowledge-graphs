import os
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

SCHEMA_FILE = Path(__file__).parent.parent / "cypher" / "schema.cypher"


def create_schema(driver) -> None:
    schema_text = SCHEMA_FILE.read_text(encoding="utf-8")
    statements = [s.strip() for s in schema_text.split(";") if s.strip()]
    with driver.session() as session:
        for stmt in statements:
            session.run(stmt)
            print(f"  OK: {stmt[:70]}...")


def main() -> None:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        print("Creating Neo4j schema constraints...")
        create_schema(driver)
        print("Schema creation complete.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
