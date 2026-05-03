import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase, Session

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"


def load_customers(session: Session, df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        session.run(
            """
            MERGE (c:Customer {id: $id})
            SET c.name = $name, c.risk_score = $risk_score, c.country = $country
            """,
            id=row["customer_id"],
            name=row["name"],
            risk_score=float(row["risk_score"]),
            country=row["country"],
        )
    print(f"  Loaded {len(df)} customers.")


def load_accounts(session: Session, df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        session.run(
            """
            MERGE (a:Account {id: $id})
            SET a.type = $type
            """,
            id=row["account_id"],
            type=row["account_type"],
        )
        session.run(
            """
            MATCH (c:Customer {id: $customer_id}), (a:Account {id: $account_id})
            MERGE (c)-[:OWNS]->(a)
            """,
            customer_id=row["customer_id"],
            account_id=row["account_id"],
        )
    print(f"  Loaded {len(df)} accounts with OWNS relationships.")


def load_transactions(session: Session, df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        session.run(
            """
            MERGE (t:Transaction {id: $id})
            SET t.amount = $amount, t.timestamp = $timestamp
            """,
            id=row["txn_id"],
            amount=float(row["amount"]),
            timestamp=row["timestamp"],
        )
        session.run(
            """
            MATCH (a:Account {id: $from_account}), (t:Transaction {id: $txn_id})
            MERGE (a)-[:SENT]->(t)
            """,
            from_account=row["from_account"],
            txn_id=row["txn_id"],
        )
        session.run(
            """
            MATCH (t:Transaction {id: $txn_id}), (a:Account {id: $to_account})
            MERGE (t)-[:TO]->(a)
            """,
            txn_id=row["txn_id"],
            to_account=row["to_account"],
        )
    print(f"  Loaded {len(df)} transactions with SENT/TO relationships.")


def load_advisors(session: Session, df: pd.DataFrame) -> None:
    for advisor_id, group in df.groupby("advisor_id"):
        advisor_name = group.iloc[0]["name"]
        session.run(
            """
            MERGE (a:Advisor {id: $id})
            SET a.name = $name
            """,
            id=advisor_id,
            name=advisor_name,
        )
        for _, row in group.iterrows():
            session.run(
                """
                MATCH (advisor:Advisor {id: $advisor_id}), (c:Customer {id: $customer_id})
                MERGE (advisor)-[:MANAGES]->(c)
                """,
                advisor_id=advisor_id,
                customer_id=row["customer_id"],
            )
    print(f"  Loaded {df['advisor_id'].nunique()} advisors with MANAGES relationships.")


def main() -> None:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        print("Loading data into Neo4j...")
        with driver.session() as session:
            load_customers(session, pd.read_csv(DATA_DIR / "customers.csv"))
            load_accounts(session, pd.read_csv(DATA_DIR / "accounts.csv"))
            load_transactions(session, pd.read_csv(DATA_DIR / "transactions.csv"))
            load_advisors(session, pd.read_csv(DATA_DIR / "advisors.csv"))
        print("All data loaded successfully.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
