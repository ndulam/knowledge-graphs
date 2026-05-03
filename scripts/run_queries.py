import os
from dotenv import load_dotenv
from neo4j import GraphDatabase, Session

load_dotenv()


def run_high_risk_customers(session: Session) -> None:
    print("\n=== High-Risk Customers (risk_score > 0.8) ===")
    result = session.run(
        """
        MATCH (c:Customer)
        WHERE c.risk_score > 0.8
        RETURN c.id AS customer_id, c.name AS name, c.risk_score AS risk_score, c.country AS country
        ORDER BY c.risk_score DESC
        """
    )
    rows = list(result)
    if not rows:
        print("  None found.")
        return
    for r in rows:
        print(f"  {r['customer_id']} | {r['name']} | risk={r['risk_score']:.2f} | {r['country']}")


def run_high_value_transactions(session: Session) -> None:
    print("\n=== High-Value Transactions (amount > 10,000) ===")
    result = session.run(
        """
        MATCH (from:Account)-[:SENT]->(t:Transaction)-[:TO]->(to:Account)
        WHERE t.amount > 10000
        RETURN from.id AS from_account, to.id AS to_account,
               t.id AS transaction_id, t.amount AS amount
        ORDER BY t.amount DESC
        """
    )
    rows = list(result)
    if not rows:
        print("  None found.")
        return
    for r in rows:
        print(f"  {r['transaction_id']} | {r['from_account']} -> {r['to_account']} | ${r['amount']:,.2f}")


def run_circular_movements(session: Session) -> None:
    print("\n=== Circular Money Movement ===")
    result = session.run(
        """
        MATCH (a:Account)-[:SENT]->(:Transaction)-[:TO]->(b:Account),
              (b)-[:SENT]->(:Transaction)-[:TO]->(a)
        RETURN DISTINCT a.id AS account_a, b.id AS account_b
        LIMIT 10
        """
    )
    rows = list(result)
    if not rows:
        print("  No circular movements detected.")
        return
    for r in rows:
        print(f"  {r['account_a']} <-> {r['account_b']}")


def run_advisor_exposure(session: Session) -> None:
    print("\n=== Advisor High-Risk Exposure ===")
    result = session.run(
        """
        MATCH (advisor:Advisor)-[:MANAGES]->(c:Customer)
        WHERE c.risk_score > 0.8
        RETURN advisor.name AS advisor,
               collect(c.name) AS high_risk_customers,
               count(c) AS exposure_count
        ORDER BY exposure_count DESC
        """
    )
    rows = list(result)
    if not rows:
        print("  No advisors with high-risk exposure.")
        return
    for r in rows:
        customers = ", ".join(r["high_risk_customers"])
        print(f"  {r['advisor']} | exposure={r['exposure_count']} | [{customers}]")


def run_customer_transaction_exposure(session: Session) -> None:
    print("\n=== Customer-to-Customer Transaction Exposure ===")
    result = session.run(
        """
        MATCH (c1:Customer)-[:OWNS]->(a1:Account)-[:SENT]->(t:Transaction)
              -[:TO]->(a2:Account)<-[:OWNS]-(c2:Customer)
        WHERE c1.id <> c2.id
        RETURN c1.name AS sender, c2.name AS receiver,
               t.amount AS amount, t.timestamp AS timestamp
        ORDER BY t.amount DESC
        LIMIT 10
        """
    )
    rows = list(result)
    if not rows:
        print("  None found.")
        return
    for r in rows:
        print(f"  {r['sender']} -> {r['receiver']} | ${r['amount']:,.2f} | {r['timestamp']}")


def main() -> None:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            run_high_risk_customers(session)
            run_high_value_transactions(session)
            run_circular_movements(session)
            run_advisor_exposure(session)
            run_customer_transaction_exposure(session)
    finally:
        driver.close()
    print("\nDemo queries complete.")


if __name__ == "__main__":
    main()
