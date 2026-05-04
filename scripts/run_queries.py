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


def run_structuring(session: Session) -> None:
    print("\n=== Structuring (Smurfing) — large inflow fanned into small outflows ===")
    result = session.run(
        """
        MATCH (a_in:Account)-[:SENT]->(inflow:Transaction)-[:TO]->(hub:Account)
        WITH hub, sum(inflow.amount) AS total_in
        MATCH (hub)-[:SENT]->(outflow:Transaction)-[:TO]->(a_out:Account)
        WITH hub, total_in, collect(outflow.amount) AS out_amounts
        WHERE total_in > 10000 AND size(out_amounts) >= 3
        WITH hub, total_in, out_amounts,
             reduce(s=0.0, x IN out_amounts | s+x) / size(out_amounts) AS avg_out
        WHERE avg_out < 5000
        OPTIONAL MATCH (c:Customer)-[:OWNS]->(hub)
        RETURN hub.id AS hub_account, c.name AS owner, c.risk_score AS risk_score,
               round(total_in) AS total_inflow, size(out_amounts) AS outflow_count,
               round(avg_out) AS avg_outflow
        ORDER BY total_inflow DESC
        """
    )
    rows = list(result)
    if not rows:
        print("  No structuring patterns detected.")
        return
    for r in rows:
        print(f"  {r['hub_account']} ({r['owner']}) | inflow=${r['total_inflow']:,.0f} | "
              f"{r['outflow_count']} outflows avg ${r['avg_outflow']:,.0f}")


def run_layering(session: Session) -> None:
    print("\n=== Layering — 3-hop transaction chains with declining amounts ===")
    result = session.run(
        """
        MATCH (a1:Account)-[:SENT]->(t1:Transaction)-[:TO]->(a2:Account)
              -[:SENT]->(t2:Transaction)-[:TO]->(a3:Account)
              -[:SENT]->(t3:Transaction)-[:TO]->(a4:Account)
        WHERE a1 <> a3 AND a2 <> a4 AND a1 <> a4
          AND t2.amount < t1.amount AND t3.amount < t2.amount
        OPTIONAL MATCH (c1:Customer)-[:OWNS]->(a1)
        OPTIONAL MATCH (c4:Customer)-[:OWNS]->(a4)
        RETURN a1.id AS hop_1, a2.id AS hop_2, a3.id AS hop_3, a4.id AS hop_4,
               t1.amount AS amount_1, t2.amount AS amount_2, t3.amount AS amount_3,
               c1.name AS origin_owner, c4.name AS final_owner
        ORDER BY amount_1 DESC
        LIMIT 10
        """
    )
    rows = list(result)
    if not rows:
        print("  No layering chains detected.")
        return
    for r in rows:
        print(f"  {r['hop_1']}→{r['hop_2']}→{r['hop_3']}→{r['hop_4']} | "
              f"${r['amount_1']:,.0f}→${r['amount_2']:,.0f}→${r['amount_3']:,.0f} | "
              f"{r['origin_owner']} → {r['final_owner']}")


def run_velocity_anomaly(session: Session) -> None:
    print("\n=== Velocity Anomaly — accounts with 3+ outgoing transactions in one day ===")
    result = session.run(
        """
        MATCH (a:Account)-[:SENT]->(t:Transaction)
        WITH a, substring(toString(t.timestamp), 0, 10) AS day,
             count(t) AS burst_count, sum(t.amount) AS total_amount
        WHERE burst_count >= 3
        OPTIONAL MATCH (c:Customer)-[:OWNS]->(a)
        RETURN a.id AS account, c.name AS owner, c.risk_score AS risk_score,
               day, burst_count, round(total_amount) AS total_amount
        ORDER BY burst_count DESC
        """
    )
    rows = list(result)
    if not rows:
        print("  No velocity anomalies detected.")
        return
    for r in rows:
        print(f"  {r['account']} ({r['owner']}, risk={r['risk_score']:.2f}) | "
              f"day={r['day']} | {r['burst_count']} txns | total=${r['total_amount']:,.0f}")


def run_geographic_risk(session: Session) -> None:
    print("\n=== Geographic Concentration Risk — cross-border flows involving high-risk customers ===")
    result = session.run(
        """
        MATCH (c1:Customer)-[:OWNS]->(a1:Account)-[:SENT]->(t:Transaction)
              -[:TO]->(a2:Account)<-[:OWNS]-(c2:Customer)
        WHERE c1.country <> c2.country
          AND (c1.risk_score > 0.7 OR c2.risk_score > 0.7)
        RETURN c1.country AS from_country, c2.country AS to_country,
               count(t) AS txn_count, round(sum(t.amount)) AS total_amount,
               collect(DISTINCT c1.name)[0..3] AS from_customers,
               collect(DISTINCT c2.name)[0..3] AS to_customers
        ORDER BY total_amount DESC
        LIMIT 15
        """
    )
    rows = list(result)
    if not rows:
        print("  No geographic risk concentrations detected.")
        return
    for r in rows:
        print(f"  {r['from_country']} → {r['to_country']} | {r['txn_count']} txns | "
              f"${r['total_amount']:,.0f} | {r['from_customers']} → {r['to_customers']}")


def run_money_mule(session: Session) -> None:
    print("\n=== Money Mule — accounts forwarding ≥85% of received funds ===")
    result = session.run(
        """
        MATCH (mule:Account)
        WITH mule,
          [(src:Account)-[:SENT]->(ti:Transaction)-[:TO]->(mule) | ti.amount] AS in_amounts,
          [(mule)-[:SENT]->(to_t:Transaction)-[:TO]->(dst:Account) | to_t.amount] AS out_amounts
        WHERE size(in_amounts) > 0 AND size(out_amounts) > 0
        WITH mule,
          reduce(s=0.0, x IN in_amounts  | s+x) AS total_in,
          reduce(s=0.0, x IN out_amounts | s+x) AS total_out
        WHERE total_in > 5000 AND (total_out * 1.0 / total_in) >= 0.85
        OPTIONAL MATCH (c:Customer)-[:OWNS]->(mule)
        RETURN mule.id AS mule_account, c.name AS owner, c.risk_score AS risk_score,
               round(total_in) AS total_received, round(total_out) AS total_forwarded,
               round(100.0 * total_out / total_in, 1) AS forward_pct
        ORDER BY forward_pct DESC
        """
    )
    rows = list(result)
    if not rows:
        print("  No money mule patterns detected.")
        return
    for r in rows:
        print(f"  {r['mule_account']} ({r['owner']}, risk={r['risk_score']:.2f}) | "
              f"received=${r['total_received']:,.0f} | forwarded=${r['total_forwarded']:,.0f} | "
              f"{r['forward_pct']}% pass-through")


def run_advisor_network_contagion(session: Session) -> None:
    print("\n=== Advisor Network Contagion — low-risk clients transacting with HRC counterparties ===")
    result = session.run(
        """
        MATCH (adv:Advisor)-[:MANAGES]->(c:Customer)-[:OWNS]->(a:Account)
              -[:SENT]->(:Transaction)-[:TO]->(a2:Account)<-[:OWNS]-(c2:Customer)
        WHERE c2.risk_score > 0.8 AND c.risk_score <= 0.8
        RETURN adv.name AS advisor, c.name AS client, c.risk_score AS client_risk,
               collect(DISTINCT c2.name) AS hrc_counterparties,
               count(DISTINCT c2) AS contagion_count
        ORDER BY contagion_count DESC
        LIMIT 20
        """
    )
    rows = list(result)
    if not rows:
        print("  No advisor network contagion detected.")
        return
    for r in rows:
        print(f"  {r['advisor']} → {r['client']} (risk={r['client_risk']:.2f}) | "
              f"HRC counterparties: {r['hrc_counterparties']}")


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
            run_structuring(session)
            run_layering(session)
            run_velocity_anomaly(session)
            run_geographic_risk(session)
            run_money_mule(session)
            run_advisor_network_contagion(session)
    finally:
        driver.close()
    print("\nDemo queries complete.")


if __name__ == "__main__":
    main()
