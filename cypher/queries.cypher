// High-risk customers
MATCH (c:Customer)
WHERE c.risk_score > 0.8
RETURN c.id AS customer_id, c.name AS name, c.risk_score AS risk_score, c.country AS country
ORDER BY c.risk_score DESC;

// High-value transactions
MATCH (from:Account)-[:SENT]->(t:Transaction)-[:TO]->(to:Account)
WHERE t.amount > 10000
RETURN from.id AS from_account, to.id AS to_account, t.id AS transaction_id, t.amount AS amount
ORDER BY t.amount DESC;

// Circular money movement
MATCH path = (a:Account)-[:SENT]->(:Transaction)-[:TO]->(b:Account),
             (b)-[:SENT]->(:Transaction)-[:TO]->(a)
RETURN path
LIMIT 10;

// Advisor exposure to high-risk customers
MATCH (advisor:Advisor)-[:MANAGES]->(c:Customer)
WHERE c.risk_score > 0.8
RETURN advisor.name AS advisor, collect(c.name) AS high_risk_customers, count(c) AS exposure_count
ORDER BY exposure_count DESC;

// Customer-to-customer transaction exposure
MATCH (c1:Customer)-[:OWNS]->(a1:Account)-[:SENT]->(t:Transaction)-[:TO]->(a2:Account)<-[:OWNS]-(c2:Customer)
RETURN c1.name AS sender_customer, c2.name AS receiver_customer, t.amount AS amount, t.timestamp AS timestamp
ORDER BY t.amount DESC;
