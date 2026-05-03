# Graph Model

## Nodes

### Customer
Represents a financial customer.

Properties:
- id
- name
- risk_score
- country

### Account
Represents a financial account owned by a customer.

Properties:
- id
- type

### Transaction
Represents a money movement event.

Properties:
- id
- amount
- timestamp

### Advisor
Represents an advisor who manages customers.

Properties:
- id
- name

## Relationships

```text
(Customer)-[:OWNS]->(Account)
(Account)-[:SENT]->(Transaction)
(Transaction)-[:TO]->(Account)
(Advisor)-[:MANAGES]->(Customer)
```

## Why graph?

Relational tables are good for storing records. Graphs are better for asking relationship questions such as:

- Who is connected to whom?
- How many hops separate two customers?
- Is money moving in a circular pattern?
- Which advisor is exposed to risky customers?
