import random
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = Path(__file__).parent.parent / "data"

_COUNTRIES = ["US", "UK", "IN", "SG", "AU", "DE", "CA", "FR", "JP", "BR"]
_ACCOUNT_TYPES = ["SAVINGS", "CHECKING", "BROKERAGE", "INVESTMENT"]
_FIRST_NAMES = ["Alice", "Bob", "Charlie", "Diana", "Ethan", "Fiona", "George",
                "Hannah", "Ivan", "Julia", "Kevin", "Laura", "Marcus", "Nina", "Oscar"]
_LAST_NAMES = ["Johnson", "Smith", "Rao", "Lee", "Brown", "Davis", "Wilson",
               "Chen", "Miller", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris"]
_ADVISOR_NAMES = ["John Miller", "Sarah Wilson", "Michael Chen", "Emily Davis", "Robert Taylor"]


def generate_data(n_customers: int = 20, n_transactions: int = 30, seed: int = 42) -> None:
    random.seed(seed)
    DATA_DIR.mkdir(exist_ok=True)

    # ── Scenario customers C001–C012 (deterministic) ──────────────────────────
    # Each maps to one of the 8 complex queries. Do not reorder.
    scenario_customers = [
        # Q6 + Q8: highest-risk node — blast-radius source and risk-cluster anchor
        {"customer_id": "C001", "name": "Alice Morgan",  "risk_score": 0.95, "country": "US"},
        # Q6: risk-cluster peer (directly transacts with C001)
        {"customer_id": "C002", "name": "Bob Carter",    "risk_score": 0.91, "country": "UK"},
        # Q7: John Miller portfolio — 4/4 clients are high-risk (100%)
        {"customer_id": "C003", "name": "Charlie Ross",  "risk_score": 0.87, "country": "SG"},
        {"customer_id": "C004", "name": "Diana Blake",   "risk_score": 0.85, "country": "DE"},
        # Q2 + Q7: smurfing outflow recipient; also in John Miller's high-risk book
        {"customer_id": "C005", "name": "Ethan Hunt",    "risk_score": 0.82, "country": "FR"},
        # Q3: low-risk contagion — transacts directly with high-risk C001
        {"customer_id": "C006", "name": "Fiona Green",   "risk_score": 0.22, "country": "AU"},
        # Q3: low-risk contagion — transacts directly with high-risk C002
        {"customer_id": "C007", "name": "George Hall",   "risk_score": 0.18, "country": "CA"},
        # Q1 + Q4: owners of the 3-hop circular ring (A008 → A009 → A010 → A008)
        {"customer_id": "C008", "name": "Hannah Park",   "risk_score": 0.55, "country": "JP"},
        {"customer_id": "C009", "name": "Ivan Cruz",     "risk_score": 0.48, "country": "BR"},
        {"customer_id": "C010", "name": "Julia Stone",   "risk_score": 0.61, "country": "IN"},
        # Q2: owns the smurfing account (large inflow → multiple small outflows)
        {"customer_id": "C011", "name": "Kevin White",   "risk_score": 0.72, "country": "US"},
        # Q5: owns the hub account (≥2 inbound sources + ≥2 outbound destinations)
        {"customer_id": "C012", "name": "Laura Kim",     "risk_score": 0.44, "country": "SG"},
    ]

    # Random background customers C013–C020 (capped at 0.75 so Q7 stays clean)
    random_customers = [
        {
            "customer_id": f"C{i:03d}",
            "name": f"{random.choice(_FIRST_NAMES)} {random.choice(_LAST_NAMES)}",
            "risk_score": round(random.uniform(0.10, 0.75), 2),
            "country": random.choice(_COUNTRIES),
        }
        for i in range(13, n_customers + 1)
    ]

    customers = scenario_customers + random_customers
    pd.DataFrame(customers).to_csv(DATA_DIR / "customers.csv", index=False)

    # ── Scenario accounts A001–A012 (one per scenario customer) ──────────────
    scenario_accounts = [
        {"account_id": "A001", "customer_id": "C001", "account_type": "CHECKING"},   # blast-radius source
        {"account_id": "A002", "customer_id": "C002", "account_type": "SAVINGS"},    # risk-cluster peer
        {"account_id": "A003", "customer_id": "C003", "account_type": "BROKERAGE"},  # high-risk portfolio
        {"account_id": "A004", "customer_id": "C004", "account_type": "INVESTMENT"}, # high-risk portfolio
        {"account_id": "A005", "customer_id": "C005", "account_type": "CHECKING"},   # high-risk portfolio
        {"account_id": "A006", "customer_id": "C006", "account_type": "SAVINGS"},    # low-risk contagion
        {"account_id": "A007", "customer_id": "C007", "account_type": "CHECKING"},   # low-risk contagion
        {"account_id": "A008", "customer_id": "C008", "account_type": "SAVINGS"},    # ring node 1
        {"account_id": "A009", "customer_id": "C009", "account_type": "CHECKING"},   # ring node 2
        {"account_id": "A010", "customer_id": "C010", "account_type": "BROKERAGE"},  # ring node 3
        {"account_id": "A011", "customer_id": "C011", "account_type": "CHECKING"},   # smurfing account
        {"account_id": "A012", "customer_id": "C012", "account_type": "SAVINGS"},    # hub account
    ]

    # One background account per random customer, 30% chance of a second
    random_accounts = []
    next_acc = 13
    for cust in random_customers:
        random_accounts.append({
            "account_id": f"A{next_acc:03d}",
            "customer_id": cust["customer_id"],
            "account_type": random.choice(_ACCOUNT_TYPES),
        })
        next_acc += 1
        if random.random() < 0.3:
            random_accounts.append({
                "account_id": f"A{next_acc:03d}",
                "customer_id": cust["customer_id"],
                "account_type": random.choice(_ACCOUNT_TYPES),
            })
            next_acc += 1

    accounts = scenario_accounts + random_accounts
    pd.DataFrame(accounts).to_csv(DATA_DIR / "accounts.csv", index=False)

    all_account_ids = [a["account_id"] for a in accounts]
    base_time = datetime(2026, 1, 1, 10, 0, 0)

    # ── Scenario transactions T001–T016 (deterministic) ──────────────────────
    scenario_transactions = [
        # Q6 + Q8 — high-risk cluster: C001 (0.95) sends large amounts to three other high-risk customers.
        # Q8: blast radius — C001 is the highest-risk node; T001/T002/T003 map its full reach.
        {"txn_id": "T001", "from_account": "A001", "to_account": "A002", "amount": 32000.00,
         "timestamp": (base_time + timedelta(hours=0)).isoformat()},
        {"txn_id": "T002", "from_account": "A001", "to_account": "A003", "amount": 15500.00,
         "timestamp": (base_time + timedelta(hours=6)).isoformat()},
        {"txn_id": "T003", "from_account": "A001", "to_account": "A004", "amount": 18200.00,
         "timestamp": (base_time + timedelta(hours=12)).isoformat()},

        # Q3 — risk contagion: low-risk customers C006 (0.22) and C007 (0.18) transact
        # directly with high-risk customers C001 (0.95) and C002 (0.91).
        {"txn_id": "T004", "from_account": "A006", "to_account": "A001", "amount": 9800.00,
         "timestamp": (base_time + timedelta(hours=18)).isoformat()},
        {"txn_id": "T005", "from_account": "A007", "to_account": "A002", "amount": 5400.00,
         "timestamp": (base_time + timedelta(hours=24)).isoformat()},

        # Q1 + Q4 — 3-hop circular ring: A008 → A009 → A010 → A008.
        # Q4: advisors Sarah Wilson (C008) and Michael Chen (C009, C010) are indirectly
        # exposed because they manage the owners of these ring accounts.
        {"txn_id": "T006", "from_account": "A008", "to_account": "A009", "amount": 12000.00,
         "timestamp": (base_time + timedelta(hours=30)).isoformat()},
        {"txn_id": "T007", "from_account": "A009", "to_account": "A010", "amount": 12000.00,
         "timestamp": (base_time + timedelta(hours=36)).isoformat()},
        {"txn_id": "T008", "from_account": "A010", "to_account": "A008", "amount": 12000.00,
         "timestamp": (base_time + timedelta(hours=42)).isoformat()},

        # Q2 — structuring / smurfing: A011 receives one large inflow ($28k > $10k threshold),
        # then fans out three smaller outflows (avg ≈ $2,600 — well below the $5k threshold).
        {"txn_id": "T009", "from_account": "A002", "to_account": "A011", "amount": 28000.00,
         "timestamp": (base_time + timedelta(hours=48)).isoformat()},
        {"txn_id": "T010", "from_account": "A011", "to_account": "A005", "amount": 2800.00,
         "timestamp": (base_time + timedelta(hours=54)).isoformat()},
        {"txn_id": "T011", "from_account": "A011", "to_account": "A007", "amount": 3100.00,
         "timestamp": (base_time + timedelta(hours=60)).isoformat()},
        {"txn_id": "T012", "from_account": "A011", "to_account": "A012", "amount": 1900.00,
         "timestamp": (base_time + timedelta(hours=66)).isoformat()},

        # Q5 — hub account: A012 receives from 3 distinct sources (T012, T013, T014)
        # and sends to 2 distinct destinations (T015, T016) → qualifies as a transaction hub.
        {"txn_id": "T013", "from_account": "A003", "to_account": "A012", "amount": 7500.00,
         "timestamp": (base_time + timedelta(hours=72)).isoformat()},
        {"txn_id": "T014", "from_account": "A005", "to_account": "A012", "amount": 6200.00,
         "timestamp": (base_time + timedelta(hours=78)).isoformat()},
        {"txn_id": "T015", "from_account": "A012", "to_account": "A006", "amount": 4100.00,
         "timestamp": (base_time + timedelta(hours=84)).isoformat()},
        {"txn_id": "T016", "from_account": "A012", "to_account": "A007", "amount": 3800.00,
         "timestamp": (base_time + timedelta(hours=90)).isoformat()},
    ]

    # Random background transactions to reach n_transactions total
    n_random = max(0, n_transactions - len(scenario_transactions))
    random_transactions = []
    for i in range(len(scenario_transactions) + 1, len(scenario_transactions) + n_random + 1):
        pair = random.sample(all_account_ids, 2)
        random_transactions.append({
            "txn_id": f"T{i:03d}",
            "from_account": pair[0],
            "to_account": pair[1],
            "amount": round(random.uniform(500, 9000), 2),
            "timestamp": (base_time + timedelta(hours=i * 6)).isoformat(),
        })

    transactions = scenario_transactions + random_transactions
    pd.DataFrame(transactions).to_csv(DATA_DIR / "transactions.csv", index=False)

    # ── Advisors ──────────────────────────────────────────────────────────────
    # Customers are assigned in order so advisor portfolios are deterministic:
    #
    #   AD001  John Miller   : C001 C002 C003 C004  → 4/4 high-risk (100%)   [Q7]
    #   AD002  Sarah Wilson  : C005 C006 C007 C008  → 1/4 high-risk; C008 owns ring node  [Q4]
    #   AD003  Michael Chen  : C009 C010 C011 C012  → 0/4 high-risk; C009+C010 own ring nodes [Q4]
    #   AD004  Emily Davis   : C013 C014 C015 C016  → random background
    #   AD005  Robert Taylor : C017 C018 C019 C020  → random background
    all_customer_ids = [c["customer_id"] for c in customers]
    advisor_rows = [
        {"advisor_id": f"AD{idx + 1:03d}", "name": name, "customer_id": cid}
        for idx, name in enumerate(_ADVISOR_NAMES)
        for cid in all_customer_ids[idx * 4 : (idx + 1) * 4]
    ]
    pd.DataFrame(advisor_rows).to_csv(DATA_DIR / "advisors.csv", index=False)

    print(
        f"Generated {len(customers)} customers, {len(accounts)} accounts, "
        f"{len(transactions)} transactions, {len(advisor_rows)} advisor-customer links.\n"
    )
    print("Embedded scenarios:")
    print("  Q1 + Q4  Circular 3-hop ring      A008 → A009 → A010 → A008  (T006-T008)")
    print("  Q2       Structuring / smurfing   A011 receives $28k (T009), fans out T010-T012 (avg $2.6k)")
    print("  Q3       Risk contagion            C006 (0.22) & C007 (0.18) transact with C001/C002 (>0.8)")
    print("  Q4       Advisor ring exposure    Sarah Wilson → C008 owns ring node A008")
    print("  Q5       Hub account              A012: 3 inbound + 2 outbound connections")
    print("  Q6       Risk cluster             C001 (0.95) ↔ C002 (0.91) via T001")
    print("  Q7       Advisor portfolio risk   John Miller: 4/4 clients high-risk (100%)")
    print("  Q8       Blast radius             C001 (0.95) → A001 → T001/T002/T003 → 3 destinations")


if __name__ == "__main__":
    generate_data()
