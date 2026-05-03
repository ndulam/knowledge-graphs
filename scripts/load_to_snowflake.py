"""
Load CSV data and generate synthetic time-series into Snowflake.

Run after generate_data.py:
    python scripts/load_to_snowflake.py
"""
from __future__ import annotations

import os
import random
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"


# ── Connection ────────────────────────────────────────────────────────────────

def build_snowflake_conn():
    """Return a Snowflake connection, or raise if credentials are missing."""
    account = os.getenv("SNOWFLAKE_ACCOUNT", "")
    user = os.getenv("SNOWFLAKE_USER", "")
    password = os.getenv("SNOWFLAKE_PASSWORD", "")
    if not account or not user or not password:
        raise RuntimeError(
            "SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, and SNOWFLAKE_PASSWORD must be set in .env"
        )
    import snowflake.connector
    return snowflake.connector.connect(
        account=account,
        user=user,
        password=password,
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        database=os.getenv("SNOWFLAKE_DATABASE", "FINANCIAL_KG"),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"),
    )


# ── Schema setup ──────────────────────────────────────────────────────────────

def setup_schema(cur) -> None:
    database = os.getenv("SNOWFLAKE_DATABASE", "FINANCIAL_KG")
    schema = os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
    cur.execute(f"USE DATABASE {database}")
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    cur.execute(f"USE SCHEMA {schema}")


def create_tables(cur) -> None:
    cur.execute("""
        CREATE OR REPLACE TABLE CUSTOMERS (
            CUSTOMER_ID  VARCHAR(10),
            NAME         VARCHAR(100),
            RISK_SCORE   FLOAT,
            COUNTRY      VARCHAR(5)
        )
    """)
    cur.execute("""
        CREATE OR REPLACE TABLE ACCOUNTS (
            ACCOUNT_ID   VARCHAR(10),
            CUSTOMER_ID  VARCHAR(10),
            ACCOUNT_TYPE VARCHAR(20)
        )
    """)
    cur.execute("""
        CREATE OR REPLACE TABLE TRANSACTIONS (
            TXN_ID        VARCHAR(10),
            FROM_ACCOUNT  VARCHAR(10),
            TO_ACCOUNT    VARCHAR(10),
            AMOUNT        FLOAT,
            TXN_TIMESTAMP TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE OR REPLACE TABLE ADVISORS (
            ADVISOR_ID   VARCHAR(10),
            NAME         VARCHAR(100),
            CUSTOMER_ID  VARCHAR(10)
        )
    """)
    # Synthetic time-series: the key Snowflake-unique data source
    cur.execute("""
        CREATE OR REPLACE TABLE MONTHLY_TX_SUMMARY (
            MONTH                DATE,
            CUSTOMER_ID          VARCHAR(10),
            CUSTOMER_NAME        VARCHAR(100),
            RISK_SCORE           FLOAT,
            COUNTRY              VARCHAR(5),
            TOTAL_SENT           FLOAT,
            TOTAL_RECEIVED       FLOAT,
            TXN_COUNT_SENT       INT,
            TXN_COUNT_RECEIVED   INT,
            UNIQUE_COUNTERPARTIES INT
        )
    """)
    print("  Tables created.")


# ── Data loading ──────────────────────────────────────────────────────────────

def _insert_df(cur, table: str, df: pd.DataFrame) -> None:
    """Bulk insert a DataFrame using executemany."""
    cols = ", ".join(df.columns.str.upper())
    placeholders = ", ".join(["%s"] * len(df.columns))
    sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
    rows = [tuple(r) for r in df.itertuples(index=False, name=None)]
    cur.executemany(sql, rows)
    print(f"  Loaded {len(rows):>4} rows -> {table}")


def load_csvs(cur) -> None:
    _insert_df(cur, "CUSTOMERS",    pd.read_csv(DATA_DIR / "customers.csv"))
    _insert_df(cur, "ACCOUNTS",     pd.read_csv(DATA_DIR / "accounts.csv"))

    tx = pd.read_csv(DATA_DIR / "transactions.csv")
    tx = tx.rename(columns={"timestamp": "txn_timestamp"})
    _insert_df(cur, "TRANSACTIONS", tx)

    _insert_df(cur, "ADVISORS",     pd.read_csv(DATA_DIR / "advisors.csv"))


# ── Synthetic time-series (the unique Snowflake contribution) ─────────────────

def generate_monthly_summary(cur) -> None:
    """
    Build 12 months (Jan–Dec 2025) of per-customer transaction stats.
    High-risk customers (risk_score > 0.80) show a realistic volume spike in
    Oct–Dec to simulate the kind of anomalous activity pattern that triggers
    a SAR review — data that only becomes visible through time-series analytics.
    """
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    random.seed(42)

    rows = []
    for _, cust in customers.iterrows():
        cid      = cust["customer_id"]
        risk     = float(cust["risk_score"])
        is_hrc   = risk >= 0.80

        # Base monthly volumes scale with risk level
        base_sent = 1500 + risk * 7000

        for month_num in range(1, 13):
            # High-risk customers spike in Q4 (Oct–Dec)
            spike = 1.0
            if is_hrc and month_num >= 10:
                spike = 2.5 + (month_num - 10) * 1.8   # 2.5× Oct, 4.3× Nov, 6.1× Dec

            noise      = random.uniform(0.75, 1.30)
            total_sent = round(base_sent * spike * noise, 2)
            total_recv = round(base_sent * spike * random.uniform(0.50, 1.20), 2)
            sent_count = random.randint(2, 9)
            recv_count = random.randint(1, 6)
            peers      = random.randint(1, min(sent_count + recv_count, 6))

            rows.append((
                date(2025, month_num, 1).isoformat(),
                cid,
                str(cust["name"]),
                risk,
                str(cust["country"]),
                total_sent,
                total_recv,
                sent_count,
                recv_count,
                peers,
            ))

    cur.executemany(
        """INSERT INTO MONTHLY_TX_SUMMARY
           (MONTH, CUSTOMER_ID, CUSTOMER_NAME, RISK_SCORE, COUNTRY,
            TOTAL_SENT, TOTAL_RECEIVED, TXN_COUNT_SENT, TXN_COUNT_RECEIVED,
            UNIQUE_COUNTERPARTIES)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        rows,
    )
    print(f"  Loaded {len(rows):>4} rows -> MONTHLY_TX_SUMMARY (12 months x {len(customers)} customers)")


# ── Entry point ───────────────────────────────────────────────────────────────

def load_to_snowflake() -> None:
    print("Connecting to Snowflake...")
    conn = build_snowflake_conn()
    cur = conn.cursor()

    print("Setting up schema...")
    setup_schema(cur)

    print("Creating tables...")
    create_tables(cur)

    print("Loading CSV data...")
    load_csvs(cur)

    print("Generating monthly time-series...")
    generate_monthly_summary(cur)

    conn.commit()
    cur.close()
    conn.close()
    print("\nSnowflake loaded successfully.")


if __name__ == "__main__":
    load_to_snowflake()
