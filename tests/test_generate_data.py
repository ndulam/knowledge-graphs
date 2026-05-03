import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import generate_data as gd


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(gd, "DATA_DIR", tmp_path)
    gd.generate_data()
    return tmp_path


def test_customers_file_created(data_dir):
    assert (data_dir / "customers.csv").exists()


def test_accounts_file_created(data_dir):
    assert (data_dir / "accounts.csv").exists()


def test_transactions_file_created(data_dir):
    assert (data_dir / "transactions.csv").exists()


def test_advisors_file_created(data_dir):
    assert (data_dir / "advisors.csv").exists()


def test_customers_columns(data_dir):
    df = pd.read_csv(data_dir / "customers.csv")
    assert set(df.columns) == {"customer_id", "name", "risk_score", "country"}


def test_accounts_columns(data_dir):
    df = pd.read_csv(data_dir / "accounts.csv")
    assert set(df.columns) == {"account_id", "customer_id", "account_type"}


def test_transactions_columns(data_dir):
    df = pd.read_csv(data_dir / "transactions.csv")
    assert set(df.columns) == {"txn_id", "from_account", "to_account", "amount", "timestamp"}


def test_advisors_columns(data_dir):
    df = pd.read_csv(data_dir / "advisors.csv")
    assert set(df.columns) == {"advisor_id", "name", "customer_id"}


def test_risk_score_range(data_dir):
    df = pd.read_csv(data_dir / "customers.csv")
    assert df["risk_score"].between(0.0, 1.0).all()


def test_transaction_amounts_positive(data_dir):
    df = pd.read_csv(data_dir / "transactions.csv")
    assert (df["amount"] > 0).all()


def test_customer_ids_unique(data_dir):
    df = pd.read_csv(data_dir / "customers.csv")
    assert df["customer_id"].is_unique


def test_account_ids_unique(data_dir):
    df = pd.read_csv(data_dir / "accounts.csv")
    assert df["account_id"].is_unique


def test_transaction_ids_unique(data_dir):
    df = pd.read_csv(data_dir / "transactions.csv")
    assert df["txn_id"].is_unique


def test_accounts_reference_valid_customers(data_dir):
    customers = pd.read_csv(data_dir / "customers.csv")
    accounts = pd.read_csv(data_dir / "accounts.csv")
    assert accounts["customer_id"].isin(customers["customer_id"]).all()


def test_advisors_reference_valid_customers(data_dir):
    customers = pd.read_csv(data_dir / "customers.csv")
    advisors = pd.read_csv(data_dir / "advisors.csv")
    assert advisors["customer_id"].isin(customers["customer_id"]).all()
