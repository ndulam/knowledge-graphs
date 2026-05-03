"""
One-click setup: runs the full data pipeline in order.

Usage:
    python scripts/setup_all.py               # runs everything
    python scripts/setup_all.py --skip-snowflake
    python scripts/setup_all.py --check-only  # validate connections, no data load
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
VENV_PYTHON = ROOT / ".venv" / ("Scripts" if sys.platform == "win32" else "bin") / "python"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

# ── Terminal colours ──────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg: str)   -> None: print(f"  {GREEN}[OK]{RESET}  {msg}")
def err(msg: str)  -> None: print(f"  {RED}[FAIL]{RESET} {msg}")
def info(msg: str) -> None: print(f"  {CYAN}[INFO]{RESET} {msg}")
def warn(msg: str) -> None: print(f"  {YELLOW}[WARN]{RESET} {msg}")
def hdr(msg: str)  -> None: print(f"\n{BOLD}{CYAN}{msg}{RESET}")


# ── Connection checks ─────────────────────────────────────────────────────────

def check_neo4j() -> bool:
    hdr("Checking Neo4j connection...")
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "password")),
        )
        driver.verify_connectivity()
        driver.close()
        ok("Neo4j is reachable")
        return True
    except Exception as exc:
        err(f"Neo4j not reachable: {exc}")
        info("Start Neo4j with:  docker compose up -d")
        return False


def check_snowflake() -> bool:
    hdr("Checking Snowflake connection...")
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        account  = os.getenv("SNOWFLAKE_ACCOUNT", "")
        user     = os.getenv("SNOWFLAKE_USER", "")
        password = os.getenv("SNOWFLAKE_PASSWORD", "")
        if not account or not user or not password:
            warn("Snowflake credentials not set in .env — skipping Snowflake steps")
            return False
        import snowflake.connector
        conn = snowflake.connector.connect(
            account=account, user=user, password=password,
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        )
        conn.cursor().execute("SELECT 1")
        conn.close()
        ok("Snowflake is reachable")
        return True
    except Exception as exc:
        err(f"Snowflake not reachable: {exc}")
        return False


def check_llm() -> bool:
    hdr("Checking LLM API key...")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    if os.getenv("ANTHROPIC_API_KEY"):
        ok("ANTHROPIC_API_KEY is set")
        return True
    if os.getenv("OPENAI_API_KEY"):
        ok("OPENAI_API_KEY is set")
        return True
    warn("No LLM API key found — AI features will require a key in the sidebar")
    return False


# ── Script runner ─────────────────────────────────────────────────────────────

def run_script(label: str, script: str) -> bool:
    print(f"\n  {BOLD}>> {label}{RESET}")
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, str(ROOT / script)],
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        ok(f"Done in {elapsed:.1f}s")
        return True
    else:
        err(f"Failed after {elapsed:.1f}s  (exit code {result.returncode})")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="One-click setup for Financial Knowledge Graph")
    parser.add_argument("--skip-snowflake", action="store_true", help="Skip Snowflake data load")
    parser.add_argument("--check-only",     action="store_true", help="Only check connections, do not load data")
    args = parser.parse_args()

    print(f"\n{BOLD}{'='*60}")
    print("  Financial Knowledge Graph — Setup")
    print(f"{'='*60}{RESET}")

    # ── Connection checks ─────────────────────────────────────────────────────
    neo4j_ok     = check_neo4j()
    snowflake_ok = False if args.skip_snowflake else check_snowflake()
    check_llm()

    if args.check_only:
        hdr("Connection check complete.")
        sys.exit(0 if neo4j_ok else 1)

    if not neo4j_ok:
        print(f"\n{RED}Neo4j must be running before setup. Aborting.{RESET}")
        print("Run:  docker compose up -d")
        sys.exit(1)

    # ── Data pipeline ─────────────────────────────────────────────────────────
    hdr("Running data pipeline...")

    steps = [
        ("1/6  Generate synthetic data",          "scripts/generate_data.py"),
        ("2/6  Create Neo4j schema & constraints", "scripts/create_schema.py"),
        ("3/6  Load data into Neo4j",              "scripts/load_to_neo4j.py"),
        ("4/6  Generate compliance PDFs",          "scripts/generate_pdfs.py"),
        ("5/6  Ingest PDFs into vector index",     "scripts/ingest_pdfs.py"),
    ]

    failures = []
    for label, script in steps:
        if not run_script(label, script):
            failures.append(label)

    if snowflake_ok:
        if not run_script("6/6  Load data into Snowflake", "scripts/load_to_snowflake.py"):
            failures.append("6/6  Load data into Snowflake")
    else:
        warn("Skipping Snowflake load (not configured or --skip-snowflake set)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*60}")
    print("  Setup Summary")
    print(f"{'='*60}{RESET}")

    if not failures:
        ok("All steps completed successfully!")
    else:
        for f in failures:
            err(f"Failed: {f}")

    hdr("Next steps:")
    print("  1. Start the UI:   streamlit run ui/app.py")
    print("  2. Open browser:   http://localhost:8501")
    print("  3. Neo4j Browser:  http://localhost:7474  (user: neo4j / password: password)")
    if not snowflake_ok and not args.skip_snowflake:
        print("  4. To enable Snowflake: add credentials to .env and re-run setup")
    print()


if __name__ == "__main__":
    main()
