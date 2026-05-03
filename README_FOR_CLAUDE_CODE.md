# How to Use These Artifacts with Claude Code

## Option 1: Start from prompt
1. Open Claude Code in an empty folder.
2. Paste the contents of `prompts/claude_code_main_prompt.md`.
3. Add: `Use CLAUDE.md and specs/project_spec.md as implementation guidance.`
4. Ask Claude Code to create the full repo.

## Option 2: Copy starter artifacts first
1. Create a local folder named `financial-knowledge-graph`.
2. Copy these artifacts into that folder.
3. Open Claude Code in the folder.
4. Ask:

```text
Read CLAUDE.md, specs/project_spec.md, and specs/acceptance_criteria.md.
Generate the complete project implementation based on these requirements.
Create or update all missing files. Then run tests and fix any issues.
```

## Recommended Claude Code workflow

```text
Step 1: Generate the full project.
Step 2: Run pytest.
Step 3: Start Neo4j using docker compose.
Step 4: Run data generation, schema creation, data load, and demo queries.
Step 5: Fix errors until the project runs cleanly.
```

## Commands expected in final project

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
docker compose up -d
python scripts/generate_data.py
python scripts/create_schema.py
python scripts/load_to_neo4j.py
python scripts/run_queries.py
pytest
```
