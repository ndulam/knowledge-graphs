"""Chunk compliance PDFs, embed with sentence-transformers, and store in Neo4j vector index."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

PDF_DIR = Path(__file__).parent.parent / "data" / "pdfs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_DIMS = 384
CHUNK_WORDS = 120   # target words per chunk
OVERLAP_WORDS = 20  # words of overlap between consecutive chunks


def _build_driver():
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.getenv("NEO4J_USERNAME", "neo4j"),
            os.getenv("NEO4J_PASSWORD", "password"),
        ),
    )


def _ensure_vector_index(driver) -> None:
    with driver.session() as session:
        session.run(
            """
            CREATE VECTOR INDEX document_chunks IF NOT EXISTS
            FOR (n:DocumentChunk) ON (n.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: $dims,
                `vector.similarity_function`: 'cosine'
            }}
            """,
            dims=VECTOR_DIMS,
        )


def _chunk_words(text: str) -> list[str]:
    """Sliding-window word chunking."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start : start + CHUNK_WORDS])
        if len(chunk) > 40:
            chunks.append(chunk)
        start += CHUNK_WORDS - OVERLAP_WORDS
    return chunks


def _store_chunks(session, rows: list[dict]) -> None:
    session.run(
        """
        UNWIND $rows AS row
        MERGE (c:DocumentChunk {id: row.id})
        SET c.text      = row.text,
            c.source    = row.source,
            c.page      = row.page,
            c.embedding = row.embedding
        """,
        rows=rows,
    )


def ingest_pdfs() -> None:
    from pypdf import PdfReader
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PDF_DIR}. Run scripts/generate_pdfs.py first.")
        return

    driver = _build_driver()
    _ensure_vector_index(driver)

    total = 0
    with driver.session() as session:
        for pdf_path in pdf_files:
            print(f"  Ingesting {pdf_path.name}...", end=" ", flush=True)
            reader = PdfReader(str(pdf_path))
            rows: list[dict] = []
            for page_num, page in enumerate(reader.pages, start=1):
                raw = (page.extract_text() or "").replace("\n", " ")
                for idx, chunk in enumerate(_chunk_words(raw)):
                    rows.append(
                        {
                            "id": f"{pdf_path.stem}_p{page_num}_c{idx}",
                            "text": chunk,
                            "source": pdf_path.name,
                            "page": page_num,
                        }
                    )
            if rows:
                texts = [r["text"] for r in rows]
                embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
                for row, emb in zip(rows, embeddings):
                    row["embedding"] = emb.tolist()
                _store_chunks(session, rows)
            print(f"{len(rows)} chunks")
            total += len(rows)

    driver.close()
    print(f"\nDone. Stored {total} DocumentChunk nodes in Neo4j vector index 'document_chunks'.")


if __name__ == "__main__":
    ingest_pdfs()
