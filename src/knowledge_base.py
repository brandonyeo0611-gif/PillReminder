"""
knowledge_base.py
==================
Run ONCE to load data/drug_interactions.txt into ChromaDB.
After running, rag_retriever.py can query it.
Safe to re-run — drops and rebuilds the collection each time.

USAGE:
    python -m src.knowledge_base
"""

import chromadb
from pathlib import Path

DB_PATH = str(Path(__file__).parents[1] / "data" / "chroma_db")
FACTS_PATH = Path(__file__).parents[1] / "data" / "drug_interactions.txt"


def build_knowledge_base():
    client = chromadb.PersistentClient(path=DB_PATH)

    # Drop and recreate for clean idempotent build
    try:
        client.delete_collection("carewatch_knowledge")
    except Exception:
        pass

    collection = client.create_collection("carewatch_knowledge")

    facts = []
    ids = []

    with open(FACTS_PATH, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line and ":" in line:
                facts.append(line)
                ids.append(f"fact_{i}")

    if not facts:
        print("ERROR: No facts found. Check data/drug_interactions.txt exists.")
        return

    collection.add(documents=facts, ids=ids)
    print(f"Loaded {len(facts)} facts into ChromaDB at {DB_PATH}")

    # Verify write succeeded immediately
    count = collection.count()
    assert count == len(facts), f"ChromaDB count mismatch: expected {len(facts)}, got {count}"
    print(f"Verified: {count} documents queryable")


if __name__ == "__main__":
    build_knowledge_base()
