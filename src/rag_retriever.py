"""
rag_retriever.py
=================
Queries the ChromaDB knowledge base built by knowledge_base.py.
Given a list of anomaly dicts from deviation_detector, returns relevant
medical context as a plain string for the LLM to use.

Returns empty string on any failure — agent always continues without RAG.

Anomaly shape from deviation_detector.check():
- Dict path:   {"activity": str, "type": str, "message": str, "severity": str}
- String path: e.g. "No baseline built yet — need 7 days of data" (no-baseline case)
Only dict items are queried. String items are skipped silently.
"""

import logging
import chromadb
from pathlib import Path

logger = logging.getLogger(__name__)
DB_PATH = str(Path(__file__).parents[1] / "data" / "chroma_db")


class RAGRetriever:
    def __init__(self):
        try:
            client = chromadb.PersistentClient(path=DB_PATH)
            self.collection = client.get_collection("carewatch_knowledge")
            self._available = True
        except Exception as e:
            logger.warning("RAG not available: %s. Run: python -m src.knowledge_base", e)
            self.collection = None
            self._available = False

    def get_context(self, anomalies: list, n_results: int = 3) -> str:
        """
        Given anomaly dicts (or mixed list with string items), return relevant facts.
        String anomalies (e.g. "No baseline built yet") are skipped silently.
        Returns empty string if RAG unavailable, collection empty, or query fails.
        """
        if not self._available or not anomalies:
            return ""

        # Guard 2: empty collection causes ValueError in ChromaDB query
        if self.collection.count() == 0:
            return ""

        # Guard 3: skip string anomalies — only process dicts
        query_terms = " ".join([
            a.get("activity", "") + " " + a.get("type", "")
            for a in anomalies
            if isinstance(a, dict)
        ]).strip()

        if not query_terms:
            return ""

        try:
            results = self.collection.query(
                query_texts=[query_terms],
                n_results=min(n_results, self.collection.count()),
            )
            docs = results.get("documents", [[]])[0]
            return "\n".join(docs)
        except Exception as e:
            logger.warning("RAG query failed: %s", e)
            return ""
