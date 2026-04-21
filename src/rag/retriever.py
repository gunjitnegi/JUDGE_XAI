"""
Retriever for JUDGE X AI RAG Pipeline.
Handles similarity search with XAI-ready output (scores, metadata, keyword overlap).
"""

import re
from typing import List, Dict, Any, Set
import numpy as np

from src.rag.embedding_manager import EmbeddingManager
from src.rag.vector_store import VectorStore


class Retriever:
    """
    Retrieves the most relevant chunks for a given query.
    Enriches results with XAI signals: similarity scores, keyword overlap, and metadata highlights.
    """

    def __init__(self, embedding_manager: EmbeddingManager, vector_store: VectorStore):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text (lowercased, de-duplicated)."""
        # Remove common stopwords for legal context
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or',
            'not', 'no', 'nor', 'so', 'yet', 'both', 'each', 'all', 'any', 'few',
            'more', 'most', 'other', 'some', 'such', 'than', 'too', 'very', 'just',
            'also', 'now', 'then', 'here', 'there', 'when', 'where', 'why', 'how',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'it',
            'its', 'he', 'she', 'they', 'them', 'his', 'her', 'their', 'our', 'your',
            'about', 'up', 'out', 'if', 'because', 'until', 'while', 'against',
        }
        words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        return {w for w in words if w not in stopwords and len(w) > 2}

    def _compute_keyword_overlap(self, query: str, chunk_text: str) -> Dict[str, Any]:
        """Compute keyword overlap between query and chunk for XAI transparency."""
        query_keywords = self._extract_keywords(query)
        chunk_keywords = self._extract_keywords(chunk_text)

        matching = query_keywords & chunk_keywords
        total_query = len(query_keywords) if query_keywords else 1

        return {
            "query_keywords": sorted(query_keywords),
            "matching_keywords": sorted(matching),
            "overlap_count": len(matching),
            "overlap_ratio": round(len(matching) / total_query, 2)
        }

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the top-K most relevant chunks for a query.

        Args:
            query: The user's question or search query.
            top_k: Number of results to return.

        Returns:
            List of result dicts with XAI-enriched fields:
              - rank, score, similarity_pct (from VectorStore)
              - chunk (text + metadata)
              - keyword_overlap (XAI: which query words appear in chunk)
        """
        # Embed the query
        query_embedding = self.embedding_manager.embed_single(query)

        # Search FAISS
        results = self.vector_store.search(query_embedding, top_k=top_k)

        # Enrich each result with keyword overlap analysis (XAI Goal #2)
        for result in results:
            overlap = self._compute_keyword_overlap(query, result["chunk"]["text"])
            result["keyword_overlap"] = overlap

        return results


if __name__ == "__main__":
    import os

    index_dir = r"c:\final_year\JUDGEXAI\data\faiss_index"

    embedder = EmbeddingManager()
    store = VectorStore.load(index_dir)
    retriever = Retriever(embedder, store)

    query = "What constitutional articles were violated by the tender condition?"
    print(f"Query: {query}\n")

    results = retriever.retrieve(query, top_k=3)

    for r in results:
        chunk = r["chunk"]
        meta = chunk["metadata"]
        kw = r["keyword_overlap"]
        print(f"--- Rank {r['rank']} | Similarity: {r['similarity_pct']:.1f}% | Page: {meta['page_number']} | Section: {meta['section']} ---")
        print(f"  Statutes: {meta['statutes_mentioned']}")
        print(f"  Keyword overlap: {kw['overlap_count']} matches ({kw['overlap_ratio']*100:.0f}%): {kw['matching_keywords']}")
        print(f"  Preview: {chunk['text'][:150]}...")
        print()
