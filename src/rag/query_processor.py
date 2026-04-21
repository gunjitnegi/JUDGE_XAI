"""
Query Processor for JUDGE X AI RAG Pipeline.
Rewrites user queries using LLM for better semantic matching against legal text.
"""

import requests
from typing import Dict, Any


class QueryProcessor:
    """
    Rewrites raw user queries into semantically richer forms
    optimized for legal document retrieval.
    """

    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate",
                 model: str = "llama3.1:8b-instruct-q4_K_M"):
        self.ollama_url = ollama_url
        self.model = model

    def rewrite_query(self, original_query: str) -> Dict[str, Any]:
        """
        Rewrite a user query for better semantic retrieval.

        The LLM expands the query with legal terminology and context
        so the embedding search matches more relevant chunks.

        Args:
            original_query: The raw user question.

        Returns:
            Dict with:
              - original_query: unchanged input
              - rewritten_query: LLM-expanded version
              - search_query: the query to actually use for embedding search
        """
        prompt = f"""You are a legal search query optimizer for Indian court judgments.

Your task: Rewrite the user's question into a detailed search query that will match relevant legal text.

Rules:
- Expand abbreviations (IPC → Indian Penal Code)
- Add related legal terms and synonyms
- Keep it as a single paragraph, no bullet points
- Do NOT answer the question, only rewrite it
- Keep it under 100 words

User question: {original_query}

Rewritten search query:"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 150}
        }

        try:
            response = requests.post(self.ollama_url, json=payload, timeout=15)
            response.raise_for_status()
            rewritten = response.json().get("response", "").strip()

            # Clean up: remove quotes if model wraps it
            rewritten = rewritten.strip('"').strip("'").strip()

            if len(rewritten) < 10:
                # Fallback: use original if LLM returned garbage
                rewritten = original_query

            return {
                "original_query": original_query,
                "rewritten_query": rewritten,
                "search_query": rewritten  # This is what gets embedded for search
            }

        except Exception as e:
            print(f"Query rewriting failed ({e}), using original query.")
            return {
                "original_query": original_query,
                "rewritten_query": original_query,
                "search_query": original_query
            }


if __name__ == "__main__":
    processor = QueryProcessor()

    test_queries = [
        "Why was he jailed?",
        "What IPC sections were applied?",
        "Was the tender condition fair?",
        "Which articles of the constitution were violated?"
    ]

    for q in test_queries:
        result = processor.rewrite_query(q)
        print(f"Original:  {result['original_query']}")
        print(f"Rewritten: {result['rewritten_query']}")
        print()
