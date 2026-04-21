"""
RAG Pipeline Orchestrator for JUDGE X AI.
Ties together: Query Rewriting → Retrieval → Answer Generation → XAI Formatting.
"""

import os
import json
import requests
from typing import Dict, Any, List, Optional

from src.rag.pdf_processor import LegalPDFProcessor
from src.rag.embedding_manager import EmbeddingManager
from src.rag.vector_store import VectorStore
from src.rag.retriever import Retriever
from src.rag.query_processor import QueryProcessor
from src.rag.statutes_manager import StatutesManager
from src.summarization.judgment_summarizer import JudgmentSummarizer


class RAGPipeline:
    """
    Full RAG pipeline for Indian legal judgment Q&A with Explainable AI.

    Pipeline flow:
    1. Query Rewriting (LLM expands the question)
    2. Retrieval (FAISS similarity search with keyword overlap)
    3. Answer Generation (LLM synthesizes answer from retrieved chunks)
    4. XAI Formatting (similarity scores, keyword overlap, source attribution, retrieval trace)
    """

    def __init__(self,
                 ollama_url: str = "http://localhost:11434/api/generate",
                 model: str = "llama3.1:8b-instruct-q4_K_M",
                 index_dir: str = r"c:\final_year\JUDGEXAI\data\faiss_index"):
        self.ollama_url = ollama_url
        self.model = model
        self.index_dir = index_dir

        # Initialize components
        self.query_processor = QueryProcessor(ollama_url=ollama_url, model=model)
        self.embedding_manager = EmbeddingManager()
        self.statutes_manager = StatutesManager()
        self.summarizer = JudgmentSummarizer(ollama_url=ollama_url, model=model)
        self.vector_store = None
        self.retriever = None
        self.current_summary = None

    def load_index(self, index_dir: Optional[str] = None) -> None:
        """Load a previously saved FAISS index."""
        dir_path = index_dir or self.index_dir
        self.vector_store = VectorStore.load(dir_path)
        self.retriever = Retriever(self.embedding_manager, self.vector_store)
        print(f"Index loaded with {self.vector_store.index.ntotal} vectors.")
        
        # Load summary if exists
        sum_path = os.path.join(dir_path, "summary.json")
        if os.path.exists(sum_path):
            with open(sum_path, 'r', encoding='utf-8') as f:
                self.current_summary = json.load(f)
            print("Loaded existing judgment summary.")
        else:
            self.current_summary = None

    def ingest_pdf(self, pdf_path: str, save: bool = True) -> int:
        """
        Process a PDF and add its chunks to the vector store.

        Args:
            pdf_path: Path to the legal judgment PDF.
            save: Whether to persist the index to disk after ingestion.

        Returns:
            Number of chunks ingested.
        """
        # Process PDF
        processor = LegalPDFProcessor(ollama_url=self.ollama_url, model=self.model)
        chunks = processor.process_pdf(pdf_path)

        # Embed
        embeddings = self.embedding_manager.embed_chunks(chunks)

        # Always create a fresh vector store for each new PDF
        self.vector_store = VectorStore(dimension=embeddings.shape[1])

        self.vector_store.add(embeddings, chunks)
        self.vector_store.source_file = os.path.basename(pdf_path)
        self.retriever = Retriever(self.embedding_manager, self.vector_store)

        if save:
            self.vector_store.save(self.index_dir)
            # Save summary as well if generated
            if self.current_summary:
                sum_path = os.path.join(self.index_dir, "summary.json")
                with open(sum_path, 'w', encoding='utf-8') as f:
                    json.dump(self.current_summary, f, indent=2)

        return len(chunks)

    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of the currently loaded document."""
        if not self.vector_store or not self.vector_store.metadata_store:
            return {"error": "No index loaded."}
        
        print("Generating structured judgment summary...")
        self.current_summary = self.summarizer.summarize(self.vector_store.metadata_store)
        
        # Auto-save if index exists
        sum_path = os.path.join(self.index_dir, "summary.json")
        with open(sum_path, 'w', encoding='utf-8') as f:
            json.dump(self.current_summary, f, indent=2)
            
        return self.current_summary

    def _generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate an answer using the LLM with retrieved context.
        Injects a statute reference table when statutes are found in retrieved chunks.

        Returns:
            Dict with 'answer'.
        """
        # Build context string with source markers
        context_parts = []
        all_statute_refs = []
        for r in context_chunks:
            chunk = r["chunk"]
            meta = chunk["metadata"]
            marker = f"[Source: Page {meta['page_number']}, Section: {meta['section']}]"
            context_parts.append(f"{marker}\n{chunk['text']}")
            # Collect statute references for enrichment
            all_statute_refs.extend(meta.get("statutes_mentioned", []))

        context = "\n\n---\n\n".join(context_parts)

        # Build statute reference table if any statutes were found
        statute_table = self.statutes_manager.enrich_context(all_statute_refs)
        statute_section = ""
        if statute_table:
            statute_section = f"\n\n{statute_table}\n"

        prompt = f"""You are a legal AI assistant specialized in Indian law. Answer the question using ONLY the provided context from a court judgment.

Rules:
1. Base your answer strictly on the provided context
2. Cite specific page numbers and sections when making claims
3. If the context doesn't contain enough information, say so
4. Be precise and use legal terminology
5. When statutes (IPC sections, BNS sections, Constitutional Articles) are mentioned, explain what they mean and what punishment/fine they prescribe using the Statute Reference Table below
6. If BNS equivalents are available, mention them alongside IPC sections
7. At the end, list which source chunks you used most and why (as a brief note)
{statute_section}
Context:
{context}

Question: {query}

Answer:"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 600}
        }

        try:
            response = requests.post(self.ollama_url, json=payload, timeout=90)
            response.raise_for_status()
            answer = response.json().get("response", "").strip()
            return {"answer": answer}
        except Exception as e:
            return {"answer": f"Error generating answer: {e}"}

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Full RAG query with XAI output.

        Args:
            question: The user's legal question.
            top_k: Number of chunks to retrieve.

        Returns:
            Complete response dict with:
              - query_info: original and rewritten query (XAI: retrieval trace step 1)
              - retrieved_chunks: ranked results with scores and keyword overlap
              - answer: LLM-generated answer
              - xai: explainability section with full retrieval trace
        """
        if self.retriever is None:
            raise RuntimeError("No index loaded. Call load_index() or ingest_pdf() first.")

        # Step 1: Query Rewriting
        print("Step 1: Rewriting query...")
        query_info = self.query_processor.rewrite_query(question)
        search_query = query_info["search_query"]

        # Step 2: Retrieval
        print("Step 2: Retrieving relevant chunks...")
        results = self.retriever.retrieve(search_query, top_k=top_k)

        # Step 3: Statute Enrichment
        print("Step 3: Enriching with statute knowledge...")
        statute_analysis = self.statutes_manager.get_all_for_chunks(results)

        # Step 4: Answer Generation
        print("Step 4: Generating answer...")
        answer_result = self._generate_answer(question, results)

        # Step 5: XAI Formatting
        print("Step 5: Formatting XAI output...")

        # Build retrieval trace (XAI Goal #4)
        retrieval_trace = [
            {
                "step": 1,
                "action": "Query Rewriting",
                "input": query_info["original_query"],
                "output": query_info["rewritten_query"]
            },
            {
                "step": 2,
                "action": "Semantic Search",
                "detail": f"Retrieved top-{top_k} chunks from FAISS index ({self.vector_store.index.ntotal} total vectors)"
            },
            {
                "step": 3,
                "action": "Statute Enrichment",
                "detail": f"Found {len(statute_analysis)} statute references across retrieved chunks"
            },
            {
                "step": 4,
                "action": "Answer Generation",
                "detail": f"Sent {len(results)} chunks + statute reference table as context to {self.model}"
            }
        ]

        # Build source attribution (XAI Goal #5)
        sources = []
        for r in results:
            chunk = r["chunk"]
            meta = chunk["metadata"]
            sources.append({
                "rank": r["rank"],
                "page": meta["page_number"],
                "section": meta["section"],
                "similarity_pct": r["similarity_pct"],
                "statutes": meta["statutes_mentioned"],
                "keyword_overlap": r["keyword_overlap"]["matching_keywords"],
                "text_preview": chunk["text"][:200] + "..."
            })

        response = {
            "query_info": query_info,
            "answer": answer_result["answer"],
            "retrieved_chunks": results,
            "xai": {
                "retrieval_trace": retrieval_trace,
                "sources": sources,
                "statute_analysis": statute_analysis
            }
        }

        return response


if __name__ == "__main__":
    pipeline = RAGPipeline()

    # Load existing index
    pipeline.load_index()

    # Test query
    question = "Why was the tender condition held to be unconstitutional?"
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}\n")

    result = pipeline.query(question, top_k=3)

    print(f"\n{'='*60}")
    print("ANSWER")
    print(f"{'='*60}")
    print(result["answer"])

    print(f"\n{'='*60}")
    print("XAI: RETRIEVAL TRACE")
    print(f"{'='*60}")
    for step in result["xai"]["retrieval_trace"]:
        print(f"  Step {step['step']}: {step['action']}")
        if "input" in step:
            print(f"    Input:  {step['input']}")
            print(f"    Output: {step['output']}")
        if "detail" in step:
            print(f"    {step['detail']}")

    print(f"\n{'='*60}")
    print("XAI: SOURCE ATTRIBUTION")
    print(f"{'='*60}")
    for s in result["xai"]["sources"]:
        print(f"  Rank {s['rank']} | Page {s['page']} | {s['section']} | Similarity: {s['similarity_pct']:.1f}%")
        print(f"    Statutes: {s['statutes']}")
        print(f"    Keyword matches: {s['keyword_overlap']}")
        print(f"    Preview: {s['text_preview'][:120]}...")
        print()
