"""
RAG Pipeline Orchestrator for JUDGE X AI.
Ties together: Query Rewriting → Retrieval → Answer Generation → XAI Formatting.
"""

import os
import json
import requests
from typing import Dict, Any, List, Optional
from pathlib import Path
from dotenv import load_dotenv

from src.rag.pdf_processor import LegalPDFProcessor
from src.rag.embedding_manager import EmbeddingManager
from src.rag.vector_store import VectorStore
from src.rag.retriever import Retriever
from src.rag.query_processor import QueryProcessor
from src.rag.statutes_manager import StatutesManager
from src.summarization.judgment_summarizer import JudgmentSummarizer

load_dotenv()

_DEFAULT_INDEX = str(Path(__file__).resolve().parents[2] / "data" / "faiss_index")


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
                 ollama_url: str = None,
                 model: str = None,
                 index_dir: str = None):
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        self.model = model or os.getenv("MODEL_NAME", "llama3.1:8b-instruct-q4_K_M")
        self.index_dir = index_dir or os.getenv("INDEX_DIR", _DEFAULT_INDEX)

        # Initialize components
        self.query_processor = QueryProcessor(ollama_url=self.ollama_url, model=self.model)
        self.embedding_manager = EmbeddingManager()
        self.statutes_manager = StatutesManager()
        self.summarizer = JudgmentSummarizer(ollama_url=self.ollama_url, model=self.model)
        self.vector_store = None
        self.retriever = None
        self.current_summary = None

    def load_index(self, index_dir: Optional[str] = None) -> None:
        """Load a previously saved FAISS index."""
        dir_path = index_dir or self.index_dir
        self.vector_store = VectorStore.load(dir_path)
        self.retriever = Retriever(self.embedding_manager, self.vector_store)
        print(f"Index loaded with {self.vector_store.collection.count()} vectors.")
        
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

        # Create or load vector store
        self.vector_store = VectorStore(dimension=embeddings.shape[1], persist_dir=self.index_dir)
        self.vector_store.add(embeddings, chunks)
        
        # Reset backend summary state for new document
        self.current_summary = None
        self.vector_store.source_file = os.path.basename(pdf_path)
        self.retriever = Retriever(self.embedding_manager, self.vector_store)

        if save:
            self.vector_store.save(self.index_dir)
            # Save summary as well if generated
            if self.current_summary:
                sum_dir = os.path.join(self.index_dir, "summaries")
                os.makedirs(sum_dir, exist_ok=True)
                sum_path = os.path.join(sum_dir, f"summary_{os.path.basename(pdf_path)}.json")
                with open(sum_path, 'w', encoding='utf-8') as f:
                    json.dump(self.current_summary, f, indent=2)

        return len(chunks)

    def generate_summary(self, case_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate a summary of the loaded document (optionally filtered by case_id)."""
        if not self.vector_store:
            return {"error": "No index loaded."}
        
        print(f"Generating structured judgment summary for {case_id or 'all'}...")
        all_chunks = self.vector_store.get_all_chunks(case_id=case_id)
        
        if not all_chunks:
            return {"error": f"No chunks found for case_id: {case_id}"}
            
        self.current_summary = self.summarizer.summarize(all_chunks)
        
        # Auto-save if index exists
        sum_dir = os.path.join(self.index_dir, "summaries")
        os.makedirs(sum_dir, exist_ok=True)
        filename = f"summary_{case_id}.json" if case_id else "summary.json"
        sum_path = os.path.join(sum_dir, filename)
        
        with open(sum_path, 'w', encoding='utf-8') as f:
            json.dump(self.current_summary, f, indent=2)
            
        return self.current_summary

    def load_summary(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Load a previously saved summary for a specific case_id."""
        sum_path = os.path.join(self.index_dir, "summaries", f"summary_{case_id}.json")
        if os.path.exists(sum_path):
            with open(sum_path, 'r', encoding='utf-8') as f:
                self.current_summary = json.load(f)
            return self.current_summary
            
        self.current_summary = None
        return None

    def _classify_query_complexity(self, query: str) -> str:
        """Classify if the query requires a LAYMAN or TECHNICAL response."""
        prompt = f"""You are a query classifier. Analyze the following user question and categorize its complexity.
If the question is conversational, uses simple English, or asks basic facts (e.g., 'who won?', 'what happened?'), respond with exactly 'LAYMAN'.
If the question uses specific legal terminology, asks about statutes, or requires technical interpretation, respond with exactly 'TECHNICAL'.

Question: "{query}"
Category (LAYMAN or TECHNICAL):"""
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 5}
        }
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=10)
            response.raise_for_status()
            result_text = response.json().get("response", "").strip().upper()
            if "TECHNICAL" in result_text:
                return "TECHNICAL"
            return "LAYMAN"
        except Exception as e:
            print(f"Classification failed: {e}")
            return "LAYMAN"  # default to simple

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

        # Inject Global Case Summary into context to answer high-level questions (like "who won")
        if self.current_summary and "sections" in self.current_summary:
            summary_context = "[GLOBAL CASE SUMMARY FOR CONTEXT]\n"
            for section, text in self.current_summary["sections"].items():
                if text and isinstance(text, str) and "No content found" not in text:
                    # Inject first 400 chars of each summary section to avoid context bloat
                    summary_context += f"**{section.upper()}**: {text[:400]}...\n"
            context_parts.insert(0, summary_context)

        context = "\n\n---\n\n".join(context_parts)

        # Build statute reference table if any statutes were found
        statute_table = self.statutes_manager.enrich_context(all_statute_refs)
        statute_section = ""
        if statute_table:
            statute_section = f"\n\n{statute_table}\n"

        # Route Prompt Tone Based on Complexity
        complexity = self._classify_query_complexity(query)
        
        if complexity == "LAYMAN":
            tone_instructions = """
CRITICAL INSTRUCTIONS FOR LANGUAGE AND TONE:
1. Use SIMPLE, EVERYDAY, LAYMAN'S English. Your answer must be easily understood by the general public.
2. DO NOT use dense legal jargon (e.g., "impugned", "inter alia", "appellant", "respondent"). Translate these into normal words (e.g., "challenged", "among other things", "the person appealing", "the defending party").
3. Explain any laws or section numbers in simple terms rather than just quoting the numbers."""
            answer_suffix = "Answer in simple, clear, layman's terms:"
        else:
            tone_instructions = """
CRITICAL INSTRUCTIONS FOR LANGUAGE AND TONE:
1. Provide a STRICT, PROFESSIONAL legal analysis.
2. Utilize precise statutory interpretation and advanced legal terminology appropriate for a lawyer or judge.
3. Be highly technical and accurately reference the legal mechanisms, sections, and conditions at play."""
            answer_suffix = "Answer with precise legal terminology:"

        prompt = f"""You are a legal AI assistant specialized in Indian law. Answer the question using ONLY the provided context from a court judgment.
{tone_instructions}

Rules:
1. Base your answer strictly on the provided context.
2. If the context does not contain enough information to answer the question, clearly state: "The provided context does not contain this information."
3. **CRITICAL FORMATTING:** Answer in continuous paragraphs ONLY. Do NOT use bullet points, numbered lists, or conversational preambles. Start your answer immediately.
4. **CRITICAL GROUNDING:** Extract explicitly stated facts ONLY. Do NOT make logical inferences, deductions, or assumptions.
5. Do NOT list source chunks, page numbers, or sections in your response. The system will handle citations automatically.
6. Do NOT bring in external legal knowledge (e.g., IPC sections, article numbers) unless explicitly written in the provided context.
7. When statutes are mentioned, explain them using the Statute Reference Table below.
8. If BNS equivalents are available, mention them alongside IPC sections.
{statute_section}
Context:
{context}

Question: {query}

{answer_suffix}"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,  # Enable streaming
            "options": {"temperature": 0.1, "num_predict": 600}
        }

        try:
            response = requests.post(self.ollama_url, json=payload, stream=True, timeout=90)
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    yield chunk.get("response", "")
        except Exception as e:
            yield f"Error generating answer: {e}"

    def query(self, question: str, top_k: int = 5, case_id: str = None) -> Dict[str, Any]:
        """
        Full RAG query with XAI output.

        Args:
            question: The user's legal question.
            top_k: Number of chunks to retrieve.
            case_id: Optional specific case to query against.

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
        
        # Step 2: Retrieval with XAI Overlap
        retrieved_chunks = self.retriever.retrieve(query_info["rewritten_query"], top_k=top_k, case_id=case_id)

        # Step 3: Statute Enrichment
        print("Step 3: Enriching with statute knowledge...")
        statute_analysis = self.statutes_manager.get_all_for_chunks(retrieved_chunks)

        # Step 4: Answer Generation
        print("Step 4: Generating answer (streaming)...")
        
        # Build XAI artifacts early so we have them ready
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
                "detail": f"Retrieved top-{top_k} chunks from FAISS index ({self.vector_store.collection.count()} total vectors)"
            },
            {
                "step": 3,
                "action": "Statute Enrichment",
                "detail": f"Found {len(statute_analysis)} statute references across retrieved chunks"
            },
            {
                "step": 4,
                "action": "Answer Generation",
                "detail": f"Sent {len(retrieved_chunks)} chunks + statute reference table as context to {self.model}"
            }
        ]

        sources = []
        for r in retrieved_chunks:
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
            
        full_answer = ""
        for chunk in self._generate_answer(question, retrieved_chunks):
            full_answer += chunk
            yield chunk

        # Yield the final complete result dictionary
        response = {
            "query_info": query_info,
            "answer": full_answer.strip(),
            "retrieved_chunks": retrieved_chunks,
            "xai": {
                "retrieval_trace": retrieval_trace,
                "sources": sources,
                "statute_analysis": statute_analysis
            }
        }
        yield response


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
