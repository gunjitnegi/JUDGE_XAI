"""
End-to-end test: PDF -> Chunks -> Embeddings -> FAISS -> Search
Tests the full Phase 1 + Phase 2 pipeline.
"""
import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rag.pdf_processor import LegalPDFProcessor
from src.rag.embedding_manager import EmbeddingManager
from src.rag.vector_store import VectorStore


def main():
    pdf_path = r"c:\final_year\JUDGEXAI\test_pdf\civil_001_VINISHMA TECHNOLOGIES PVT_ LTD_versusSTATE OF CHHATTISGARH _ ANR_-_2025_ 10 S_C_.pdf"
    index_dir = r"c:\final_year\JUDGEXAI\data\faiss_index"

    # ---- STEP 1: Process PDF ----
    print("=" * 60)
    print("STEP 1: Processing PDF")
    print("=" * 60)
    processor = LegalPDFProcessor()
    chunks = processor.process_pdf(pdf_path)
    print(f"Got {len(chunks)} chunks\n")

    # ---- STEP 2: Generate Embeddings ----
    print("=" * 60)
    print("STEP 2: Generating Embeddings")
    print("=" * 60)
    embedder = EmbeddingManager()
    embeddings = embedder.embed_chunks(chunks)
    print(f"Embedding shape: {embeddings.shape}\n")

    # ---- STEP 3: Store in FAISS ----
    print("=" * 60)
    print("STEP 3: Storing in FAISS")
    print("=" * 60)
    store = VectorStore(dimension=embeddings.shape[1])
    store.add(embeddings, chunks)
    store.save(index_dir)
    print()

    # ---- STEP 4: Test Search ----
    print("=" * 60)
    print("STEP 4: Test Search")
    print("=" * 60)
    test_query = "Why was the tender condition held to be unconstitutional?"
    print(f"Query: {test_query}\n")

    query_emb = embedder.embed_single(test_query)
    results = store.search(query_emb, top_k=3)

    for r in results:
        chunk = r["chunk"]
        meta = chunk["metadata"]
        print(f"--- Rank {r['rank']} | Similarity: {r['similarity_pct']}% | Page: {meta['page_number']} | Section: {meta['section']} ---")
        print(f"Statutes: {meta['statutes_mentioned']}")
        print(f"Text preview: {chunk['text'][:200]}...")
        print()

    # ---- STEP 5: Test Load from Disk ----
    print("=" * 60)
    print("STEP 5: Verify Load from Disk")
    print("=" * 60)
    loaded_store = VectorStore.load(index_dir)
    loaded_results = loaded_store.search(query_emb, top_k=1)
    print(f"Top result from loaded index: Rank {loaded_results[0]['rank']}, "
          f"similarity={loaded_results[0]['similarity_pct']}%")
    print("\n ALL TESTS PASSED!")


if __name__ == "__main__":
    main()
