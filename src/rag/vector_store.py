"""
FAISS Vector Store for JUDGE X AI RAG Pipeline.
Stores embeddings alongside chunk metadata for retrieval with full XAI traceability.
"""

import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple


class VectorStore:
    """
    FAISS-backed vector store that maps embeddings to rich chunk metadata.
    
    Supports:
    - Adding chunks with their embeddings and metadata
    - Similarity search returning ranked results with scores
    - Saving/loading the index + metadata to/from disk
    """

    def __init__(self, dimension: int, index_type: str = "flat"):
        """
        Initialize the vector store.
        
        Args:
            dimension: Embedding vector dimension (e.g., 768 for nomic-embed-text).
            index_type: Type of FAISS index. 'flat' for exact search (best for < 100k vectors).
        """
        self.dimension = dimension
        self.index_type = index_type

        if index_type == "flat":
            # L2 (Euclidean) distance — smaller = more similar
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "cosine":
            # Normalize + inner product = cosine similarity
            self.index = faiss.IndexFlatIP(dimension)
            self._normalize = True
        else:
            raise ValueError(f"Unsupported index type: {index_type}. Use 'flat' or 'cosine'.")

        self._normalize = index_type == "cosine"
        self.metadata_store: List[Dict[str, Any]] = []  # Parallel list of chunk metadata
        self.source_file: Optional[str] = None  # Source PDF filename

    def add(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]) -> None:
        """
        Add embeddings and their corresponding chunk metadata to the store.
        
        Args:
            embeddings: numpy array of shape (n, dimension).
            chunks: List of chunk dicts (from LegalPDFProcessor). Must have same length as embeddings.
        """
        if len(embeddings) != len(chunks):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(chunks)} chunks")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dim {embeddings.shape[1]} != expected {self.dimension}")

        if self._normalize:
            faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.metadata_store.extend(chunks)
        print(f"Added {len(chunks)} vectors. Total vectors in index: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the index for the most similar chunks to the query embedding.
        
        Args:
            query_embedding: 1D numpy array of shape (dimension,).
            top_k: Number of top results to return.
            
        Returns:
            List of result dicts, each containing:
              - 'chunk': the original chunk dict (text + metadata)
              - 'score': similarity score (lower L2 distance = more similar for 'flat',
                         higher IP = more similar for 'cosine')
              - 'rank': 1-indexed rank
        """
        if self.index.ntotal == 0:
            print("Warning: Index is empty. No results to return.")
            return []

        # Reshape to (1, dimension) for FAISS
        query = query_embedding.reshape(1, -1).astype(np.float32)

        if self._normalize:
            faiss.normalize_L2(query)

        distances, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
            if idx == -1:
                continue  # FAISS returns -1 for unfilled slots

            # Convert L2 distance to a similarity percentage (0-100) for XAI transparency
            if self.index_type == "flat":
                # L2 distance: 0 = identical. Convert to similarity score.
                similarity_pct = round(max(0, 100 * (1 / (1 + dist))), 2)
            else:
                # Inner product after normalization = cosine similarity (-1 to 1)
                similarity_pct = round(float(dist) * 100, 2)

            results.append({
                "rank": rank,
                "score": float(dist),
                "similarity_pct": similarity_pct,
                "chunk": self.metadata_store[idx]
            })

        return results

    def save(self, directory: str, name: str = "legal_index") -> None:
        """
        Save the FAISS index and metadata to disk.
        
        Args:
            directory: Directory to save files in.
            name: Base name for the saved files.
        """
        os.makedirs(directory, exist_ok=True)

        index_path = os.path.join(directory, f"{name}.faiss")
        meta_path = os.path.join(directory, f"{name}_metadata.json")

        faiss.write_index(self.index, index_path)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                "dimension": self.dimension,
                "index_type": self.index_type,
                "total_vectors": self.index.ntotal,
                "source_file": self.source_file,
                "chunks": self.metadata_store
            }, f, indent=2)

        print(f"Saved index ({self.index.ntotal} vectors) to {index_path}")
        print(f"Saved metadata to {meta_path}")

    @classmethod
    def load(cls, directory: str, name: str = "legal_index") -> "VectorStore":
        """
        Load a previously saved FAISS index and metadata from disk.
        
        Args:
            directory: Directory containing the saved files.
            name: Base name used when saving.
            
        Returns:
            A VectorStore instance with the loaded index and metadata.
        """
        index_path = os.path.join(directory, f"{name}.faiss")
        meta_path = os.path.join(directory, f"{name}_metadata.json")

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"Index or metadata not found in {directory}")

        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        store = cls(dimension=meta["dimension"], index_type=meta["index_type"])
        store.index = faiss.read_index(index_path)
        store.metadata_store = meta["chunks"]
        store.source_file = meta.get("source_file")

        print(f"Loaded index with {store.index.ntotal} vectors from {index_path}")
        return store


if __name__ == "__main__":
    # Quick integration test with dummy data
    dim = 768
    store = VectorStore(dimension=dim)

    # Create fake embeddings and chunks
    fake_embeddings = np.random.rand(5, dim).astype(np.float32)
    fake_chunks = [
        {"chunk_id": i + 1, "text": f"Sample legal text {i + 1}", "metadata": {"page_number": i + 1, "section": "FACTS"}}
        for i in range(5)
    ]

    store.add(fake_embeddings, fake_chunks)

    # Search with a random query
    query = np.random.rand(dim).astype(np.float32)
    results = store.search(query, top_k=3)

    print("\nSearch Results:")
    for r in results:
        print(f"  Rank {r['rank']}: chunk_id={r['chunk']['chunk_id']}, "
              f"similarity={r['similarity_pct']}%, score={r['score']:.4f}")

    # Test save/load
    store.save("c:/final_year/JUDGEXAI/data", name="test_index")
    loaded = VectorStore.load("c:/final_year/JUDGEXAI/data", name="test_index")
    print(f"\nReloaded index has {loaded.index.ntotal} vectors")
