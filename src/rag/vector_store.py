import os
import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any, Optional

class VectorStore:
    """
    ChromaDB-backed vector store for JUDGE X AI RAG Pipeline.
    Supports rich metadata filtering and persistent storage.
    """

    def __init__(self, dimension: int, index_type: str = "cosine", persist_dir: str = ".chroma"):
        """
        Initialize the vector store using ChromaDB.
        
        Args:
            dimension: Embedding vector dimension (e.g., 768).
            index_type: Distance metric (cosine by default).
            persist_dir: Directory to save the ChromaDB database.
        """
        self.dimension = dimension
        self.index_type = index_type
        self.persist_dir = persist_dir
        
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        
        space = "cosine" if index_type == "cosine" else "l2"
        self.collection = self.client.get_or_create_collection(
            name="judgments",
            metadata={"hnsw:space": space}
        )

    def clear(self) -> None:
        """Clear all documents from the vector store."""
        try:
            self.client.delete_collection("judgments")
            space = "cosine" if self.index_type == "cosine" else "l2"
            self.collection = self.client.get_or_create_collection(
                name="judgments",
                metadata={"hnsw:space": space}
            )
        except Exception as e:
            print(f"Failed to clear collection: {e}")

    def add(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]) -> None:
        """
        Add embeddings and their corresponding chunk metadata to ChromaDB.
        """
        if len(embeddings) != len(chunks):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(chunks)} chunks")

        # Chroma expects lists of lists/floats for embeddings
        embeddings_list = embeddings.tolist()
        
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = str(chunk.get("chunk_id", f"auto_id_{i}"))
            case_id = str(chunk.get("metadata", {}).get("case_id", "unknown_case"))
            role = str(chunk.get("metadata", {}).get("role", "other"))
            text = chunk.get("text", "")
            
            # Store full chunk JSON string in metadata to preserve original structure
            import json
            safe_meta = {
                "case_id": case_id,
                "role": role,
                "chunk_json": json.dumps(chunk)
            }
            
            documents.append(text)
            metadatas.append(safe_meta)
            ids.append(f"{case_id}_{chunk_id}_{i}")

        self.collection.add(
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(chunks)} vectors to ChromaDB collection 'judgments'.")

    def search(self, query_embedding: np.ndarray, top_k: int = 5, where: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search the collection using the query embedding, with optional metadata filtering.
        """
        query_list = query_embedding.tolist()
        
        results = self.collection.query(
            query_embeddings=[query_list],
            n_results=top_k,
            where=where
        )
        
        out_results = []
        if not results['ids'] or not results['ids'][0]:
            return out_results
            
        import json
        for rank, (doc_id, dist, meta) in enumerate(zip(results['ids'][0], results['distances'][0], results['metadatas'][0]), start=1):
            
            # Distance conversion based on metric
            if self.index_type == "cosine":
                # Chroma cosine distance is 1.0 - cosine_similarity. So sim = 1.0 - dist
                sim = max(0.0, 1.0 - dist)
                similarity_pct = round(sim * 100, 2)
            else:
                similarity_pct = round(max(0, 100 * (1 / (1 + dist))), 2)

            original_chunk = json.loads(meta['chunk_json'])
            
            out_results.append({
                "rank": rank,
                "score": float(dist),
                "similarity_pct": similarity_pct,
                "chunk": original_chunk
            })

        return out_results

    def get_all_chunks(self, case_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve all original chunk dictionaries stored in the collection."""
        where = {"case_id": case_id} if case_id else None
        results = self.collection.get(where=where)
        import json
        out_chunks = []
        if results and results.get('metadatas'):
            for meta in results['metadatas']:
                if 'chunk_json' in meta:
                    out_chunks.append(json.loads(meta['chunk_json']))
        return out_chunks

    def get_all_cases(self) -> List[str]:
        """Retrieve a list of unique case IDs stored in the collection."""
        results = self.collection.get()
        cases = set()
        if results and results.get('metadatas'):
            for meta in results['metadatas']:
                if 'case_id' in meta:
                    cases.add(meta['case_id'])
        return sorted(list(cases))

    def save(self, directory: str, name: str = "legal_index") -> None:
        """
        ChromaDB is automatically persisted to self.persist_dir.
        This method is kept for backwards compatibility.
        """
        print(f"ChromaDB automatically persists data to {self.persist_dir}.")

    @classmethod
    def load(cls, directory: str, name: str = "legal_index") -> "VectorStore":
        """
        Loads the VectorStore pointing to the persistent ChromaDB directory.
        """
        # We assume the user wants to load from .chroma or the specified directory
        store = cls(dimension=768, persist_dir=directory)
        print(f"Loaded ChromaDB collection with {store.collection.count()} vectors.")
        return store
