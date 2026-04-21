"""
Embedding Manager for JUDGE X AI RAG Pipeline.
Uses nomic-embed-text via Ollama for generating vector embeddings of legal text chunks.
"""

import requests
import numpy as np
from typing import List, Dict, Any


class EmbeddingManager:
    """Manages text embedding generation using Ollama's nomic-embed-text model."""

    def __init__(self, ollama_url: str = "http://localhost:11434/api/embed", model: str = "nomic-embed-text"):
        self.ollama_url = ollama_url
        self.model = model
        self._dimension = None  # Will be set after first embedding call

    @property
    def dimension(self) -> int:
        """Return the embedding dimension. Generates a test embedding if not yet known."""
        if self._dimension is None:
            test_emb = self.embed_single("test")
            self._dimension = len(test_emb)
        return self._dimension

    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text string."""
        payload = {
            "model": self.model,
            "input": text
        }

        try:
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            # Ollama /api/embed returns {"embeddings": [[...], ...]}
            embedding = data["embeddings"][0]
            return np.array(embedding, dtype=np.float32)
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama at {self.ollama_url}: {e}")
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected response format from Ollama: {e}")

    def embed_batch(self, texts: List[str], batch_size: int = 10) -> np.ndarray:
        """
        Generate embeddings for a list of texts in batches.
        
        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts to send per API call.
            
        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = {
                "model": self.model,
                "input": batch
            }

            try:
                response = requests.post(self.ollama_url, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                batch_embeddings = data["embeddings"]
                all_embeddings.extend(batch_embeddings)
                print(f"  Embedded batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1} ({len(batch)} texts)")
            except requests.exceptions.RequestException as e:
                raise ConnectionError(f"Failed to embed batch starting at index {i}: {e}")
            except (KeyError, IndexError) as e:
                raise ValueError(f"Unexpected response format from Ollama for batch at index {i}: {e}")

        return np.array(all_embeddings, dtype=np.float32)

    def embed_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 10) -> np.ndarray:
        """
        Convenience method: embed a list of chunk dicts (as produced by LegalPDFProcessor).
        Extracts the 'text' field from each chunk.
        
        Args:
            chunks: List of chunk dictionaries with 'text' key.
            batch_size: Number of chunks to embed per API call.
            
        Returns:
            numpy array of shape (len(chunks), embedding_dim).
        """
        texts = [chunk["text"] for chunk in chunks]
        print(f"Embedding {len(texts)} chunks using {self.model}...")
        embeddings = self.embed_batch(texts, batch_size=batch_size)
        print(f"Done. Embedding matrix shape: {embeddings.shape}")
        return embeddings


if __name__ == "__main__":
    # Quick smoke test
    manager = EmbeddingManager()
    print(f"Embedding dimension: {manager.dimension}")

    test_texts = [
        "The appellant was convicted under IPC Section 420 for cheating.",
        "The High Court dismissed the writ petition challenging the tender condition.",
        "Article 14 of the Constitution guarantees equality before law."
    ]

    embeddings = manager.embed_batch(test_texts)
    print(f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")

    # Verify cosine similarity between related texts
    from numpy.linalg import norm
    def cosine_sim(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))

    print(f"\nCosine similarities:")
    for i in range(len(test_texts)):
        for j in range(i + 1, len(test_texts)):
            sim = cosine_sim(embeddings[i], embeddings[j])
            print(f"  Text {i+1} vs Text {j+1}: {sim:.4f}")
