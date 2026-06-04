#!/usr/bin/env python3
"""
RAG Evaluator for JUDGEXAI

This script evaluates the RAG Pipeline by comparing its generated answers against
a ground-truth QA dataset. It calculates Semantic Similarity using sentence-transformers
and Lexical Overlap (ROUGE-style) to measure answer accuracy.
"""

import os
import sys
import json
import argparse
import random
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Ensure project root is in sys.path
sys.path.append(r"c:\final_year\JUDGEXAI")

# Local imports
from src.rag.rag_pipeline import RAGPipeline

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class RAGEvaluator:
    def __init__(self, dataset_path: str, ollama_url: str = "http://localhost:11434/api/generate", model: str = "llama3.1:8b-instruct-q4_K_M"):
        self.dataset_path = dataset_path
        self.pipeline = RAGPipeline(ollama_url=ollama_url, model=model)
        
        # Load embedding model for semantic similarity grading
        if HAS_SENTENCE_TRANSFORMERS:
            print("Loading SentenceTransformer model for semantic grading...")
            self.eval_model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            print("Warning: sentence-transformers not found. Semantic Similarity will not be calculated.")
            self.eval_model = None

    def _calculate_lexical_overlap(self, pred: str, truth: str) -> float:
        """Calculates a simple token overlap ratio (Jaccard similarity style)."""
        pred_tokens = set(pred.lower().replace('.', '').replace(',', '').split())
        truth_tokens = set(truth.lower().replace('.', '').replace(',', '').split())
        
        if not pred_tokens or not truth_tokens:
            return 0.0
            
        intersection = pred_tokens.intersection(truth_tokens)
        union = pred_tokens.union(truth_tokens)
        return len(intersection) / len(union)

    def _calculate_semantic_similarity(self, pred: str, truth: str) -> float:
        """Calculates cosine similarity between prediction and ground truth embeddings."""
        if not self.eval_model:
            return 0.0
        
        pred_emb = self.eval_model.encode(pred)
        truth_emb = self.eval_model.encode(truth)
        
        sim = cos_sim(pred_emb, truth_emb).item()
        return max(0.0, sim)  # Clamp negative to 0

    def load_samples(self, n_samples: int) -> List[Dict[str, Any]]:
        """Loads a random subset of QA pairs from the dataset."""
        samples = []
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    if "qa_pairs" in data:
                        for pair in data["qa_pairs"]:
                            samples.append({
                                "case_id": data.get("case_id", "unknown"),
                                "question": pair["question"],
                                "ground_truth": pair["answer"],
                            })
        except FileNotFoundError:
            print(f"Dataset not found at {self.dataset_path}")
            return []

        if len(samples) > n_samples:
            random.seed(42)
            samples = random.sample(samples, n_samples)
            
        return samples

    def evaluate(self, n_samples: int = 50, output_csv: str = "evaluation_results.csv", index_dir: str = None):
        """Runs the evaluation pipeline."""
        print(f"Loading {n_samples} samples from dataset...")
        samples = self.load_samples(n_samples)
        
        if not samples:
            print("No samples loaded. Exiting.")
            return

        print("Loading RAG Index...")
        try:
            if index_dir:
                self.pipeline.load_index(index_dir=index_dir)
            else:
                self.pipeline.load_index()
        except Exception as e:
            print(f"Failed to load FAISS index: {e}")
            print("Please ensure you have ingested a PDF and created a FAISS index first.")
            return

        results = []
        total_semantic = 0.0
        total_lexical = 0.0
        total_latency = 0.0

        print(f"\nStarting Evaluation on {len(samples)} questions...\n")

        for i, sample in enumerate(samples):
            question = sample["question"]
            truth = sample["ground_truth"]
            
            print(f"[{i+1}/{len(samples)}] Question: {question}")
            
            start_time = time.time()
            try:
                # Query the RAG Pipeline
                response = self.pipeline.query(question, top_k=3)
                pred_answer = response["answer"]
                
                # Check if it failed to find context
                if "Error" in pred_answer or not response.get("retrieved_chunks"):
                    pred_answer = "Failed to retrieve context."
                    
            except Exception as e:
                print(f"  -> Error querying pipeline: {e}")
                pred_answer = "Error during generation."

            latency = time.time() - start_time
            
            # Calculate metrics
            sem_sim = self._calculate_semantic_similarity(pred_answer, truth)
            lex_overlap = self._calculate_lexical_overlap(pred_answer, truth)
            
            total_semantic += sem_sim
            total_lexical += lex_overlap
            total_latency += latency
            
            print(f"  -> Latency: {latency:.2f}s | Semantic Sim: {sem_sim:.2f} | Lexical Overlap: {lex_overlap:.2f}")

            results.append({
                "case_id": sample["case_id"],
                "question": question,
                "ground_truth": truth,
                "predicted_answer": pred_answer,
                "semantic_similarity": sem_sim,
                "lexical_overlap": lex_overlap,
                "latency_seconds": latency
            })

        # Save to CSV
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False, encoding='utf-8')
        
        # Print Summary
        avg_sem = total_semantic / len(samples)
        avg_lex = total_lexical / len(samples)
        avg_lat = total_latency / len(samples)
        
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Total Questions Tested : {len(samples)}")
        print(f"Average Latency        : {avg_lat:.2f} seconds")
        print(f"Average Semantic Sim   : {avg_sem:.2f} (0 to 1)")
        print(f"Average Lexical Overlap: {avg_lex:.2f} (0 to 1)")
        print(f"\nFull results saved to: {output_csv}")
        print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JUDGEXAI RAG Evaluator")
    parser.add_argument("--samples", type=int, default=5, help="Number of random samples to test")
    parser.add_argument("--dataset", type=str, default=r"c:\final_year\JUDGEXAI\data\processed\legal_qa_dataset.jsonl", help="Path to QA dataset")
    parser.add_argument("--output", type=str, default="evaluation_results.csv", help="Output CSV path")
    parser.add_argument("--index", type=str, default=r"c:\final_year\JUDGEXAI\data\faiss_index", help="Path to FAISS index directory")
    
    args = parser.parse_args()
    
    evaluator = RAGEvaluator(dataset_path=args.dataset)
    evaluator.evaluate(n_samples=args.samples, output_csv=args.output, index_dir=args.index)
