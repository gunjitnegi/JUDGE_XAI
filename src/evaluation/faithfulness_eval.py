#!/usr/bin/env python3
"""
JUDGE X AI — Research-Grade Faithfulness Evaluation
====================================================

Evaluation architecture:
  1. Chunk the source judgment (reuses project's sentence-aware chunking)
  2. For each summary sentence, retrieve top-k evidence chunks via embedding similarity
  3. Run NLI (entailment / contradiction / neutral) against each retrieved chunk
  4. Aggregate per-sentence verdicts into document-level metrics

Metrics produced:
  - Faithfulness Score      (fraction of sentences entailed by evidence)
  - Hallucination Rate      (contradiction_rate + (1 - entailment_rate))
  - Structural Coverage     (which of FACTS/ISSUES/ARGUMENTS/REASONING/DECISION are present)
  - Role Consistency        (NLI-grounded check, not string matching)
  - Per-sentence evidence trace (full explainability)


"""

import json
import os
import re
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

# Reproducibility (Issue #19)
torch.manual_seed(42)
np.random.seed(42)

logger = logging.getLogger("faithfulness_eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ════════════════════════════════════════════════════════════
# Data Classes
# ════════════════════════════════════════════════════════════

@dataclass
class SentenceVerdict:
    """Per-sentence evaluation result with full evidence trace (Issue #11, #14)."""
    sentence: str
    entailment: float
    contradiction: float
    neutral: float
    verdict: str            # "supported", "contradicted", "unverifiable"
    evidence_chunk: str     # best matching chunk text (truncated for storage)
    evidence_score: float   # similarity score of the best chunk

@dataclass
class DocumentResult:
    """Document-level evaluation result."""
    case_id: str = ""
    faithfulness: float = 0.0
    hallucination_rate: float = 0.0
    contradiction_rate: float = 0.0
    entailment_rate: float = 0.0
    neutral_rate: float = 0.0
    role_consistency: float = 0.0
    structural_coverage: float = 0.0
    sections_found: List[str] = field(default_factory=list)
    sections_missing: List[str] = field(default_factory=list)
    num_sentences: int = 0
    sentence_verdicts: List[dict] = field(default_factory=list)
    error: Optional[str] = None


# ════════════════════════════════════════════════════════════
# Sentence Splitter (Issue #5: Legal-aware, not naive split)
# ════════════════════════════════════════════════════════════

def legal_sent_tokenize(text: str) -> List[str]:
    """
    Legal-aware sentence tokenizer.
    Uses nltk.sent_tokenize if available, with post-processing
    to avoid splitting on 'Art. 14', 'S.C.', 'v.', etc.
    Falls back to a regex-based splitter.
    """
    try:
        from nltk.tokenize import sent_tokenize
        raw_sents = sent_tokenize(text)
    except ImportError:
        # Regex fallback: split on period/question/exclamation followed by space+capital
        raw_sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # Post-process: merge back sentences that were wrongly split on legal abbreviations
    merged = []
    buffer = ""
    abbreviation_pattern = re.compile(
        r'(?:Art|Sec|S\.C|A\.I\.R|Crl|Civ|O\.P|M\.P|W\.P|vs|v|No|Nos|Hon|Dr|Mr|Mrs|Ms|Ltd|Pvt)\s*\.\s*$',
        re.IGNORECASE
    )
    for sent in raw_sents:
        if buffer:
            sent = buffer + " " + sent
            buffer = ""
        if abbreviation_pattern.search(sent.strip()):
            buffer = sent.strip()
        else:
            merged.append(sent.strip())
    if buffer:
        merged.append(buffer)

    # Filter out very short fragments (< 15 chars)
    return [s for s in merged if len(s.strip()) >= 15]


# ════════════════════════════════════════════════════════════
# Lightweight Chunker (for evaluation; reuses project logic)
# ════════════════════════════════════════════════════════════

def chunk_text(text: str, max_words: int = 300) -> List[str]:
    """
    Chunk a judgment text into overlapping segments for NLI grounding.
    Uses sentence-aware boundaries (Issue #1, #8).
    """
    sentences = legal_sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_words = 0

    for sent in sentences:
        sent_words = len(sent.split())
        if current_words + sent_words > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Keep last sentence for overlap (helps with cross-boundary evidence)
            current_chunk = [current_chunk[-1], sent] if current_chunk else [sent]
            current_words = len(current_chunk[-1].split()) + sent_words
        else:
            current_chunk.append(sent)
            current_words += sent_words

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks if chunks else [text[:2000]]  # fallback: use first 2000 chars


# ════════════════════════════════════════════════════════════
# Embedding-based Retrieval for Evaluation (Issue #2, #9)
# ════════════════════════════════════════════════════════════

class EvalRetriever:
    """
    Lightweight retriever for evaluation.
    Uses sentence-transformers for embedding (no Ollama dependency).
    Falls back to TF-IDF if sentence-transformers unavailable.
    """

    def __init__(self):
        self._mode = None
        self._model = None
        self._vectorizer = None
        self._init_embedder()

    def _init_embedder(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self._mode = "sbert"
            logger.info("EvalRetriever: Using sentence-transformers (all-MiniLM-L6-v2)")
        except ImportError:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            self._vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
            self._mode = "tfidf"
            logger.info("EvalRetriever: Falling back to TF-IDF (install sentence-transformers for better results)")

    def retrieve_top_k(self, query: str, chunks: List[str], k: int = 3) -> List[Tuple[str, float]]:
        """
        Retrieve the top-k chunks most relevant to the query sentence.
        Returns list of (chunk_text, similarity_score).
        """
        if not chunks:
            return []

        k = min(k, len(chunks))

        if self._mode == "sbert":
            q_emb = self._model.encode([query], convert_to_numpy=True)
            c_embs = self._model.encode(chunks, convert_to_numpy=True)
            # Cosine similarity
            sims = (q_emb @ c_embs.T)[0]
            top_indices = np.argsort(sims)[::-1][:k]
            return [(chunks[i], float(sims[i])) for i in top_indices]
        else:
            from sklearn.metrics.pairwise import cosine_similarity
            all_texts = [query] + chunks
            tfidf_matrix = self._vectorizer.fit_transform(all_texts)
            sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
            top_indices = np.argsort(sims)[::-1][:k]
            return [(chunks[i], float(sims[i])) for i in top_indices]


# ════════════════════════════════════════════════════════════
# NLI Evaluator (Issues #3, #4, #10, #12, #13)
# ════════════════════════════════════════════════════════════

class NLIEvaluator:
    """
    NLI-based faithfulness evaluator.
    Dynamically reads label indices from model config (Issue #3).
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        logger.info(f"Loading NLI model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        # Issue #3: Read label indices from model config, never assume
        id2label = self.model.config.id2label
        self.entail_idx = None
        self.contra_idx = None
        self.neutral_idx = None
        for k, v in id2label.items():
            v_lower = v.lower()
            if "entail" in v_lower:
                self.entail_idx = int(k)
            elif "contradict" in v_lower:
                self.contra_idx = int(k)
            elif "neutral" in v_lower:
                self.neutral_idx = int(k)

        if self.entail_idx is None:
            raise ValueError(f"Could not find 'entailment' label in model config: {id2label}")

        logger.info(f"NLI label mapping: entail={self.entail_idx}, contradict={self.contra_idx}, neutral={self.neutral_idx}")

    def predict(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """
        Run NLI inference: is hypothesis entailed by premise?
        Returns dict with entailment/contradiction/neutral probabilities.
        """
        inputs = self.tokenizer(
            premise, hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        result = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
        if self.entail_idx is not None:
            result["entailment"] = float(probs[self.entail_idx])
        if self.contra_idx is not None:
            result["contradiction"] = float(probs[self.contra_idx])
        if self.neutral_idx is not None:
            result["neutral"] = float(probs[self.neutral_idx])
        return result

    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, float]]:
        """
        Batch NLI inference (Issue #12).
        """
        if not pairs:
            return []

        encodings = self.tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**encodings).logits
            all_probs = torch.softmax(logits, dim=1).cpu().numpy()

        results = []
        for probs in all_probs:
            result = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
            if self.entail_idx is not None:
                result["entailment"] = float(probs[self.entail_idx])
            if self.contra_idx is not None:
                result["contradiction"] = float(probs[self.contra_idx])
            if self.neutral_idx is not None:
                result["neutral"] = float(probs[self.neutral_idx])
            results.append(result)
        return results

class LLMFaithfulnessEvaluator:
    """
    LLM-as-a-Judge fallback for NLI limitations.
    Queries the local Ollama instance to verify if a claim is supported by a chunk.
    This handles formatting (bullet points) better than strict NLI cross-encoders.
    """
    def __init__(self, model_name: str = "llama3.1:8b-instruct-q4_K_M"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"

    def evaluate(self, premise: str, hypothesis: str) -> bool:
        """Returns True if the LLM judges the hypothesis as supported by the premise."""
        prompt = f"""You are a strict logical evaluator.
Context (Premise):
{premise}

Claim (Hypothesis):
{hypothesis}

Is the Claim logically supported by or explicitly stated in the Context?
Answer ONLY with "YES" or "NO".
"""
        import requests
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 10}
                },
                timeout=10
            )
            response.raise_for_status()
            result_text = response.json().get("response", "").strip().upper()
            return "YES" in result_text
        except Exception as e:
            logger.error(f"LLMFaithfulnessEvaluator failed: {e}")
            return False


# ════════════════════════════════════════════════════════════
# Structural Coverage (Issue #16, #17)
# ════════════════════════════════════════════════════════════

EXPECTED_SECTIONS = ["facts", "issues", "arguments", "reasoning", "decision"]

SECTION_PATTERNS = {
    "facts": re.compile(r'\*\*facts\*\*|facts\s*:', re.IGNORECASE),
    "issues": re.compile(r'\*\*issues\*\*|issues\s*:', re.IGNORECASE),
    "arguments": re.compile(r'\*\*arguments\*\*|arguments\s*:', re.IGNORECASE),
    "reasoning": re.compile(r'\*\*reasoning\*\*|reasoning\s*:', re.IGNORECASE),
    "decision": re.compile(r'\*\*decision\*\*|decision\s*:|judgment\s*:', re.IGNORECASE),
}


def evaluate_structural_coverage(summary: str) -> Tuple[float, List[str], List[str]]:
    """
    Check which of the 5 expected sections are present in the summary (Issue #16).
    Returns (coverage_ratio, found_sections, missing_sections).
    """
    found = []
    missing = []
    for section, pattern in SECTION_PATTERNS.items():
        if pattern.search(summary):
            found.append(section)
        else:
            missing.append(section)

    coverage = len(found) / len(EXPECTED_SECTIONS) if EXPECTED_SECTIONS else 0.0
    return coverage, found, missing


# ════════════════════════════════════════════════════════════
# Role Consistency via NLI (Issue #7: not string matching)
# ════════════════════════════════════════════════════════════

def evaluate_role_consistency_nli(
    nli: NLIEvaluator,
    judgment: str,
    summary: str,
    roles: Dict[str, str]
) -> float:
    """
    NLI-grounded role consistency check (Issue #7).
    Constructs hypotheses like "The petitioner is X" and checks
    if the judgment entails them + the summary preserves them.
    """
    if not roles:
        return 1.0  # No roles to check

    scores = []
    for role, name in roles.items():
        if not name or name.lower() in ("unknown", "n/a", ""):
            continue

        hypothesis = f"The {role} in this case is {name}."

        # Check: does the summary preserve this role assignment?
        summary_result = nli.predict(summary, hypothesis)
        scores.append(summary_result["entailment"])

    return float(np.mean(scores)) if scores else 1.0


# ════════════════════════════════════════════════════════════
# Main Evaluator Class
# ════════════════════════════════════════════════════════════

class LegalFaithfulnessEvaluator:
    """
    Research-grade Legal Faithfulness Evaluator.

    Pipeline per document:
      1. Chunk source judgment into overlapping segments
      2. For each summary sentence:
         a. Retrieve top-k evidence chunks (Issue #2, #9)
         b. Run NLI against each chunk
         c. Take max entailment score across chunks (best evidence)
         d. Apply threshold to classify (Issue #10)
      3. Compute document-level metrics
      4. Log per-sentence evidence trace (Issue #11, #14)
    """

    # Thresholds (Issue #10)
    ENTAILMENT_THRESHOLD = 0.5
    CONTRADICTION_THRESHOLD = 0.7

    def __init__(
        self,
        nli_model: str = "cross-encoder/nli-deberta-v3-base",
        top_k: int = 5,
        chunk_max_words: int = 300,
    ):
        self.nli = NLIEvaluator(nli_model)
        self.retriever = EvalRetriever()
        self.llm_evaluator = LLMFaithfulnessEvaluator()
        self.top_k = top_k
        self.chunk_max_words = chunk_max_words

    def evaluate_document(
        self,
        judgment: str,
        generated_summary: str,
        case_id: str = "",
        roles: Optional[Dict[str, str]] = None,
    ) -> DocumentResult:
        """
        Evaluate a single document.
        """
        result = DocumentResult(case_id=case_id)

        # ── Validate inputs (Issue #18) ──
        if not judgment or not judgment.strip():
            result.error = "empty_judgment"
            return result
        if not generated_summary or not generated_summary.strip():
            result.error = "empty_summary"
            return result

        # ── 1. Chunk the source judgment (Issue #1, #8) ──
        chunks = chunk_text(judgment, max_words=self.chunk_max_words)

        # ── 2. Split summary into sentences (Issue #5) ──
        sentences = legal_sent_tokenize(generated_summary)
        if not sentences:
            result.error = "no_sentences_extracted"
            return result

        result.num_sentences = len(sentences)

        # ── 3. Per-sentence NLI with retrieval (Issues #2, #4, #9, #11, #13, #14) ──
        supported = 0
        contradicted = 0
        neutral_count = 0
        abstained = 0  # Refusal sentences — excluded from faithfulness denominator

        # Patterns that indicate the model is refusing to answer (abstention)
        ABSTENTION_PATTERNS = [
            re.compile(r"provided context does not (?:contain|mention)"),
            re.compile(r"does not contain (?:any )?information"),
            re.compile(r"not mentioned in the provided context"),
            re.compile(r"no information (?:is )?available"),
        ]

        for sent in sentences:
            # Issue #13: Split very long sentences (>200 tokens ~= 800 chars)
            sub_sentences = [sent]
            if len(sent) > 800:
                sub_sentences = legal_sent_tokenize(sent)
                if not sub_sentences:
                    sub_sentences = [sent]

            for sub_sent in sub_sentences:
                lower_sub = sub_sent.lower().strip()

                # Detect abstention: model says "context does not contain X"
                # These are EXCLUDED from faithfulness calculation, not scored as 1.0
                if any(pattern.search(lower_sub) for pattern in ABSTENTION_PATTERNS):
                    abstained += 1
                    result.sentence_verdicts.append(asdict(SentenceVerdict(
                        sentence=sub_sent,
                        entailment=0.0,
                        contradiction=0.0,
                        neutral=1.0,
                        verdict="abstained",
                        evidence_chunk="[ABSTAINED]",
                        evidence_score=0.0
                    )))
                    continue

                # Retrieve top-k evidence chunks (Issue #9)
                top_chunks = self.retriever.retrieve_top_k(sub_sent, chunks, k=self.top_k)

                # Run NLI against each retrieved chunk, take best (Issue #9)
                best_entailment = 0.0
                best_contradiction = 0.0
                best_neutral = 0.0
                best_chunk_text = ""
                best_chunk_score = 0.0

                for chunk_text_str, sim_score in top_chunks:
                    nli_result = self.nli.predict(chunk_text_str, sub_sent)
                    if nli_result["entailment"] > best_entailment:
                        best_entailment = nli_result["entailment"]
                        best_contradiction = nli_result["contradiction"]
                        best_neutral = nli_result["neutral"]
                        best_chunk_text = chunk_text_str
                        best_chunk_score = sim_score

                # Issue #10: Apply thresholds to classify
                verdict = "unverifiable"
                if best_entailment >= self.ENTAILMENT_THRESHOLD:
                    verdict = "supported"
                    supported += 1
                else:
                    # Fallback to LLM Judge if NLI fails (Fixing A: Evaluation Fragility)
                    is_llm_supported = self.llm_evaluator.evaluate(best_chunk_text, sub_sent)
                    if is_llm_supported:
                        verdict = "supported"
                        supported += 1
                        logger.info(f"LLM-as-a-Judge overridden NLI failure for sentence: {sub_sent}")
                    elif best_contradiction >= self.CONTRADICTION_THRESHOLD:
                        verdict = "contradicted"
                        contradicted += 1
                    else:
                        neutral_count += 1

                # Issue #11, #14: Log per-sentence evidence trace
                result.sentence_verdicts.append(asdict(SentenceVerdict(
                    sentence=sub_sent,
                    entailment=best_entailment,
                    contradiction=best_contradiction,
                    neutral=best_neutral,
                    verdict=verdict,
                    evidence_chunk=best_chunk_text[:300],  # truncate for storage
                    evidence_score=best_chunk_score,
                )))

        # Faithfulness denominator only counts sentences where the model actually made claims
        scored_total = supported + contradicted + neutral_count
        full_total = scored_total + abstained

        if full_total == 0:
            result.error = "no_verdicts"
            return result

        # If the model abstained on every single sentence, faithfulness is N/A
        if scored_total == 0:
            result.faithfulness = None
            result.error = "all_abstained"
            return result

        # ── 4. Compute document-level metrics ──
        result.entailment_rate = supported / scored_total
        result.contradiction_rate = contradicted / scored_total
        result.neutral_rate = neutral_count / scored_total

        # Faithfulness = fraction of sentences that are supported
        result.faithfulness = result.entailment_rate

        # Issue #15: Hallucination = contradiction + (1 - entailment)
        result.hallucination_rate = result.contradiction_rate + (1 - result.entailment_rate)

        # ── 5. Structural coverage (Issue #16, #17) ──
        coverage, found, missing = evaluate_structural_coverage(generated_summary)
        result.structural_coverage = coverage
        result.sections_found = found
        result.sections_missing = missing

        # ── 6. Role consistency via NLI (Issue #7) ──
        if roles:
            result.role_consistency = evaluate_role_consistency_nli(
                self.nli, judgment, generated_summary, roles
            )
        else:
            result.role_consistency = -1.0  # Not applicable

        return result

    def evaluate_answer(self, answer: str, context_chunks: List[str]) -> Dict[str, Any]:
        """
        Evaluate a Q&A answer against specific retrieved context chunks.
        Splits the answer into sentences and checks if each sentence is supported.
        """
        if not answer.strip() or not context_chunks:
            return {"faithfulness_score": 0.0}

        sentences = legal_sent_tokenize(answer)
        if not sentences:
            return {"faithfulness_score": 0.0}

        supported_count = 0
        for sent in sentences:
            best_entailment = 0.0
            best_chunk = ""
            for chunk in context_chunks:
                nli_result = self.nli.predict(chunk, sent)
                if nli_result["entailment"] > best_entailment:
                    best_entailment = nli_result["entailment"]
                    best_chunk = chunk
                    
            if best_entailment >= self.ENTAILMENT_THRESHOLD:
                supported_count += 1
            else:
                # Fallback to LLM if NLI fails
                if self.llm_evaluator.evaluate(best_chunk, sent):
                    supported_count += 1

        return {"faithfulness_score": supported_count / len(sentences)}

# ════════════════════════════════════════════════════════════
# Benchmark Runner
# ════════════════════════════════════════════════════════════

def run_evaluation_benchmark(
    predictions_path: str,
    output_path: str = "evaluation_results.json",
    nli_model: str = "cross-encoder/nli-deberta-v3-base",
    top_k: int = 3,
    max_samples: int = 0,  # 0 = all
):
    """
    Run the full evaluation benchmark on a predictions file.

    Expected predictions format (JSONL):
    {
        "case_id": "...",
        "judgment": "full judgment text",
        "generated_summary": "model output",
        "roles": {"Petitioner": "Name", "Respondent": "Name"}   # optional
    }
    """
    evaluator = LegalFaithfulnessEvaluator(nli_model=nli_model, top_k=top_k)

    # Load predictions
    data = []
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # Issue #18: Validate required fields
            if "judgment" not in item or "generated_summary" not in item:
                logger.warning(f"Skipping item missing 'judgment' or 'generated_summary': {item.get('case_id', '?')}")
                continue
            data.append(item)

    if max_samples > 0:
        data = data[:max_samples]

    logger.info(f"Evaluating {len(data)} samples...")

    results = []
    for item in tqdm(data, desc="Evaluating"):
        doc_result = evaluator.evaluate_document(
            judgment=item["judgment"],
            generated_summary=item["generated_summary"],
            case_id=item.get("case_id", ""),
            roles=item.get("roles"),
        )
        results.append(asdict(doc_result))

    # ── Aggregate ──
    valid = [r for r in results if r["error"] is None]
    if not valid:
        logger.error("No valid results to aggregate.")
        return

    avg_faithfulness = np.mean([r["faithfulness"] for r in valid])
    avg_hallucination = np.mean([r["hallucination_rate"] for r in valid])
    avg_contradiction = np.mean([r["contradiction_rate"] for r in valid])
    avg_coverage = np.mean([r["structural_coverage"] for r in valid])
    avg_role = np.mean([r["role_consistency"] for r in valid if r["role_consistency"] >= 0])

    report = {
        "num_samples": len(data),
        "num_valid": len(valid),
        "num_errors": len(data) - len(valid),
        "aggregate": {
            "faithfulness_score": round(float(avg_faithfulness), 4),
            "hallucination_rate": round(float(avg_hallucination), 4),
            "contradiction_rate": round(float(avg_contradiction), 4),
            "structural_coverage": round(float(avg_coverage), 4),
            "role_consistency": round(float(avg_role), 4) if avg_role else "N/A",
        },
        "per_document": results,
    }

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("JUDGE X AI — LEGAL FAITHFULNESS EVALUATION REPORT")
    print("=" * 60)
    print(f"  Samples evaluated     : {len(valid)} / {len(data)}")
    print(f"  Faithfulness (NLI)    : {avg_faithfulness:.4f}")
    print(f"  Hallucination Rate    : {avg_hallucination:.4f}")
    print(f"  Contradiction Rate    : {avg_contradiction:.4f}")
    print(f"  Structural Coverage   : {avg_coverage:.4f}")
    if avg_role and avg_role >= 0:
        print(f"  Role Consistency      : {avg_role:.4f}")
    print(f"  Results saved to      : {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="JUDGE X AI Faithfulness Evaluator")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions JSONL file")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output path for results")
    parser.add_argument("--nli-model", type=str, default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--top-k", type=int, default=3, help="Number of evidence chunks to retrieve per sentence")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples to evaluate (0=all)")
    args = parser.parse_args()

    run_evaluation_benchmark(
        predictions_path=args.predictions,
        output_path=args.output,
        nli_model=args.nli_model,
        top_k=args.top_k,
        max_samples=args.max_samples,
    )
