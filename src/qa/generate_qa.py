#!/usr/bin/env python3
"""
generate_qa.py – Generate RAG-ready Question-Answer pairs from labeled legal paragraphs.

Logic:
1. Load role-labeled judgments.
2. Filter for high-value rhetorical roles (reasoning, statutory, final_decision).
3. Cluster paragraphs into context windows.
4. Use LLM to generate complex legal Q&A pairs with paragraph citations.
"""

import json
import time
import re
import logging
from pathlib import Path
from datetime import datetime
import ollama
from tqdm import tqdm

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────

INPUT = Path(r"C:\final_year\JUDGEXAI\data\processed\role_labelled judgements.jsonl")
OUTPUT = Path(r"C:\final_year\JUDGEXAI\data\processed\legal_qa_dataset.jsonl")
LOG_FILE = Path(f"logs/qa_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

MODEL = "llama3.1:8b-instruct-q4_K_M"
OLLAMA_OPTIONS = {
    "temperature": 0.1,
    "top_p": 1.0,
    "num_predict": 1024, # More tokens for QA pairs
    "num_ctx": 12288,    # Larger context for multiple paragraphs
    "num_gpu": -1,
    "num_thread": 6,
}

# Rhetorical roles that are "Askable" (contain knowledge)
TARGET_ROLES = {"reasoning", "statutory", "final_decision", "issues"}

# ────────────────────────────────────────────────
# PROMPT
# ────────────────────────────────────────────────

QA_PROMPT_TEMPLATE = """
You are a Senior Legal Advocate and Researcher specializing in Indian Law.
Your task is to generate high-quality, professional Question-Answer (QA) pairs based ONLY on the provided extracts from a court judgment.

CONTEXT (Paragraphs from Judgment):
{context_text}

STRICT GUIDELINES:
1. Generate exactly {num_pairs} QA pair(s).
2. The Questions must be complex and professional (e.g., "What was the Court's rationale for holding Section 4(4) ultra vires?").
3. The Answers must be precise, authoritative, and include citations to the paragraph numbers.
4. You MUST reference the specific Paragraph IDs in the "reference_paras" field.
5. Do not use generic phrases like "The judgment says...". Use legal phrasing.

OUTPUT FORMAT (Strict JSON):
{{
  "qa_pairs": [
    {{
      "question": "The question text...",
      "answer": "The detailed answer with citations...",
      "reference_paras": [id1, id2],
      "rhetorical_roles": ["role1", "role2"]
    }}
  ]
}}

No preamble. No markdown. Return ONLY the JSON.
"""

# ────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────

def setup_logging():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def extract_json(text: str) -> str:
    """Robustly extract JSON block from LLM noise."""
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        return text[start:end+1]
    return text

def clean_json_response(raw_text: str):
    """Attempt to parse JSON, with fallback logic."""
    try:
        clean = extract_json(raw_text)
        return json.loads(clean)
    except Exception as e:
        logging.warning(f"JSON Parse failed: {e}")
        # Try a regex-based 'fix' for common LLM JSON errors (like unescaped quotes)
        try:
            # Basic cleanup: remove problematic control chars
            clean = re.sub(r'[\x00-\x1F\x7F]', '', clean)
            return json.loads(clean)
        except:
            return None

# ────────────────────────────────────────────────
# CORE LOGIC
# ────────────────────────────────────────────────

def process_case(case_data: dict):
    case_id = case_data.get("case_id") or "unknown"
    paragraphs = case_data.get("paragraphs", [])
    
    # 1. Filter for knowledge paragraphs
    knowledge_paras = []
    for p in paragraphs:
        roles = set(p.get("paragraph_roles", []))
        if roles & TARGET_ROLES:
            knowledge_paras.append({
                "id": p["para_id"],
                "text": p["text"],
                "roles": list(roles)
            })
    
    if not knowledge_paras:
        return None

    # 2. Build context string
    context_items = []
    for p in knowledge_paras:
        context_items.append(f"Para ID {p['id']} [{', '.join(p['roles'])}]: {p['text']}")
    
    context_text = "\n\n".join(context_items)
    
    # 3. Request QA from LLM
    num_to_gen = 2 if len(knowledge_paras) > 10 else 1
    prompt = QA_PROMPT_TEMPLATE.format(context_text=context_text, num_pairs=num_to_gen)
    
    try:
        response = ollama.generate(model=MODEL, prompt=prompt, options=OLLAMA_OPTIONS)
        raw_output = response["response"].strip()
        
        parsed = clean_json_response(raw_output)
        if parsed and "qa_pairs" in parsed:
            return {
                "case_id": case_id,
                "judgment_title": case_data.get("case_title", "Unknown"),
                "qa_pairs": parsed["qa_pairs"]
            }
    except Exception as e:
        logging.error(f"Error generating QA for {case_id}: {e}")
    
    return None

def main():
    setup_logging()
    logging.info(f"Starting Legal QA Generation using {MODEL}")

    if not INPUT.exists():
        logging.error(f"Input file not found: {INPUT}")
        return

    # Checkpoint logic
    processed_ids = set()
    if OUTPUT.exists():
        with open(OUTPUT, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    processed_ids.add(d["case_id"])
                except: continue
        logging.info(f"Resuming: {len(processed_ids)} cases already processed.")

    # Read input
    with open(INPUT, "r", encoding="utf-8") as fin:
        all_cases = [json.loads(line) for line in fin]
    
    to_process = [c for c in all_cases if (c.get("case_id") or "unknown") not in processed_ids]
    logging.info(f"Queue: {len(to_process)} judgments to analyze.")

    with open(OUTPUT, "a", encoding="utf-8") as fout:
        for case in tqdm(to_process, desc="Generating QA Pairs"):
            result = process_case(case)
            if result:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
            
            # Rate limiting / Safety sleep
            time.sleep(0.2)

    logging.info("QA Generation complete. Output saved to data/processed/legal_qa_dataset.jsonl")

if __name__ == "__main__":
    main()
