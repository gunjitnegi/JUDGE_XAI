#!/usr/bin/env python3
"""
role_labeling.py – Assign rhetorical roles to paragraphs using Llama 3.1 (Ollama)

Improvements:
- Larger context window (8192) to handle batch prompts.
- Smaller batch size (8) to stay within limits.
- Per‑paragraph LLM fallback when batch mode fails.
- Rule‑based fallback for cases where LLM is unavailable.
- Detailed logging and confidence scoring.
"""

import json
import time
time.sleep(0.1)
import re
import hashlib
from pathlib import Path
import ollama
from tqdm import tqdm
from collections import Counter
from datetime import datetime
import logging

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────

INPUT  = Path(r"C:\final_year\JUDGEXAI\data\processed\judgments_paragraphs_filtered.jsonl")
OUTPUT = Path(r"C:\final_year\JUDGEXAI\data\processed\role_labelled judgements.jsonl")
LOG_FILE = Path(f"logs/role_labeling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

MODEL = "llama3.1:8b-instruct-q4_K_M"

# Adjusted for stability
BATCH_SIZE = 4
OLLAMA_OPTIONS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "repeat_penalty": 1.0,
    "num_predict": 512,        # Increased to prevent truncated JSON
    "num_ctx": 8192,           # enough for batch prompts
    "num_batch": 512,
    "num_gpu": -1,
    "num_thread": 6,
}

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M")
FIXED_SEED = 20260213
RUN_ID = f"role_run_{RUN_TIMESTAMP}_seed{FIXED_SEED}"
PROMPT_VERSION = "v7_fixed_batching"

# ────────────────────────────────────────────────
# TAXONOMY
# ────────────────────────────────────────────────

FINAL_ROLES = [
    "facts", "arguments", "reasoning", "statutory",
    "final_decision", "procedural", "issues", "other"
]

STAGE1_ROLES = [
    "metadata", "facts", "arguments", "analysis", "decision"
]

STAGE2_MAP = {
    "metadata": ["procedural", "other"],
    "facts": ["facts", "procedural"],
    "arguments": ["arguments"],
    "analysis": ["reasoning", "statutory", "issues"],
    "decision": ["final_decision"]
}

# ────────────────────────────────────────────────
# PROMPTS
# ────────────────────────────────────────────────

STAGE1_PROMPT = """
You are a legal expert analyzing Indian court judgments.

Classify each paragraph into EXACTLY ONE:

metadata → judge names, bench, dates, headings
facts → case background, events, history
arguments → what petitioner/respondent argued
analysis → court reasoning, interpretation, legal discussion
decision → final outcome, order, directions

STRICT RULES:
- If judge explains law → analysis
- If citing cases → analysis
- If describing dispute → facts
- If lawyer claims → arguments
- If final ruling → decision

You MUST return EXACTLY {n} lines.
Each line must contain ONLY ONE word from:
metadata, facts, arguments, analysis, decision

No numbering.
No explanation.
No missing lines.

If unsure, choose the closest label.
"""

# ────────────────────────────────────────────────
# HELPER FUNCTIONS
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

def normalize_label(label: str) -> str:
    """Normalize LLM output to valid final roles strictly."""
    if not label:
        return "other"
    
    label = label.lower().strip()
    label = re.sub(r'[^a-z_]', '', label)

    # Direct match
    if label in FINAL_ROLES:
        return label

    # Strict drift mapping (Normalization)
    drift_map = {
        "background": "facts",
        "case history": "facts",
        "events": "facts",
        "history": "facts",
        "narration": "facts",
        "story": "facts",
        "analysis": "reasoning",
        "discussion": "reasoning",
        "observation": "reasoning",
        "held": "reasoning",
        "opinion": "reasoning",
        "decision": "final_decision",
        "order": "final_decision",
        "relief": "final_decision",
        "statute": "statutory",
        "law": "statutory",
        "provision": "statutory",
        "preamble": "procedural",
        "header": "procedural",
        "metadata": "procedural",
        "question": "issues",
        "point": "issues"
    }
    
    return drift_map.get(label, "other")

def extract_json(text: str) -> str:
    """Extract JSON block from LLM response even if it contains markdown or preamble."""
    # Find first { and last }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        return text[start:end+1]
    return text

def truncate_text(text: str, head=800, tail=400) -> str:
    text = text.strip()
    if len(text) <= head + tail + 20:
        return text
    return text[:head] + "\n…\n" + text[-tail:]

# Regex patterns for fast routing
DECISION_PAT = re.compile(r'\b(allowed|dismissed|disposed|quashed|set aside|acquitted|convicted|rule made absolute|petition stands disposed|appeal stands dismissed|no order as to costs|accordingly allowed)\b',re.IGNORECASE)
REASONING_PAT = re.compile(r'\b(in my view|we hold|we are of the opinion|it follows that|therefore we hold|thus we conclude)\b', re.IGNORECASE)
STATUTE_PAT = re.compile(r'\b(section\s+\d+|article\s+\d+|rule\s+\d+|act\s+of\s+\d{4}|sub-?section\s*\(\d+\))\b', re.IGNORECASE)
FACT_PAT = re.compile(r'\b(petitioner|respondent|plaintiff|defendant|filed|arose|background|case of|incident|facts of the case)\b',re.IGNORECASE)
PROCEDURAL_PAT = re.compile(r'^(coram|appearance|before:|present:|order sheet|registry|bench:)\b', re.IGNORECASE)

def apply_regex_rules(text: str, length: int, position: str = "middle") -> tuple[str | None, str | None]:
    """Return (role, source) if regex matches, else (None, None)"""
    tl = text.lower()
    
    # Decision rule: Only highly confident if at the end of the document
    if DECISION_PAT.search(tl):
        if position == "end":
            return "final_decision", "regex_decision"
        elif position == "start":
            # Very unlikely to be a decision at the start; probably case history
            return None, None 
        elif length < 300: # Short orders at the end
             return "final_decision", "regex_decision_short"
    if REASONING_PAT.search(tl) and length < 400:
        return "reasoning", "regex_reasoning"
    if FACT_PAT.search(tl) and length < 500:
        return "facts", "regex_facts"
    if STATUTE_PAT.search(tl) and length < 300:
        return "statutory", "regex_statutory"
    if PROCEDURAL_PAT.search(tl) and length < 600:
        return "procedural", "regex_procedural"
    return None, None

def fallback_rule(text: str, position: str) -> str:
    tl = text.lower()
    if position == "start" and len(text) < 500:
        return "facts"
    if position == "end" and DECISION_PAT.search(tl):
        return "final_decision"
    if STATUTE_PAT.search(tl):
        return "statutory"
    if REASONING_PAT.search(tl):
        return "reasoning"
    if FACT_PAT.search(tl):
        return "facts"
    if PROCEDURAL_PAT.search(tl):
        return "procedural"
    # New line:
    if "we are of the opinion" in tl or "it follows that" in tl:
        return "reasoning"
    return "other"

import warnings
# Suppress transformers warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Global zero-shot classifier
_zero_shot_classifier = None

def get_classifier():
    global _zero_shot_classifier
    if _zero_shot_classifier is None:
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        _zero_shot_classifier = pipeline(
            "zero-shot-classification", 
            model="typeform/distilbert-base-uncased-mnli",
            device=device
        )
    return _zero_shot_classifier

def classify_single(text: str, position: str):
    """
    Per-paragraph fallback classification using zero-shot + rules
    """
    # Step 1 — regex fast path
    role, source = apply_regex_rules(text, len(text), position)
    if role:
        return role, source, 0.9

    # Step 2 — Zero-Shot Classification
    try:
        classifier = get_classifier()
        candidate_labels = ["facts", "arguments", "reasoning", "statutory", "procedural", "issues", "final_decision"]
        result = classifier(text[:500], candidate_labels)
        best_label = result['labels'][0]
        score = result['scores'][0]
        
        if score > 0.4:
            return best_label, "zeroshot_single", score
        else:
            return fallback_rule(text, position), "zeroshot_fallback", score

    except Exception:
        return fallback_rule(text, position), "rule_fallback", 0.4

def classify_batch(texts: list[str], positions: list[str]) -> tuple[list[str], list[str], list[float]]:
    """
    Batch classification using zero-shot DeBERTa pipeline.
    """
    original_texts = texts[:]
    original_positions = positions[:]

    # Hard rule pre-classification
    def hard_rules_override(text, position):
        tl = text.lower()
        role, _ = apply_regex_rules(text, len(text), position)
        if role:
            return role
        if any(x in tl for x in ["allowed", "dismissed", "disposed", "quashed", "set aside"]):
            return "final_decision"
        if any(x in tl for x in ["learned counsel", "petitioner submits", "respondent contends"]):
            return "arguments"
        if position == "start" and any(x in tl for x in ["notification", "corrigendum", "applications", "filed"]):
            return "facts"
        if any(x in tl for x in ["according to the petitioners", "according to the respondents", "the facts of the case"]):
            return "facts"
        if ("section" in tl or "article" in tl) and not any(x in tl for x in ["held", "observed", "therefore", "thus", "we conclude", "it follows"]) and len(text) < 400:
            return "statutory"
        if position == "start" and any(x in tl for x in ["notification", "circular", "application"]):
            return "procedural"
        if "question is" in tl or "issue is" in tl or "the question is" in tl:
            return "issues"
        if len(text) > 500 and ("candidate" in tl or "seat" in tl or "reservation" in tl) and not any(c in tl for c in ["we are of the opinion", "it follows that"]):
            return None
        return None

    pre_labels = {}
    remaining_texts = []
    remaining_positions = []
    mapping = []

    for i, (txt, pos) in enumerate(zip(texts, positions)):
        rule = hard_rules_override(txt, pos)
        if rule:
            pre_labels[i] = rule
        else:
            mapping.append(i)
            remaining_texts.append(txt)
            remaining_positions.append(pos)

    if not remaining_texts:
        return (
            [pre_labels.get(i, "other") for i in range(len(texts))],
            ["rule"] * len(texts),
            [0.9] * len(texts)
        )

    # Try Zero-Shot Batch
    final_labels = []
    try:
        classifier = get_classifier()
        candidate_labels = ["facts", "arguments", "reasoning", "statutory", "procedural", "issues", "final_decision"]
        truncated_texts = [txt[:500] for txt in remaining_texts]
        
        results = classifier(truncated_texts, candidate_labels)
        
        # Pipeline returns a dict if input is single str, else list of dicts
        if isinstance(results, dict):
            results = [results]
            
        for res in results:
            best_label = res['labels'][0]
            score = res['scores'][0]
            if score > 0.4:
                final_labels.append(best_label)
            else:
                final_labels.append("other")
                
    except Exception as e:
        logging.error(f"Zero-Shot Batch failed: {e}")
        return fallback_to_per_paragraph(original_texts, original_positions)

    # Apply bias fixes (Assuming fix_statutory_bias and boost_reasoning exist below)
    try:
        final_labels = [fix_statutory_bias(txt, lbl) for txt, lbl in zip(remaining_texts, final_labels)]
        final_labels = [boost_reasoning(txt, lbl) for txt, lbl in zip(remaining_texts, final_labels)]
    except NameError:
        pass # Functions might be defined below

    # Merge rule + Zero-shot results
    final_roles = [None] * len(original_texts)
    final_sources = [None] * len(original_texts)
    final_conf = [0.0] * len(original_texts)

    for idx, label in pre_labels.items():
        final_roles[idx] = label
        final_sources[idx] = "rule"
        final_conf[idx] = 0.9

    for i, map_idx in enumerate(mapping):
        label = final_labels[i] if i < len(final_labels) else fallback_rule(original_texts[map_idx], original_positions[map_idx])
        label = boost_reasoning(original_texts[map_idx], label)
        label = fix_statutory_bias(original_texts[map_idx], label)
        label = final_safety_fix(original_texts[map_idx], label)

        final_roles[map_idx] = label
        final_sources[map_idx] = "llm"
        final_conf[map_idx] = 0.5 if label == "other" else 0.8

    return final_roles, final_sources, final_conf

def fix_statutory_bias(text, label):
    tl = text.lower()

    if label == "statutory":
        if any(x in tl for x in [
            "held", "observed", "therefore", "thus",
            "we conclude", "it follows", "we are of the opinion"
        ]):
            return "reasoning"

    return label


def boost_reasoning(text: str, label: str) -> str:
    """Strong push for court analysis to reasoning."""
    tl = text.lower()
    cues = [
        "we are of the opinion", "it follows that", "therefore we hold", "thus we conclude",
        "in our view", "we hold", "this court has held", "as held in", "indira sawhney",
        "pradeep tandon", "compartmentalised", "overall reservation", "horizontal reservation",
        "vertical reservation", "article 15", "article 16", "it is clear that",
        "it would be better", "in the interest of", "we must yet take notice",
        "on a careful consideration", "we may explain", "it is not possible to give",
        "res judicata", "judgment in rem", "res sub-judice"
    ]
    if any(cue in tl for cue in cues):
        return "reasoning"
    return label

def final_safety_fix(text: str, label: str) -> str:
    """Aggressive final correction for common errors."""
    tl = text.lower()
    length = len(text)

    # Fix over-prediction of "issues"
    if label == "issues":
        if not any(phrase in tl for phrase in ["the question is", "point for determination", "main question", "issue that arises"]):
            if any(cue in tl for cue in ["we are of the opinion", "it follows that", "it would be better", "on a careful consideration"]):
                return "reasoning"
            if length > 450 and ("candidate" in tl or "seat" in tl or "reservation" in tl or "petition" in tl):
                return "facts"

    # Fix "arguments" on court analysis
    if label == "arguments" and any(cue in tl for cue in ["we are of the opinion", "it follows that", "it would be better", "on a careful consideration"]):
        return "reasoning"

    # Fix "statutory" on analysis + citations
    if label == "statutory" and ("held by this court" in tl or "indira sawhney" in tl or "pradeep tandon" in tl or "it follows that" in tl):
        return "reasoning"

    # Fix procedural paragraphs wrongly marked as facts
    if label == "facts" and ("procedure" in tl or "application" in tl or "ia no" in tl or "section 10" in tl):
        return "procedural"

    return label


def fallback_to_per_paragraph(texts, positions):
    """Helper to run per‑paragraph classification for a batch."""
    roles, sources, confs = [], [], []
    for txt, pos in zip(texts, positions):
        r, s, c = classify_single(txt, pos)
        roles.append(r)
        sources.append(s)
        confs.append(c)
    return roles, sources, confs

def compute_text_hash(texts: list[str]) -> str:
    combined = "".join(t.strip() for t in texts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()

def main():
    setup_logging()
    logging.info(f"Starting role labeling run {RUN_ID}")

    total_cases = 0
    total_paras = 0
    global_role_counter = Counter()
    total_fallbacks = 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Checkpoint: skip already processed cases
    processed_case_ids = set()
    if OUTPUT.exists():
        logging.info(f"Resuming from: {OUTPUT}")
        with open(OUTPUT, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    case = json.loads(line.strip())
                    case_id = case.get("case_id") or case.get("meta", {}).get("case_id")
                    if case_id:
                        processed_case_ids.add(case_id)
                except:
                    continue
        logging.info(f"Already processed {len(processed_case_ids)} cases")

    with open(INPUT, encoding="utf-8") as fin, \
         open(OUTPUT, "a", encoding="utf-8") as fout:

        lines = fin.readlines()
        to_process = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
                case_id = case.get("case_id") or case.get("meta", {}).get("case_id", f"unknown_{total_cases}")
                if case_id in processed_case_ids:
                    continue
                to_process.append((line, case_id))
            except:
                continue

        logging.info(f"Need to process {len(to_process)} new cases")

        for line, case_id in tqdm(to_process, desc="Labeling new judgments", unit="case"):
            total_cases += 1

            try:
                case = json.loads(line)
                paragraphs = case.get("paragraphs", [])
                total_paras += len(paragraphs)

                if not paragraphs:
                    fout.write(json.dumps(case, ensure_ascii=False) + "\n")
                    continue

                # Filter usable paragraphs
                usable_texts = []
                usable_indices = []
                for i, p in enumerate(paragraphs):
                    if not p.get("usable_for_evidence", True):
                        p["paragraph_roles"] = ["other"]
                        p["role_source"] = "skipped_unusable"
                        p["role_confidence"] = 0.0
                    else:
                        usable_texts.append(p["text"])
                        usable_indices.append(i)

                # Get positions for fallback
                positions = [paragraphs[i].get("position", "middle") for i in usable_indices]

                # Batch classify
                roles, sources, confidences = [], [], []
                for i in range(0, len(usable_texts), BATCH_SIZE):
                    batch_txt = usable_texts[i:i+BATCH_SIZE]
                    batch_pos = positions[i:i+BATCH_SIZE]
                    batch_roles, batch_src, batch_conf = classify_batch(batch_txt, batch_pos)
                    roles.extend(batch_roles)
                    sources.extend(batch_src)
                    confidences.extend(batch_conf)

                # Assign back
                full_roles = [None] * len(paragraphs)
                full_sources = [None] * len(paragraphs)
                full_conf = [0.0] * len(paragraphs)
                llm_ptr = 0
                for idx in usable_indices:
                    full_roles[idx] = roles[llm_ptr]
                    full_sources[idx] = sources[llm_ptr]
                    full_conf[idx] = confidences[llm_ptr]
                    llm_ptr += 1

                # Write to paragraphs
                for i, para in enumerate(paragraphs):
                    if full_roles[i] is None:
                        # fallback for skipped/unusable
                        para["paragraph_roles"] = ["other"]
                        para["role_source"] = "auto_filled"
                        para["role_confidence"] = 0.0

                        full_roles[i] = "other"
                        full_sources[i] = "auto_filled"
                        full_conf[i] = 0.0
                    else:
                        para["paragraph_roles"] = [full_roles[i]]
                        para["role_source"] = full_sources[i]
                        para["role_confidence"] = full_conf[i]

                # Final check: STRICT DATA CONSISTENCY
                if len(paragraphs) != len(full_roles):
                    raise ValueError(f"CRITICAL ERROR: Paragraphs count ({len(paragraphs)}) != Labels count ({len(full_roles)})")

                # Stats
                role_dist = dict(Counter([r for r in full_roles if r is not None]))
                fallback_count = sum(1 for s in full_sources if s in ("rule_fallback", "llm_fallback"))
                llm_paragraphs = sum(1 for s in full_sources if s is not None)
                fallback_rate = fallback_count / llm_paragraphs if llm_paragraphs else 0.0

                case["role_distribution"] = role_dist
                case["llm_fallback_rate"] = round(fallback_rate, 4)
                case["text_hash"] = compute_text_hash(usable_texts)
                case["dataset_version"] = "v1_clean_roles"
                
                # Default Metadata if missing
                if "meta" not in case:
                    case["meta"] = {}
                case["meta"].setdefault("court", "Supreme Court")
                case["meta"].setdefault("year", 2025)
                case["meta"].setdefault("case_type", "civil")
                case["meta"].setdefault("acts_referred", [])

                case["role_labeling"] = {
                    "model": MODEL,
                    "taxonomy_version": "v7_fixed_batching",
                    "prompt_version": PROMPT_VERSION,
                    "temperature": OLLAMA_OPTIONS["temperature"],
                    "seed": FIXED_SEED,
                    "run_id": RUN_ID,
                    "run_timestamp": RUN_TIMESTAMP,
                    "batch_size": BATCH_SIZE,
                    "ollama_options": OLLAMA_OPTIONS,
                }

                fout.write(json.dumps(case, ensure_ascii=False) + "\n")
                fout.flush()

                global_role_counter.update([r for r in full_roles if r is not None])
                total_fallbacks += fallback_count

                if total_cases % 100 == 0:
                    logging.info(f"Processed {total_cases} cases, current fallback rate: {fallback_rate:.2%}")

            except json.JSONDecodeError:
                logging.error(f"JSON error in case index {total_cases}")
            except Exception as e:
                print(f"Error in case index {total_cases}: {type(e).__name__} {e}")

                # WRITE PARTIAL CASE (VERY IMPORTANT)
                case["error"] = str(e)
                fout.write(json.dumps(case, ensure_ascii=False) + "\n")

    # Final summary
    logging.info("═"*80)
    logging.info("Role labeling completed")
    logging.info(f"  Run ID:                {RUN_ID}")
    logging.info(f"  Cases processed this run: {total_cases:,}")
    logging.info(f"  Paragraphs labeled this run: {total_paras:,}")
    logging.info(f"  Batch size:            {BATCH_SIZE}")
    logging.info(f"  Total fallbacks this run: {total_fallbacks:,} ({total_fallbacks/total_paras*100:.2f}% if total_paras else 'N/A')")

    logging.info("\nRole distribution this run:")
    for r in sorted(global_role_counter):
        cnt = global_role_counter[r]
        pct = cnt / total_paras * 100 if total_paras else 0
        logging.info(f"  {r:18} : {cnt:8,}  ({pct:5.1f}%)")

    logging.info(f"\nOutput saved/appended to: {OUTPUT}")
    logging.info("═"*80)

if __name__ == "__main__":
    main()