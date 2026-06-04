#!/usr/bin/env python3
"""
prepare_dataset.py — Convert structured summaries into Alpaca instruction-tuning format.

Input:  data/processed/summarization_dataset.jsonl
Output: data/finetuning/train.jsonl, data/finetuning/val.jsonl
        data/finetuning/full_dataset.jsonl (for Kaggle upload)

This script creates instruction-response pairs that teach the model
to produce structured legal summaries from judgment paragraphs.
"""

import json
import random
from pathlib import Path

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
INPUT_PATH = Path("data/processed/summarization_dataset.jsonl")
OUTPUT_DIR = Path("data/finetuning")
TRAIN_PATH = OUTPUT_DIR / "train.jsonl"
VAL_PATH = OUTPUT_DIR / "val.jsonl"
FULL_PATH = OUTPUT_DIR / "full_dataset.jsonl"

TRAIN_SPLIT = 0.9  # 90% train, 10% validation
SEED = 42
MAX_INPUT_CHARS = 12000   # Truncate very long inputs to fit in 4096 tokens
MAX_OUTPUT_CHARS = 4000   # Cap output length

SYSTEM_PROMPT = """You are JUDGE X AI, an expert legal analyst specializing in Indian court judgments. 
Your task is to produce a structured summary of a court judgment by analyzing the provided text.
You must identify and separate: Facts, Issues, Arguments, Reasoning, and the Final Decision.
Be precise, use legal terminology, and cite specific statutes or articles when mentioned."""

INSTRUCTION = """Analyze the following Indian court judgment and produce a structured summary with these sections:
1. **Facts**: Key background events and case history
2. **Issues**: Legal questions before the court
3. **Arguments**: What each side argued
4. **Reasoning**: The court's legal analysis and interpretation
5. **Decision**: The final order and directions

Judgment Text:
{input_text}"""


def truncate_smart(text: str, max_chars: int) -> str:
    """Truncate text at a sentence boundary to avoid cutting mid-sentence."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Find last period or newline
    last_period = truncated.rfind('.')
    last_newline = truncated.rfind('\n')
    cut_point = max(last_period, last_newline)
    if cut_point > max_chars * 0.7:  # Only use if we keep >70% of content
        return truncated[:cut_point + 1]
    return truncated + "..."


def build_output(summary: dict) -> str:
    """Build the target structured summary output."""
    parts = []
    
    if summary.get("facts"):
        facts = truncate_smart(summary["facts"], 1500)
        parts.append(f"**Facts**:\n{facts}")
    
    if summary.get("issues"):
        issues = truncate_smart(summary["issues"], 800)
        parts.append(f"**Issues**:\n{issues}")
    
    if summary.get("arguments"):
        args = truncate_smart(summary["arguments"], 1200)
        parts.append(f"**Arguments**:\n{args}")
    
    if summary.get("reasoning"):
        reasoning = truncate_smart(summary["reasoning"], 2000)
        parts.append(f"**Reasoning**:\n{reasoning}")
    
    if summary.get("decision"):
        decision = truncate_smart(summary["decision"], 1000)
        parts.append(f"**Decision**:\n{decision}")
    
    return "\n\n".join(parts)


def convert_to_alpaca(case: dict) -> dict | None:
    """Convert a single case to Alpaca instruction format."""
    summary = case.get("structured_summary", {})
    input_text = case.get("input_text", "")
    
    if not input_text or not summary:
        return None
    
    # Build the output
    output = build_output(summary)
    if len(output) < 200:  # Skip cases with very short summaries
        return None
    
    # Truncate input
    input_text = truncate_smart(input_text, MAX_INPUT_CHARS)
    output = truncate_smart(output, MAX_OUTPUT_CHARS)
    
    return {
        "instruction": INSTRUCTION.format(input_text=input_text),
        "input": "",  # Already embedded in instruction
        "output": output,
        "system": SYSTEM_PROMPT,
        "case_id": case.get("case_id"),
        "case_type": case.get("case_type"),
    }


def main():
    print(f"Preparing fine-tuning dataset from: {INPUT_PATH}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load and convert
    samples = []
    skipped = 0
    
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            case = json.loads(line)
            converted = convert_to_alpaca(case)
            if converted:
                samples.append(converted)
            else:
                skipped += 1
    
    print(f"Converted: {len(samples)} | Skipped: {skipped}")
    
    # Shuffle and split
    random.seed(SEED)
    random.shuffle(samples)
    
    split_idx = int(len(samples) * TRAIN_SPLIT)
    train_data = samples[:split_idx]
    val_data = samples[split_idx:]
    
    # Write files
    for path, data in [(TRAIN_PATH, train_data), (VAL_PATH, val_data), (FULL_PATH, samples)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(data)} samples to {path}")
    
    # Stats
    avg_input_len = sum(len(s["instruction"]) for s in samples) / len(samples)
    avg_output_len = sum(len(s["output"]) for s in samples) / len(samples)
    
    print(f"\n{'='*60}")
    print(f"Dataset Statistics:")
    print(f"  Total samples    : {len(samples)}")
    print(f"  Train            : {len(train_data)}")
    print(f"  Validation       : {len(val_data)}")
    print(f"  Avg input chars  : {avg_input_len:.0f}")
    print(f"  Avg output chars : {avg_output_len:.0f}")
    print(f"  Avg input tokens : ~{avg_input_len/4:.0f} (estimated)")
    print(f"  Avg output tokens: ~{avg_output_len/4:.0f} (estimated)")
    print(f"{'='*60}")
    print(f"\nUpload 'data/finetuning/full_dataset.jsonl' to Kaggle for training!")


if __name__ == "__main__":
    main()
