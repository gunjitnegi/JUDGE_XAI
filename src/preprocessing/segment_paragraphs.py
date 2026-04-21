#!/usr/bin/env python3
"""
segment_paragraphs.py – Robust paragraph segmentation for Indian court judgments

Improvements:
- Split on double newlines, numbered paragraphs, headings, bullet points.
- Enforce maximum paragraph length (default 1500 chars) by splitting on sentence boundaries.
- Preserve character‑level alignment with original text for traceability.
- Merge very short fragments into previous paragraph.
- Cap at MAX_PARAGRAPHS (600) to avoid pathological cases.
- Output usable_for_evidence flag based on alignment success.
"""

import json
import re
from pathlib import Path
from tqdm import tqdm

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
INPUT_PATH = Path("data/processed/judgments_cleaned.jsonl")
OUTPUT_PATH = Path("data/processed/judgments_paragraphs.jsonl")
SAMPLES_PATH = Path("data/reports/segmentation_samples.jsonl")

# Maximum characters per paragraph – longer ones are split on sentence boundaries
MAX_PARA_LEN = 1500
# Merge fragments shorter than this into the previous paragraph
MIN_FRAGMENT_LENGTH = 50
# Hard limit on total paragraphs per judgment (safety)
MAX_PARAGRAPHS = 800

SEGMENTATION_VERSION = "v5_legal_aware"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)


def normalize_for_match(s: str) -> str:
    """Collapse whitespace for fuzzy matching when exact match fails."""
    return re.sub(r"\s+", " ", s.strip())


def split_on_sentences(text: str, max_len: int) -> list[str]:
    """
    Split a long paragraph into chunks not exceeding max_len,
    preferring sentence boundaries (., ?, !) or legal markers.
    """
    if len(text) <= max_len:
        return [text]

    # First try to split on common legal paragraph markers
    markers = [
        r'(?<=\.)\s+(?=\d+\.)',           # ". 1."  (numbered paragraph start)
        r'(?<=\.)\s+(?=[A-Z])',            # ". Next sentence" (capital letter)
        r'(?<=\.)\s+(?=\([a-z]\)\s)',      # ". (a) "  (sub‑paragraph)
        r'(?<=;)\s+(?=\d+\.)',              # "; 1." 
    ]
    combined_pattern = "|".join(markers)
    parts = re.split(combined_pattern, text)
    if len(parts) > 1:
        # Reassemble parts into chunks respecting max_len
        chunks = []
        current = ""
        for part in parts:
            if len(current) + len(part) + 1 <= max_len:
                current += (" " + part if current else part)
            else:
                if current:
                    chunks.append(current.strip())
                current = part
        if current:
            chunks.append(current.strip())
        # If all chunks are still too long, fall back to sentence splitting
        if all(len(chunk) > max_len for chunk in chunks):
            # fall through to sentence split
            pass
        else:
            return chunks

    # Fallback: split on sentence boundaries (., ?, !)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_len:
            current += (" " + sent if current else sent)
        else:
            if current:
                chunks.append(current.strip())
            current = sent
    if current:
        chunks.append(current.strip())

    # If any chunk is still too long, we have to hard‑split (rare)
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_len:
            final_chunks.append(chunk)
        else:
            # Emergency split – cut at max_len (last resort)
            final_chunks.extend([chunk[i:i+max_len] for i in range(0, len(chunk), max_len)])
    return final_chunks


def split_into_paragraphs(text: str) -> list[dict]:
    """
    Main segmentation routine.
    Returns list of paragraph dicts with fields:
        para_id, text, length, marker, position,
        original_char_start, original_char_end,
        alignment_failed, usable_for_evidence,
        plus metadata flags.
    """
    if not text:
        return []

    original_text = text  # never modify this – used for alignment

    # Normalise line endings and collapse multiple blank lines
    working_text = re.sub(r"\r\n", "\n", original_text)
    working_text = re.sub(r"\n{3,}", "\n\n", working_text)

    # Split on clear paragraph boundaries
    # 1. double newline
    # 2. newline followed by a number + dot (e.g., "10.") or square bracket "[1]"
    # 3. newline followed by a roman numeral (i), (ii)
    # 4. newline followed by a heading-like word + colon (e.g., "Facts:")
    # 5. newline followed by a bullet (•) or dash
    split_pattern = r'\n\s*\n|\n(?=\d+\.\s)|\n(?=\[\d+\]\s)|\n(?=\([ivx]+\)\s)|\n(?=[A-Z][a-z]+:\s)|\n(?=[•\-]+\s)'
    raw_paragraphs = re.split(split_pattern, working_text)
    raw_paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    # Merge very short fragments into previous paragraph
    merged = []
    for para in raw_paragraphs:
        if len(para) < MIN_FRAGMENT_LENGTH and merged:
            merged[-1] += " " + para
        else:
            merged.append(para)

    # Further split any paragraph that exceeds MAX_PARA_LEN
    final_paras = []
    for para in merged:
        if len(para) <= MAX_PARA_LEN:
            final_paras.append(para)
        else:
            # Split into chunks
            chunks = split_on_sentences(para, MAX_PARA_LEN)
            final_paras.extend(chunks)

    # Safety cap
    if len(final_paras) > MAX_PARAGRAPHS:
        final_paras = final_paras[:MAX_PARAGRAPHS]

    # Now build paragraph objects with alignment to original_text
    paragraphs = []
    current_char_pos = 0
    para_counter = 1

    for para in final_paras:
        # Try to detect a marker (e.g., "1.", "(a)", "Facts:")
        marker_match = re.match(r"^(\(?\d+\)?\.|[a-zA-Z]\)|[A-Z][a-z]+:)", para)
        marker = marker_match.group(0) if marker_match else None

        # Locate this paragraph in the original text, starting from current_char_pos
        start = original_text.find(para, current_char_pos)
        alignment_failed = False
        usable_for_evidence = False

        if start != -1:
            alignment_failed = False
            usable_for_evidence = True
            orig_start = start
            orig_end = start + len(para)
            current_char_pos = orig_end
        else:
            # Fallback: try normalized matching
            norm_para = normalize_for_match(para)
            norm_text_from_pos = normalize_for_match(original_text[current_char_pos:])
            norm_start = norm_text_from_pos.find(norm_para)
            if norm_start != -1:
                start = current_char_pos + norm_start
                alignment_failed = False
                usable_for_evidence = True
                orig_start = start
                orig_end = start + len(para)
                current_char_pos = orig_end
            else:
                # Cannot align – still output but mark as failed
                alignment_failed = True
                usable_for_evidence = False
                orig_start = None
                orig_end = None
                # Advance cursor roughly to avoid infinite loops
                current_char_pos += len(para)

        # Build paragraph record
        para_dict = {
            "para_id": para_counter,
            "text": para,
            "length": len(para),
            "paragraph_roles": ["unlabeled"],
            "marker": marker,
            "starts_with_reasoning_cue": bool(
                re.match(
                    r"(?i)^(in the result|therefore|accordingly|hence|thus|we hold|we find|in conclusion)",
                    para,
                )
            ),
            "contains_citation": bool(
                re.search(r"\bAIR\b|\bSCC\b|\bvs\.\b", para)
            ),
            "contains_statute": bool(
                re.search(r"\bSection\s+\d+|\bArticle\s+\d+", para, re.IGNORECASE)
            ),
            "original_char_start": orig_start,
            "original_char_end": orig_end,
            "alignment_failed": alignment_failed,
            "usable_for_evidence": usable_for_evidence,
            "sentences": [],  # to be filled later if needed
        }
        paragraphs.append(para_dict)
        para_counter += 1

    # Add position tags (start / middle / end)
    total = len(paragraphs)
    for i, p in enumerate(paragraphs):
        if total == 0:
            break
        if i < total * 0.2:
            p["position"] = "start"
        elif i > total * 0.8:
            p["position"] = "end"
        else:
            p["position"] = "middle"

    return paragraphs


def main():
    total = 0
    total_paragraphs = 0
    max_paragraphs = 0
    min_paragraphs = float("inf")
    cases_with_zero_paras = 0
    cases_with_alignment_fail = 0
    cases_with_low_alignment_quality = 0
    all_records = []

    # Read all lines
    with open(INPUT_PATH, "r", encoding="utf-8") as fin:
        lines = list(fin)

    # Open output file once in write mode
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for line in tqdm(lines, desc="Segmenting judgments"):
            total += 1
            try:
                record = json.loads(line.strip())
                cleaned_text = record.get("cleaned_text", "")

                paragraphs = split_into_paragraphs(cleaned_text)
                num_p = len(paragraphs)

                total_paragraphs += num_p
                if num_p > max_paragraphs:
                    max_paragraphs = num_p
                if num_p < min_paragraphs:
                    min_paragraphs = num_p

                if num_p == 0:
                    cases_with_zero_paras += 1

                alignment_fails = sum(
                    1 for p in paragraphs if p.get("alignment_failed", False)
                )
                if alignment_fails > 0:
                    cases_with_alignment_fail += 1

                out_record = {
                    "case_id": record["case_id"],
                    "case_type": record["case_type"],
                    "paragraphs": paragraphs,
                    "num_paragraphs": num_p,
                    "total_text_length": len(cleaned_text),
                    "segmentation_quality_heuristic": 0.0,  # disabled for now
                    "segmentation_version": SEGMENTATION_VERSION,
                }

                # Flag if more than 30% of paragraphs failed alignment
                if num_p > 0 and (alignment_fails / num_p) > 0.3:
                    cases_with_low_alignment_quality += 1
                    out_record["low_alignment_quality"] = True

                fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")

                all_records.append((num_p, out_record))

            except Exception as e:
                print(f"Error at case {record.get('case_id', 'unknown')}: {e}")
                continue

    # Final statistics
    avg_paras = total_paragraphs / total if total > 0 else 0
    min_paragraphs = min_paragraphs if min_paragraphs != float("inf") else 0

    print(f"\nFinished. Total cases processed: {total}")
    print("Segmentation statistics:")
    print(f"  Total paragraphs across all cases : {total_paragraphs:,d}")
    print(f"  Average paragraphs per case       : {avg_paras:.1f}")
    print(f"  Max paragraphs in one case         : {max_paragraphs}")
    print(f"  Min paragraphs in one case         : {min_paragraphs}")
    print(f"  Cases with zero paragraphs         : {cases_with_zero_paras}")
    print(f"  Cases with alignment failures      : {cases_with_alignment_fail}")
    print(f"  Cases with low alignment quality   : {cases_with_low_alignment_quality}")
    print(f"  Output saved to                    : {OUTPUT_PATH}")

    # Write a sample file for inspection (first 50, top 25 longest, top 25 shortest)
    if all_records:
        first_50 = all_records[:50]
        sorted_max = sorted(all_records, key=lambda x: x[0], reverse=True)[:25]
        sorted_min = sorted(all_records, key=lambda x: x[0])[:25]

        samples = first_50 + sorted_max + sorted_min

        with open(SAMPLES_PATH, "w", encoding="utf-8") as sf:
            for _, rec in samples:
                sf.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"  Samples saved to                   : {SAMPLES_PATH}")


if __name__ == "__main__":
    main()