#!/usr/bin/env python3
"""
minimal-research-grade-judgment-cleaner.py

Focused ONLY on cleaning Indian court judgments:
- Remove headers, footers, metadata, appearance, coram, counsel lists
- Remove repeated lines / page artifacts / common noise
- Normalize whitespace and unicode
- Output: case_id, case_type (normalized), cleaned_text, removal_ratio

Very minimal filtering — only obviously empty documents are skipped.
"""
import json
import re
import unicodedata
from pathlib import Path
from bs4 import BeautifulSoup
from collections import Counter

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────
INPUT_PATH  = Path("data/processed/judgments_filtered.jsonl")
OUTPUT_PATH = Path("data/processed/judgments_cleaned.jsonl")
STATS_PATH  = Path("data/reports/cleaning_stats.json")

MIN_RAW_LENGTH   = 100
MIN_CLEAN_LENGTH = 100

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
STATS_PATH.parent.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────
# PATTERNS
# ────────────────────────────────────────────────
STRONG_BODY_START_MARKERS = [
    r'(?i)^\s*J\s*U\s*D\s*G\s*M\s*E\s*N\s*T\s*$',
    r'(?i)^\s*JUDGMENT\s*$',
    r'(?i)^\s*O\s*R\s*D\s*E\s*R\s*$',
    r'(?i)^\s*ORDER\s*$',
    r'(?i)^ORAL\s+(JUDGMENT|ORDER)',
    r'(?i)The\s+Judgment\s+of\s+the\s+Court\s+was\s+delivered\s+by',
    r'(?i)^REASONS\s+FOR\s+(ORDER|JUDGMENT)',
]

WEAK_BODY_START_HINTS = [
    r'(?i)\b(held|we\s+hold|in\s+our\s+view|opinion|considered|ratio)\b',
    r'(?i)(section|article|rule|order|clause)\s+\d+',
    r'(?i)\(\d{4}\)\s*\d+\s*(scc|air|pat|blr|jl|crilj|compcas|taxman|itr)',
    r'(?i)(writ|petition|appeal|revision|application).*?(is|stands|hereby)',
]

PREAMBLE_PATTERNS = [
    r'(?im)^.*(coram|bench|present|hon\'?ble|justice|mr\.|ms\.|dr\.|acting\s*chief\s*justice).*$',
    r'(?im)^.*(appearance|advocate|advocates|standing\s*counsel|counsel\s+for|for\s+the\s+(appellant|petitioner|respondent|defendant|accused)).*$',
    r'(?im)^.*(learned\s+(senior\s+)?advocate|counsel\s+appeared|argued\s+by|argued\s+through).*$',
    # Removed aggressive petitioner/respondent pattern to preserve core legal text
    r'(?im)^.*(date\s+of\s+(decision|hearing|judgment|order)|pronounced\s+on).*$',
]

HEADER_PATTERNS = [
    r'(?i)^(supreme\s*court\s*of\s*india|high\s*court\s*of\s*\w+|district\s*court\s*\w*)',
    r'(?i)^(civil|criminal)\s*(writ|petition|appeal|revision|application|slp)\s*no\.\s*\d+',
    r'(?i)^cwjc\s*no\.\d+\s*of\s*\d+',
    r'(?i)^[a-z]+\s+high\s+court\s+\w+\s+no\.\d+\s+of\s+\d+',
    r'(?i)^patna\s+high\s+court\s+cwjc\s+no\.\d+\s+of\s+\d+',
    r'(?i)^(\w+\s+)?high\s+court\s+(\w+\s+)?(\w+\s+no\.\s*\d+\s+of\s+\d+)?',
]

FOOTER_PATTERNS = [
    r'(?i)^page\s*\d+(\s*of\s*\d+)?$',
    r'(?i)^indian\s*kanoon',
    r'(?i)^https?://indiankanoon\.org',
    r'(?i)^reportable\s*/\s*non.?reportable',
    r'(?i)^©\s*\d{4}\s*indian\s*kanoon',
]

APPEARANCE_END_HINTS = [
    r'^={3,}$', r'^-{3,}$', r'^_{3,}$',
    r'^\s*$',
]

def find_body_start_idx(lines: list[str]) -> int:
    last_metadata_line = 0
    in_appearance_block = False

    for i, line in enumerate(lines):
        stripped_lower = line.strip().lower()

        if any(re.search(p, stripped_lower) for p in [
            r'(?i)appearance', r'(?i)advocat', r'(?i)counsel',
            r'(?i)for the.*(petitioner|respondent)', r'(?i)paag', r'(?i)aag'
        ]):
            in_appearance_block = True
            last_metadata_line = i

        for pat in STRONG_BODY_START_MARKERS:
            if re.search(pat, line.strip()):
                return max(0, i - 2)

        if in_appearance_block and i > last_metadata_line + 4:
            for pat in WEAK_BODY_START_HINTS:
                if re.search(pat, line, re.I):
                    return max(0, i - 3)

        if in_appearance_block and any(re.match(p, line.strip()) for p in APPEARANCE_END_HINTS):
            return i + 1

    # fallback
    if last_metadata_line > 4 and last_metadata_line < len(lines) - 6:
        return min(last_metadata_line + 6, len(lines) - 1)

    return len(lines)


def remove_repeated_lines(text: str, max_repeats: int = 5, min_line_len: int = 30) -> str:
    lines = text.splitlines()
    line_counts = Counter(lines)
    cleaned_lines = [
        line for line in lines
        if len(line.strip()) < min_line_len or line_counts[line] <= max_repeats
    ]
    return "\n".join(cleaned_lines)


def strip_preamble(text: str) -> str:
    if not text.strip():
        return ""

    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator="\n", strip=True)

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    start_idx = find_body_start_idx(lines)

    # Safer fallback: skip at most first 20 lines when detection fails badly
    if start_idx >= len(lines) - 3:
        start_idx = min(20, len(lines))

    core_text = "\n".join(lines[start_idx:])

    # remove lingering judge/coram lines
    core_text = re.sub(r'(?im)^.*(coram|bench|hon\'?ble|justice|mr\.|ms\.|dr\.).*$', '', core_text)

    return core_text.strip()


def normalize_and_clean(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    # remove only footers/noise (preamble/headers already handled in strip_preamble)
    for pat in FOOTER_PATTERNS:
        text = re.sub(pat, "", text, flags=re.I | re.M)

    text = re.sub(r"[_=\-★☆♦—━]{4,}", "\n", text)
    text = remove_repeated_lines(text, max_repeats=5, min_line_len=30)
    # Preserve paragraph structure
    text = re.sub(r'\r\n', '\n', text)

    # Collapse only excessive newlines, not structure
    text = re.sub(r'\n{4,}', '\n\n', text)

    # Normalize spaces but preserve line breaks
    text = re.sub(r'[ \t]{2,}', ' ', text)


    return text.strip()


def normalize_case_type(case_type):
    case_type = str(case_type).lower()
    if "civil" in case_type:
        return "civil"
    if "criminal" in case_type:
        return "criminal"
    if "constitution" in case_type:
        return "constitutional"
    return "other"


# ────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────
def main():
    print("Minimal cleaning pipeline — only text normalization & noise removal\n")

    total = kept = skipped = 0
    skip_short_raw = skip_short_cleaned = 0
    chars_in_kept = chars_out_kept = 0
    total_removed_ratio = max_removed_ratio = high_removal_count = 0

    with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                record = json.loads(line)

                raw_text = (
                    record.get("text") or
                    record.get("judgment_text") or
                    record.get("content") or
                    record.get("full_text") or
                    ""
                ).strip()

                if len(raw_text) < MIN_RAW_LENGTH:
                    skipped += 1
                    skip_short_raw += 1
                    continue

                # Cleaning steps
                no_header = strip_preamble(raw_text)
                cleaned   = normalize_and_clean(no_header)

                if len(cleaned) < MIN_CLEAN_LENGTH:
                    skipped += 1
                    skip_short_cleaned += 1
                    continue

                # Now we keep it
                kept += 1
                chars_in_kept += len(raw_text)
                chars_out_kept += len(cleaned)

                # Removal ratio stats (only on kept documents)
                raw_len = len(raw_text)
                cleaned_len = len(cleaned)
                removed_ratio = 0.0
                if raw_len > 0:
                    removed_ratio = 1 - (cleaned_len / raw_len)
                    total_removed_ratio += removed_ratio
                    max_removed_ratio = max(max_removed_ratio, removed_ratio)
                    if removed_ratio > 0.5:
                        high_removal_count += 1

                # Output record — now includes removal_ratio
                out_record = {
                    "case_id":      record.get("case_id", line_num),
                    "case_type":    normalize_case_type(record.get("case_type", "unknown")),
                    "cleaned_text": cleaned,
                    "removal_ratio": round(removed_ratio, 4)   # 4 decimal places for precision
                }

                fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")

                if total % 200 == 0:
                    print(f"{total:6d}  | kept {kept:5d}  | skipped {skipped:5d}", end="\r")

            except Exception as e:
                print(f"Error at line {line_num}: {e}")
                skipped += 1
                continue

    # Save comprehensive stats
    stats = {
        "total": total,
        "kept": kept,
        "skipped": skipped,
        "skip_short_raw": skip_short_raw,
        "skip_short_cleaned": skip_short_cleaned,
        "avg_input_length": chars_in_kept // max(kept, 1),
        "avg_output_length": chars_out_kept // max(kept, 1),
        "avg_removal_ratio": (total_removed_ratio / kept) if kept else 0,
        "max_removal_ratio": max_removed_ratio,
        "cases_over_50_percent_removed": high_removal_count
    }
    with open(STATS_PATH, "w", encoding="utf-8") as stats_file:
        json.dump(stats, stats_file, indent=2)

    print("\n\nFinished.")
    print(f"Total records seen : {total:,d}")
    print(f"Kept               : {kept:,d}")
    print(f"Skipped            : {skipped:,d}")
    if kept > 0:
        print(f"Avg input length   : {chars_in_kept // kept:,} chars")
        print(f"Avg output length  : {chars_out_kept // kept:,} chars")
        print(f"Avg removal ratio  : {total_removed_ratio / kept:.2%}")
        print(f"Max removal ratio  : {max_removed_ratio:.2%}")
        print(f"Cases >50% removed : {high_removal_count:,d}")
    print(f"Skip reasons: short_raw={skip_short_raw}, short_cleaned={skip_short_cleaned}")
    print(f"\nOutput → {OUTPUT_PATH}")
    print(f"Stats  → {STATS_PATH}")


if __name__ == "__main__":
    main()