#!/usr/bin/env python3
"""
Filter Indian court judgment paragraphs (JSONL → cleaned JSONL).
Hard-coded paths version – ready to run directly.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

# ────────────────────────────────────────────────
#               CONFIGURATION
# ────────────────────────────────────────────────

INPUT  = Path("data/processed/judgments_paragraphs.jsonl")
OUTPUT = Path("data/processed/judgments_paragraphs_filtered.jsonl")
REPORT = Path("data/reports/filter_report.json")

CONFIG = {
    "min_paragraphs": 5,
    "max_paragraphs": 800,                     # aligned with segment_paragraphs.py
    "max_alignment_fail_ratio": 0.30,
    "max_paragraph_length_chars": 2000,
    "max_allowed_long_paragraphs": 5,          # soft threshold
    "min_total_length_chars": 800,
    "truncate_long_paragraphs": True,          # now applied only on kept cases
    "require_reasoning_or_facts": True,
    "add_filter_metadata": True,
    "filter_version": "2025-02-v2",
    "dataset_version": "phase2_filtered_v1",
}

# ────────────────────────────────────────────────

logger = logging.getLogger(__name__)


def validate_case(
    record: Dict[str, Any], config: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    PURE VALIDATION — does NOT modify the record
    Returns (is_valid, list_of_failure_reasons)
    """
    reasons: List[str] = []
    paragraphs: List[Dict] = record.get("paragraphs", [])

    if not isinstance(paragraphs, list):
        reasons.append("paragraphs_not_list")
        return False, reasons

    num_paragraphs = len(paragraphs)

    # Too few paragraphs
    if num_paragraphs < config["min_paragraphs"]:
        reasons.append(f"too_few_paragraphs_{num_paragraphs}")

    # Too many paragraphs (OCR garbage / annexures)
    if num_paragraphs > config["max_paragraphs"]:
        reasons.append(f"too_many_paragraphs_{num_paragraphs}")

    # Alignment failure ratio
    if num_paragraphs > 0:
        alignment_fails = sum(
            1 for p in paragraphs if p.get("alignment_failed", False) is True
        )
        fail_ratio = alignment_fails / num_paragraphs
        if fail_ratio > config["max_alignment_fail_ratio"]:
            reasons.append(f"high_align_fail_{fail_ratio:.3f}")

    # Soft long-paragraph threshold
    long_paras = sum(
        1 for p in paragraphs
        if (p.get("length", 0) or len(p.get("text", ""))) > config["max_paragraph_length_chars"]
    )
    if long_paras > config["max_allowed_long_paragraphs"]:
        reasons.append(f"too_many_long_paragraphs_{long_paras}")

    # Very short overall document
    total_length = sum(p.get("length", 0) or len(p.get("text", "")) for p in paragraphs)
    if total_length < config["min_total_length_chars"]:
        reasons.append(f"doc_too_short_{total_length}")

    # Must have at least some reasoning or facts
    if config["require_reasoning_or_facts"]:
        has_reasoning = any("reasoning" in p.get("paragraph_roles", []) for p in paragraphs)
        has_facts     = any("facts"     in p.get("paragraph_roles", []) for p in paragraphs)
        if not (has_reasoning or has_facts):
            # only remove if also very short
            if total_length < 2000:
                reasons.append("no_reasoning_or_facts")


    return len(reasons) == 0, reasons


def truncate_paragraphs(record: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Mutates record in-place — ONLY call this on cases that already passed validation
    """
    if not config.get("truncate_long_paragraphs", False):
        return

    max_len = config["max_paragraph_length_chars"]

    for p in record.get("paragraphs", []):
        text = p.get("text", "")
        if len(text) > max_len:
            p["text"] = text[:max_len]
            p["length"] = len(p["text"])
            p["truncated"] = True


def inject_filter_metadata(record: Dict[str, Any], config: Dict[str, Any]) -> None:
    if not config.get("add_filter_metadata", False):
        return

    now = datetime.utcnow().isoformat() + "Z"
    record["filter"] = {
        "version": config["filter_version"],
        "filtered_at": now,
        "config_digest": str(hash(str(sorted(config.items()))))[:10],
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not INPUT.is_file():
        logger.error(f"Input file does NOT exist: {INPUT}")
        logger.error("→ Create / download the file first.")
        return

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    removal_stats = defaultdict(int)
    total = kept = 0

    with (
        INPUT.open("r", encoding="utf-8") as fin,
        OUTPUT.open("w", encoding="utf-8") as fout,
        tqdm(desc="Filtering", unit=" cases", dynamic_ncols=True) as pbar,
    ):
        for i, line in enumerate(fin, 1):
            total += 1
            line = line.rstrip("\n")
            if not line.strip():
                removal_stats["empty_line"] += 1
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                removal_stats["json_decode_error"] += 1
                continue

            is_valid, reasons = validate_case(record, CONFIG)

            if is_valid:
                truncate_paragraphs(record, CONFIG)       # ← now only here
                inject_filter_metadata(record, CONFIG)
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
            else:
                for r in reasons:
                    removal_stats[r] += 1

            pbar.update(1)

    removed = total - kept

    report = {
        "run_at": datetime.utcnow().isoformat() + "Z",
        "input": str(INPUT.absolute()),
        "output": str(OUTPUT.absolute()),
        "config": CONFIG,
        "dataset_version": CONFIG["dataset_version"],
        "total": total,
        "kept": kept,
        "removed": removed,
        "removal_rate_percent": round(removed / total * 100, 2) if total else 0.0,
        "removal_breakdown": dict(sorted(removal_stats.items(), key=lambda x: -x[1])),
    }

    with REPORT.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("")
    logger.info("Filtering finished")
    logger.info("─" * 50)
    logger.info(f"Input  → {INPUT}")
    logger.info(f"Output → {OUTPUT}")
    logger.info(f"Report → {REPORT}")
    logger.info(f"Total  : {total:>7,d}")
    logger.info(f"Kept   : {kept:>7,d}  ({kept/total*100:5.1f}%)")
    logger.info(f"Removed: {removed:>7,d}  ({removed/total*100:5.1f}%)")
    logger.info("")

    if removal_stats:
        logger.info("Top removal reasons:")
        for reason, cnt in sorted(removal_stats.items(), key=lambda x: -x[1])[:12]:
            logger.info(f"  {cnt:>6,d} × {reason}")
    else:
        logger.info("No cases were removed.")


if __name__ == "__main__":
    main()