import json
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_dataset")

INPUT_PATH = Path(r"C:\final_year\JUDGEXAI\data\processed\role_labelled judgements.jsonl")
OUT_DIR = Path(r"C:\final_year\JUDGEXAI\data\kaggle")

DS1_ROLES = OUT_DIR / "kaggle_rhetorical_roles.jsonl"
DS2_SUMMARIES = OUT_DIR / "kaggle_structured_summaries.jsonl"
DS3_QA = OUT_DIR / "kaggle_qa_rag.jsonl"

FINAL_ROLES = ["facts", "issues", "arguments", "reasoning", "final_decision", "statutory", "procedural", "other"]

def build_datasets():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    total = 0
    with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
         open(DS1_ROLES, "w", encoding="utf-8") as f_roles, \
         open(DS2_SUMMARIES, "w", encoding="utf-8") as f_sums, \
         open(DS3_QA, "w", encoding="utf-8") as f_qa:
         
        for line in fin:
            if not line.strip(): continue
            try:
                case = json.loads(line)
            except:
                continue
                
            paragraphs = case.get("paragraphs", [])
            if not paragraphs: continue
            
            # Consistency
            if len(paragraphs) != len(paragraphs): # Should be verified upstream but just in case
                continue
            
            case_id = case.get("case_id", "")
            meta = case.get("meta", {})
            court = meta.get("court", "Supreme Court")
            year = meta.get("year", 2025)
            dataset_version = case.get("dataset_version", "v1_clean_roles")
            
            # DS1: Rhetorical Roles
            ds1_obj = {
                "case_id": case_id,
                "dataset_version": dataset_version,
                "court": court,
                "year": year,
                "paragraphs": []
            }
            
            sections = defaultdict(list)
            
            for p in paragraphs:
                role = p.get("paragraph_roles", ["other"])[0]
                text = p.get("text", "").strip()
                ds1_obj["paragraphs"].append({
                    "text": text,
                    "role": role
                })
                sections[role].append(text)
                
            f_roles.write(json.dumps(ds1_obj, ensure_ascii=False) + "\n")
            
            # DS2: Structured Summaries
            summary = {
                "facts": "\n\n".join(sections["facts"]),
                "issues": "\n\n".join(sections["issues"]),
                "arguments": "\n\n".join(sections["arguments"]),
                "reasoning": "\n\n".join(sections["reasoning"]),
                "decision": "\n\n".join(sections["final_decision"])
            }
            
            full_text = "\n\n".join([p["text"] for p in paragraphs])
            
            if len(summary["facts"]) > 100 and len(summary["reasoning"]) > 100:
                ds2_obj = {
                    "case_id": case_id,
                    "court": court,
                    "year": year,
                    "full_judgment": full_text,
                    "structured_summary": summary
                }
                f_sums.write(json.dumps(ds2_obj, ensure_ascii=False) + "\n")
                
                # DS3: QA RAG Dataset
                qa_pairs = [
                    ("What are the facts of the case?", summary["facts"], sections["facts"]),
                    ("What issues were raised?", summary["issues"], sections["issues"]),
                    ("What were the arguments presented?", summary["arguments"], sections["arguments"]),
                    ("What was the reasoning of the court?", summary["reasoning"], sections["reasoning"]),
                    ("What did the court decide?", summary["decision"], sections["final_decision"])
                ]
                
                for q, a, support in qa_pairs:
                    if len(a) > 50: # Only write if there is a meaningful answer
                        ds3_obj = {
                            "case_id": case_id,
                            "question": q,
                            "answer": a,
                            "supporting_paragraphs": support
                        }
                        f_qa.write(json.dumps(ds3_obj, ensure_ascii=False) + "\n")
            total += 1
            if total % 100 == 0:
                logger.info(f"Processed {total} cases into datasets...")
                
    logger.info(f"Finished dataset generation. Processed {total} cases.")
    logger.info(f"Rhetorical Roles: {DS1_ROLES}")
    logger.info(f"Structured Summaries: {DS2_SUMMARIES}")
    logger.info(f"QA RAG: {DS3_QA}")

if __name__ == "__main__":
    build_datasets()
