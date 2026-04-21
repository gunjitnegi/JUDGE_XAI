from datasets import load_dataset
import json
from pathlib import Path

output_path = Path("data/raw/judgments/injudgements.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

print("Downloading OpenNyAI InJudgements dataset...")

dataset = load_dataset("opennyaiorg/InJudgements_dataset", split="train")

print("Columns:", dataset.column_names)

print("Saving dataset to JSONL...")

with open(output_path, "w", encoding="utf-8") as f:
    for idx, row in enumerate(dataset):
        record = {
            "case_id": idx,
            "title": row.get("Titles"),
            "court": row.get("Court_Name"),
            "case_type": row.get("Case_Type"),
            "court_type": row.get("Court_Type"),
            "text": row.get("Text")
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("Download complete:", output_path)
