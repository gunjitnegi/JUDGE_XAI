import json
from pathlib import Path

input_path = Path("data/raw/judgments/injudgements.jsonl")
output_path = Path("data/processed/judgments_filtered.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

def map_case_type(raw_type):
    if not raw_type:
        return "other"

    t = raw_type.lower()

    # Civil categories
    if any(k in t for k in [
        "land",
        "property",
        "service",
        "tax",
        "civil"
    ]):
        return "civil"

    # Criminal
    if "criminal" in t:
        return "criminal"

    # Constitutional
    if "constitution" in t:
        return "constitutional"

    return "other"


print("Classifying judgments...")

count_total = 0
count_kept = 0

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:

    for line in fin:
        count_total += 1
        record = json.loads(line)

        raw_type = record.get("case_type")
        mapped_type = map_case_type(raw_type)

        if mapped_type in ["civil", "criminal", "constitutional"]:
            output_record = {
                "case_id": record.get("case_id"),
                "title": record.get("title"),
                "court": record.get("court"),
                "court_type": record.get("court_type"),
                "case_type": mapped_type,
                "text": record.get("text")
            }

            fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            count_kept += 1

print(f"Total cases: {count_total}")
print(f"Kept cases: {count_kept}")
print("Filtered dataset saved to:", output_path)
