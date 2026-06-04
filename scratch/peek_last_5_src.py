import json
from pathlib import Path

path = Path(r"C:\final_year\JUDGEXAI\data\processed\role_labelled judgements.jsonl")

with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()
    last_5 = lines[-5:]

for i, line in enumerate(last_5):
    data = json.loads(line)
    print(f"\n--- Case {i+1} ({data.get('case_id', 'unknown')}) ---")
    for p in data.get("paragraphs", [])[:4]: # Show first 4 paras
        print(f"Para {p['para_id']} [{p['paragraph_roles']}] (Src: {p.get('role_source')}): {p['text'][:120]}...")
    print(f"Role Distribution: {data.get('role_distribution')}")
