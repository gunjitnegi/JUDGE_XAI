import pandas as pd
import json
from pathlib import Path

statutes_dir = Path("data/raw/statutes")
output_path = statutes_dir / "statutes.jsonl"

records = []

# -------- IPC --------
ipc_file = statutes_dir / "ipc_sections.csv"
if ipc_file.exists():
    print("Loading IPC:", ipc_file)
    df = pd.read_csv(ipc_file)

    for _, row in df.iterrows():
        records.append({
            "law": "IPC",
            "section": str(row.get("Section", "")).strip(),
            "title": str(row.get("Title", "")).strip(),
            "text": str(row.get("Description", "")).strip()
        })
else:
    print("IPC file not found")

# -------- BNS --------
bns_file = statutes_dir / "bns_sections.csv"
if bns_file.exists():
    print("Loading BNS:", bns_file)
    df = pd.read_csv(bns_file)

    for _, row in df.iterrows():
        records.append({
            "law": "BNS",
            "section": str(row.get("Section", "")).strip(),
            "title": str(row.get("Title", "")).strip(),
            "text": str(row.get("Description", "")).strip()
        })
else:
    print("BNS file not found")

# -------- Save --------
with open(output_path, "w", encoding="utf-8") as f:
    for r in records:
        if r["section"] and r["text"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Saved {len(records)} statute records to {output_path}")
