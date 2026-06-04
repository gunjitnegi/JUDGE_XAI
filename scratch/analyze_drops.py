import json
from pathlib import Path
from collections import defaultdict, Counter

INPUT = Path(r"C:\final_year\JUDGEXAI\data\processed\role_labelled judgements.jsonl")

no_facts = 0
no_reasoning = 0
no_both = 0
empty_paras = 0
total = 0
kept = 0

with open(INPUT, "r", encoding="utf-8") as f:
    for line in f:
        total += 1
        case = json.loads(line)
        paras = case.get("paragraphs", [])
        
        if not paras:
            empty_paras += 1
            continue
        
        facts_text = ""
        reasoning_text = ""
        for p in paras:
            role = p.get("paragraph_roles", ["other"])[0]
            if role == "facts":
                facts_text += p.get("text", "")
            elif role == "reasoning":
                reasoning_text += p.get("text", "")
        
        has_facts = len(facts_text) > 100
        has_reasoning = len(reasoning_text) > 100
        
        if has_facts and has_reasoning:
            kept += 1
        elif not has_facts and not has_reasoning:
            no_both += 1
        elif not has_facts:
            no_facts += 1
        elif not has_reasoning:
            no_reasoning += 1

print(f"Total cases: {total}")
print(f"Kept (facts+reasoning): {kept} ({kept/total*100:.1f}%)")
print(f"Dropped - No facts: {no_facts}")
print(f"Dropped - No reasoning: {no_reasoning}")  
print(f"Dropped - Neither: {no_both}")
print(f"Dropped - Empty paragraphs: {empty_paras}")
