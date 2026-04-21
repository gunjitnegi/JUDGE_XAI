import json
from pathlib import Path
from collections import defaultdict

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────
INPUT_PATH = Path("data/processed/role_labelled judgements.jsonl")
OUTPUT_PATH = Path("data/processed/summarization_dataset.jsonl")

FINAL_ROLES = ["facts", "issues", "arguments", "reasoning", "final_decision", "statutory", "procedural"]

def build_structured_summary(paragraphs):
    """
    Groups labeled paragraphs and creates a structured summary object.
    """
    sections = defaultdict(list)
    statutes = set()
    
    for p in paragraphs:
        role = p.get("paragraph_roles", ["other"])[0]
        text = p.get("text", "").strip()
        
        if role in FINAL_ROLES:
            sections[role].append(text)
        
        # Collect statutes if present in metadata
        if p.get("contains_statute"):
            # This is a heuristic, we could use a more advanced parser here
            pass

    # Create the structured object
    # We take the first few paragraphs of each section for the "summary" 
    # or the full content if it's short.
    structured = {
        "facts": "\n\n".join(sections["facts"]),
        "issues": "\n\n".join(sections["issues"]),
        "arguments": "\n\n".join(sections["arguments"]),
        "reasoning": "\n\n".join(sections["reasoning"]),
        "decision": "\n\n".join(sections["final_decision"]),
        "statutes": list(statutes)
    }
    
    return structured

def main():
    print(f"Building structured summarization dataset from: {INPUT_PATH}")
    
    total = 0
    kept = 0
    
    if not INPUT_PATH.exists():
        print(f"Error: Input file {INPUT_PATH} not found!")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

        for line in fin:
            total += 1
            try:
                case = json.loads(line)
                paragraphs = case.get("paragraphs", [])
                
                if not paragraphs:
                    continue
                    
                # Extract the full text for reference
                full_text = "\n\n".join([p["text"] for p in paragraphs])
                
                # Build the structured view
                summary = build_structured_summary(paragraphs)
                
                # Only keep if we have at least facts and reasoning (the core of a judgment)
                if len(summary["facts"]) > 100 and len(summary["reasoning"]) > 100:
                    output = {
                        "case_id": case.get("case_id"),
                        "case_type": case.get("case_type"),
                        "input_text": full_text,
                        "structured_summary": summary,
                        "meta": {
                            "num_paragraphs": len(paragraphs),
                            "role_distribution": case.get("role_distribution", {})
                        }
                    }
                    fout.write(json.dumps(output, ensure_ascii=False) + "\n")
                    kept += 1
            except Exception as e:
                print(f"Error processing case: {e}")
                continue

    print(f"\nProcessing complete:")
    print(f"  Total cases seen        : {total}")
    print(f"  Structured sets created : {kept}")
    print(f"  Output saved to         : {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
