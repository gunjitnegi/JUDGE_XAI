import sys
sys.path.insert(0, r"C:\final_year\JUDGEXAI\venv\Lib\site-packages")
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 1. Configuration
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
# Ensure this path is correct relative to the script location
ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "../../models/legal_llm_adapter")

print("--- JUDGE X AI LOCAL TEST ---")

# 2. Check for CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device detected: {device.upper()}")

if device == "cpu":
    print("⚠️ WARNING: No GPU detected. Loading in 4-bit is NOT possible on CPU.")
    print("Loading base model in FP16 (this might take a lot of RAM)...")
    load_args = {"torch_dtype": torch.float16}
else:
    print("✅ GPU detected. Loading in 4-bit...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    load_args = {"quantization_config": quantization_config, "device_map": "auto"}

try:
    print(f"⏳ Loading Tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    print(f"⏳ Loading Base Model...")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **load_args)

    print(f"⚖️ Attaching Legal Adapter from: {ADAPTER_PATH}")
    if not os.path.exists(ADAPTER_PATH):
        print(f"❌ ERROR: Adapter path not found: {ADAPTER_PATH}")
        sys.exit(1)
        
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    
    if device == "cpu":
        model = model.to("cpu")

    # 3. Extract text from the target PDF file
    pdf_path = r"c:\final_year\JUDGEXAI\test_pdf\civil_001_VINISHMA TECHNOLOGIES PVT_ LTD_versusSTATE OF CHHATTISGARH _ ANR_-_2025_ 10 S_C_.pdf"
    print(f"📄 Extracting text from PDF: {pdf_path}")
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    
    # Extract first 5 pages to stay safely within context limits (approx 3000 words)
    test_judgment_pages = []
    for page_num in range(min(5, len(doc))):
        test_judgment_pages.append(doc[page_num].get_text())
    test_judgment = "\n".join(test_judgment_pages).strip()
    print(f"✅ Extracted {len(test_judgment.split())} words from the first {min(5, len(doc))} pages.")

    # 4. Construct EXACT Prompt used during Fine-Tuning
    SYSTEM_PROMPT = """You are JUDGE X AI, an expert legal analyst specializing in Indian court judgments. 
Your task is to produce a structured summary of a court judgment by analyzing the provided text.
You must identify and separate: Facts, Issues, Arguments, Reasoning, and the Final Decision.
Be precise, use legal terminology, and cite specific statutes or articles when mentioned."""

    INSTRUCTION = f"""Analyze the following Indian court judgment and produce a structured summary with these sections:
1. **Facts**: Key background events and case history
2. **Issues**: Legal questions before the court
3. **Arguments**: What each side argued
4. **Reasoning**: The court's legal analysis and interpretation
5. **Decision**: The final order and directions

Judgment Text:
{test_judgment}"""

    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### System:\n{SYSTEM_PROMPT}\n\n### Instruction:\n{INSTRUCTION}\n\n### Response:\n"

    print("🚀 Generating Summary (this may take a few minutes)...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=1024, 
            temperature=0.3,
            do_sample=True,
            repetition_penalty=1.15
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n" + "="*40)
    print("FINAL LEGAL SUMMARY:")
    print("="*40)
    # Extract just the response part
    if "### Response:" in response:
        print(response.split("### Response:")[-1].strip())
    else:
        print(response)

except Exception as e:
    print(f"\n❌ CRITICAL ERROR: {str(e)}")
