#!/usr/bin/env python3
"""
JUDGE X AI — Legal Summarization Fine-Tuning (Standard Research Method)
=====================================================================
Run this in a Kaggle Notebook with GPU T4 (single).

This version uses the standard HuggingFace 'transformers' and 'peft' 
libraries. It is slightly slower than Unsloth but 100% stable and 
free from dimension-mismatch bugs.
"""

# ════════════════════════════════════════════════════════════
# CELL 1 — Environment Setup
# ════════════════════════════════════════════════════════════
# --- Paste into Cell 1 ---
"""
import os, gc, torch

# ── 1. Environment variables ─────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"]    = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── 2. Helper ────────────────────────────────────────────
def gpu_report(tag=""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU {tag}] Allocated {alloc:.2f} GB / {total:.2f} GB")

torch.cuda.empty_cache(); gc.collect()
gpu_report("after cleanup")

# ── 3. Install Standard Libraries ────────────────────────
!pip install -q -U transformers peft accelerate bitsandbytes datasets trl
"""

# ════════════════════════════════════════════════════════════
# CELL 2 — Load Model (4-bit Quantized)
# ════════════════════════════════════════════════════════════
# --- Paste into Cell 2 ---
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" # Base model
MAX_SEQ_LEN = 512

# ── 1. Quantization Config ──────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# ── 2. Load Tokenizer ───────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── 3. Load Model ───────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.float16,
)

print(f"✅ Model loaded!")
gpu_report("after load")
"""

# ════════════════════════════════════════════════════════════
# CELL 3 — Add LoRA Adapters (Standard PEFT)
# ════════════════════════════════════════════════════════════
# --- Paste into Cell 3 ---
"""
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Prepare model for 4-bit training
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# Standard LoRA Config
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Attention only for T4 stability
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
gpu_report("after LoRA")
"""

# ════════════════════════════════════════════════════════════
# CELL 4 — Dataset Preparation (with Safety Buffer)
# ════════════════════════════════════════════════════════════
# --- Paste into Cell 4 ---
"""
import json
from datasets import Dataset

DATASET_PATH = "/kaggle/input/judgex-legal-dataset/full_dataset.jsonl"
TEMPLATE = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### System:\n{system}\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}"
EOS = tokenizer.eos_token

def tokenize_function(examples):
    texts = []
    for system, instruction, output in zip(examples["system"], examples["instruction"], examples["output"]):
        # Budget management
        out_tokens = tokenizer.encode(output, add_special_tokens=False)
        sys_tokens = tokenizer.encode(system, add_special_tokens=False)
        reserved = len(out_tokens) + len(sys_tokens) + 100 
        inst_budget = MAX_SEQ_LEN - reserved
        
        inst_tokens = tokenizer.encode(instruction, add_special_tokens=False)
        if len(inst_tokens) > inst_budget:
            inst_tokens = inst_tokens[:inst_budget]
            instruction = tokenizer.decode(inst_tokens, skip_special_tokens=True)
            
        text = TEMPLATE.format(system=system, instruction=instruction, output=output) + EOS
        texts.append(text)
    
    # Manual Tokenization
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
    )
    # For Causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Load & Split
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f]

dataset = Dataset.from_list(raw_data).train_test_split(test_size=0.1, seed=42)

# Tokenize the whole dataset at once
train_ds = dataset["train"].map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
val_ds = dataset["test"].map(tokenize_function, batched=True, remove_columns=dataset["test"].column_names)

print(f"✅ Data tokenized: {len(train_ds)} samples")
"""

# ════════════════════════════════════════════════════════════
# CELL 5 — Training Configuration
# ════════════════════════════════════════════════════════════
# --- Paste into Cell 5 ---
"""
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./judgex-stable-outputs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    max_steps=500, 
    logging_steps=10,
    fp16=True,
    optim="paged_adamw_8bit",
    save_strategy="steps",
    save_steps=100,
    report_to="none",
    remove_unused_columns=False, # Important for PEFT
)

trainer = Trainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=training_args,
)
"""

# ════════════════════════════════════════════════════════════
# CELL 6 — Run Training!
# ════════════════════════════════════════════════════════════
# --- Paste into Cell 6 ---
"""
import gc, torch
torch.cuda.empty_cache(); gc.collect()

print("🚀 Starting Stable Fine-Tuning...")
trainer.train()
print("✅ Training Finished!")
"""

# ════════════════════════════════════════════════════════════
# CELL 7 — Save & Export
# ════════════════════════════════════════════════════════════
# --- Paste into Cell 7 ---
"""
# Save the final adapter
model.save_pretrained("final_adapter")
tokenizer.save_pretrained("final_adapter")
print("✅ Adapter saved to 'final_adapter/' folder")
"""
