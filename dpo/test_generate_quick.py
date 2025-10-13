#!/usr/bin/env python3
"""Quick test: generate 10 completions and save to CSV."""

import csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_DIR = "./dpo_lora_paper_seed456"
MAX_NEW_TOKENS = 64

print("Loading model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token

print("Generating 10 test completions...")
records = []

for i in range(10):
    prompt = 'Complete the following sentence in natural, coherent English (8–15 words long): "The architect was"'
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=1.0,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    records.append({"RunID": i+1, "Text": text, "Balanced": False})
    print(f"  {i+1}. {text[:60]}...")

print("\nSaving to test_output.csv...")
try:
    with open("test_dpo_output.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, 
            fieldnames=records[0].keys(), 
            quoting=csv.QUOTE_MINIMAL,
            doublequote=True
        )
        writer.writeheader()
        writer.writerows(records)
    print("✅ Success! CSV saved.")
except Exception as e:
    print(f"❌ Failed: {e}")

