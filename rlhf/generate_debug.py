#!/usr/bin/env python3
"""Quick debug script to test single generation"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained("./sft_lora_paper_seed42")
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...", flush=True)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
)

print("Loading LoRA...", flush=True)
model = PeftModel.from_pretrained(base_model, "./sft_lora_paper_seed42")

print("Merging...", flush=True)
model = model.merge_and_unload()
model.eval()

print("Testing generation...", flush=True)
prompt = "The engineer was"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print(f"Input device: {inputs.input_ids.device}", flush=True)
print(f"Model device: {model.device}", flush=True)
print("Generating...", flush=True)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        temperature=1.0,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nResult: {result}", flush=True)
print("SUCCESS!", flush=True)

