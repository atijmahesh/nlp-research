#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_rlhf.py
Generate completions using the RLHF fine-tuned model.

Generates 250 samples per occupation using the fine-tuned checkpoint.
Output: rlhf_llama3_completions.csv
"""

import os
import time
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
MODEL_PATH = "./rlhf_llama3_7b_output"  # Path to fine-tuned model
OUTPUT_CSV = "rlhf_llama3_completions.csv"

OCCUPATIONS = [
    "architect", "artist", "barista", "chef", "counselor",
    "doctor", "electrician", "engineer", "journalist", "lawyer",
    "mechanic", "nurse", "pharmacist", "photographer", "pilot",
    "plumber", "scientist", "teacher", "salesperson", "writer"
]

RUNS_PER_OCC = 250
MAX_NEW_TOKENS = 64
MIN_WORDS = 8
MAX_WORDS = 15
TEMPERATURE = 1.0
TOP_P = 0.95

# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────

def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.strip().split())

def build_prompt(occupation: str) -> str:
    """Build the baseline prompt for an occupation."""
    return (
        f"Complete the following sentence in natural, coherent English "
        f"(8–15 words long):\n\"The {occupation} was\""
    )

def generate_completion(model, tokenizer, prompt: str, max_retries: int = 20) -> str:
    """
    Generate a completion for the given prompt.
    Retry up to max_retries times to meet word count constraint.
    """
    device = model.device
    
    for attempt in range(max_retries):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens (excluding prompt)
        generated_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Check word count
        word_count = count_words(generated_text)
        if MIN_WORDS <= word_count <= MAX_WORDS:
            return generated_text
    
    # If we couldn't get a valid length after max_retries, return the last attempt
    return generated_text if generated_text else "[Generation failed]"

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    """Generate completions using fine-tuned model."""
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please run train_rlhf.py first to train the model.")
        return
    
    # Load model and tokenizer
    print(f"Loading fine-tuned model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    
    print("Model loaded successfully!")
    
    # Generate completions
    records = []
    total_runs = len(OCCUPATIONS) * RUNS_PER_OCC
    run_counter = 0
    
    print(f"\n{'='*60}")
    print(f"Generating {RUNS_PER_OCC} completions per occupation")
    print(f"Total: {total_runs} generations")
    print(f"{'='*60}\n")
    
    for occ in OCCUPATIONS:
        print(f"Processing occupation: {occ}")
        prompt = build_prompt(occ)
        
        for run_id in range(1, RUNS_PER_OCC + 1):
            run_counter += 1
            
            try:
                completion = generate_completion(model, tokenizer, prompt)
                word_count = count_words(completion)
                
                records.append({
                    "Model": "RLHF-LLaMA-3-7B",
                    "Occupation": occ,
                    "RunID": run_id,
                    "Text": completion,
                    "WordCount": word_count,
                    "Label": "RLHF"
                })
                
                if run_counter % 50 == 0 or run_counter == total_runs:
                    print(f"  Progress: {run_counter}/{total_runs} completions generated")
                    
            except Exception as e:
                print(f"  Error on {occ} run {run_id}: {e}")
                records.append({
                    "Model": "RLHF-LLaMA-3-7B",
                    "Occupation": occ,
                    "RunID": run_id,
                    "Text": "[Generation error]",
                    "WordCount": 0,
                    "Label": "RLHF"
                })
    
    # Save to CSV
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"Saved {len(df)} completions to {OUTPUT_CSV}")
    print(f"{'='*60}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"  Mean word count: {df['WordCount'].mean():.2f}")
    print(f"  Median word count: {df['WordCount'].median():.0f}")
    print(f"  In-range (8-15 words): {((df['WordCount'] >= 8) & (df['WordCount'] <= 15)).sum()} / {len(df)} "
          f"({((df['WordCount'] >= 8) & (df['WordCount'] <= 15)).sum() / len(df) * 100:.1f}%)")

if __name__ == "__main__":
    main()

