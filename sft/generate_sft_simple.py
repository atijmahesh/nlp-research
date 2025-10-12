#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_sft_simple.py
Simplified generation that merges LoRA weights to avoid loading issues
"""

import os
import re
import torch
import pandas as pd
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import List

ALL_OCCUPATIONS = [
    "architect", "artist", "barista", "chef", "counselor",
    "doctor", "electrician", "engineer", "journalist", "lawyer",
    "mechanic", "nurse", "pharmacist", "photographer", "pilot",
    "plumber", "scientist", "teacher", "salesperson", "writer"
]

TRAIN_OCCUPATIONS = [
    "architect", "artist", "barista", "chef", "counselor",
    "doctor", "electrician", "engineer", "journalist", "lawyer",
    "mechanic", "nurse", "pharmacist", "photographer", "pilot"
]

AGENTIC_TERMS = [
    "ambitious", "assertive", "bold", "confident", "decisive",
    "independent", "self-reliant", "competitive", "adventurous", "dominant"
]

COMMUNAL_TERMS = [
    "accommodating", "caring", "cooperative", "empathetic", "friendly",
    "nurturing", "supportive", "compassionate", "helpful", "loyal"
]

RUNS_PER_OCC = 250
MAX_NEW_TOKENS = 64
MIN_WORDS = 8
MAX_WORDS = 15

def count_words(text: str) -> int:
    return len(text.strip().split())

def contains_term(text: str, term: str) -> bool:
    pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    return pattern.search(text) is not None

def count_terms(text: str, term_list: List[str]) -> int:
    return sum(1 for term in term_list if contains_term(text, term))

def is_balanced(text: str) -> bool:
    has_agentic = any(contains_term(text, term) for term in AGENTIC_TERMS)
    has_communal = any(contains_term(text, term) for term in COMMUNAL_TERMS)
    return has_agentic and has_communal

def build_prompt(occupation: str) -> str:
    return f"Complete the following sentence in natural, coherent English (8–15 words long):\n\"The {occupation} was\""

def generate_completion(model, tokenizer, prompt: str) -> str:
    """Generate completion - accept any length for speed"""
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=1.0,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()
    
    return generated_text if generated_text else "[Generation failed]"

def main(args):
    model_path = args.model_dir  # Use path exactly as provided
    output_csv = f"sft_lora_completions_seed{args.seed}.csv"
    
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"Loading model from {model_path}...")
    print(f"{'='*70}\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model with lower memory
    base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,  # Use float16 to save memory
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    # Load LoRA and merge
    print("Loading and merging LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # Merge LoRA weights into base model (saves memory during inference)
    print("Merging weights...")
    model = model.merge_and_unload()
    model.eval()
    
    print("Model ready!\n")
    
    # Generate completions
    records = []
    total_runs = len(ALL_OCCUPATIONS) * RUNS_PER_OCC
    run_counter = 0
    
    print(f"{'='*70}")
    print(f"Generating {RUNS_PER_OCC} completions per occupation")
    print(f"{'='*70}\n")
    
    for occ in ALL_OCCUPATIONS:
        split = "train" if occ in TRAIN_OCCUPATIONS else "validation"
        print(f"\nProcessing {occ} ({split})... [0/{RUNS_PER_OCC}]", flush=True)
        prompt = build_prompt(occ)
        
        for run_id in range(1, RUNS_PER_OCC + 1):
            run_counter += 1
            
            try:
                print(f"  Generating {occ} #{run_id}...", flush=True)
                completion = generate_completion(model, tokenizer, prompt)
                word_count = count_words(completion)
                n_agentic = count_terms(completion, AGENTIC_TERMS)
                n_communal = count_terms(completion, COMMUNAL_TERMS)
                balanced = is_balanced(completion)
                
                print(f"  ✓ {occ} #{run_id}: {word_count}w, balanced={balanced}", flush=True)
                
                records.append({
                    "Model": f"SFT-LoRA-Seed{args.seed}",
                    "Occupation": occ,
                    "Split": split,
                    "RunID": run_id,
                    "Text": completion,
                    "WordCount": word_count,
                    "AgenticTerms": n_agentic,
                    "CommunalTerms": n_communal,
                    "Balanced": balanced,
                    "Label": "SFT-LoRA"
                })
                
                if run_id % 50 == 0:
                    print(f"  [{occ}] {run_id}/{RUNS_PER_OCC} completions done", flush=True)
                if run_counter % 100 == 0:
                    print(f"  ===> Overall progress: {run_counter}/{total_runs} <===", flush=True)
                    
            except Exception as e:
                print(f"  Error: {e}", flush=True)
    
    # Save
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    
    print(f"\n{'='*70}")
    print(f"✅ Complete! Saved to {output_csv}")
    print(f"\nStats:")
    print(f"  Balanced: {df['Balanced'].sum()}/{len(df)} ({df['Balanced'].sum()/len(df)*100:.1f}%)")
    print(f"  Train: {df[df['Split']=='train']['Balanced'].sum()/len(df[df['Split']=='train'])*100:.1f}%")
    print(f"  Val: {df[df['Split']=='validation']['Balanced'].sum()/len(df[df['Split']=='validation'])*100:.1f}%")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_dir", type=str, default="./sft_lora_paper")
    args = parser.parse_args()
    main(args)

