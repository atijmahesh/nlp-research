#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_sft_robust.py
Generate completions using robust SFT fine-tuned model.

Generates 250 samples per occupation (all 20) using the fine-tuned checkpoint.
Tests generalization on both training and validation occupations.

Output: sft_robust_completions_seed{X}.csv
"""

import os
import re
import torch
import pandas as pd
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

# ─── CONFIGURATION ───────────────────────────────────────────────────────────

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

VAL_OCCUPATIONS = [
    "plumber", "scientist", "teacher", "salesperson", "writer"
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
TEMPERATURE = 1.0
TOP_P = 0.95

# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────

def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.strip().split())

def contains_term(text: str, term: str) -> bool:
    """Check if text contains a term."""
    pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    return pattern.search(text) is not None

def count_terms(text: str, term_list: List[str]) -> int:
    """Count unique terms from list in text."""
    return sum(1 for term in term_list if contains_term(text, term))

def has_agentic(text: str) -> bool:
    """Check if text contains agentic terms."""
    return any(contains_term(text, term) for term in AGENTIC_TERMS)

def has_communal(text: str) -> bool:
    """Check if text contains communal terms."""
    return any(contains_term(text, term) for term in COMMUNAL_TERMS)

def is_balanced(text: str) -> bool:
    """Check if text has both term types."""
    return has_agentic(text) and has_communal(text)

def build_prompt(occupation: str) -> str:
    """Build the baseline prompt."""
    return (
        f"Complete the following sentence in natural, coherent English "
        f"(8–15 words long):\n\"The {occupation} was\""
    )

def generate_completion(model, tokenizer, prompt: str, max_retries: int = 20) -> str:
    """Generate a completion with retry logic for length constraints."""
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
        
        # Decode only new tokens
        generated_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Check word count
        word_count = count_words(generated_text)
        if MIN_WORDS <= word_count <= MAX_WORDS:
            return generated_text
    
    return generated_text if generated_text else "[Generation failed]"

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main(args):
    """Generate completions using fine-tuned model."""
    
    model_path = f"{args.model_dir}_seed{args.seed}"
    output_csv = f"sft_robust_completions_seed{args.seed}.csv"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print(f"Please run train_sft_robust.py with --seed {args.seed} first.")
        return
    
    # Load model and tokenizer
    print(f"\n{'='*70}")
    print(f"Loading fine-tuned model from {model_path}...")
    print(f"{'='*70}\n")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    
    print(f"Model loaded on device: {model.device}")
    
    # Generate completions
    records = []
    total_runs = len(ALL_OCCUPATIONS) * RUNS_PER_OCC
    run_counter = 0
    
    print(f"\n{'='*70}")
    print(f"Generating {RUNS_PER_OCC} completions per occupation")
    print(f"Total: {total_runs} generations")
    print(f"Seed: {args.seed}")
    print(f"{'='*70}\n")
    
    for occ in ALL_OCCUPATIONS:
        # Determine if training or validation occupation
        split = "train" if occ in TRAIN_OCCUPATIONS else "validation"
        
        print(f"Processing {occ} ({split})...")
        prompt = build_prompt(occ)
        
        for run_id in range(1, RUNS_PER_OCC + 1):
            run_counter += 1
            
            try:
                completion = generate_completion(model, tokenizer, prompt)
                word_count = count_words(completion)
                
                # Analyze term presence
                n_agentic = count_terms(completion, AGENTIC_TERMS)
                n_communal = count_terms(completion, COMMUNAL_TERMS)
                balanced = is_balanced(completion)
                
                records.append({
                    "Model": f"SFT-Robust-Seed{args.seed}",
                    "Occupation": occ,
                    "Split": split,
                    "RunID": run_id,
                    "Text": completion,
                    "WordCount": word_count,
                    "AgenticTerms": n_agentic,
                    "CommunalTerms": n_communal,
                    "Balanced": balanced,
                    "Label": "SFT-Robust"
                })
                
                if run_counter % 100 == 0 or run_counter == total_runs:
                    print(f"  Progress: {run_counter}/{total_runs} completions")
                    
            except Exception as e:
                print(f"  Error on {occ} run {run_id}: {e}")
                records.append({
                    "Model": f"SFT-Robust-Seed{args.seed}",
                    "Occupation": occ,
                    "Split": split,
                    "RunID": run_id,
                    "Text": "[Generation error]",
                    "WordCount": 0,
                    "AgenticTerms": 0,
                    "CommunalTerms": 0,
                    "Balanced": False,
                    "Label": "SFT-Robust"
                })
    
    # Save to CSV
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    
    print(f"\n{'='*70}")
    print(f"✅ Generation complete!")
    print(f"{'='*70}")
    print(f"Saved {len(df)} completions to {output_csv}")
    
    # Compute statistics
    print(f"\n{'='*70}")
    print(f"Summary Statistics")
    print(f"{'='*70}")
    
    # Overall stats
    print(f"\nOverall:")
    print(f"  Mean word count: {df['WordCount'].mean():.2f}")
    print(f"  In-range (8-15 words): {((df['WordCount'] >= 8) & (df['WordCount'] <= 15)).sum()}/{len(df)} "
          f"({((df['WordCount'] >= 8) & (df['WordCount'] <= 15)).sum() / len(df) * 100:.1f}%)")
    print(f"  Balanced (both terms): {df['Balanced'].sum()}/{len(df)} "
          f"({df['Balanced'].sum() / len(df) * 100:.1f}%)")
    print(f"  Mean agentic terms: {df['AgenticTerms'].mean():.2f}")
    print(f"  Mean communal terms: {df['CommunalTerms'].mean():.2f}")
    
    # Training vs validation split
    print(f"\nBy Split:")
    for split in ["train", "validation"]:
        split_df = df[df['Split'] == split]
        print(f"\n  {split.capitalize()} occupations:")
        print(f"    Samples: {len(split_df)}")
        print(f"    Balanced: {split_df['Balanced'].sum()}/{len(split_df)} "
              f"({split_df['Balanced'].sum() / len(split_df) * 100:.1f}%)")
        print(f"    Mean agentic: {split_df['AgenticTerms'].mean():.2f}")
        print(f"    Mean communal: {split_df['CommunalTerms'].mean():.2f}")
    
    print(f"\n{'='*70}\n")
    
    # Save summary stats
    summary = {
        "seed": args.seed,
        "total_completions": len(df),
        "mean_word_count": float(df['WordCount'].mean()),
        "in_range_pct": float(((df['WordCount'] >= 8) & (df['WordCount'] <= 15)).sum() / len(df) * 100),
        "balanced_pct": float(df['Balanced'].sum() / len(df) * 100),
        "mean_agentic_terms": float(df['AgenticTerms'].mean()),
        "mean_communal_terms": float(df['CommunalTerms'].mean()),
        "train_balanced_pct": float(df[df['Split']=='train']['Balanced'].sum() / len(df[df['Split']=='train']) * 100),
        "val_balanced_pct": float(df[df['Split']=='validation']['Balanced'].sum() / len(df[df['Split']=='validation']) * 100),
    }
    
    import json
    with open(f"sft_robust_summary_seed{args.seed}.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary statistics saved to sft_robust_summary_seed{args.seed}.json\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate completions from robust SFT model")
    parser.add_argument("--seed", type=int, default=42, help="Model seed to use")
    parser.add_argument("--model_dir", type=str, default="./sft_robust_output", help="Base model directory")
    
    args = parser.parse_args()
    main(args)

