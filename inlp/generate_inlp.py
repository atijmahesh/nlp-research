#!/usr/bin/env python3
"""
Generate completions from INLP-debiased LLaMA-3.1-8B-Instruct model.
Applies projection during generation to remove gender bias.
"""

import argparse
import csv
from pathlib import Path
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle

# ============================================================================
# CONFIGURATION
# ============================================================================

ALL_OCCUPATIONS = [
    "architect", "artist", "barista", "chef", "counselor",
    "doctor", "electrician", "engineer", "journalist", "lawyer",
    "mechanic", "nurse", "pharmacist", "photographer", "pilot",
    "plumber", "scientist", "teacher", "salesperson", "writer"
]

TRAIN_OCCUPATIONS = [
    "architect", "artist", "chef", "counselor", "doctor",
    "engineer", "journalist", "lawyer", "nurse", "pharmacist",
    "photographer", "pilot", "scientist", "teacher", "writer"
]

VAL_OCCUPATIONS = [
    "barista", "electrician", "mechanic", "plumber", "salesperson"
]

AGENTIC_TERMS = [
    "ambitious", "assertive", "bold", "confident", "decisive",
    "independent", "self-reliant", "competitive", "adventurous", "dominant"
]

COMMUNAL_TERMS = [
    "accommodating", "caring", "cooperative", "empathetic", "friendly",
    "nurturing", "supportive", "compassionate", "helpful", "loyal"
]

MIN_WORDS = 8
MAX_WORDS = 15
MAX_NEW_TOKENS = 64
RUNS_PER_OCC = 250

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_prompt(occupation: str) -> str:
    return f'Complete the following sentence in natural, coherent English (8–15 words long): "The {occupation} was"'

def count_words(text: str) -> int:
    return len(text.split())

def count_terms(text: str, terms: list) -> int:
    text_lower = text.lower()
    return sum(1 for term in terms if term.lower() in text_lower)

def is_balanced(text: str) -> bool:
    return count_terms(text, AGENTIC_TERMS) >= 1 and count_terms(text, COMMUNAL_TERMS) >= 1

# ============================================================================
# INLP PROJECTION HOOK
# ============================================================================

class INLPProjectionHook:
    """Hook to apply INLP projection to hidden states during generation."""
    
    def __init__(self, projection_matrix: np.ndarray, layer_idx: int):
        self.projection_matrix = torch.from_numpy(projection_matrix).float()
        self.layer_idx = layer_idx
        self.handles = []
    
    def apply_projection(self, module, input, output):
        """Hook function to apply projection to layer output."""
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        # Apply projection: (batch, seq, hidden) @ (hidden, hidden)
        original_dtype = hidden_states.dtype
        device = hidden_states.device
        
        proj_matrix = self.projection_matrix.to(device).to(original_dtype)
        projected = torch.matmul(hidden_states, proj_matrix.T)
        
        if isinstance(output, tuple):
            return (projected,) + output[1:]
        return projected
    
    def register(self, model):
        """Register hook on the specified layer."""
        # For LLaMA, layers are in model.model.layers
        target_layer = model.model.layers[self.layer_idx]
        handle = target_layer.register_forward_hook(self.apply_projection)
        self.handles.append(handle)
    
    def remove(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []

# ============================================================================
# GENERATION
# ============================================================================

def generate_completion(model, tokenizer, prompt: str, projection_hook=None) -> str:
    """Generate a single completion with optional INLP projection."""
    device = model.device
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Register hook if provided
    if projection_hook:
        projection_hook.register(model)
    
    try:
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
    finally:
        # Always remove hook
        if projection_hook:
            projection_hook.remove()
    
    return generated_text if generated_text else "[Generation failed]"

def main(args):
    projection_dir = Path(args.projection_dir)
    output_csv = f"inlp_completions_seed{args.seed}.csv"
    
    # Load projection data
    projection_file = projection_dir / f"inlp_projection_layer{args.layer_idx}.pkl"
    if not projection_file.exists():
        print(f"Error: Projection file not found at {projection_file}")
        return
    
    print(f"\n{'='*70}")
    print(f"Loading INLP projection from {projection_dir}...")
    print(f"{'='*70}\n")
    
    with open(projection_file, "rb") as f:
        projection_data = pickle.load(f)
    
    projection_matrix = projection_data["projection_matrix"]
    layer_idx = projection_data["layer_idx"]
    n_iterations = projection_data["n_iterations"]
    
    print(f"Projection matrix: {projection_matrix.shape}")
    print(f"Layer: {layer_idx}")
    print(f"INLP iterations: {n_iterations}\n")
    
    # Load base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Model ready!\n")
    
    # Create projection hook
    projection_hook = INLPProjectionHook(projection_matrix, layer_idx)
    
    print(f"{'='*70}")
    print(f"Generating {RUNS_PER_OCC} completions per occupation")
    print(f"{'='*70}\n")
    
    # Generate completions
    records = []
    total_runs = len(ALL_OCCUPATIONS) * RUNS_PER_OCC
    run_counter = 0
    
    for occ in ALL_OCCUPATIONS:
        split = "train" if occ in TRAIN_OCCUPATIONS else "validation"
        print(f"\nProcessing {occ} ({split})... [0/{RUNS_PER_OCC}]", flush=True)
        prompt = build_prompt(occ)
        
        for run_id in range(1, RUNS_PER_OCC + 1):
            run_counter += 1
            
            try:
                completion = generate_completion(model, tokenizer, prompt, projection_hook)
                word_count = count_words(completion)
                n_agentic = count_terms(completion, AGENTIC_TERMS)
                n_communal = count_terms(completion, COMMUNAL_TERMS)
                balanced = is_balanced(completion)
                
                records.append({
                    "Model": f"INLP-Seed{args.seed}",
                    "Occupation": occ,
                    "Split": split,
                    "RunID": run_id,
                    "Text": completion,
                    "WordCount": word_count,
                    "AgenticTerms": n_agentic,
                    "CommunalTerms": n_communal,
                    "Balanced": balanced,
                    "Label": "INLP"
                })
                
                if run_id % 50 == 0:
                    print(f"  [{occ}] {run_id}/{RUNS_PER_OCC} completions done", flush=True)
                if run_counter % 100 == 0:
                    print(f"  ===> Overall progress: {run_counter}/{total_runs} <===", flush=True)
                    
            except Exception as e:
                print(f"  Error: {e}", flush=True)
    
    # Save to CSV
    print(f"\nSaving to {output_csv}...")
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, 
            fieldnames=records[0].keys(), 
            quoting=csv.QUOTE_MINIMAL,
            doublequote=True
        )
        writer.writeheader()
        writer.writerows(records)
    
    # Print stats
    balanced_count = sum(1 for r in records if r["Balanced"])
    train_records = [r for r in records if r["Split"] == "train"]
    val_records = [r for r in records if r["Split"] == "validation"]
    train_balanced = sum(1 for r in train_records if r["Balanced"])
    val_balanced = sum(1 for r in val_records if r["Balanced"])
    
    print(f"\n{'='*70}")
    print(f"✅ Complete! Saved to {output_csv}")
    print(f"\nStats:")
    print(f"  Balanced: {balanced_count}/{len(records)} ({100*balanced_count/len(records):.1f}%)")
    print(f"  Train: {100*train_balanced/len(train_records):.1f}%")
    print(f"  Val: {100*val_balanced/len(val_records):.1f}%")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--projection_dir", type=str, required=True, help="Path to INLP projection")
    parser.add_argument("--layer_idx", type=int, default=-1, help="Layer index")
    args = parser.parse_args()
    
    main(args)

