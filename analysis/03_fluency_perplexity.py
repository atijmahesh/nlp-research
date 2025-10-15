#!/usr/bin/env python3
"""
Script 3: Fluency Analysis via Perplexity
Computes perplexity under a reference LM (GPT-2) as a fluency proxy.
"""

import csv
import os
import json
import torch
from collections import defaultdict
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
from config import *

def calculate_perplexity(text, model, tokenizer, device='cuda'):
    """
    Calculate perplexity of text under the model.
    Lower perplexity = more fluent/natural text.
    """
    try:
        encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        input_ids = encodings.input_ids.to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        
        return perplexity
    
    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        return None

def analyze_fluency(results_dir='../results', batch_size=32):
    """Analyze fluency for all methods using GPT-2 perplexity."""
    
    # Load GPT-2 model
    print("Loading GPT-2 model for perplexity calculation...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    
    fluency_results = {}
    
    for method_name, file_key in DATA_FILES.items():
        filepath = os.path.join(results_dir, file_key)
        
        if not os.path.exists(filepath):
            continue
        
        print(f"\nProcessing {method_name}...")
        
        perplexities = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Process with progress bar
        for row in tqdm(rows, desc=f"{method_name}"):
            text = row.get('completion', row.get('Output', row.get('Text', '')))
            
            if not text.strip():
                continue
            
            ppl = calculate_perplexity(text, model, tokenizer, device)
            
            if ppl is not None and ppl < 10000:  # Filter out extreme outliers
                perplexities.append(ppl)
        
        if perplexities:
            # Calculate statistics
            mean_ppl = sum(perplexities) / len(perplexities)
            median_ppl = sorted(perplexities)[len(perplexities) // 2]
            min_ppl = min(perplexities)
            max_ppl = max(perplexities)
            
            # Standard deviation
            variance = sum((x - mean_ppl) ** 2 for x in perplexities) / len(perplexities)
            std_ppl = variance ** 0.5
            
            fluency_results[method_name] = {
                'n_samples': len(perplexities),
                'mean_perplexity': mean_ppl,
                'median_perplexity': median_ppl,
                'std_perplexity': std_ppl,
                'min_perplexity': min_ppl,
                'max_perplexity': max_ppl,
                'perplexities': perplexities,  # Store for statistical tests
            }
            
            print(f"  Mean PPL: {mean_ppl:.2f}, Median PPL: {median_ppl:.2f}")
    
    return fluency_results

def aggregate_by_method(fluency_results):
    """Aggregate fluency metrics by method category."""
    
    aggregated = {}
    
    for method_category, file_keys in METHOD_GROUPS.items():
        relevant_results = [fluency_results[k] for k in file_keys if k in fluency_results]
        
        if not relevant_results:
            continue
        
        # Calculate means across runs
        mean_ppls = [r['mean_perplexity'] for r in relevant_results]
        median_ppls = [r['median_perplexity'] for r in relevant_results]
        
        aggregated[method_category] = {
            'mean_perplexity': sum(mean_ppls) / len(mean_ppls),
            'std_mean_perplexity': 0 if len(mean_ppls) == 1 else 
                                  (sum((x - sum(mean_ppls)/len(mean_ppls))**2 
                                  for x in mean_ppls) / len(mean_ppls))**0.5,
            'median_perplexity': sum(median_ppls) / len(median_ppls),
            'min_perplexity': min(mean_ppls),
            'max_perplexity': max(mean_ppls),
            'n_runs': len(relevant_results),
        }
    
    return aggregated

def save_results(fluency_results, aggregated, output_dir='../analysis_results'):
    """Save results to JSON and CSV files."""
    
    os.makedirs(f'{output_dir}/tables', exist_ok=True)
    
    # Save detailed results (without raw perplexity arrays for size)
    fluency_summary = {k: {key: val for key, val in v.items() if key != 'perplexities'} 
                       for k, v in fluency_results.items()}
    
    with open(f'{output_dir}/tables/fluency_detailed.json', 'w') as f:
        json.dump(fluency_summary, f, indent=2)
    
    # Save aggregated results as CSV
    with open(f'{output_dir}/tables/fluency_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Mean Perplexity', 'Std Dev', 'Median', 'Min', 'Max', 'N Runs'])
        
        for method, stats in sorted(aggregated.items()):
            writer.writerow([
                method,
                f"{stats['mean_perplexity']:.2f}",
                f"{stats['std_mean_perplexity']:.2f}",
                f"{stats['median_perplexity']:.2f}",
                f"{stats['min_perplexity']:.2f}",
                f"{stats['max_perplexity']:.2f}",
                stats['n_runs'],
            ])
    
    print(f"\n✅ Results saved to {output_dir}/tables/")

def print_summary(aggregated):
    """Print a formatted summary table."""
    
    print("\n" + "="*80)
    print("FLUENCY ANALYSIS (GPT-2 Perplexity)")
    print("="*80)
    print(f"{'Method':<15} {'Mean PPL':>12} {'Std':>10} {'Median':>10} {'Min':>10} {'Max':>10}")
    print("-"*80)
    
    for method in ['Prompt-Only', 'Gen-Filter', 'Ctrl-G', 'SFT', 'DPO', 'INLP']:
        if method in aggregated:
            stats = aggregated[method]
            print(f"{method:<15} {stats['mean_perplexity']:>12.2f} "
                  f"{stats['std_mean_perplexity']:>10.2f} "
                  f"{stats['median_perplexity']:>10.2f} "
                  f"{stats['min_perplexity']:>10.2f} "
                  f"{stats['max_perplexity']:>10.2f}")
    
    print("="*80)
    print("Note: Lower perplexity = more fluent/natural text")

if __name__ == '__main__':
    import sys
    
    print("Starting fluency analysis...")
    print("⚠️  This will take a while (needs to load GPT-2 and process all samples)")
    
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available. This will be slow on CPU.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    fluency_results = analyze_fluency()
    aggregated = aggregate_by_method(fluency_results)
    
    print_summary(aggregated)
    save_results(fluency_results, aggregated)
    
    print("\n✅ Fluency analysis complete!")

