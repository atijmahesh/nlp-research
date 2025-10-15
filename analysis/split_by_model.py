#!/usr/bin/env python3
"""
Split combined prompt-only and gen-filter CSVs by individual model.
"""

import csv
import os
from collections import defaultdict

def split_csv_by_model(input_file, output_dir, method_name):
    """Split a CSV file by model into separate files."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Read and group by model
    model_data = defaultdict(list)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            # Get model name (handle different capitalizations)
            model = row.get('model', row.get('Model', ''))
            model_data[model].append(row)
    
    # Write separate files for each model
    for model, rows in model_data.items():
        # Sanitize model name for filename
        model_safe = model.replace('/', '_').replace('-', '_')
        output_file = f"{output_dir}/{method_name}_{model_safe}.csv"
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"✅ Created {output_file} with {len(rows)} records")
    
    return list(model_data.keys())

def main():
    base_dir = '../results'
    
    print("="*80)
    print("SPLITTING CSVs BY MODEL")
    print("="*80)
    
    # Split prompt-only combined files
    print("\n1. Splitting prompt-only files...")
    
    prompt_gpt4o = f'{base_dir}/prompt-only/prompt_only_gpt4o_combined.csv'
    if os.path.exists(prompt_gpt4o):
        models = split_csv_by_model(prompt_gpt4o, f'{base_dir}/prompt-only', 'prompt_only')
        print(f"   GPT-4o models: {models}")
    
    prompt_llama = f'{base_dir}/prompt-only/prompt_only_llama_combined.csv'
    if os.path.exists(prompt_llama):
        models = split_csv_by_model(prompt_llama, f'{base_dir}/prompt-only', 'prompt_only')
        print(f"   LLaMA models: {models}")
    
    # Split gen-filter raw files
    print("\n2. Splitting gen-filter files...")
    
    genfilter_gpt4o = f'{base_dir}/gen-filter/genfilter_gpt4o_raw.csv'
    if os.path.exists(genfilter_gpt4o):
        models = split_csv_by_model(genfilter_gpt4o, f'{base_dir}/gen-filter', 'genfilter')
        print(f"   GPT-4o models: {models}")
    
    genfilter_llama = f'{base_dir}/gen-filter/genfilter_llama_raw.csv'
    if os.path.exists(genfilter_llama):
        models = split_csv_by_model(genfilter_llama, f'{base_dir}/gen-filter', 'genfilter')
        print(f"   LLaMA models: {models}")
    
    print("\n" + "="*80)
    print("✅ Split complete! Update config.py with new file paths.")
    print("="*80)

if __name__ == '__main__':
    main()

