#!/usr/bin/env python3
"""
Script 2: Lexical Diversity Analysis
Computes Shannon entropy over synonym frequencies and path diversity.
"""

import csv
import os
import json
import math
from collections import defaultdict, Counter
from config import *

def extract_synonyms(text, synonym_list):
    """Extract all synonyms present in text (case-insensitive)."""
    text_lower = text.lower()
    found = []
    for term in synonym_list:
        if term in text_lower:
            found.append(term)
    return found

def shannon_entropy(term_counts):
    """
    Calculate Shannon entropy: H = -Σ p(x) * log2(p(x))
    Higher entropy = more diverse vocabulary
    """
    if not term_counts:
        return 0.0
    
    total = sum(term_counts.values())
    entropy = 0.0
    
    for count in term_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    
    return entropy

def analyze_diversity(results_dir='../results'):
    """Analyze lexical diversity for all methods."""
    
    diversity_results = {}
    
    for method_name, file_key in DATA_FILES.items():
        filepath = os.path.join(results_dir, file_key)
        
        if not os.path.exists(filepath):
            continue
        
        print(f"Processing {method_name}...")
        
        agentic_counts = Counter()
        communal_counts = Counter()
        path_pairs = set()  # Unique (agentic, communal) combinations
        total_samples = 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                text = row.get('completion', row.get('Output', row.get('Text', row.get('sample', ''))))
                total_samples += 1
                
                # Extract synonyms
                agentic_found = extract_synonyms(text, AGENTIC_TERMS)
                communal_found = extract_synonyms(text, COMMUNAL_TERMS)
                
                # Count occurrences
                for term in agentic_found:
                    agentic_counts[term] += 1
                
                for term in communal_found:
                    communal_counts[term] += 1
                
                # Track unique paths (all combinations)
                for a_term in agentic_found:
                    for c_term in communal_found:
                        path_pairs.add((a_term, c_term))
        
        # Calculate entropies
        agentic_entropy = shannon_entropy(agentic_counts)
        communal_entropy = shannon_entropy(communal_counts)
        
        # Calculate coverage (% of synonym list used)
        agentic_coverage = len(agentic_counts) / len(AGENTIC_TERMS) * 100
        communal_coverage = len(communal_counts) / len(COMMUNAL_TERMS) * 100
        
        # Path diversity
        path_diversity = len(path_pairs)
        
        diversity_results[method_name] = {
            'total_samples': total_samples,
            'agentic_entropy': agentic_entropy,
            'communal_entropy': communal_entropy,
            'combined_entropy': (agentic_entropy + communal_entropy) / 2,
            'agentic_coverage': agentic_coverage,
            'communal_coverage': communal_coverage,
            'agentic_unique_terms': len(agentic_counts),
            'communal_unique_terms': len(communal_counts),
            'path_diversity': path_diversity,
            'agentic_counts': dict(agentic_counts),
            'communal_counts': dict(communal_counts),
        }
    
    return diversity_results

def aggregate_by_method(diversity_results):
    """Aggregate diversity metrics by method category."""
    
    aggregated = {}
    
    for method_category, file_keys in METHOD_GROUPS.items():
        relevant_results = [diversity_results[k] for k in file_keys if k in diversity_results]
        
        if not relevant_results:
            continue
        
        # Calculate means
        agentic_entropies = [r['agentic_entropy'] for r in relevant_results]
        communal_entropies = [r['communal_entropy'] for r in relevant_results]
        combined_entropies = [r['combined_entropy'] for r in relevant_results]
        path_diversities = [r['path_diversity'] for r in relevant_results]
        
        aggregated[method_category] = {
            'mean_agentic_entropy': sum(agentic_entropies) / len(agentic_entropies),
            'mean_communal_entropy': sum(communal_entropies) / len(communal_entropies),
            'mean_combined_entropy': sum(combined_entropies) / len(combined_entropies),
            'std_combined_entropy': 0 if len(combined_entropies) == 1 else 
                                   (sum((x - sum(combined_entropies)/len(combined_entropies))**2 
                                   for x in combined_entropies) / len(combined_entropies))**0.5,
            'mean_path_diversity': sum(path_diversities) / len(path_diversities),
            'max_path_diversity': max(path_diversities),
            'n_runs': len(relevant_results),
        }
    
    return aggregated

def save_results(diversity_results, aggregated, output_dir='../analysis_results'):
    """Save results to JSON and CSV files."""
    
    os.makedirs(f'{output_dir}/tables', exist_ok=True)
    
    # Save detailed results as JSON
    with open(f'{output_dir}/tables/diversity_detailed.json', 'w') as f:
        json.dump(diversity_results, f, indent=2)
    
    # Save aggregated results as CSV
    with open(f'{output_dir}/tables/diversity_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Agentic Entropy', 'Communal Entropy', 
                        'Combined Entropy', 'Std Dev', 'Path Diversity', 'N Runs'])
        
        for method, stats in sorted(aggregated.items()):
            writer.writerow([
                method,
                f"{stats['mean_agentic_entropy']:.3f}",
                f"{stats['mean_communal_entropy']:.3f}",
                f"{stats['mean_combined_entropy']:.3f}",
                f"{stats['std_combined_entropy']:.3f}",
                f"{stats['mean_path_diversity']:.1f}",
                stats['n_runs'],
            ])
    
    print(f"\n✅ Results saved to {output_dir}/tables/")

def print_summary(aggregated):
    """Print a formatted summary table."""
    
    print("\n" + "="*80)
    print("LEXICAL DIVERSITY ANALYSIS")
    print("="*80)
    print(f"{'Method':<15} {'Agentic H':>12} {'Communal H':>12} {'Combined H':>12} {'Paths':>8}")
    print("-"*80)
    
    for method in ['Prompt-Only', 'Gen-Filter', 'Ctrl-G (OR)', 'Ctrl-G (AND)', 'SFT', 'DPO', 'INLP']:
        if method in aggregated:
            stats = aggregated[method]
            print(f"{method:<15} {stats['mean_agentic_entropy']:>12.3f} "
                  f"{stats['mean_communal_entropy']:>12.3f} "
                  f"{stats['mean_combined_entropy']:>12.3f} "
                  f"{stats['mean_path_diversity']:>8.1f}")
    
    print("="*80)
    print("Note: H = Shannon entropy (higher = more diverse vocabulary)")

if __name__ == '__main__':
    print("Starting lexical diversity analysis...")
    
    diversity_results = analyze_diversity()
    aggregated = aggregate_by_method(diversity_results)
    
    print_summary(aggregated)
    save_results(diversity_results, aggregated)
    
    print("\n✅ Lexical diversity analysis complete!")

