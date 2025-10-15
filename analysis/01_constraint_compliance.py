#!/usr/bin/env python3
"""
Script 1: Constraint Compliance Analysis
Computes the percentage of samples containing both agentic and communal terms.
"""

import csv
import os
import json
from collections import defaultdict
from config import *

def has_term(text, terms):
    """Check if text contains any term from the list (case-insensitive)."""
    text_lower = text.lower()
    return any(term in text_lower for term in terms)

def analyze_compliance(results_dir='../results'):
    """Analyze constraint compliance for all methods."""
    
    compliance_results = {}
    detailed_results = defaultdict(lambda: defaultdict(list))
    
    for method_name, file_key in DATA_FILES.items():
        filepath = os.path.join(results_dir, file_key)
        
        if not os.path.exists(filepath):
            print(f"⚠️  Warning: {filepath} not found, skipping...")
            continue
        
        print(f"Processing {method_name}...")
        
        total = 0
        has_agentic = 0
        has_communal = 0
        has_both = 0
        has_neither = 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Get completion text (handle different column names)
                text = row.get('completion', row.get('Output', row.get('Text', '')))
                occupation = row.get('occupation', row.get('Occupation', ''))
                
                total += 1
                
                agentic_present = has_term(text, AGENTIC_TERMS)
                communal_present = has_term(text, COMMUNAL_TERMS)
                
                if agentic_present and communal_present:
                    has_both += 1
                    detailed_results[method_name][occupation].append('both')
                elif agentic_present:
                    has_agentic += 1
                    detailed_results[method_name][occupation].append('agentic_only')
                elif communal_present:
                    has_communal += 1
                    detailed_results[method_name][occupation].append('communal_only')
                else:
                    has_neither += 1
                    detailed_results[method_name][occupation].append('neither')
        
        if total > 0:
            compliance_results[method_name] = {
                'total': total,
                'has_both': has_both,
                'has_agentic_only': has_agentic,
                'has_communal_only': has_communal,
                'has_neither': has_neither,
                'compliance_rate': (has_both / total) * 100,
                'agentic_rate': ((has_agentic + has_both) / total) * 100,
                'communal_rate': ((has_communal + has_both) / total) * 100,
            }
    
    return compliance_results, detailed_results

def aggregate_by_method(compliance_results):
    """Aggregate results by method category (averaging across seeds)."""
    
    aggregated = {}
    
    for method_category, file_keys in METHOD_GROUPS.items():
        relevant_results = [compliance_results[k] for k in file_keys if k in compliance_results]
        
        if not relevant_results:
            continue
        
        # Calculate means
        total = sum(r['total'] for r in relevant_results)
        has_both = sum(r['has_both'] for r in relevant_results)
        compliance_rates = [r['compliance_rate'] for r in relevant_results]
        
        aggregated[method_category] = {
            'total_samples': total,
            'total_balanced': has_both,
            'mean_compliance': sum(compliance_rates) / len(compliance_rates),
            'std_compliance': 0 if len(compliance_rates) == 1 else 
                             (sum((x - sum(compliance_rates)/len(compliance_rates))**2 for x in compliance_rates) / len(compliance_rates))**0.5,
            'min_compliance': min(compliance_rates),
            'max_compliance': max(compliance_rates),
            'n_runs': len(relevant_results),
        }
    
    return aggregated

def save_results(compliance_results, aggregated, output_dir='../analysis_results'):
    """Save results to JSON and CSV files."""
    
    os.makedirs(f'{output_dir}/tables', exist_ok=True)
    
    # Save detailed results as JSON
    with open(f'{output_dir}/tables/compliance_detailed.json', 'w') as f:
        json.dump(compliance_results, f, indent=2)
    
    # Save aggregated results as CSV
    with open(f'{output_dir}/tables/compliance_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Total Samples', 'Balanced Samples', 
                        'Mean Compliance (%)', 'Std Dev (%)', 'Min (%)', 'Max (%)', 'N Runs'])
        
        for method, stats in sorted(aggregated.items()):
            writer.writerow([
                method,
                stats['total_samples'],
                stats['total_balanced'],
                f"{stats['mean_compliance']:.2f}",
                f"{stats['std_compliance']:.2f}",
                f"{stats['min_compliance']:.2f}",
                f"{stats['max_compliance']:.2f}",
                stats['n_runs'],
            ])
    
    print(f"\n✅ Results saved to {output_dir}/tables/")

def print_summary(aggregated):
    """Print a formatted summary table."""
    
    print("\n" + "="*80)
    print("CONSTRAINT COMPLIANCE ANALYSIS")
    print("="*80)
    print(f"{'Method':<15} {'Samples':>10} {'Balanced':>10} {'Mean (%)':>10} {'Std (%)':>10}")
    print("-"*80)
    
    for method in ['Prompt-Only', 'Gen-Filter', 'Ctrl-G', 'SFT', 'DPO', 'INLP']:
        if method in aggregated:
            stats = aggregated[method]
            print(f"{method:<15} {stats['total_samples']:>10,} {stats['total_balanced']:>10,} "
                  f"{stats['mean_compliance']:>10.2f} {stats['std_compliance']:>10.2f}")
    
    print("="*80)

if __name__ == '__main__':
    print("Starting constraint compliance analysis...")
    
    compliance_results, detailed_results = analyze_compliance()
    aggregated = aggregate_by_method(compliance_results)
    
    print_summary(aggregated)
    save_results(compliance_results, aggregated)
    
    print("\n✅ Constraint compliance analysis complete!")

