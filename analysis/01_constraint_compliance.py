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
                text = row.get('completion', row.get('Output', row.get('Text', row.get('sample', ''))))
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
            has_at_least_one = has_both + has_agentic + has_communal
            
            compliance_results[method_name] = {
                'total': total,
                'has_both': has_both,
                'has_agentic_only': has_agentic,
                'has_communal_only': has_communal,
                'has_neither': has_neither,
                'has_at_least_one': has_at_least_one,
                'compliance_rate': (has_both / total) * 100,
                'agentic_only_rate': (has_agentic / total) * 100,
                'communal_only_rate': (has_communal / total) * 100,
                'neither_rate': (has_neither / total) * 100,
                'at_least_one_rate': (has_at_least_one / total) * 100,
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
        has_at_least_one = sum(r['has_at_least_one'] for r in relevant_results)
        
        compliance_rates = [r['compliance_rate'] for r in relevant_results]
        at_least_one_rates = [r['at_least_one_rate'] for r in relevant_results]
        agentic_only_rates = [r['agentic_only_rate'] for r in relevant_results]
        communal_only_rates = [r['communal_only_rate'] for r in relevant_results]
        neither_rates = [r['neither_rate'] for r in relevant_results]
        
        aggregated[method_category] = {
            'total_samples': total,
            'total_balanced': has_both,
            'total_at_least_one': has_at_least_one,
            'mean_compliance': sum(compliance_rates) / len(compliance_rates),
            'std_compliance': 0 if len(compliance_rates) == 1 else 
                             (sum((x - sum(compliance_rates)/len(compliance_rates))**2 for x in compliance_rates) / len(compliance_rates))**0.5,
            'mean_at_least_one': sum(at_least_one_rates) / len(at_least_one_rates),
            'mean_agentic_only': sum(agentic_only_rates) / len(agentic_only_rates),
            'mean_communal_only': sum(communal_only_rates) / len(communal_only_rates),
            'mean_neither': sum(neither_rates) / len(neither_rates),
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
        writer.writerow(['Method', 'Total Samples', 'Both (AND)', '≥1 (OR)', 'Agentic Only', 'Communal Only', 'Neither',
                        'Both %', 'Std %', '≥1 %', 'N Runs'])
        
        for method, stats in sorted(aggregated.items()):
            writer.writerow([
                method,
                stats['total_samples'],
                stats['total_balanced'],
                stats['total_at_least_one'],
                f"{stats['mean_agentic_only']:.2f}%",
                f"{stats['mean_communal_only']:.2f}%",
                f"{stats['mean_neither']:.2f}%",
                f"{stats['mean_compliance']:.2f}",
                f"{stats['std_compliance']:.2f}",
                f"{stats['mean_at_least_one']:.2f}",
                stats['n_runs'],
            ])
    
    print(f"\n✅ Results saved to {output_dir}/tables/")

def print_summary(aggregated):
    """Print a formatted summary table."""
    
    print("\n" + "="*100)
    print("CONSTRAINT COMPLIANCE ANALYSIS (Agentic AND/OR Communal Terms)")
    print("="*100)
    print(f"{'Method':<15} {'Samples':>8} {'Both':>8} {'≥1':>8} {'A-only':>8} {'C-only':>8} {'Neither':>8}")
    print(f"{'':15} {'':8} {'(AND %)':>8} {'(OR %)':>8} {'(%)':>8} {'(%)':>8} {'(%)':>8}")
    print("-"*100)
    
    for method in ['Prompt-Only', 'Gen-Filter', 'Ctrl-G (OR)', 'Ctrl-G (AND)', 'SFT', 'DPO', 'INLP']:
        if method in aggregated:
            stats = aggregated[method]
            print(f"{method:<15} {stats['total_samples']:>8,} "
                  f"{stats['mean_compliance']:>8.2f} "
                  f"{stats['mean_at_least_one']:>8.2f} "
                  f"{stats['mean_agentic_only']:>8.2f} "
                  f"{stats['mean_communal_only']:>8.2f} "
                  f"{stats['mean_neither']:>8.2f}")
    
    print("="*100)
    print("Note: Both = samples with agentic AND communal, ≥1 = at least one trait (OR)")
    print("      A-only = agentic only, C-only = communal only")

if __name__ == '__main__':
    print("Starting constraint compliance analysis...")
    
    compliance_results, detailed_results = analyze_compliance()
    aggregated = aggregate_by_method(compliance_results)
    
    print_summary(aggregated)
    save_results(compliance_results, aggregated)
    
    print("\n✅ Constraint compliance analysis complete!")

