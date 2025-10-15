#!/usr/bin/env python3
"""
Script 4: Statistical Tests
Performs Wilcoxon signed-rank tests, Cohen's d, and bootstrap confidence intervals.
"""

import csv
import os
import json
import numpy as np
from scipy import stats
from collections import defaultdict
from config import *

def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size.
    d = (mean1 - mean2) / pooled_std
    
    |d| < 0.2: small effect
    |d| < 0.5: medium effect
    |d| >= 0.8: large effect
    """
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    d = (mean1 - mean2) / pooled_std
    return d

def bootstrap_ci(data, n_bootstrap=1000, ci=95):
    """
    Calculate bootstrap confidence interval.
    Returns (lower, upper) bounds.
    """
    means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    
    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    
    return lower, upper

def load_compliance_rates():
    """Load compliance rates from previous analysis."""
    
    compliance_file = '../analysis_results/tables/compliance_detailed.json'
    
    if not os.path.exists(compliance_file):
        print("⚠️  Warning: Run 01_constraint_compliance.py first!")
        return {}
    
    with open(compliance_file, 'r') as f:
        data = json.load(f)
    
    # Extract compliance rates per method
    compliance_rates = {}
    for method, stats in data.items():
        compliance_rates[method] = stats['compliance_rate']
    
    return compliance_rates

def perform_pairwise_tests(compliance_rates):
    """
    Perform pairwise comparisons between methods.
    Since we only have compliance rates (not individual samples),
    we'll compare across seeds for fine-tuned methods.
    """
    
    # Group by method
    method_data = defaultdict(list)
    
    for method_key, rate in compliance_rates.items():
        # Extract base method name
        if 'sft' in method_key:
            method_data['SFT'].append(rate)
        elif 'dpo' in method_key:
            method_data['DPO'].append(rate)
        elif 'inlp' in method_key:
            method_data['INLP'].append(rate)
        elif 'prompt_only' in method_key:
            method_data['Prompt-Only'].append(rate)
        elif 'genfilter' in method_key:
            method_data['Gen-Filter'].append(rate)
        elif 'ctrlg' in method_key:
            method_data['Ctrl-G'].append(rate)
    
    # Perform pairwise comparisons
    comparisons = []
    methods = ['Prompt-Only', 'Gen-Filter', 'Ctrl-G', 'SFT', 'DPO', 'INLP']
    
    for i, method1 in enumerate(methods):
        if method1 not in method_data:
            continue
        
        for method2 in methods[i+1:]:
            if method2 not in method_data:
                continue
            
            data1 = method_data[method1]
            data2 = method_data[method2]
            
            # Skip if not enough samples for statistical test
            if len(data1) < 2 or len(data2) < 2:
                continue
            
            # Wilcoxon test (for paired comparisons) or Mann-Whitney (independent)
            # Since different methods have different samples, use Mann-Whitney
            statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            
            # Cohen's d
            effect_size = cohens_d(data1, data2)
            
            comparisons.append({
                'method1': method1,
                'method2': method2,
                'mean1': np.mean(data1),
                'mean2': np.mean(data2),
                'diff': np.mean(data1) - np.mean(data2),
                'statistic': statistic,
                'p_value': p_value,
                'cohens_d': effect_size,
                'significant': p_value < 0.05,
            })
    
    return comparisons, method_data

def calculate_confidence_intervals(method_data):
    """Calculate bootstrap confidence intervals for each method."""
    
    ci_results = {}
    
    for method, rates in method_data.items():
        if len(rates) >= 2:
            lower, upper = bootstrap_ci(rates, n_bootstrap=1000)
            ci_results[method] = {
                'mean': np.mean(rates),
                'ci_lower': lower,
                'ci_upper': upper,
                'n': len(rates),
            }
        else:
            ci_results[method] = {
                'mean': rates[0] if rates else 0,
                'ci_lower': None,
                'ci_upper': None,
                'n': len(rates),
            }
    
    return ci_results

def save_results(comparisons, ci_results, output_dir='../analysis_results'):
    """Save statistical test results."""
    
    os.makedirs(f'{output_dir}/tables', exist_ok=True)
    
    # Save pairwise comparisons
    with open(f'{output_dir}/tables/pairwise_comparisons.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method 1', 'Method 2', 'Mean 1', 'Mean 2', 'Difference', 
                        'P-Value', "Cohen's d", 'Significant', 'Effect Size'])
        
        for comp in comparisons:
            effect_label = 'Large' if abs(comp['cohens_d']) >= 0.8 else \
                          'Medium' if abs(comp['cohens_d']) >= 0.5 else \
                          'Small' if abs(comp['cohens_d']) >= 0.2 else 'Negligible'
            
            writer.writerow([
                comp['method1'],
                comp['method2'],
                f"{comp['mean1']:.2f}",
                f"{comp['mean2']:.2f}",
                f"{comp['diff']:.2f}",
                f"{comp['p_value']:.4f}",
                f"{comp['cohens_d']:.3f}",
                'Yes' if comp['significant'] else 'No',
                effect_label,
            ])
    
    # Save confidence intervals
    with open(f'{output_dir}/tables/confidence_intervals.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Mean', '95% CI Lower', '95% CI Upper', 'N'])
        
        for method, stats in sorted(ci_results.items()):
            writer.writerow([
                method,
                f"{stats['mean']:.2f}",
                f"{stats['ci_lower']:.2f}" if stats['ci_lower'] else 'N/A',
                f"{stats['ci_upper']:.2f}" if stats['ci_upper'] else 'N/A',
                stats['n'],
            ])
    
    print(f"\n✅ Results saved to {output_dir}/tables/")

def print_summary(comparisons, ci_results):
    """Print formatted summary of statistical tests."""
    
    print("\n" + "="*80)
    print("STATISTICAL TESTS: Pairwise Comparisons")
    print("="*80)
    print(f"{'Method 1':<15} {'Method 2':<15} {'Diff':>8} {'P-Value':>10} {'d':>8} {'Sig':>5}")
    print("-"*80)
    
    for comp in comparisons:
        sig_marker = '***' if comp['p_value'] < 0.001 else \
                     '**' if comp['p_value'] < 0.01 else \
                     '*' if comp['p_value'] < 0.05 else 'ns'
        
        print(f"{comp['method1']:<15} {comp['method2']:<15} "
              f"{comp['diff']:>8.2f} {comp['p_value']:>10.4f} "
              f"{comp['cohens_d']:>8.3f} {sig_marker:>5}")
    
    print("\n" + "="*80)
    print("CONFIDENCE INTERVALS (95%)")
    print("="*80)
    print(f"{'Method':<15} {'Mean':>10} {'CI Lower':>10} {'CI Upper':>10}")
    print("-"*80)
    
    for method in ['Prompt-Only', 'Gen-Filter', 'Ctrl-G', 'SFT', 'DPO', 'INLP']:
        if method in ci_results:
            stats = ci_results[method]
            ci_lower_str = f"{stats['ci_lower']:.2f}" if stats['ci_lower'] else 'N/A'
            ci_upper_str = f"{stats['ci_upper']:.2f}" if stats['ci_upper'] else 'N/A'
            
            print(f"{method:<15} {stats['mean']:>10.2f} {ci_lower_str:>10} {ci_upper_str:>10}")
    
    print("="*80)
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")

if __name__ == '__main__':
    print("Starting statistical tests...")
    
    compliance_rates = load_compliance_rates()
    
    if not compliance_rates:
        print("❌ Error: No compliance data found. Run 01_constraint_compliance.py first!")
        exit(1)
    
    comparisons, method_data = perform_pairwise_tests(compliance_rates)
    ci_results = calculate_confidence_intervals(method_data)
    
    print_summary(comparisons, ci_results)
    save_results(comparisons, ci_results)
    
    print("\n✅ Statistical tests complete!")

