#!/usr/bin/env python3
"""
Script 5: Visualizations
Creates publication-quality plots for the paper.
"""

import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import *

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

def load_data(output_dir='../analysis_results'):
    """Load analysis results from JSON files."""
    
    data = {}
    
    files = {
        'compliance': f'{output_dir}/tables/compliance_detailed.json',
        'diversity': f'{output_dir}/tables/diversity_detailed.json',
        'fluency': f'{output_dir}/tables/fluency_detailed.json',
    }
    
    for key, filepath in files.items():
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data[key] = json.load(f)
        else:
            print(f"⚠️  Warning: {filepath} not found")
    
    return data

def aggregate_for_plotting(data):
    """Aggregate data by method category for plotting."""
    
    aggregated = {
        'Prompt-Only': {'compliance': [], 'diversity': [], 'fluency': []},
        'Gen-Filter': {'compliance': [], 'diversity': [], 'fluency': []},
        'Ctrl-G': {'compliance': [], 'diversity': [], 'fluency': []},
        'SFT': {'compliance': [], 'diversity': [], 'fluency': []},
        'DPO': {'compliance': [], 'diversity': [], 'fluency': []},
        'INLP': {'compliance': [], 'diversity': [], 'fluency': []},
    }
    
    # Map file keys to method categories
    method_mapping = {}
    for category, file_keys in METHOD_GROUPS.items():
        for key in file_keys:
            method_mapping[key] = category
    
    # Extract compliance rates
    if 'compliance' in data:
        for method_key, stats in data['compliance'].items():
            if method_key in method_mapping:
                category = method_mapping[method_key]
                aggregated[category]['compliance'].append(stats['compliance_rate'])
    
    # Extract diversity metrics
    if 'diversity' in data:
        for method_key, stats in data['diversity'].items():
            if method_key in method_mapping:
                category = method_mapping[method_key]
                aggregated[category]['diversity'].append(stats['combined_entropy'])
    
    # Extract fluency metrics
    if 'fluency' in data:
        for method_key, stats in data['fluency'].items():
            if method_key in method_mapping:
                category = method_mapping[method_key]
                aggregated[category]['fluency'].append(stats['mean_perplexity'])
    
    return aggregated

def plot_compliance_comparison(aggregated, output_dir):
    """Bar chart comparing constraint compliance across methods."""
    
    methods = ['Prompt-Only', 'Gen-Filter', 'Ctrl-G', 'SFT', 'DPO', 'INLP']
    means = []
    stds = []
    
    for method in methods:
        if aggregated[method]['compliance']:
            means.append(np.mean(aggregated[method]['compliance']))
            stds.append(np.std(aggregated[method]['compliance']))
        else:
            means.append(0)
            stds.append(0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c', '#1abc9c']
    bars = ax.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 2,
                f'{mean:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Constraint Compliance (%)', fontweight='bold')
    ax.set_xlabel('Method', fontweight='bold')
    ax.set_title('Constraint Compliance: % Samples with Both Agentic & Communal Terms', 
                 fontweight='bold', pad=15)
    ax.set_ylim(0, 110)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/compliance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: compliance_comparison.png")

def plot_diversity_comparison(aggregated, output_dir):
    """Bar chart comparing lexical diversity across methods."""
    
    methods = ['Prompt-Only', 'Gen-Filter', 'Ctrl-G', 'SFT', 'DPO', 'INLP']
    means = []
    stds = []
    
    for method in methods:
        if aggregated[method]['diversity']:
            means.append(np.mean(aggregated[method]['diversity']))
            stds.append(np.std(aggregated[method]['diversity']))
        else:
            means.append(0)
            stds.append(0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c', '#1abc9c']
    bars = ax.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.05,
                f'{mean:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Shannon Entropy (Combined)', fontweight='bold')
    ax.set_xlabel('Method', fontweight='bold')
    ax.set_title('Lexical Diversity: Shannon Entropy Over Synonym Frequencies', 
                 fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/diversity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: diversity_comparison.png")

def plot_fluency_comparison(aggregated, output_dir):
    """Bar chart comparing fluency (perplexity) across methods."""
    
    methods = ['Prompt-Only', 'Gen-Filter', 'Ctrl-G', 'SFT', 'DPO', 'INLP']
    means = []
    stds = []
    
    for method in methods:
        if aggregated[method]['fluency']:
            means.append(np.mean(aggregated[method]['fluency']))
            stds.append(np.std(aggregated[method]['fluency']))
        else:
            means.append(0)
            stds.append(0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c', '#1abc9c']
    bars = ax.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                f'{mean:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Perplexity (GPT-2)', fontweight='bold')
    ax.set_xlabel('Method', fontweight='bold')
    ax.set_title('Fluency: Mean Perplexity Under Reference LM (Lower = Better)', 
                 fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/fluency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: fluency_comparison.png")

def plot_tradeoff_scatter(aggregated, output_dir):
    """Scatter plot: Compliance vs Fluency trade-off."""
    
    methods = ['Prompt-Only', 'Gen-Filter', 'Ctrl-G', 'SFT', 'DPO', 'INLP']
    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c', '#1abc9c']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for method, color in zip(methods, colors):
        compliance = aggregated[method]['compliance']
        fluency = aggregated[method]['fluency']
        
        if compliance and fluency:
            mean_compliance = np.mean(compliance)
            mean_fluency = np.mean(fluency)
            
            ax.scatter(mean_compliance, mean_fluency, s=200, color=color, 
                      alpha=0.7, edgecolor='black', linewidth=1.5, label=method)
            
            # Add method label
            ax.annotate(method, (mean_compliance, mean_fluency), 
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Constraint Compliance (%)', fontweight='bold')
    ax.set_ylabel('Perplexity (Lower = More Fluent)', fontweight='bold')
    ax.set_title('Constraint-Fluency Trade-off', fontweight='bold', pad=15)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/tradeoff_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: tradeoff_scatter.png")

def plot_multi_seed_variance(aggregated, output_dir):
    """Box plot showing variance across seeds for fine-tuned methods."""
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    fine_tuned_methods = ['SFT', 'DPO', 'INLP']
    metrics = ['compliance', 'diversity', 'fluency']
    metric_labels = ['Compliance (%)', 'Diversity (Entropy)', 'Perplexity']
    
    for ax, metric, label in zip(axes, metrics, metric_labels):
        data_to_plot = []
        labels = []
        
        for method in fine_tuned_methods:
            if aggregated[method][metric]:
                data_to_plot.append(aggregated[method][metric])
                labels.append(method)
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # Color boxes
            colors = ['#9b59b6', '#e74c3c', '#1abc9c']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel(label, fontweight='bold')
            ax.set_title(f'{label} Across Seeds', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Cross-Seed Variance for Fine-Tuned Methods', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/multi_seed_variance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: multi_seed_variance.png")

def create_all_visualizations(output_dir='../analysis_results'):
    """Generate all visualizations."""
    
    os.makedirs(f'{output_dir}/figures', exist_ok=True)
    
    print("Loading analysis data...")
    data = load_data(output_dir)
    
    if not data:
        print("❌ Error: No data found. Run analysis scripts 01-03 first!")
        return
    
    print("Aggregating data for plotting...")
    aggregated = aggregate_for_plotting(data)
    
    print("\nGenerating visualizations...")
    
    plot_compliance_comparison(aggregated, output_dir)
    plot_diversity_comparison(aggregated, output_dir)
    
    if aggregated['SFT']['fluency']:  # Only if fluency analysis was run
        plot_fluency_comparison(aggregated, output_dir)
        plot_tradeoff_scatter(aggregated, output_dir)
    
    plot_multi_seed_variance(aggregated, output_dir)
    
    print(f"\n✅ All visualizations saved to {output_dir}/figures/")

if __name__ == '__main__':
    create_all_visualizations()

