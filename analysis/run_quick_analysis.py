#!/usr/bin/env python3
"""
Quick analysis script (without fluency) for fast results.
Runs everything except perplexity calculation.
"""

import subprocess
import sys
import os

def run_script(script_name):
    """Run a Python script."""
    try:
        subprocess.run([sys.executable, script_name], check=True)
        return True
    except:
        return False

def main():
    print("="*80)
    print("QUICK ANALYSIS (Without Fluency)")
    print("="*80)
    print("\nRunning: Compliance, Diversity, Stats, and Visualizations")
    print("Skipping: Fluency analysis (requires GPU and takes hours)\n")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    scripts = [
        '01_constraint_compliance.py',
        '02_lexical_diversity.py',
        '04_statistical_tests.py',
        '05_visualizations.py',
    ]
    
    for script in scripts:
        print(f"\nRunning {script}...")
        if not run_script(script):
            print(f"❌ Error running {script}")
            return
    
    print("\n" + "="*80)
    print("✅ Quick analysis complete!")
    print("="*80)
    print("\nResults saved to: analysis_results/")
    print("\nTo run full analysis (including fluency), use: python run_all_analysis.py")

if __name__ == '__main__':
    main()

