#!/usr/bin/env python3
"""
Master script to run all analysis scripts in sequence (non-interactive).
"""

import subprocess
import sys
import os

def run_script(script_name, description):
    """Run a Python script and handle errors."""
    print("\n" + "="*80)
    print(f"RUNNING: {description}")
    print("="*80)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✅ {description} completed successfully!")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_name}")
        print(f"Exit code: {e.returncode}")
        return False
    
    except FileNotFoundError:
        print(f"❌ Script not found: {script_name}")
        return False

def main():
    """Run all analysis scripts without prompts."""
    
    print("="*80)
    print("NLP RESEARCH: COMPREHENSIVE ANALYSIS PIPELINE (AUTO MODE)")
    print("="*80)
    print("\nRunning all scripts automatically (no prompts):")
    print("  1. Constraint Compliance")
    print("  2. Lexical Diversity")
    print("  3. Fluency (Perplexity) - GPU required, ~2-4 hours")
    print("  4. Statistical Tests")
    print("  5. Visualizations")
    print()
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run scripts in order
    scripts = [
        ('01_constraint_compliance.py', 'Constraint Compliance Analysis'),
        ('02_lexical_diversity.py', 'Lexical Diversity Analysis'),
        ('03_fluency_perplexity.py', 'Fluency Analysis (Perplexity)'),
        ('04_statistical_tests.py', 'Statistical Tests'),
        ('05_visualizations.py', 'Visualizations'),
    ]
    
    results = []
    
    for script, description in scripts:
        success = run_script(script, description)
        results.append((description, success))
        
        if not success:
            print(f"\n⚠️  Warning: {description} failed!")
            print(f"Continuing with remaining scripts...")
    
    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS PIPELINE SUMMARY")
    print("="*80)
    
    for description, success in results:
        status = "✅ Success" if success else "❌ Failed"
        print(f"{status:12} {description}")
    
    print("="*80)
    
    total = len(results)
    successful = sum(1 for _, s in results if s)
    
    print(f"\nCompleted: {successful}/{total} scripts")
    
    if successful == total:
        print("\n🎉 All analysis complete! Results in analysis_results/")
    else:
        print("\n⚠️  Some scripts failed. Check errors above.")

if __name__ == '__main__':
    main()

