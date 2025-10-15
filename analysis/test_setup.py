#!/usr/bin/env python3
"""
Quick test to verify analysis setup and data availability.
"""

import os
import sys
from config import *

def test_data_availability():
    """Check if all expected data files exist."""
    
    print("="*80)
    print("ANALYSIS SETUP TEST")
    print("="*80)
    
    results_dir = '../results'
    
    if not os.path.exists(results_dir):
        print(f"\n❌ ERROR: Results directory not found: {results_dir}")
        print("   This script should be run from the analysis/ directory")
        return False
    
    print(f"\n✅ Results directory found: {results_dir}")
    print("\nChecking data files:")
    print("-"*80)
    
    found = 0
    missing = 0
    
    for method_name, file_key in DATA_FILES.items():
        filepath = os.path.join(results_dir, file_key)
        
        if os.path.exists(filepath):
            # Get file size
            size_bytes = os.path.getsize(filepath)
            size_mb = size_bytes / (1024 * 1024)
            print(f"✅ {method_name:20} {file_key:50} ({size_mb:.2f} MB)")
            found += 1
        else:
            print(f"❌ {method_name:20} {file_key:50} NOT FOUND")
            missing += 1
    
    print("-"*80)
    print(f"\nSummary: {found} files found, {missing} files missing")
    
    if missing > 0:
        print("\n⚠️  Warning: Some data files are missing.")
        print("   Analysis will skip missing files.")
    
    print("\n" + "="*80)
    print("CONFIGURATION CHECK")
    print("="*80)
    print(f"Agentic terms: {len(AGENTIC_TERMS)}")
    print(f"Communal terms: {len(COMMUNAL_TERMS)}")
    print(f"Occupations: {len(OCCUPATIONS)}")
    print(f"Train occupations: {len(TRAIN_OCCUPATIONS)}")
    print(f"Val occupations: {len(VALIDATION_OCCUPATIONS)}")
    print(f"Method groups: {len(METHOD_GROUPS)}")
    
    print("\n✅ Configuration loaded successfully!")
    
    return True

if __name__ == '__main__':
    success = test_data_availability()
    sys.exit(0 if success else 1)

