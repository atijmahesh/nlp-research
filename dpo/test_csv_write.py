#!/usr/bin/env python3
"""Test CSV writing with problematic characters."""

import csv

# Test data that might have special characters
test_records = [
    {"Model": "DPO-Test", "Text": 'He said, "Hello, world!"', "Balanced": True},
    {"Model": "DPO-Test", "Text": "Text with\nnewline", "Balanced": False},
    {"Model": "DPO-Test", "Text": "Text with, comma", "Balanced": True},
    {"Model": "DPO-Test", "Text": "Text with ' quote", "Balanced": False},
]

print("Testing CSV write with QUOTE_ALL...")
try:
    with open("test_output.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=test_records[0].keys(), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(test_records)
    print("✅ Success with QUOTE_ALL!")
except Exception as e:
    print(f"❌ Failed with QUOTE_ALL: {e}")

print("\nTesting CSV write with QUOTE_ALL + escapechar...")
try:
    with open("test_output2.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, 
            fieldnames=test_records[0].keys(), 
            quoting=csv.QUOTE_ALL,
            escapechar='\\'
        )
        writer.writeheader()
        writer.writerows(test_records)
    print("✅ Success with QUOTE_ALL + escapechar!")
except Exception as e:
    print(f"❌ Failed with QUOTE_ALL + escapechar: {e}")

print("\nTesting CSV write with QUOTE_MINIMAL + doublequote...")
try:
    with open("test_output3.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, 
            fieldnames=test_records[0].keys(), 
            quoting=csv.QUOTE_MINIMAL,
            doublequote=True
        )
        writer.writeheader()
        writer.writerows(test_records)
    print("✅ Success with QUOTE_MINIMAL + doublequote!")
except Exception as e:
    print(f"❌ Failed with QUOTE_MINIMAL + doublequote: {e}")

