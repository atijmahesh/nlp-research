#!/usr/bin/env python3
"""Check trl version and DPOTrainer signature."""

import trl
import inspect

print(f"trl version: {trl.__version__}")
print(f"\nDPOTrainer signature:")
print(inspect.signature(trl.DPOTrainer.__init__))

