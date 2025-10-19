# Contributing to Compositional Bias Control

Thank you for your interest in contributing to this research project! This guide will help you get started.

## 🎯 Project Goals

This project investigates how different control strategies handle compositional constraints in bias mitigation. We welcome contributions that:

1. **Extend the evaluation**: New domains, languages, or constraint types
2. **Improve methods**: Hybrid approaches, efficiency optimizations
3. **Fix bugs**: Code improvements, documentation clarifications
4. **Add analysis**: New metrics, visualizations, or statistical tests

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/compositional-bias-control.git
cd compositional-bias-control

# Add upstream remote
git remote add upstream https://github.com/atijmahesh/compositional-bias-control.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

## 📝 Code Style

We follow standard Python conventions:

### Formatting
```bash
# Format code with Black (line length 100)
black --line-length 100 your_file.py

# Check style with flake8
flake8 your_file.py --max-line-length 100
```

### Docstrings
Use Google-style docstrings:

```python
def train_model(model, data, epochs=3):
    """Train a language model on the given data.
    
    Args:
        model: The model to train
        data: Training dataset
        epochs: Number of training epochs (default: 3)
    
    Returns:
        Trained model with updated weights
    
    Raises:
        ValueError: If epochs < 1
    """
    pass
```

### Naming Conventions
- **Functions/variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`

## 🧪 Testing

### Run Existing Tests
```bash
# Test analysis setup
cd analysis/
python test_setup.py

# Test individual components
python 01_constraint_compliance.py --test-mode
```

### Add New Tests
Place tests in a `tests/` directory (to be created):

```python
# tests/test_compliance.py
import pytest
from analysis.config import AGENTIC_TERMS, COMMUNAL_TERMS

def test_constraint_detection():
    """Test that constraint detection works correctly."""
    text = "The doctor was confident and caring"
    assert has_agentic_term(text, AGENTIC_TERMS)
    assert has_communal_term(text, COMMUNAL_TERMS)
```

## 📊 Adding New Methods

To add a new control method:

### 1. Create Method Directory

```bash
mkdir my-method/
cd my-method/
```

### 2. Implement Core Scripts

**Training script** (if applicable):
```python
# train_mymethod.py
"""
Train MyMethod for compositional bias control.

Usage:
    python train_mymethod.py --seed 42 --output_dir ./mymethod_seed42
"""
import argparse
# ... implementation
```

**Generation script**:
```python
# generate_mymethod.py
"""
Generate completions using MyMethod.

Outputs CSV with columns: model, occupation, run_id, completion
"""
import argparse
import csv
# ... implementation
```

### 3. Add Documentation

Create `MYMETHOD_README.md`:

```markdown
# MyMethod

## Overview
Brief description of the method and its approach.

## Training
Step-by-step instructions with example commands.

## Generation
How to generate completions.

## Hyperparameters
Document all key hyperparameters and their defaults.

## Expected Results
Compliance, diversity, fluency benchmarks.
```

### 4. Update Analysis Config

Edit `analysis/config.py`:

```python
DATA_FILES = {
    # ... existing files
    'mymethod_seed42': 'my-method/mymethod_completions_seed42.csv',
    'mymethod_seed123': 'my-method/mymethod_completions_seed123.csv',
    'mymethod_seed456': 'my-method/mymethod_completions_seed456.csv',
}

METHOD_GROUPS = {
    # ... existing methods
    'MyMethod': ['mymethod_seed42', 'mymethod_seed123', 'mymethod_seed456'],
}
```

### 5. Update Main README

Add your method to the repository structure and quick start sections.

## 📈 Adding New Analysis

To add a new metric or visualization:

### 1. Create Analysis Script

```python
# analysis/06_my_analysis.py
"""
Description of what this analysis measures.

Outputs:
    - analysis_results/tables/my_metric_summary.csv
    - analysis_results/figures/my_visualization.png
"""

import pandas as pd
from config import DATA_FILES, METHOD_GROUPS, RESULTS_DIR

def compute_my_metric(text):
    """Compute custom metric on text."""
    # ... implementation
    return score

if __name__ == '__main__':
    # Load data, compute metrics, save results
    pass
```

### 2. Add to Pipeline

Update `analysis/run_all_analysis_auto.py`:

```python
scripts = [
    '01_constraint_compliance.py',
    '02_lexical_diversity.py',
    '03_fluency_perplexity.py',
    '04_statistical_tests.py',
    '05_visualizations.py',
    '06_my_analysis.py',  # Add here
]
```

## 🔬 Reproducing Experiments

When adding new experiments:

1. **Use random seeds**: For reproducibility, use seeds 42, 123, 456
2. **Document hardware**: Specify GPU type, memory, training time
3. **Log hyperparameters**: Save all hyperparameters to JSON
4. **Version control data**: Track data versions if they change

Example:
```python
import json
import torch

# Log experiment config
config = {
    'seed': args.seed,
    'model': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'learning_rate': 2e-4,
    'batch_size': 4,
    'epochs': 3,
    'gpu': torch.cuda.get_device_name(0),
    'cuda_version': torch.version.cuda,
}

with open(f'{output_dir}/config.json', 'w') as f:
    json.dump(config, f, indent=2)
```

## 📋 Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines (Black, flake8)
- [ ] Added docstrings to new functions
- [ ] Updated relevant README files
- [ ] Added or updated tests (if applicable)
- [ ] Tested on GPU (if fine-tuning changes)
- [ ] Updated `CITATION.cff` if adding significant methods

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature (method, analysis)
- [ ] Documentation update
- [ ] Performance improvement

## Testing
Describe how you tested the changes.

## Results (if applicable)
Include metrics (compliance, diversity, fluency) for new methods.

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed code
- [ ] Commented complex sections
- [ ] Updated documentation
- [ ] No new warnings
```

### Review Process

1. Automated checks (style, basic tests) must pass
2. Maintainer reviews code and results
3. Address feedback and update PR
4. Once approved, maintainer will merge

## 🐛 Reporting Bugs

### Bug Report Template

**Title:** Clear, descriptive title

**Description:**
- What happened?
- What did you expect to happen?
- How can we reproduce it?

**Environment:**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.10.6]
- GPU: [e.g., RTX A6000, 48GB]
- CUDA version: [e.g., 11.8]
- Relevant package versions: [transformers, torch, etc.]

**Code to Reproduce:**
```python
# Minimal code that triggers the bug
```

**Error Message:**
```
Full traceback or error message
```

## 💡 Feature Requests

We welcome feature requests! Please include:

1. **Problem:** What problem does this solve?
2. **Solution:** Proposed approach
3. **Alternatives:** Other solutions you considered
4. **Context:** Use case, related work

## 📞 Getting Help

- **Questions:** Open a GitHub Discussion
- **Bugs:** Open a GitHub Issue
- **Security:** Email maintainer directly (see README)

## 🙏 Code of Conduct

- Be respectful and constructive
- Welcome newcomers and help them learn
- Focus on what is best for the community
- Show empathy towards other contributors

## 📚 Resources

### Related Research
- [Winogender Schemas](https://github.com/rudinger/winogender-schemas)
- [LABE Benchmark](https://github.com/ewanlee/LABE)
- [Ctrl-G](https://github.com/zorazrw/Ctrl-G)

### Technical Documentation
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PEFT (LoRA)](https://huggingface.co/docs/peft)
- [TRL (DPO)](https://huggingface.co/docs/trl)

---

Thank you for contributing to fair and compositional language generation! 🎉

