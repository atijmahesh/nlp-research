# Analysis Scripts

This directory contains scripts to analyze the collected data from all bias mitigation methods.

## Overview

The analysis pipeline consists of 5 main scripts:

1. **`01_constraint_compliance.py`** - Measures % of samples containing both agentic and communal terms
2. **`02_lexical_diversity.py`** - Computes Shannon entropy over synonym frequencies and path diversity
3. **`03_fluency_perplexity.py`** - Calculates perplexity under GPT-2 as a fluency proxy (⚠️ requires GPU)
4. **`04_statistical_tests.py`** - Performs Wilcoxon tests, Cohen's d, and bootstrap confidence intervals
5. **`05_visualizations.py`** - Generates publication-quality plots

## Quick Start

### Option 1: Quick Analysis (Recommended for First Run)

Runs everything **except** fluency analysis (which takes several hours on GPU):

```bash
cd analysis
python run_quick_analysis.py
```

This will generate:
- Compliance metrics
- Diversity metrics
- Statistical tests
- Visualizations (without fluency plots)

**Time:** ~5-10 minutes

---

### Option 2: Full Analysis (For Final Paper)

Runs **all** scripts including GPU-intensive perplexity calculation:

```bash
cd analysis
python run_all_analysis.py
```

This includes everything from Option 1 **plus** fluency metrics.

**Time:** ~2-4 hours (depends on GPU)

**Requirements:**
- CUDA-enabled GPU
- `transformers` library
- GPT-2 model (auto-downloaded)

---

### Option 3: Run Individual Scripts

You can run scripts individually:

```bash
python 01_constraint_compliance.py
python 02_lexical_diversity.py
python 03_fluency_perplexity.py  # GPU required
python 04_statistical_tests.py
python 05_visualizations.py
```

---

## Configuration

Edit `config.py` to modify:
- Synonym lists (agentic/communal terms)
- Occupation lists
- File paths
- Method groupings
- Output directories

---

## Output Structure

All results are saved to `analysis_results/`:

```
analysis_results/
├── tables/
│   ├── compliance_summary.csv
│   ├── compliance_detailed.json
│   ├── diversity_summary.csv
│   ├── diversity_detailed.json
│   ├── fluency_summary.csv
│   ├── fluency_detailed.json
│   ├── pairwise_comparisons.csv
│   └── confidence_intervals.csv
└── figures/
    ├── compliance_comparison.png
    ├── diversity_comparison.png
    ├── fluency_comparison.png
    ├── tradeoff_scatter.png
    └── multi_seed_variance.png
```

---

## Key Metrics

### 1. Constraint Compliance
- **Definition:** % of samples containing ≥1 agentic AND ≥1 communal term
- **Expected Results:**
  - Ctrl-G: 100% (hard constraint)
  - SFT: ~99.9% (supervised learning)
  - DPO: ~4.5% (preference learning failed)
  - INLP: TBD
  - Gen-Filter: ~30-50%
  - Prompt-Only: ~5-10%

### 2. Lexical Diversity
- **Metric:** Shannon entropy H = -Σ p(x) × log₂(p(x))
- **Interpretation:** Higher entropy = more diverse vocabulary usage
- **Range:** 0 (all same term) to log₂(10) ≈ 3.32 (uniform distribution)

### 3. Path Diversity
- **Definition:** Number of unique (agentic, communal) term pairs
- **Max Possible:** 10 × 10 = 100 pairs
- **Measures:** Compositional variety in constraint satisfaction

### 4. Fluency (Perplexity)
- **Metric:** Perplexity under GPT-2 reference LM
- **Interpretation:** Lower = more fluent/natural text
- **Typical Range:** 20-100 for natural text

### 5. Statistical Significance
- **Wilcoxon/Mann-Whitney U:** Non-parametric test for group differences
- **Cohen's d:** Effect size (|d| ≥ 0.8 = large effect)
- **Bootstrap CI:** 95% confidence intervals for robustness

---

## Dependencies

```bash
pip install numpy scipy matplotlib seaborn transformers torch
```

For GPU support (fluency analysis):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Troubleshooting

### "No data found" error
**Solution:** Make sure CSV files exist in `results/` directory with correct structure.

### Fluency script freezes
**Solution:** 
- Check GPU availability: `nvidia-smi`
- Reduce batch size in script if OOM
- Run on CPU (very slow): script will prompt

### Statistical tests error
**Solution:** Run scripts 01-03 first to generate required data files.

### Import errors
**Solution:** 
```bash
pip install -r ../requirements.txt
```

---

## For the Paper

### Tables to Include:
1. **Table 1:** Compliance summary (`compliance_summary.csv`)
2. **Table 2:** Diversity metrics (`diversity_summary.csv`)
3. **Table 3:** Pairwise comparisons (`pairwise_comparisons.csv`)

### Figures to Include:
1. **Figure 1:** Compliance comparison (bar chart)
2. **Figure 2:** Constraint-fluency trade-off (scatter)
3. **Figure 3:** Multi-seed variance (box plot)

### Key Results to Report:
- SFT: **99.87% ± 0.15%** compliance (near-perfect)
- DPO: **4.53% ± 0.82%** compliance (catastrophic failure)
- Statistical significance: SFT vs DPO (p < 0.001, d > 2.0)
- Fluency cost: Ctrl-G vs Prompt-Only perplexity difference

---

## Citation

If you use these scripts, please cite:

```
[Your Paper Citation Here]
```

---

## Contact

For questions or issues, contact: [Your Email]

