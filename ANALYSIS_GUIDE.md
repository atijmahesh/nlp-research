# Analysis Guide

## Summary of Created Analysis Scripts

I've created a comprehensive analysis pipeline in the `analysis/` directory with **10 files**:

### Core Analysis Scripts (5)
1. **`config.py`** - Central configuration (synonyms, file paths, method groupings)
2. **`01_constraint_compliance.py`** - Measures % samples with both agentic & communal terms
3. **`02_lexical_diversity.py`** - Shannon entropy and path diversity metrics
4. **`03_fluency_perplexity.py`** - Perplexity under GPT-2 (⚠️ requires GPU, ~2-4 hours)
5. **`04_statistical_tests.py`** - Wilcoxon, Cohen's d, bootstrap CIs
6. **`05_visualizations.py`** - Publication-quality plots (5 figures)

### Runner Scripts (2)
7. **`run_all_analysis.py`** - Full pipeline (includes fluency)
8. **`run_quick_analysis.py`** - Fast pipeline (skips fluency)

### Documentation & Utils (3)
9. **`README.md`** - Complete documentation
10. **`requirements.txt`** - Python dependencies
11. **`test_setup.py`** - Verify data availability

---

## Quick Start on Remote Server

### Step 1: Push to Remote
```bash
# On local
cd /Users/atij.mahesh/Desktop/Development/nlp-research
git add analysis/
git commit -m "Add comprehensive analysis pipeline (compliance, diversity, fluency, stats, viz)"
git push
```

### Step 2: Pull on Remote
```bash
# On remote server
cd /local4/atij/nlp-research
git pull
```

### Step 3: Install Dependencies
```bash
cd /local4/atij/nlp-research/analysis
pip install -r requirements.txt
```

### Step 4: Run Quick Analysis (Recommended First)
```bash
# Fast analysis without fluency (~5-10 min)
python run_quick_analysis.py
```

This generates:
- ✅ Compliance metrics
- ✅ Diversity metrics  
- ✅ Statistical tests
- ✅ Visualizations (without fluency plots)

**Output:** `analysis_results/tables/` and `analysis_results/figures/`

### Step 5: Run Full Analysis (For Final Paper)
```bash
# Full analysis with GPU-based fluency (~2-4 hours)
python run_all_analysis.py
```

This adds:
- ✅ Perplexity under GPT-2
- ✅ Fluency comparison plots
- ✅ Constraint-fluency trade-off scatter plot

---

## Expected Results

Based on your preliminary data:

| Method | Compliance | Entropy | Perplexity | Status |
|--------|-----------|---------|-----------|--------|
| Prompt-Only | 5-10% | Medium | Baseline | ✅ |
| Gen-Filter | 30-50% | Medium | Baseline | ✅ |
| Ctrl-G | 100% | ? | Higher? | ✅ |
| **SFT** | **99.87% ± 0.15%** | High | Low | ✅ |
| **DPO** | **4.53% ± 0.82%** | Low | Low | ✅ |
| **INLP** | TBD | TBD | TBD | ✅ |

### Key Finding for Paper:
🎯 **SFT (99.87%) dramatically outperforms DPO (4.53%)** for compositional constraint satisfaction, revealing that **preference learning fails for complex logical constraints** that require explicit positive examples.

---

## Output Files for Paper

### Tables (CSV)
1. `compliance_summary.csv` - Main results table
2. `diversity_summary.csv` - Lexical diversity
3. `fluency_summary.csv` - Perplexity results
4. `pairwise_comparisons.csv` - Statistical significance
5. `confidence_intervals.csv` - Bootstrap CIs

### Figures (PNG, 300 DPI)
1. `compliance_comparison.png` - Bar chart of compliance rates
2. `diversity_comparison.png` - Shannon entropy comparison
3. `fluency_comparison.png` - Perplexity comparison
4. `tradeoff_scatter.png` - Constraint vs fluency trade-off
5. `multi_seed_variance.png` - Cross-seed variance (box plots)

---

## Troubleshooting

### If scripts fail:
1. **Check data paths:** Run `python test_setup.py` to verify files
2. **Missing dependencies:** `pip install -r requirements.txt`
3. **GPU issues (fluency):** Use `run_quick_analysis.py` instead
4. **Import errors:** Make sure you're in `analysis/` directory

### Manual execution:
```bash
cd analysis
python 01_constraint_compliance.py
python 02_lexical_diversity.py
# Skip 03 if no GPU
python 04_statistical_tests.py
python 05_visualizations.py
```

---

## Next Steps After Analysis

1. ✅ **Review Results:** Check `analysis_results/tables/*.csv`
2. ✅ **Verify Figures:** Open `analysis_results/figures/*.png`
3. ✅ **Update Protocol:** Add final metrics to `RESEARCH_PROTOCOL_UPDATED.md`
4. 📝 **Write Paper:** Use tables/figures in manuscript
5. 📊 **Present Results:** Use visualizations in slides

---

## Data Summary (Current Status)

Total collected: **62,561 completions** across 6 methods

- Prompt-Only: 6,300 (GPT-4o + LLaMA)
- Gen-Filter: 1,021 (filtered)
- Ctrl-G: 10,240 (hard constraints)
- SFT: 15,000 (3 seeds × 5,000)
- DPO: 15,000 (3 seeds × 5,000)
- INLP: 15,000 (3 seeds × 5,000)

✅ All data collection complete!

---

## Timeline Estimate

| Task | Time | GPU | Notes |
|------|------|-----|-------|
| Quick Analysis | 5-10 min | No | Compliance, diversity, stats, viz |
| Full Analysis | 2-4 hours | Yes | Adds fluency (perplexity) |
| Review Results | 30 min | No | Check tables and figures |
| Paper Writing | 1-2 days | No | Results, discussion, polish |

**Total to submission-ready draft:** ~3-5 days

---

## Questions?

If you encounter issues:
1. Check `analysis/README.md` for detailed docs
2. Run `python test_setup.py` to diagnose
3. Try `run_quick_analysis.py` first (no GPU needed)

Good luck with your preprint! 🚀

