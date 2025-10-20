# Quick Reference Guide

**Repository:** compositional-bias-control  
**Paper:** "Compositional Bias Control in Large Language Models: Preference Learning Fails, Supervision Succeeds"

---

## 📂 Essential Files

### Documentation
- **README.md** - Main project overview (START HERE)
- **CONTRIBUTING.md** - How to contribute
- **LICENSE** - MIT License
- **CITATION.cff** - Citation metadata
- **REPOSITORY_CHECKLIST.md** - Publication readiness tracker

### Configuration
- **requirements.txt** - Python dependencies
- **.gitignore** - Excluded files/patterns
- **analysis/config.py** - Analysis configuration (data paths, methods, trait terms)

---

## 🔬 Running Experiments

### Baselines (No GPU)
```bash
# Prompt-only
cd prompt-only/ && python prompt-only-gpt4o.py

# Generate-and-filter  
cd gen-filter/ && python gen-filter-gpt4o.py
```

### Fine-tuning (GPU Required)
```bash
# SFT (99.87% compliance)
cd sft/
CUDA_VISIBLE_DEVICES=0 python train_sft_lora.py --seed 42 --output_dir ./sft_seed42
CUDA_VISIBLE_DEVICES=0 python generate_sft_simple.py --seed 42 --model_dir ./sft_seed42

# DPO (4.53% compliance)
cd dpo/
CUDA_VISIBLE_DEVICES=0 python train_dpo_lora.py --seed 42 --output_dir ./dpo_seed42
CUDA_VISIBLE_DEVICES=0 python generate_dpo.py --seed 42 --model_dir ./dpo_seed42

# INLP (0.09% compliance, fast!)
cd inlp/
CUDA_VISIBLE_DEVICES=0 python train_inlp.py --seed 42 --output_dir ./inlp_seed42
CUDA_VISIBLE_DEVICES=0 python generate_inlp.py --seed 42 --projection_dir ./inlp_seed42
```

### Ctrl-G (CPU or GPU)
```bash
cd ctrl-g/
python generate_ctrlg_gpt2.py  # Generates AND variant by default
```

---

## 📊 Running Analysis

```bash
cd analysis/

# Full analysis (compliance, diversity, fluency, stats, visualizations)
python run_all_analysis_auto.py

# Quick analysis (skip fluency for speed)
python run_quick_analysis.py

# Individual analyses
python 01_constraint_compliance.py  # AND/OR compliance
python 02_lexical_diversity.py       # Shannon entropy, path diversity  
python 03_fluency_perplexity.py      # GPT-2 perplexity (slow, requires GPU)
python 04_statistical_tests.py       # Mann-Whitney U, Cohen's d
python 05_visualizations.py          # 5 publication figures
```

**Outputs:**
- `analysis_results/tables/` - CSV files with metrics
- `analysis_results/figures/` - PNG visualizations

---

## 📝 Key Results

| Method | AND Compliance | Diversity (Entropy) | Fluency (PPL) |
|--------|----------------|---------------------|---------------|
| **SFT** | 99.87% ± 0.15 | 3.284 (100 pairs) | 67.77 |
| **DPO** | 4.53% ± 0.82 | 1.845 (24 pairs) | 76.77 |
| **Ctrl-G (AND)** | 100% | 1.313 (13 pairs) | 29.53 |
| **INLP** | 0.09% ± 0.05 | 1.956 (4 pairs) | 33.57 |
| Prompt-Only | 0% | 0.79-1.18 | 65-111 |
| Gen-Filter | 0% | 0.82-1.21 | 64-110 |

---

## 🎯 Key Insights

1. **SFT succeeds** because it provides explicit positive examples of balanced compositions
2. **DPO fails** because binary preferences encode ranking, not logical conjunctions  
3. **Ctrl-G guarantees** compliance but sacrifices diversity (13 vs. 100 unique pairs)
4. **INLP overcorrects**, removing 95% of all trait terms

---

## 🔧 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size for fine-tuning
python train_sft_lora.py --per_device_train_batch_size 2 --gradient_accumulation_steps 8

# Or use gradient checkpointing (already enabled by default)
```

### API Rate Limits (prompt-only, gen-filter)
```bash
# Add delays in scripts or use higher-tier API keys
# GPT-4o: ~$0.50 per 100 completions
# LLaMA (Together AI): ~$0.10 per 100 completions
```

### Analysis Requires GPU?
```bash
# Only fluency analysis (03_fluency_perplexity.py) requires GPU
# Other analyses work on CPU
python run_quick_analysis.py  # Skips fluency
```

---

## 📚 Method Documentation

Each method has a dedicated README:
- **prompt-only/README.md** - Baseline generation
- **gen-filter/README.md** - Post-hoc filtering
- **ctrl-g/README.md** - DFA constraints
- **sft/SFT_ROBUST_README.md** - Supervised fine-tuning
- **dpo/DPO_README.md** - Preference optimization
- **inlp/INLP_README.md** - Nullspace projection

---

## 🌟 Most Important Commands

### Reproduce Paper Results
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run analysis on existing results
cd analysis/ && python run_all_analysis_auto.py

# 3. Train SFT (if retraining)
cd sft/
for seed in 42 123 456; do
    CUDA_VISIBLE_DEVICES=0 python train_sft_lora.py --seed $seed
    CUDA_VISIBLE_DEVICES=0 python generate_sft_simple.py --seed $seed
done
```

### Verify Setup
```bash
# Check Python version
python --version  # Should be 3.8+

# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check data files exist
cd analysis/ && python -c "from config import DATA_FILES; print(f'{len(DATA_FILES)} data files configured')"
```

---

## 🐛 Common Issues

**Q: `ModuleNotFoundError: No module named 'transformers'`**  
A: `pip install -r requirements.txt`

**Q: `RuntimeError: CUDA out of memory`**  
A: Reduce batch size or use smaller models

**Q: `FileNotFoundError: results/sft/...csv`**  
A: Run generation scripts first, or update `analysis/config.py` with correct paths

**Q: Analysis outputs "0 samples"**  
A: Check CSV file paths in `analysis/config.py` match your actual file locations

---

## 📞 Getting Help

1. Check method-specific READMEs in each directory
2. Read CONTRIBUTING.md for development guidelines
3. Open an issue on GitHub (after repository is public)
4. Email: [your email]

---

## ✅ Pre-Publication Checklist

Before making repository public:
- [ ] Update GitHub URL in CITATION.cff
- [ ] Update contact email in README.md
- [ ] Add topics to GitHub repo: `nlp`, `bias-mitigation`, `fairness`, `large-language-models`
- [ ] Enable Issues and Discussions
- [ ] Add paper link when available
- [ ] (Optional) Upload data to HuggingFace/Zenodo and link in README

---

**Last Updated:** October 19, 2025  
**Status:** Publication-ready (95/100)  
**Commit:** `749f3e2` Repository cleanup for publication

