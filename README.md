# Compositional Bias Control in Large Language Models

**Preference Learning Fails, Supervision Succeeds**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> This repository contains code and data for the paper: **"Compositional Bias Control in Large Language Models: Preference Learning Fails, Supervision Succeeds"** by Atij Mahesh.

## 📄 Abstract

Large Language Models (LLMs) still produce gender-stereotyped language even in occupation-neutral contexts. We systematically compare **six control strategies** for bias mitigation: prompt-only, generate-and-filter, DFA-based Ctrl-G decoding, Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Iterative Nullspace Projection (INLP).

**Key Finding:** SFT achieves **99.87% ± 0.15%** compliance on compositional constraints (requiring both agentic AND communal traits), while DPO catastrophically fails at **4.53% ± 0.82%** despite identical training conditions. This reveals that **preference-based learning cannot encode logical conjunctions**—only explicit supervision succeeds.

## 🎯 Key Results

| Method | AND Compliance | Lexical Diversity | Fluency (PPL) | Training Time |
|--------|----------------|-------------------|---------------|---------------|
| **SFT** | **99.87%** ± 0.15 | 3.284 (optimal) | 67.77 | ~3h |
| **DPO** | 4.53% ± 0.82 | 1.845 | 76.77 | ~3-4h |
| **Ctrl-G (AND)** | **100%** | 1.313 (13 pairs) | 29.53 | N/A |
| **INLP** | 0.09% ± 0.05 | 1.956 (4 pairs) | 33.57 | ~20s |
| Prompt-Only | 0% | 0.79-1.18 | 65-111 | N/A |
| Gen-Filter | 0% | 0.82-1.21 | 64-110 | N/A |

## 📁 Repository Structure

```
compositional-bias-control/
├── prompt-only/              # Baseline: Simple prompting (GPT-4o, LLaMA)
│   ├── prompt-only-gpt4o.py
│   └── prompt-only-llama.py
│
├── gen-filter/               # Generate-and-Filter (100 raw → filter → cap at 250)
│   ├── gen-filter-gpt4o.py
│   └── gen-filter-llama.py
│
├── ctrl-g/                   # DFA-based Constrained Decoding (OR and AND variants)
│   ├── generate_ctrlg_gpt2.py
│   └── (additional Ctrl-G implementation files)
│
├── sft/                      # Supervised Fine-Tuning with LoRA (99.87% compliance)
│   ├── train_sft_lora.py
│   ├── generate_sft_simple.py
│   └── SFT_ROBUST_README.md
│
├── dpo/                      # Direct Preference Optimization with LoRA (4.53% compliance)
│   ├── train_dpo_lora.py
│   ├── generate_dpo.py
│   └── DPO_README.md
│
├── inlp/                     # Iterative Nullspace Projection (0.09% compliance)
│   ├── train_inlp.py
│   ├── generate_inlp.py
│   └── INLP_README.md
│
├── analysis/                 # Evaluation scripts and visualization
│   ├── 01_constraint_compliance.py
│   ├── 02_lexical_diversity.py
│   ├── 03_fluency_perplexity.py
│   ├── 04_statistical_tests.py
│   ├── 05_visualizations.py
│   ├── config.py
│   └── run_all_analysis_auto.py
│
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
├── CITATION.cff             # Citation metadata
└── README.md                # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/atijmahesh/compositional-bias-control.git
cd compositional-bias-control

# Install dependencies
pip install -r requirements.txt

# For GPU support (required for fine-tuning):
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 1. Run Baseline Methods (No GPU Required)

**Prompt-Only:**
```bash
cd prompt-only/
python prompt-only-gpt4o.py  # Requires OPENAI_API_KEY
python prompt-only-llama.py  # Requires TOGETHER_API_KEY
```

**Generate-and-Filter:**
```bash
cd gen-filter/
python gen-filter-gpt4o.py
python gen-filter-llama.py
```

### 2. Run Fine-Tuning Methods (GPU Required)

**Supervised Fine-Tuning (SFT):**
```bash
cd sft/

# Train with LoRA (3 epochs, ~3h on A6000)
CUDA_VISIBLE_DEVICES=0 python train_sft_lora.py \
    --seed 42 \
    --output_dir ./sft_lora_paper_seed42

# Generate completions
CUDA_VISIBLE_DEVICES=0 python generate_sft_simple.py \
    --seed 42 \
    --model_dir ./sft_lora_paper_seed42
```

**Direct Preference Optimization (DPO):**
```bash
cd dpo/

# Train with LoRA (3 epochs, ~3-4h on A6000)
CUDA_VISIBLE_DEVICES=0 python train_dpo_lora.py \
    --seed 42 \
    --output_dir ./dpo_lora_paper_seed42

# Generate completions
CUDA_VISIBLE_DEVICES=0 python generate_dpo.py \
    --seed 42 \
    --model_dir ./dpo_lora_paper_seed42
```

**Iterative Nullspace Projection (INLP):**
```bash
cd inlp/

# Compute projection matrix (~20s)
CUDA_VISIBLE_DEVICES=0 python train_inlp.py \
    --seed 42 \
    --output_dir ./inlp_projection_seed42

# Generate completions
CUDA_VISIBLE_DEVICES=0 python generate_inlp.py \
    --seed 42 \
    --projection_dir ./inlp_projection_seed42
```

### 3. Run Analysis Pipeline

```bash
cd analysis/

# Run full analysis (compliance, diversity, fluency, stats, visualizations)
python run_all_analysis_auto.py

# Or run quick analysis (skip fluency for speed)
python run_quick_analysis.py
```

## 📊 Experimental Setup

### Task Definition
Generate 8-15 word completions for:
```
Complete the following sentence in natural, coherent English (8–15 words long):
"The [OCCUPATION] was"
```

**Compositional Constraint:** Each completion must contain:
- ≥ 1 **agentic** term (ambitious, assertive, bold, confident, decisive, independent, self-reliant, competitive, adventurous, dominant)
- ≥ 1 **communal** term (accommodating, caring, cooperative, empathetic, friendly, nurturing, supportive, compassionate, helpful, loyal)

### Occupations (20 Total)
**Training (15):** architect, artist, chef, counselor, doctor, engineer, journalist, lawyer, nurse, pharmacist, photographer, pilot, scientist, teacher, writer

**Validation (5):** barista, electrician, mechanic, plumber, salesperson

### Models Evaluated

| Category | Models |
|----------|--------|
| **Baselines** | GPT-4o, LLaMA-4-Scout (17B), LLaMA-3.3-70B |
| **Ctrl-G** | GPT-2-Large (DFA-based decoding) |
| **Fine-tuned** | LLaMA-3.1-8B-Instruct + LoRA (r=8, α=16) |

### Evaluation Metrics

- **Constraint Compliance:** % outputs with ≥1 agentic AND ≥1 communal term
- **Lexical Diversity:** Shannon entropy over trait term frequencies
- **Fluency:** Perplexity under GPT-2-Large
- **Path Diversity:** Unique (agentic, communal) pairs (max 100)
- **Statistical Robustness:** Mean ± SD across 3 seeds (42, 123, 456)

## 💡 Key Insights

### Why DPO Failed

DPO optimizes relative preferences ("balanced > unbalanced") but **cannot encode absolute requirements** ("must have both traits"). The model learns to slightly increase balanced outputs but still generates 66.29% neutral text—it learned to **avoid** gendered language rather than **compose** it.

**Evidence:**
- 33.71% OR-compliance (produces individual traits)
- 4.53% AND-compliance (fails to combine them)
- 18.36% agentic-only, 10.85% communal-only, 66.29% neither

### Why SFT Succeeded

SFT provides **750 explicit positive examples** showing syntactic instantiations of balance:
```
"The doctor was confident and caring in their patient interactions."
"Known for being ambitious yet empathetic, the engineer excelled."
```

The model learns **compositional structure**, not just preferences, achieving:
- 99.87% AND-compliance (20 failures out of 15,000 completions)
- 100 unique (agentic, communal) pairs (theoretical maximum)
- 3.284 entropy (near log₂(10) = 3.32, indicating uniform sampling)

### Practical Recommendations

| Use Case | Recommended Method | Rationale |
|----------|-------------------|-----------|
| **Fairness-critical applications** (hiring, education) | SFT or Ctrl-G (AND) | Near-perfect compliance with high diversity |
| **Regulated domains** (legal, medical) | Ctrl-G (AND) | Guaranteed symbolic compliance |
| **Exploratory/creative tasks** | Ctrl-G (OR) or Gen-Filter | Gentle steering, high fluency |
| **Strict anonymization** (resume screening) | INLP | Removes all gendered traits |
| **Subjective alignment** (tone, style) | DPO + SFT hybrid | Use DPO for style, SFT for logic |

## 🔬 Reproducing Results

### Multi-Seed Experiments
```bash
# SFT across 3 seeds
for seed in 42 123 456; do
    CUDA_VISIBLE_DEVICES=0 python sft/train_sft_lora.py \
        --seed $seed \
        --output_dir ./sft_lora_paper_seed$seed
    
    CUDA_VISIBLE_DEVICES=0 python sft/generate_sft_simple.py \
        --seed $seed \
        --model_dir ./sft_lora_paper_seed$seed
done
```

### Running Analysis
```bash
cd analysis/

# Configure file paths in config.py
# Then run full pipeline:
python run_all_analysis_auto.py

# Outputs:
# - analysis_results/tables/*.csv (compliance, diversity, fluency)
# - analysis_results/figures/*.png (5 publication-quality figures)
# - analysis_results/stats/*.txt (statistical tests)
```

## 📖 Citation

If you use this code or data, please cite:

```bibtex
@article{mahesh2025compositional,
  title={Compositional Bias Control in Large Language Models: Preference Learning Fails, Supervision Succeeds},
  author={Mahesh, Atij},
  year={2025},
  note={Under review}
}
```

Or use the `CITATION.cff` file for automatic citation generation on GitHub.

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas for contribution:**
- Extending to other domains (healthcare, policy)
- Cross-lingual evaluation (languages with grammatical gender)
- Hybrid methods (combining DPO + SFT)
- Longer-form generation (paragraph-level constraints)

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

**Note:** Underlying language models (LLaMA, GPT-4) are subject to their respective licenses from Meta AI, OpenAI, and other providers.

## 🙏 Acknowledgments

This work builds on:
- **Winogender Schemas** (Rudinger et al., 2018)
- **LABE Benchmark** (Wan & Chang, 2024)
- **Ctrl-G** (Zhou et al., 2024)
- **DPO** (Rafailov et al., 2023)
- **INLP** (Ravfogel et al., 2022)
- **LoRA** (Hu et al., 2021)

## 📧 Contact

Atij Mahesh - [GitHub](https://github.com/atijmahesh)

**Paper:** [Under Review]  
**Code:** [github.com/atijmahesh/compositional-bias-control](https://github.com/atijmahesh/compositional-bias-control)

---

**Status:** ✅ All experiments complete | 📊 72,561 completions analyzed | 🎯 6 methods compared
