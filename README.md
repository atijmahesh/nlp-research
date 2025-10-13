# NLP Research: Gender Bias in Language Models

**Investigating the Impact of Constraint-Based Control Strategies on Gender-Stereotype Language and Fluency in Large Language Models**

## Project Structure

```
nlp-research/
├── prompt-only/          # Baseline: Simple prompting (5 samples/occupation)
│   ├── prompt-only-gpt4o.py
│   └── prompt-only-llama.py
│
├── gen-filter/           # Generate-and-Filter (100 raw → filter → cap at 250)
│   ├── gen-filter-gpt4o.py
│   └── gen-filter-llama.py
│
├── sft/                  # Supervised Fine-Tuning with LoRA
│   ├── train_sft_lora.py          # Training script (~3h on A6000)
│   ├── generate_sft_simple.py     # Generate 250 completions per occupation
│   └── SFT_ROBUST_README.md       # Detailed SFT methodology
│
├── dpo/                  # Direct Preference Optimization with LoRA
│   ├── train_dpo_lora.py          # Training script (~3-4h on A6000)
│   ├── generate_dpo.py            # Generate 250 completions per occupation
│   └── DPO_README.md              # Detailed DPO methodology
│
├── inlp/                 # Iterative Nullspace Projection
│   ├── train_inlp.py              # Compute projection matrix (~30min)
│   ├── generate_inlp.py           # Generate with projection applied
│   └── INLP_README.md             # Detailed INLP methodology
│
├── listModels.py         # Utility: List available OpenAI models
└── README.md             # This file
```

## Research Overview

This study evaluates **six control strategies** to mitigate gender bias in LLMs:

1. **Prompt-Only** ✅ - Simple prompting with length constraints
2. **Generate-and-Filter** ✅ - Post-hoc filtering for stereotypical terms
3. **Ctrl-G Decoding** ✅ - DFA-based constrained generation (separate implementation)
4. **SFT Fine-tuning** ✅ - Supervised learning with LoRA to encourage balanced outputs
5. **DPO Fine-tuning** ✅ - Preference learning with LoRA (chosen vs rejected outputs)
6. **INLP** 🔄 - Linear projection to remove gender subspace (post-hoc debiasing)

### Key Metrics
- **Constraint Compliance**: % samples with both agentic AND communal terms
- **Lexical Diversity**: Shannon entropy over synonym frequencies
- **Fluency**: Perplexity under reference LM
- **Path Diversity**: Unique (agentic, communal) pairs
- **Bias-Shift**: Jensen-Shannon divergence

## Materials

### 20 Neutral Occupations
architect, artist, barista, chef, counselor, doctor, electrician, engineer, journalist, lawyer, mechanic, nurse, pharmacist, photographer, pilot, plumber, scientist, teacher, salesperson, writer

### Synonym Lists (Gaucher et al., 2011)
**Agentic (10)**: ambitious, assertive, bold, confident, decisive, independent, self-reliant, competitive, adventurous, dominant

**Communal (10)**: accommodating, caring, cooperative, empathetic, friendly, nurturing, supportive, compassionate, helpful, loyal

### Prompt Template
```
Complete the following sentence in natural, coherent English (8–15 words long):
"The [OCCUPATION] was"
```

## Quick Start

### 1. Setup Environment
```bash
# For prompt-only and gen-filter (no GPU needed)
pip install openai together pandas python-dotenv

# For SFT fine-tuning (GPU required)
cd rlhf/
pip install torch transformers peft datasets accelerate bitsandbytes
```

### 2. Run Experiments

**Prompt-Only Baseline:**
```bash
cd prompt-only/
python prompt-only-gpt4o.py  # or prompt-only-llama.py
```

**Generate-and-Filter:**
```bash
cd gen-filter/
python gen-filter-gpt4o.py  # or gen-filter-llama.py
```

**SFT Training (Remote GPU Server Required):**
```bash
# On remote server with A6000/A100
cd sft/

# Train for one seed
CUDA_VISIBLE_DEVICES=0 python train_sft_lora.py \
    --seed 42 \
    --output_dir ./sft_lora_paper_seed42

# Generate completions
CUDA_VISIBLE_DEVICES=0 python generate_sft_simple.py \
    --seed 42 \
    --model_dir ./sft_lora_paper_seed42_seed42
```

**DPO Training (Remote GPU Server Required):**
```bash
# On remote server with A6000/A100
cd dpo/

# Train for one seed
CUDA_VISIBLE_DEVICES=0 python train_dpo_lora.py \
    --seed 42 \
    --output_dir ./dpo_lora_paper_seed42

# Generate completions
CUDA_VISIBLE_DEVICES=0 python generate_dpo.py \
    --seed 42 \
    --model_dir ./dpo_lora_paper_seed42
```

**INLP (Remote GPU Server Required):**
```bash
# On remote server with A6000/A100
cd inlp/

# Compute projection matrix (fast: ~30 min)
CUDA_VISIBLE_DEVICES=0 python train_inlp.py \
    --seed 42 \
    --n_iterations 300 \
    --layer_idx -1 \
    --output_dir ./inlp_projection_seed42

# Generate completions
CUDA_VISIBLE_DEVICES=0 python generate_inlp.py \
    --seed 42 \
    --projection_dir ./inlp_projection_seed42 \
    --layer_idx -1
```

## Model Roster

### Prompt-Only & Gen-Filter
- **GPT-4o**: `chatgpt-4o-latest` (OpenAI)
- **LLaMA-4-Scout**: `meta-llama/Llama-4-Scout-17B-16E-Instruct` (Together AI)
- **LLaMA-3-70B**: `meta-llama/Llama-3.3-70B-Instruct-Turbo` (Together AI)

### SFT Fine-tuning
- **Base Model**: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **Method**: Supervised Fine-Tuning with LoRA (rank-8, alpha-16)
- **Training Data**: 750 programmatic examples (50 per occupation, 18 templates)
- **Hardware**: RTX A6000 (48GB) or A100 40GB/80GB
- **Training Time**: ~3 hours per seed
- **Seeds**: 42, 123, 456 (for reproducibility)

### DPO Fine-tuning
- **Base Model**: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **Method**: Direct Preference Optimization with LoRA (rank-8, alpha-16)
- **Training Data**: 750 preference pairs (chosen=balanced, rejected=unbalanced)
- **Hardware**: RTX A6000 (48GB) or A100 40GB/80GB
- **Training Time**: ~3-4 hours per seed
- **Seeds**: 42, 123, 456 (for reproducibility)

### INLP
- **Base Model**: `meta-llama/Meta-Llama-3.1-8B-Instruct` (unchanged)
- **Method**: Iterative Nullspace Projection (300 iterations)
- **Training Data**: 39 gendered word pairs (he/she, man/woman, etc.)
- **Hardware**: RTX A6000 (48GB) or A100 40GB/80GB
- **Training Time**: ~30 minutes per seed (10× faster than fine-tuning!)
- **Seeds**: 42, 123, 456 (for reproducibility)

### Ctrl-G Decoding
- Implemented separately with DFA constraints
- GPT-2 Ctrl-G, LLaMA 4.0+HMM, LLaMA 3.1-8B+HMM

## GPU Requirements

| Method | GPU Required | Memory | Time |
|--------|--------------|--------|------|
| Prompt-Only | ❌ No | - | ~1 hour |
| Gen-Filter | ❌ No | - | ~2 hours |
| SFT Training | ✅ Yes | ~10-15GB | ~3 hours |
| SFT Generation | ✅ Yes | ~10-15GB | ~3 hours |
| DPO Training | ✅ Yes | ~12-18GB | ~3-4 hours |
| DPO Generation | ✅ Yes | ~10-15GB | ~3 hours |
| INLP Training | ✅ Yes | ~10-15GB | **~30 min** ⚡ |
| INLP Generation | ✅ Yes | ~10-15GB | ~3 hours |

## SFT Performance

**Results (Seed 42):**
- **99.7% balanced outputs** (4983/5000 completions)
- Train occupations: 99.7% compliance
- Validation occupations: 99.6% compliance

This demonstrates that lightweight supervised fine-tuning successfully teaches the model to include both agentic and communal terms without hard constraints or post-filtering.

## Expected Outputs

### Prompt-Only
- `prompt_only_gpt4o_completions.csv`
- `prompt_only_llama_completions.csv`

### Generate-and-Filter
- `genfilter_gpt4o_raw.csv` + `genfilter_gpt4o_filtered.csv`
- `genfilter_llama_raw.csv` + `genfilter_llama_filtered.csv`

### SFT
- `sft/sft_lora_paper_seed{42,123,456}_seed{42,123,456}/` (trained models)
- `sft/sft_lora_completions_seed{42,123,456}.csv` (generated samples)

### DPO
- `dpo/dpo_lora_paper_seed{42,123,456}/` (trained models)
- `dpo/dpo_lora_completions_seed{42,123,456}.csv` (generated samples)

### INLP
- `inlp/inlp_projection_seed{42,123,456}/` (projection matrices)
- `inlp/inlp_completions_seed{42,123,456}.csv` (generated samples)

## Key Findings

**SFT vs Other Methods:**
- **Better compliance than Gen-Filter** (~99.7% vs ~30-50%)
- **Better fluency than Ctrl-G** (no hard constraints during generation)
- **Simpler than RLHF** (supervised learning vs reinforcement learning)
- **Efficient with LoRA** (~0.3% of parameters, 10-15GB VRAM)

## References

- Dathathri, S., et al. (2020). Plug and Play Language Models. ICLR.
- Gaucher, D., et al. (2011). Evidence That Gendered Wording in Job Advertisements Exists. JPSP.
- Hu, E., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.
- Rafailov, R., et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. arXiv:2305.18290.
- Ravfogel, S., et al. (2022). Null It Out: Guarding Protected Attributes. EMNLP.
- Rudinger, R., et al. (2018). Gender Bias in Coreference Resolution: Winogender Schemas.

## Next Steps

1. ✅ Collect prompt-only baselines
2. ✅ Run generate-and-filter
3. ✅ Implement Ctrl-G decoding (separate)
4. ✅ Train SFT models (seeds 42, 123, 456)
5. ✅ Generate SFT completions for all 3 seeds
6. ✅ Implement DPO training and generation scripts
7. ✅ Train DPO models (seeds 42, 123, 456)
8. 🔄 Generate DPO completions for all 3 seeds (in progress)
9. ✅ Implement INLP (linear projection debiasing)
10. ⏳ Train INLP projections (seeds 42, 123, 456)
11. ⏳ Generate INLP completions for all 3 seeds
12. ⏳ Run evaluation metrics across all methods
13. ⏳ Compare SFT vs DPO vs INLP: compliance, diversity, fluency, efficiency
14. ⏳ Statistical analysis and visualization
15. ⏳ Write manuscript

## License

Research project - see individual model licenses for usage terms.
