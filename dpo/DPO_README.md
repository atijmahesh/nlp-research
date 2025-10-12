# DPO (Direct Preference Optimization) for Bias Mitigation

## Overview

This directory implements **Direct Preference Optimization (DPO)** as a second debiasing method, comparing it to SFT.

**Paper:** Rafailov et al. (2023) - [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)

## What is DPO?

DPO is a **preference learning** method that directly optimizes a language model to prefer certain outputs over others, without requiring a separate reward model (unlike RLHF/PPO).

### Key Differences from SFT

| Aspect | SFT | DPO |
|--------|-----|-----|
| **Training Data** | Single outputs | Preference pairs (chosen vs rejected) |
| **Learning Signal** | Cross-entropy loss | Preference classification loss |
| **Objective** | Mimic good examples | Prefer good over bad examples |
| **Complexity** | Simpler | Slightly more complex |
| **Sample Efficiency** | Lower | Higher (learns from contrasts) |

### How DPO Works

1. **Preference Pairs**: For each occupation, create pairs:
   - **Chosen**: Balanced outputs (agentic + communal terms)
   - **Rejected**: Unbalanced outputs (only agentic OR communal)

2. **Training**: Optimize the policy to maximize:
   ```
   L_DPO = -log(σ(β * [log π_θ(chosen) - log π_ref(chosen) 
                        - log π_θ(rejected) + log π_ref(rejected)]))
   ```
   Where:
   - `π_θ` = policy model (being trained)
   - `π_ref` = reference model (frozen copy of base model)
   - `β` = temperature parameter (controls strength of preference)
   - `σ` = sigmoid function

3. **Intuition**: Push the model to assign higher probability to balanced outputs relative to the reference model, and lower probability to unbalanced outputs.

## Training Data

### Preference Generation

**Balanced (Chosen) Examples:**
- "The architect was **ambitious** and **caring** in their work."
- "Known for being **confident** yet **empathetic**, the engineer excelled."
- "The **bold** and **supportive** scientist built strong relationships."

**Unbalanced (Rejected) Examples:**
- "The architect was **ambitious** and driven to succeed." (agentic only)
- "The engineer was **caring** and focused on others." (communal only)
- "The **bold** scientist dominated their field." (agentic only)

### Dataset Size
- **Training**: 750 preference pairs (15 occupations × 50 pairs)
- **Validation**: 250 preference pairs (5 occupations × 50 pairs)

## Training Configuration

### Model
- **Base**: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **LoRA**: rank=8, alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
- **Reference Model**: Frozen copy of base model

### Hyperparameters
- **β (DPO temperature)**: 0.1
- **Epochs**: 3
- **Batch size**: 2 per device (16 effective with gradient accumulation)
- **Learning rate**: 5e-5
- **Seeds**: 42, 123, 456 (for reproducibility)

### Hardware
- **GPU**: RTX A6000 (48GB) or A100 (40GB/80GB)
- **Memory**: ~12-15GB per model (policy + reference)
- **Time**: ~3-4 hours per seed

## Usage

### 1. Training

```bash
cd dpo/

# Train seed 42
CUDA_VISIBLE_DEVICES=0 python train_dpo_lora.py \
    --seed 42 \
    --output_dir ./dpo_lora_paper_seed42

# Train seed 123
CUDA_VISIBLE_DEVICES=1 python train_dpo_lora.py \
    --seed 123 \
    --output_dir ./dpo_lora_paper_seed123

# Train seed 456
CUDA_VISIBLE_DEVICES=2 python train_dpo_lora.py \
    --seed 456 \
    --output_dir ./dpo_lora_paper_seed456
```

### 2. Generation

```bash
# Generate completions for seed 42
CUDA_VISIBLE_DEVICES=0 python generate_dpo.py \
    --seed 42 \
    --model_dir ./dpo_lora_paper_seed42

# Expected output: dpo_lora_completions_seed42.csv
# 5,000 completions (20 occupations × 250 runs)
```

### 3. Run All Seeds

```bash
# On remote server with multiple GPUs
for seed in 42 123 456; do
    CUDA_VISIBLE_DEVICES=$((seed % 8)) nohup python train_dpo_lora.py \
        --seed $seed \
        --output_dir ./dpo_lora_paper_seed${seed} \
        > train_dpo_${seed}.log 2>&1 &
done

# Wait for training to complete, then generate
for seed in 42 123 456; do
    CUDA_VISIBLE_DEVICES=$((seed % 8)) nohup python generate_dpo.py \
        --seed $seed \
        --model_dir ./dpo_lora_paper_seed${seed} \
        > gen_dpo_${seed}.log 2>&1 &
done
```

## Expected Results

Based on DPO literature, we expect:
- **High compliance**: ~95-99% balanced outputs (similar to SFT)
- **Better sample efficiency**: Learns faster from fewer examples
- **Robust generalization**: Strong performance on validation occupations
- **Maintained fluency**: No degradation compared to base model

## Comparison to SFT

### Research Question
**Does preference learning (DPO) outperform supervised learning (SFT) for bias mitigation?**

**Metrics:**
1. **Constraint compliance**: % with both term types
2. **Lexical diversity**: Shannon entropy
3. **Fluency**: Perplexity under reference LM
4. **Training efficiency**: Convergence speed, final loss
5. **Generalization**: Train vs validation performance

### Hypothesis
DPO may achieve:
- **Similar or better compliance** (learns from contrasts)
- **Similar fluency** (both maintain naturalness)
- **Faster convergence** (more efficient learning signal)

## Output Files

```
dpo/
├── train_dpo_lora.py                    # Training script
├── generate_dpo.py                      # Generation script
├── DPO_README.md                        # This file
├── dpo_lora_paper_seed42/               # Trained model (seed 42)
├── dpo_lora_paper_seed123/              # Trained model (seed 123)
├── dpo_lora_paper_seed456/              # Trained model (seed 456)
├── dpo_lora_completions_seed42.csv      # Completions (seed 42)
├── dpo_lora_completions_seed123.csv     # Completions (seed 123)
└── dpo_lora_completions_seed456.csv     # Completions (seed 456)
```

## References

1. Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. *arXiv preprint arXiv:2305.18290*.

2. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.

3. Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS 2022*. (RLHF baseline)

## Next Steps

1. ✅ Implement DPO training and generation
2. ⏳ Train 3 seeds on remote GPU
3. ⏳ Generate 15,000 completions (3 seeds × 5,000 each)
4. ⏳ Compare DPO vs SFT: compliance, diversity, fluency
5. ⏳ Implement INLP (third debiasing method)
6. ⏳ Statistical analysis and manuscript

