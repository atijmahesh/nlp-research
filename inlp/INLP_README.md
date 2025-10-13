## INLP (Iterative Nullspace Projection) for Bias Mitigation

## Overview

This directory implements **Iterative Nullspace Projection (INLP)** as a third debiasing method, using linear algebra to remove gender bias from model representations.

**Paper:** Ravfogel et al. (2022) - [Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection](https://aclanthology.org/2022.acl-long.395/)

## What is INLP?

INLP is a **post-hoc debiasing** method that removes protected attribute information (e.g., gender) from model representations through iterative linear projections.

### Key Differences from SFT and DPO

| Aspect | SFT | DPO | INLP |
|--------|-----|-----|------|
| **Approach** | Fine-tune weights | Fine-tune weights | Modify representations |
| **Training Data** | Balanced examples | Preference pairs | Gendered word pairs |
| **Mechanism** | Supervised learning | Preference learning | Linear projection |
| **Model Weights** | Changed | Changed | **Unchanged** |
| **Computational Cost** | High (GPU hours) | High (GPU hours) | **Low (~30 min)** |

### How INLP Works

1. **Extract Embeddings**: Get hidden representations for gendered word pairs
   - Male: "he", "father", "man", "businessman", etc.
   - Female: "she", "mother", "woman", "businesswoman", etc.

2. **Train Classifier**: Train a linear classifier to predict gender from embeddings

3. **Find Gender Direction**: The classifier's decision boundary identifies the "gender direction" in embedding space

4. **Project Out**: Remove this direction by projecting embeddings onto its orthogonal complement (nullspace)

5. **Iterate**: Repeat 300 times to remove all traces of gender information

**Mathematical Formulation:**
```
For each iteration:
  1. Train classifier w to predict gender: w^T x
  2. Compute projection matrix: P = I - w(w^T w)^{-1}w^T  
  3. Project embeddings: x' = Px
  4. Accumulate all w vectors
```

After N iterations, the final projection matrix removes all N gender directions.

## Training Data

### Gendered Word Pairs (39 pairs)

**Pronouns**: he/she, him/her, his/hers, himself/herself

**Family**: man/woman, father/mother, son/daughter, brother/sister, etc.

**Professional**: actor/actress, waiter/waitress, businessman/businesswoman, etc.

**Titles**: mr/mrs, sir/madam, king/queen, prince/princess, etc.

## Training Configuration

### Method
- **Base**: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **No fine-tuning**: Model weights remain unchanged
- **Projection**: Applied at last hidden layer (layer -1)
- **Iterations**: 300 (removes 300 gender directions)

### Hyperparameters
- **Classifier**: Logistic Regression (sklearn)
- **Max iterations**: 1000 per classifier
- **Layer**: -1 (last hidden layer, 4096-dim embeddings)
- **Seeds**: 42, 123, 456 (for reproducibility)

### Hardware
- **GPU**: RTX A6000 (48GB) or A100 (40GB/80GB)
- **Memory**: ~10-15GB (loads model once)
- **Time**: ~30 minutes per seed (much faster than SFT/DPO!)

## Usage

### 1. Training (Compute Projection Matrix)

```bash
cd inlp/

# Train seed 42
CUDA_VISIBLE_DEVICES=0 python train_inlp.py \
    --seed 42 \
    --n_iterations 300 \
    --layer_idx -1 \
    --output_dir ./inlp_projection_seed42

# Train seed 123
CUDA_VISIBLE_DEVICES=1 python train_inlp.py \
    --seed 123 \
    --n_iterations 300 \
    --layer_idx -1 \
    --output_dir ./inlp_projection_seed123

# Train seed 456
CUDA_VISIBLE_DEVICES=2 python train_inlp.py \
    --seed 456 \
    --n_iterations 300 \
    --layer_idx -1 \
    --output_dir ./inlp_projection_seed456
```

This computes and saves the projection matrix to remove gender directions.

### 2. Generation

```bash
# Generate completions for seed 42
CUDA_VISIBLE_DEVICES=0 python generate_inlp.py \
    --seed 42 \
    --projection_dir ./inlp_projection_seed42 \
    --layer_idx -1

# Expected output: inlp_completions_seed42.csv
# 5,000 completions (20 occupations × 250 runs)
```

### 3. Run All Seeds

```bash
# On remote server with multiple GPUs
for seed in 42 123 456; do
    CUDA_VISIBLE_DEVICES=$((seed % 8)) nohup python train_inlp.py \
        --seed $seed \
        --n_iterations 300 \
        --layer_idx -1 \
        --output_dir ./inlp_projection_seed${seed} \
        > train_inlp_${seed}.log 2>&1 &
done

# Wait for training (~30 min), then generate
for seed in 42 123 456; do
    CUDA_VISIBLE_DEVICES=$((seed % 8)) nohup python generate_inlp.py \
        --seed $seed \
        --projection_dir ./inlp_projection_seed${seed} \
        --layer_idx -1 \
        > gen_inlp_${seed}.log 2>&1 &
done
```

## Expected Results

Based on INLP literature:
- **Moderate compliance**: ~40-60% balanced outputs (weaker than SFT/DPO)
- **Maintains fluency**: No degradation (model weights unchanged)
- **Fast computation**: 300× faster than SFT/DPO training
- **Interpretable**: Linear projection, mathematically grounded

## Comparison to Other Methods

### Research Questions
1. **Effectiveness**: Does linear debiasing match fine-tuning (SFT/DPO)?
2. **Efficiency**: Can we achieve good results without expensive training?
3. **Trade-offs**: Compliance vs computational cost?

**Hypothesis:**
- INLP will have **lower compliance** than SFT/DPO (~40-60% vs ~99%)
- INLP will be **much faster** (~30 min vs ~3-4 hours)
- INLP will **preserve fluency** better (no weight changes)

## Technical Details

### Projection at Generation Time

During generation, the projection is applied **dynamically** as a forward hook:

```python
class INLPProjectionHook:
    def apply_projection(self, module, input, output):
        hidden_states = output[0]
        projected = hidden_states @ projection_matrix.T
        return (projected,) + output[1:]
```

This allows using the base model without permanently modifying it.

### Why Layer -1?

The last hidden layer (layer -1) contains the most semantic information, where gender stereotypes are most likely encoded. Earlier layers focus on syntax/tokens.

## Output Files

```
inlp/
├── train_inlp.py                        # Training script
├── generate_inlp.py                     # Generation script  
├── INLP_README.md                       # This file
├── inlp_projection_seed42/              # Projection data (seed 42)
│   ├── inlp_projection_layer-1.pkl     # Projection matrix
│   └── inlp_config.pkl                 # Configuration
├── inlp_projection_seed123/             # Projection data (seed 123)
├── inlp_projection_seed456/             # Projection data (seed 456)
├── inlp_completions_seed42.csv          # Completions (seed 42)
├── inlp_completions_seed123.csv         # Completions (seed 123)
└── inlp_completions_seed456.csv         # Completions (seed 456)
```

## References

1. Ravfogel, S., Elazar, Y., Gonen, H., Twiton, M., & Goldberg, Y. (2022). Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection. *ACL 2022*.

2. Bolukbasi, T., et al. (2016). Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings. *NeurIPS 2016*.

3. Liang, P. P., et al. (2020). Towards Debiasing Sentence Representations. *ACL 2020*.

## Advantages

- ✅ **Fast**: 10× faster than fine-tuning
- ✅ **Interpretable**: Linear algebra, clear mechanism
- ✅ **Reversible**: Can remove/reapply projection
- ✅ **No training data needed**: Uses predefined word pairs
- ✅ **Preserves fluency**: Model weights unchanged

## Limitations

- ⚠️ **Lower effectiveness**: Linear methods can't capture complex biases
- ⚠️ **Requires word pairs**: Manual curation needed
- ⚠️ **Layer-specific**: Only modifies one layer
- ⚠️ **May lose information**: Removes all gender-correlated features

## Next Steps

1. ✅ Implement INLP training and generation
2. ⏳ Train INLP projections (3 seeds)
3. ⏳ Generate 15,000 completions (3 seeds × 5,000 each)
4. ⏳ Compare INLP vs SFT vs DPO: compliance, fluency, efficiency
5. ⏳ Statistical analysis and manuscript

