# RLHF Training Guide

## Overview
This guide covers fine-tuning LLaMA-3-7B-Instruct with PPO (Proximal Policy Optimization) to generate balanced occupational descriptions containing both agentic and communal terms.

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements_rlhf.txt
```

### 2. GPU Requirements
- **Recommended**: A100 40GB or A100 80GB
- **Minimum**: RTX 3090 / RTX 4090 (24GB) with 8-bit quantization
- **Training time**: ~12 hours on A100

### 3. Model Access
Ensure you have access to `meta-llama/Llama-3.1-7B-Instruct` on HuggingFace:
```bash
huggingface-cli login
```

## Training

### Full Precision Training (Recommended for A100)
```bash
python train_rlhf.py
```

### 8-bit Quantized Training (For lower memory GPUs)
Edit `train_rlhf.py` and change:
```python
model, ref_model, tokenizer = setup_model_and_tokenizer(use_8bit=True)
```

Then run:
```bash
python train_rlhf.py
```

### Remote SSH Server
If training on a remote server:
```bash
# SSH into server
ssh your-gpu-server

# Clone repo or upload files
# Install dependencies
pip install -r requirements_rlhf.txt

# Run training in background with logging
nohup python train_rlhf.py > training.log 2>&1 &

# Monitor progress
tail -f training.log

# Or use tmux/screen for better control
tmux new -s rlhf_training
python train_rlhf.py
# Ctrl+B, then D to detach
# tmux attach -t rlhf_training to reattach
```

## Monitoring Training

### TensorBoard
Training logs are saved to TensorBoard:
```bash
tensorboard --logdir=runs
```

### Key Metrics to Watch
- **Mean Reward**: Should increase over time (target: ~0.7-1.0)
- **KL Divergence**: Should stay around 0.03 (target_kl)
- **Sample Quality**: Check printed samples every 50 steps

### Signs of Reward Hacking
- Repetitive outputs (same synonyms)
- Unnatural sentence structure
- Very high rewards but poor fluency

**Solutions**:
- Increase repetition penalty weight in reward function
- Adjust learning rate or target_kl
- Add more diversity bonuses

## Generation

After training completes:
```bash
python generate_rlhf.py
```

This will:
- Load the fine-tuned model from `./rlhf_llama3_7b_output`
- Generate 250 completions per occupation (5,000 total)
- Save to `rlhf_llama3_completions.csv`

## Output Files

```
rlhf_llama3_7b_output/          # Final fine-tuned model
├── config.json
├── model.safetensors
├── tokenizer.json
└── training_metrics.csv         # Reward and KL over time

rlhf_checkpoints/                # Intermediate checkpoints
├── checkpoint-100/
├── checkpoint-200/
└── ...

rlhf_llama3_completions.csv     # Generated samples for evaluation
```

## Reward Function Details

The reward function implements the research protocol:

**Base Rewards:**
- `+1.0`: Contains ≥1 agentic AND ≥1 communal term
- `+0.3`: Contains only one term class
- `-0.2`: Contains no terms

**Bonuses:**
- `+0.05` per extra unique term (beyond first of each class)
- Max bonus: `+0.2`

**Penalties:**
- `-0.1`: Length outside 8-15 words
- `-0.1`: Any synonym repeated >2 times

**Example:**
```
Text: "The nurse was confident and caring in the emergency room."
- Has agentic: "confident" ✓
- Has communal: "caring" ✓
- Base reward: +1.0
- Extra terms: 0
- Length: 10 words ✓
- No repetitions ✓
- Final reward: 1.0
```

## Hyperparameters

```python
BATCH_SIZE = 64                  # Total batch size
MINI_BATCH_SIZE = 16             # Mini-batch for gradient updates
PPO_EPOCHS = 4                   # PPO update epochs per batch
LEARNING_RATE = 1e-5             # Adam learning rate
TARGET_KL = 0.03                 # KL divergence target
TRAINING_STEPS = 2000            # Total training steps
```

## Troubleshooting

### Out of Memory
1. Enable 8-bit quantization
2. Reduce `BATCH_SIZE` to 32 or 16
3. Reduce `MINI_BATCH_SIZE` to 8
4. Enable gradient checkpointing (already enabled)

### Training Instability
1. Reduce `LEARNING_RATE` to 5e-6
2. Increase `TARGET_KL` to 0.05
3. Check for NaN values in rewards

### Poor Sample Quality
1. Check reward function is working correctly
2. Verify model is generating diverse outputs
3. Adjust temperature/top_p if needed
4. Increase diversity bonuses

### Model Not Loading
1. Check HuggingFace authentication
2. Verify model name is correct
3. Ensure sufficient disk space (~15GB)

## Next Steps

After generation:
1. Run evaluation metrics (constraint compliance, diversity, fluency)
2. Compare with baseline, Ctrl-G, and Gen+Filter methods
3. Analyze trade-offs between constraint adherence and fluency
4. Generate visualizations and statistical comparisons

