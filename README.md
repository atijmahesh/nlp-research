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
├── rlhf/                 # RLHF Fine-tuning with PPO
│   ├── train_rlhf.py              # Training script (~12h on A100)
│   ├── generate_rlhf.py           # Generate 250 samples per occupation
│   ├── test_reward_function.py   # Test reward logic
│   ├── requirements_rlhf.txt     # Dependencies
│   └── RLHF_README.md            # Detailed RLHF guide
│
├── listModels.py         # Utility: List available OpenAI models
└── README.md             # This file
```

## Research Overview

This study evaluates **four control strategies** to mitigate gender bias in LLMs:

1. **Prompt-Only** ✅ - Simple prompting with length constraints
2. **Generate-and-Filter** ✅ - Post-hoc filtering for stereotypical terms
3. **Ctrl-G Decoding** ✅ - DFA-based constrained generation (separate implementation)
4. **RLHF Fine-tuning** 🔄 - Reinforcement learning to encourage balanced outputs

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
# Copy .env.example to .env and add your API keys
cp .env.example .env

# For prompt-only and gen-filter
pip install openai together pandas python-dotenv

# For RLHF (GPU required)
cd rlhf/
pip install -r requirements_rlhf.txt
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

**RLHF Training (Remote GPU Server Required):**
```bash
# On remote server with A100/RTX 4090
cd rlhf/
python train_rlhf.py           # Train (~12 hours)
python generate_rlhf.py        # Generate samples
```

## Model Roster

### Prompt-Only & Gen-Filter
- **GPT-4o**: `chatgpt-4o-latest` (OpenAI)
- **LLaMA-4-Scout**: `meta-llama/Llama-4-Scout-17B-16E-Instruct` (Together AI)
- **LLaMA-3-70B**: `meta-llama/Llama-3.3-70B-Instruct-Turbo` (Together AI)

### RLHF Fine-tuning
- **Base Model**: `meta-llama/Llama-3.1-7B-Instruct`
- **Method**: PPO with reward function for balanced terms
- **Hardware**: A100 40GB/80GB recommended

### Ctrl-G Decoding
- Implemented separately with DFA constraints
- GPT-2 Ctrl-G, LLaMA 4.0+HMM, LLaMA 3.1-8B+HMM

## GPU Requirements

| Method | GPU Required | Memory | Time |
|--------|--------------|--------|------|
| Prompt-Only | ❌ No | - | ~1 hour |
| Gen-Filter | ❌ No | - | ~2 hours |
| RLHF Training | ✅ Yes | 40GB+ | ~12 hours |
| RLHF Generation | ✅ Yes | 16GB+ | ~2 hours |

## Expected Outputs

### Prompt-Only
- `prompt_only_gpt4o_completions.csv`
- `prompt_only_llama_completions.csv`

### Generate-and-Filter
- `genfilter_gpt4o_raw.csv` + `genfilter_gpt4o_filtered.csv`
- `genfilter_llama_raw.csv` + `genfilter_llama_filtered.csv`

### RLHF
- `rlhf/rlhf_llama3_7b_output/` (trained model)
- `rlhf/rlhf_llama3_completions.csv` (generated samples)
- `rlhf/rlhf_llama3_7b_output/training_metrics.csv` (training logs)

## References

- Dathathri, S., et al. (2020). Plug and Play Language Models. ICLR.
- Gaucher, D., et al. (2011). Evidence That Gendered Wording in Job Advertisements Exists. JPSP.
- Ravfogel, S., et al. (2022). Null It Out: Guarding Protected Attributes. EMNLP.
- Rudinger, R., et al. (2018). Gender Bias in Coreference Resolution: Winogender Schemas.

## Next Steps

1. ✅ Collect prompt-only baselines
2. ✅ Run generate-and-filter
3. ✅ Implement Ctrl-G decoding (separate)
4. 🔄 Train RLHF model (in progress)
5. ⏳ Run evaluation metrics across all methods
6. ⏳ Statistical analysis and visualization
7. ⏳ Write manuscript

## License

Research project - see individual model licenses for usage terms.

