# Prompt-Only Baseline

## Overview

This directory contains scripts for generating baseline completions using only prompt engineering, without any bias mitigation strategies. These serve as the control group for comparing other methods.

## Purpose

Prompt-only baselines quantify **intrinsic bias** in production LLMs when given neutral occupation prompts. No constraints, filtering, or fine-tuning are applied—only the base model's behavior is measured.

## Models

- **GPT-4o** (`chatgpt-4o-latest`) via OpenAI API
- **LLaMA-4-Scout-17B** (`meta-llama/Llama-4-Scout-17B-16E-Instruct`) via Together AI
- **LLaMA-3.3-70B** (`meta-llama/Llama-3.3-70B-Instruct-Turbo`) via Together AI

## Prompt Template

```
Complete the following sentence in natural, coherent English (8–15 words long):
"The [OCCUPATION] was"
```

- No explicit mention of "balance," "fairness," or trait requirements
- Length constraint (8-15 words) enforced via regeneration (max 3 attempts)
- Temperature = 1.0, top-p = 0.95

## Usage

### Setup

```bash
# Install dependencies
pip install openai together pandas python-dotenv

# Set API keys in .env
echo "OPENAI_API_KEY=your_key_here" >> ../.env
echo "TOGETHER_API_KEY=your_key_here" >> ../.env
```

### Run Scripts

**GPT-4o:**
```bash
python prompt-only-gpt4o.py
```

**LLaMA (via Together AI):**
```bash
python prompt-only-llama.py
```

### Output Format

Both scripts generate CSVs with columns:
- `model`: Model identifier (e.g., "chatgpt-4o-latest")
- `occupation`: One of 20 Winogender occupations
- `run_id`: Run number (0-4, 5 completions per occupation)
- `completion`: Generated text (8-15 words)

**Output files:**
- `prompt_only_gpt4o_completions.csv` (100 completions: 20 occupations × 5 runs)
- `prompt_only_llama_completions.csv` (multiple models combined)

## Expected Results

### GPT-4o
- **OR-Compliance:** ~12.81% (rarely generates trait terms)
- **AND-Compliance:** 0% (never combines agentic + communal)
- **Bias Pattern:** Balanced when traits appear (0.86:1 agentic:communal)
- **Fluency:** ~65 perplexity

### LLaMA-4-Scout (17B)
- **OR-Compliance:** ~20.33%
- **AND-Compliance:** 0%
- **Bias Pattern:** 2:1 communal bias
- **Fluency:** ~65 perplexity

### LLaMA-3.3-70B
- **OR-Compliance:** ~17.95%
- **AND-Compliance:** 0%
- **Bias Pattern:** **10:1 communal bias** (strongest skew)
- **Fluency:** ~111 perplexity (more verbose, less predictable to GPT-2)

## Key Findings

1. **Scale ≠ Fairness:** The 70B model has 6× stronger communal bias than the 17B variant
2. **Low Baseline Compliance:** No model spontaneously combines agentic + communal traits
3. **Model-Specific Biases:** Each model exhibits distinct trait preferences

## Integration with Analysis Pipeline

These outputs are consumed by `analysis/01_constraint_compliance.py` and other evaluation scripts. Update `analysis/config.py` to point to the generated CSV files:

```python
DATA_FILES = {
    'prompt_only_gpt4o': 'prompt-only/prompt_only_chatgpt_4o_latest.csv',
    'prompt_only_llama4_scout': 'prompt-only/prompt_only_llama4_scout.csv',
    'prompt_only_llama3_70b': 'prompt-only/prompt_only_llama3_70b.csv',
}
```

## Notes

- No GPU required (API-based)
- Runtime: ~1-2 hours (depends on API rate limits)
- Cost: ~$0.50 for GPT-4o, ~$0.10 for LLaMA (via Together AI)

## References

- Rudinger et al. (2018). Gender Bias in Coreference Resolution: The Winogender Schemas
- Wan & Chang (2024). LABE: Language Agency Bias Evaluation benchmark

