# Generate-and-Filter

## Overview

This directory implements a **post-hoc filtering** approach to bias mitigation: generate many completions freely, then filter for outputs containing at least one agentic **OR** one communal term.

This approximates industry pipelines that prioritize fluency over balanced content.

## Purpose

Generate-and-filter serves as a **weak baseline** to test whether simple post-processing can:
1. Increase trait term frequency vs. prompt-only
2. Achieve compositional balance (spoiler: it cannot)

## Method

```
For each occupation:
    1. Generate 100 raw completions (no constraints)
    2. Filter: Keep only outputs with ≥1 agentic OR ≥1 communal term
    3. Cap at 250 retained samples per model
```

**Key limitation:** This uses **OR logic**, not AND. It increases trait term usage but does not enforce balanced combinations.

## Models

Same as prompt-only:
- **GPT-4o** (`chatgpt-4o-latest`)
- **LLaMA-4-Scout-17B**
- **LLaMA-3.3-70B**

## Usage

### Setup

```bash
# Same as prompt-only
pip install openai together pandas python-dotenv

# Ensure API keys in .env
```

### Run Scripts

**GPT-4o:**
```bash
python gen-filter-gpt4o.py
```

**LLaMA:**
```bash
python gen-filter-llama.py
```

### Output Files

Each script generates **two CSVs**:

1. **Raw outputs** (before filtering):
   - `genfilter_gpt4o_raw.csv`
   - `genfilter_llama_raw.csv`
   - Columns: `Model`, `Occupation`, `RunID`, `Text`

2. **Filtered outputs**:
   - `genfilter_gpt4o_filtered.csv`
   - `genfilter_llama_filtered.csv`
   - Only includes samples with ≥1 trait term

## Expected Results

### Compliance

| Model | Raw Samples | Retained (OR) | OR-Compliance | AND-Compliance |
|-------|-------------|---------------|---------------|----------------|
| GPT-4o | 2000 | ~256 | ~12.75% | **0%** |
| LLaMA-4-Scout | 2000 | ~411 | ~20.55% | **0%** |
| LLaMA-3.3-70B | 2000 | ~359 | ~17.95% | **0%** |

**Key insight:** Filtering only selects outputs the model was already likely to generate. It cannot redistribute traits or create balanced combinations that don't naturally occur.

### Comparison to Prompt-Only

| Metric | GPT-4o Prompt | GPT-4o Filter | Change |
|--------|---------------|---------------|--------|
| OR-Compliance | 12.81% | 12.75% | -0.06% |
| AND-Compliance | 0% | 0% | 0% |

Filtering produces **nearly identical** results to prompt-only, confirming that post-hoc selection cannot fix intrinsic bias.

## Why Generate-and-Filter Fails

1. **Selection bias:** Can only choose from the base model's natural distribution
2. **OR vs AND:** Filtering for OR (≥1 trait) doesn't enforce AND (both traits)
3. **No learning:** The model never learns to compose balanced descriptions

From the paper (Section 5.4):
> "Generate-and-filter models simulate prompt-only compliance (e.g., GPT-4o 12.75 vs. 12.81%). Post-hoc selection can't redistribute traits because balanced outputs don't spontaneously appear."

## Limitations

- **High cost:** Generating 100× more samples than needed
- **Low yield:** Most outputs are discarded
- **No composition:** Cannot enforce conjunctive constraints
- **Bias amplification:** May reinforce existing skews

## Use Cases

Despite poor compositional control, gen-filter may suit:
- **Exploratory tasks:** Where some trait diversity helps, but balance isn't critical
- **Creative writing:** Gentle steering without hard constraints
- **Low-stakes applications:** Where fluency > fairness

## Integration with Analysis

Update `analysis/config.py`:

```python
DATA_FILES = {
    'genfilter_gpt4o': 'gen-filter/genfilter_chatgpt_4o_latest.csv',
    'genfilter_llama4_scout': 'gen-filter/genfilter_llama4_scout.csv',
    'genfilter_llama3_70b': 'gen-filter/genfilter_llama3_70b.csv',
}
```

The analysis pipeline compares gen-filter to:
- **Prompt-only:** Same OR-compliance (proves filtering is ineffective)
- **SFT:** 99.87% AND-compliance (shows supervision is necessary)
- **Ctrl-G (OR):** 99.69% OR-compliance (shows constraints work)

## Notes

- No GPU required (API-based)
- Runtime: ~2-3 hours (5× slower than prompt-only due to volume)
- Cost: ~$2.50 for GPT-4o, ~$0.50 for LLaMA

## References

- Sheng et al. (2020). The Woman Worked as a Babysitter: On Biases in Language Generation
- Wan & Chang (2024). LABE: Measuring gendered agency in LLMs

