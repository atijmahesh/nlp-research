# Ctrl-G: DFA-Based Constrained Decoding

## Overview

This directory implements **Ctrl-G** (Controlled Generation), a hard constraint method using Deterministic Finite Automata (DFA) to enforce logical formulas during decoding. This guarantees 100% compliance with specified constraints but may reduce fluency and lexical diversity.

## Purpose

Ctrl-G serves as a **symbolic baseline** to compare:
- **Hard constraints** (DFA-enforced) vs. **soft learning** (SFT, DPO)
- **OR logic** (≥1 trait) vs. **AND logic** (both traits required)
- **Guarantee** (perfect compliance) vs. **flexibility** (generalization, diversity)

## Method

Ctrl-G constructs a DFA that accepts only outputs matching a specified formula:

```python
# OR constraint: agentic OR communal
prod = DFA_prod([agentic_DFA, communal_DFA], mode="union")

# AND constraint: agentic AND communal
prod = DFA_prod([agentic_DFA, communal_DFA], mode="intersection")
```

During generation, the model's vocabulary is dynamically masked to ensure only valid transitions through the DFA are allowed.

## Variants

### Ctrl-G (OR)
- **Formula:** `(agentic+ | communal+)` 
- **Requires:** ≥1 agentic term **OR** ≥1 communal term
- **Compliance:** 99.69% OR (due to length constraints causing some failures)
- **AND Compliance:** 0% (never combines both traits)
- **Bias Pattern:** 2.6:1 agentic bias (71.88% agentic-only, 27.81% communal-only)

### Ctrl-G (AND)
- **Formula:** `(agentic+) & (communal+)`
- **Requires:** ≥1 agentic term **AND** ≥1 communal term
- **Compliance:** 100% (by construction)
- **Lexical Diversity:** 13 unique pairs (low due to beam search favoring high-probability paths)
- **Fluency:** 29.53 perplexity (45% higher than OR variant)

## Base Model

- **GPT-2-Large** (774M parameters)
- Beam search (beam=16)
- Terminate on period token
- Rank outputs by log-probability

## Usage

### Installation

```bash
# Install Ctrl-G package
cd ctrl-g/
pip install -e .

# Or install from parent directory
cd ..
pip install -e ctrl-g/
```

### Generate with OR Constraints

```python
python ctrl-g/generate_ctrlg_gpt2.py --mode or
```

This generates completions with ≥1 agentic OR ≥1 communal term.

### Generate with AND Constraints

```python
python ctrl-g/generate_ctrlg_gpt2.py --mode and
```

This generates completions with ≥1 agentic AND ≥1 communal term.

### Output Format

CSV with columns:
- `occupation`: One of 20 Winogender occupations
- `sample`: Generated completion
- `run_id`: Run number (0-499, 500 samples per occupation)

**Output files:**
- `ctrlg_prefix_completions.csv` (OR variant)
- `ctrlg_and_completions.csv` (AND variant)

## Expected Results

### Ctrl-G (OR)

| Metric | Value |
|--------|-------|
| OR-Compliance | 99.69% |
| AND-Compliance | 0% |
| Agentic-Only | 71.88% |
| Communal-Only | 27.81% |
| Neither | 0.31% (failures) |
| Fluency (PPL) | 20.31 |
| Entropy | 0.945 |
| Unique Pairs | 0 |

**Insight:** Disjunctive constraints increase trait term usage but don't enforce composition. The model exhibits strong agentic bias (2.6:1).

### Ctrl-G (AND)

| Metric | Value |
|--------|-------|
| OR-Compliance | 100% |
| AND-Compliance | 100% |
| Agentic-Only | 0% |
| Communal-Only | 0% |
| Neither | 0% |
| Fluency (PPL) | 29.53 |
| Entropy | 1.313 |
| Unique Pairs | 13 |

**Insight:** Perfect compliance achieved, but at the cost of repetitive phrasing. Beam search favors high-probability clichés like "confident and caring."

## Comparison to Learning Methods

| Method | AND Compliance | Unique Pairs | Fluency | Training Time |
|--------|----------------|--------------|---------|---------------|
| **Ctrl-G (AND)** | 100% | 13 | 29.53 | N/A |
| **SFT** | 99.87% | 100 | 67.77 | ~3h |
| **DPO** | 4.53% | 24 | 76.77 | ~3-4h |

**Key Finding:** SFT achieves nearly identical compliance (99.87% vs. 100%) with 7× higher diversity. This demonstrates that learning-based methods can approach symbolic guarantees while maintaining expressiveness.

## Limitations

1. **Upfront specification:** Constraints must be fully enumerated before generation
2. **Brittleness:** Small changes (OR → AND) yield drastically different outputs
3. **Low diversity:** Beam search exploits high-probability DFA paths, producing repetitive text
4. **No generalization:** Cannot adapt to novel constraint compositions without recompiling DFA

## Use Cases

Ctrl-G excels when:
- **Compliance is critical:** Regulatory, legal, or medical text requiring auditable guarantees
- **Constraints are well-defined:** Template-based generation with explicit rules
- **Diversity is secondary:** Acceptable to sacrifice lexical variety for correctness

Avoid Ctrl-G when:
- **Open-ended generation:** Creative writing, exploratory tasks
- **Evolving requirements:** Fairness criteria that change over time
- **High diversity needed:** Educational content requiring varied examples

## Integration with Analysis

Update `analysis/config.py`:

```python
DATA_FILES = {
    'ctrlg_or': 'ctrl-g/ctrlg_prefix_completions.csv',
    'ctrlg_and': 'ctrl-g/ctrlg_and_completions.csv',
}

METHOD_GROUPS = {
    'Ctrl-G (OR)': ['ctrlg_or'],
    'Ctrl-G (AND)': ['ctrlg_and'],
}
```

## Technical Details

### DFA Construction

```python
from ctrlg import DFA, Alphabet

# Define alphabet (vocabulary)
vocab = tokenizer.get_vocab()
acb = Alphabet(vocab)

# Build DFA for agentic terms
pats_a = [f".*\\b{term}\\b.*" for term in AGENTIC_TERMS]
agentic_dfa = acb.build(pats_a)

# Build DFA for communal terms
pats_c = [f".*\\b{term}\\b.*" for term in COMMUNAL_TERMS]
communal_dfa = acb.build(pats_c)

# Combine with AND logic
prod = DFA_prod([agentic_dfa, communal_dfa], mode="intersection")
```

### Constrained Beam Search

At each decoding step:
1. Compute next-token logits from GPT-2
2. Check which tokens maintain DFA validity
3. Mask invalid tokens (set logits to -inf)
4. Sample from valid tokens only

This guarantees the output satisfies the constraint.

## Notes

- **Runtime:** ~2-3 hours for 10,000 samples (500 per occupation × 20 occupations)
- **Hardware:** Can run on CPU (no GPU required, but slower)
- **Memory:** ~4GB for GPT-2-Large

## References

- Zhou et al. (2024). Ctrl-G: Controlled generation with deterministic finite automata for enforcing compositional constraints in large language models. arXiv:2404.02511
- Dathathri et al. (2020). Plug and Play Language Models. ICLR
- Original Ctrl-G: https://github.com/zorazrw/Ctrl-G

## Credit

This implementation is based on the Ctrl-G framework by Zhou et al. (2024). We adapted the generation script for our occupational bias task and added AND constraint support.

