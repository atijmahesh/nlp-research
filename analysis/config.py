#!/usr/bin/env python3
"""
Configuration file for analysis scripts.
Contains synonym lists, file paths, and shared constants.
"""

# Synonym lists (from Gaucher et al., 2011)
AGENTIC_TERMS = [
    'ambitious', 'assertive', 'bold', 'confident', 'decisive',
    'independent', 'self-reliant', 'competitive', 'adventurous', 'dominant'
]

COMMUNAL_TERMS = [
    'accommodating', 'caring', 'cooperative', 'empathetic', 'friendly',
    'nurturing', 'supportive', 'compassionate', 'helpful', 'loyal'
]

# All synonyms combined
ALL_SYNONYMS = AGENTIC_TERMS + COMMUNAL_TERMS

# Occupations (20 from Winogender Schemas)
OCCUPATIONS = [
    'architect', 'artist', 'barista', 'chef', 'counselor',
    'doctor', 'electrician', 'engineer', 'journalist', 'lawyer',
    'mechanic', 'nurse', 'pharmacist', 'photographer', 'pilot',
    'plumber', 'scientist', 'teacher', 'salesperson', 'writer'
]

# Train/validation split
TRAIN_OCCUPATIONS = [
    'architect', 'artist', 'chef', 'counselor', 'doctor',
    'engineer', 'journalist', 'lawyer', 'nurse', 'pharmacist',
    'photographer', 'pilot', 'scientist', 'teacher', 'writer'
]

VALIDATION_OCCUPATIONS = [
    'barista', 'electrician', 'mechanic', 'plumber', 'salesperson'
]

# File paths (relative to results/)
DATA_FILES = {
    'prompt_only_gpt4o': 'prompt-only/prompt_only_gpt4o_combined.csv',
    'prompt_only_llama': 'prompt-only/prompt_only_llama_combined.csv',
    'genfilter_gpt4o': 'gen-filter/genfilter_gpt4o_filtered.csv',
    'genfilter_llama': 'gen-filter/genfilter_llama_filtered.csv',
    'ctrlg': 'ctrl-g/ctrlg_prefix_completions.csv',
    'sft_seed42': 'sft/sft_lora_completions_seed42.csv',
    'sft_seed123': 'sft/sft_lora_completions_seed123.csv',
    'sft_seed456': 'sft/sft_lora_completions_seed456.csv',
    'dpo_seed42': 'dpo/dpo_lora_completions_seed42.csv',
    'dpo_seed123': 'dpo/dpo_lora_completions_seed123.csv',
    'dpo_seed456': 'dpo/dpo_lora_completions_seed456.csv',
    'inlp_seed42': 'inlp/inlp_completions_seed42.csv',
    'inlp_seed123': 'inlp/inlp_completions_seed123.csv',
    'inlp_seed456': 'inlp/inlp_completions_seed456.csv',
}

# Method groupings
METHOD_GROUPS = {
    'Prompt-Only': ['prompt_only_gpt4o', 'prompt_only_llama'],
    'Gen-Filter': ['genfilter_gpt4o', 'genfilter_llama'],
    'Ctrl-G': ['ctrlg'],
    'SFT': ['sft_seed42', 'sft_seed123', 'sft_seed456'],
    'DPO': ['dpo_seed42', 'dpo_seed123', 'dpo_seed456'],
    'INLP': ['inlp_seed42', 'inlp_seed123', 'inlp_seed456'],
}

# Seeds for fine-tuned methods
SEEDS = [42, 123, 456]

# Output directories
OUTPUT_DIR = 'analysis_results'
FIGURES_DIR = f'{OUTPUT_DIR}/figures'
TABLES_DIR = f'{OUTPUT_DIR}/tables'

