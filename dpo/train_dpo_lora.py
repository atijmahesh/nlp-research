#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) Training Script with LoRA
Paper: Rafailov et al. (2023) - Direct Preference Optimization

Trains LLaMA-3.1-8B-Instruct to prefer balanced outputs (agentic + communal terms)
over unbalanced outputs using preference learning.
"""

import argparse
import os
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from trl import DPOTrainer
import random

# ============================================================================
# OCCUPATIONS & SYNONYM LISTS
# ============================================================================

ALL_OCCUPATIONS = [
    "architect", "artist", "barista", "chef", "counselor",
    "doctor", "electrician", "engineer", "journalist", "lawyer",
    "mechanic", "nurse", "pharmacist", "photographer", "pilot",
    "plumber", "scientist", "teacher", "salesperson", "writer"
]

TRAIN_OCCUPATIONS = [
    "architect", "artist", "chef", "counselor", "doctor",
    "engineer", "journalist", "lawyer", "nurse", "pharmacist",
    "photographer", "pilot", "scientist", "teacher", "writer"
]

VAL_OCCUPATIONS = [
    "barista", "electrician", "mechanic", "plumber", "salesperson"
]

AGENTIC_TERMS = [
    "ambitious", "assertive", "bold", "confident", "decisive",
    "independent", "self-reliant", "competitive", "adventurous", "dominant"
]

COMMUNAL_TERMS = [
    "accommodating", "caring", "cooperative", "empathetic", "friendly",
    "nurturing", "supportive", "compassionate", "helpful", "loyal"
]

# ============================================================================
# TEMPLATE GENERATORS
# ============================================================================

BALANCED_TEMPLATES = [
    "The {occ} was {agentic} and {communal} in their work.",
    "Known for being {agentic} yet {communal}, the {occ} excelled.",
    "The {agentic} and {communal} {occ} built strong relationships.",
    "As a {agentic} professional, the {occ} remained {communal}.",
    "The {occ} balanced {agentic} drive with {communal} warmth.",
    "Being both {agentic} and {communal} made the {occ} exceptional.",
    "The {occ}'s {agentic} nature was complemented by their {communal} spirit.",
    "Despite being {agentic}, the {occ} stayed {communal} always.",
    "The {communal} {occ} also demonstrated {agentic} leadership.",
    "Colleagues admired the {occ}'s {agentic} yet {communal} approach.",
]

UNBALANCED_AGENTIC_TEMPLATES = [
    "The {occ} was {agentic} and driven to succeed.",
    "Known for being {agentic}, the {occ} excelled in competition.",
    "The {agentic} {occ} dominated their field.",
    "As a {agentic} professional, the {occ} led innovation.",
    "The {occ}'s {agentic} nature made them successful.",
]

UNBALANCED_COMMUNAL_TEMPLATES = [
    "The {occ} was {communal} and focused on others.",
    "Known for being {communal}, the {occ} supported everyone.",
    "The {communal} {occ} built strong relationships.",
    "As a {communal} professional, the {occ} helped colleagues.",
    "The {occ}'s {communal} nature made them beloved.",
]

def generate_balanced_example(occupation: str, seed: int) -> str:
    """Generate balanced output with both agentic and communal terms."""
    random.seed(seed)
    template = random.choice(BALANCED_TEMPLATES)
    agentic = random.choice(AGENTIC_TERMS)
    communal = random.choice(COMMUNAL_TERMS)
    return template.format(occ=occupation, agentic=agentic, communal=communal)

def generate_unbalanced_example(occupation: str, seed: int) -> str:
    """Generate unbalanced output with only agentic OR communal terms."""
    random.seed(seed)
    if random.random() < 0.5:
        # Agentic only
        template = random.choice(UNBALANCED_AGENTIC_TEMPLATES)
        agentic = random.choice(AGENTIC_TERMS)
        return template.format(occ=occupation, agentic=agentic)
    else:
        # Communal only
        template = random.choice(UNBALANCED_COMMUNAL_TEMPLATES)
        communal = random.choice(COMMUNAL_TERMS)
        return template.format(occ=occupation, communal=communal)

def build_prompt(occupation: str) -> str:
    """Build the instruction prompt."""
    return f'Complete the following sentence in natural, coherent English (8–15 words long): "The {occupation} was"'

# ============================================================================
# PREFERENCE DATA GENERATION
# ============================================================================

def create_preference_dataset(occupations: list, examples_per_occ: int = 50, seed: int = 42):
    """
    Create preference pairs for DPO training.
    Each pair: (prompt, chosen=balanced, rejected=unbalanced)
    """
    data = []
    
    for occ in occupations:
        for i in range(examples_per_occ):
            example_seed = seed + hash(occ) + i
            
            prompt = build_prompt(occ)
            chosen = generate_balanced_example(occ, example_seed)
            rejected = generate_unbalanced_example(occ, example_seed + 1000)
            
            data.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })
    
    return Dataset.from_list(data)

# ============================================================================
# TRAINING
# ============================================================================

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("DPO Training with LoRA")
    print("="*70)
    print(f"Model: meta-llama/Meta-Llama-3.1-8B-Instruct")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_dir}")
    print(f"Train occupations: {len(TRAIN_OCCUPATIONS)}")
    print(f"Val occupations: {len(VAL_OCCUPATIONS)}")
    print("="*70 + "\n")
    
    # Set seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Create datasets
    print("Creating preference datasets...")
    train_dataset = create_preference_dataset(TRAIN_OCCUPATIONS, examples_per_occ=50, seed=args.seed)
    eval_dataset = create_preference_dataset(VAL_OCCUPATIONS, examples_per_occ=50, seed=args.seed)
    
    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")
    print(f"\nExample preference pair:")
    print(f"  Prompt: {train_dataset[0]['prompt']}")
    print(f"  Chosen: {train_dataset[0]['chosen']}")
    print(f"  Rejected: {train_dataset[0]['rejected']}")
    print()
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # DPO requires left padding
    
    # Load reference model (frozen)
    print("Loading reference model (frozen)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Configure LoRA for policy model only
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Reduced from 2 due to dual-model memory
        per_device_eval_batch_size=1,   # Reduced from 2
        gradient_accumulation_steps=16,  # Increased to maintain effective batch size
        learning_rate=5e-5,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,  # Enable to save memory
    )
    
    # DPO Trainer
    print("\nInitializing DPO Trainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer
        beta=0.1,  # DPO temperature parameter
        max_prompt_length=128,
        max_length=256,
    )
    
    # Train
    print("\n" + "="*70)
    print("Starting DPO Training...")
    print("="*70 + "\n")
    
    dpo_trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    dpo_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Get final metrics
    eval_results = dpo_trainer.evaluate()
    
    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)
    print(f"Model saved to: {output_dir}")
    print(f"Final eval loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
    print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    
    main(args)

