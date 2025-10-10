#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_sft_lora.py
Memory-Efficient Supervised Fine-Tuning with LoRA

Uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
Only trains 0.1% of model parameters, drastically reducing memory usage.

Expected memory: ~10-15GB (vs. 47GB for full fine-tuning)
Expected training time: ~2-3 hours per seed on A100
"""

import os
import re
import json
import torch
import random
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import Counter

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
import argparse

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
load_dotenv()

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# All 20 occupations
ALL_OCCUPATIONS = [
    "architect", "artist", "barista", "chef", "counselor",
    "doctor", "electrician", "engineer", "journalist", "lawyer",
    "mechanic", "nurse", "pharmacist", "photographer", "pilot",
    "plumber", "scientist", "teacher", "salesperson", "writer"
]

# Split: 15 for training, 5 for validation
TRAIN_OCCUPATIONS = [
    "architect", "artist", "barista", "chef", "counselor",
    "doctor", "electrician", "engineer", "journalist", "lawyer",
    "mechanic", "nurse", "pharmacist", "photographer", "pilot"
]

VAL_OCCUPATIONS = [
    "plumber", "scientist", "teacher", "salesperson", "writer"
]

AGENTIC_TERMS = [
    "ambitious", "assertive", "bold", "confident", "decisive",
    "independent", "self-reliant", "competitive", "adventurous", "dominant"
]

COMMUNAL_TERMS = [
    "accommodating", "caring", "cooperative", "empathetic", "friendly",
    "nurturing", "supportive", "compassionate", "helpful", "loyal"
]

# ─── EXAMPLE GENERATION (same as robust version) ─────────────────────────────

class BalancedExampleGenerator:
    """Generate diverse balanced examples programmatically."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        self.templates = [
            "The {occ} was {ag} and {com}.",
            "The {occ} was {ag} yet {com}.",
            "The {occ} was {ag} while being {com}.",
            "The {occ} was both {ag} and {com}.",
            "The {occ} was {ag} in {ctx1} and {com} with {ctx2}.",
            "The {occ} was {ag} during {ctx1} yet {com} toward {ctx2}.",
            "The {occ} was {com} with {ctx2} and {ag} in {ctx1}.",
            "The {occ} was {com} toward {ctx2} while being {ag} about {ctx1}.",
            "The {occ} was {ag1}, {ag2}, and {com} to everyone.",
            "The {occ} was {com1} and {com2} yet {ag} when needed.",
            "The {occ} was {ag} in leadership and {com1}, {com2} with the team.",
            "The {occ} was {ag} about professional goals and {com} with colleagues.",
            "The {occ} was known for being {ag} and remarkably {com}.",
            "The {occ} was {com} in daily interactions while remaining {ag}.",
            "The {occ} was {ag} when making decisions but {com} in implementation.",
            "The {occ} was {ag} in pursuing excellence and {com} toward others.",
            "The {occ} was {com} when working with people and {ag} about outcomes.",
            "The {occ} was {ag} in professional matters yet {com} personally.",
        ]
        
        self.contexts = {
            "work": ["work", "the job", "projects", "tasks", "responsibilities"],
            "people": ["clients", "customers", "patients", "students", "colleagues", "team members"],
            "skills": ["technical skills", "craft", "practice", "expertise"],
            "decisions": ["decisions", "choices", "judgment", "planning"],
        }
    
    def get_context(self, occupation: str, context_type: str) -> str:
        contexts = self.contexts[context_type]
        if occupation in ["doctor", "nurse", "pharmacist"]:
            if context_type == "people":
                return random.choice(["patients", "families", "staff"])
        elif occupation in ["teacher", "counselor"]:
            if context_type == "people":
                return random.choice(["students", "learners"])
        return random.choice(contexts)
    
    def generate_examples(self, occupations: List[str], n_per_occupation: int = 20) -> List[str]:
        examples = []
        for occ in occupations:
            for _ in range(n_per_occupation):
                template = random.choice(self.templates)
                example = template.format(
                    occ=occ,
                    ag=random.choice(AGENTIC_TERMS),
                    ag1=random.choice(AGENTIC_TERMS),
                    ag2=random.choice([t for t in AGENTIC_TERMS if t != random.choice(AGENTIC_TERMS)]),
                    com=random.choice(COMMUNAL_TERMS),
                    com1=random.choice(COMMUNAL_TERMS),
                    com2=random.choice([t for t in COMMUNAL_TERMS if t != random.choice(COMMUNAL_TERMS)]),
                    ctx1=self.get_context(occ, random.choice(["work", "skills", "decisions"])),
                    ctx2=self.get_context(occ, "people"),
                )
                examples.append(example)
        return examples

# ─── VALIDATION FUNCTIONS ────────────────────────────────────────────────────

def contains_term(text: str, term: str) -> bool:
    pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    return pattern.search(text) is not None

def is_balanced(text: str) -> bool:
    has_agentic = any(contains_term(text, term) for term in AGENTIC_TERMS)
    has_communal = any(contains_term(text, term) for term in COMMUNAL_TERMS)
    return has_agentic and has_communal

def validate_dataset(examples: List[str]) -> Dict:
    stats = {
        "total": len(examples),
        "balanced": sum(1 for ex in examples if is_balanced(ex)),
    }
    return stats

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ─── MAIN TRAINING WITH LORA ─────────────────────────────────────────────────

def main(args):
    set_seed(args.seed)
    
    output_dir = f"{args.output_dir}_seed{args.seed}"
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*70}")
    print(f"MEMORY-EFFICIENT SFT WITH LORA")
    print(f"{'='*70}")
    print(f"Model: {MODEL_NAME}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_dir}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"{'='*70}\n")
    
    # Generate examples
    print("Generating training examples...")
    generator = BalancedExampleGenerator(seed=args.seed)
    train_examples = generator.generate_examples(TRAIN_OCCUPATIONS, n_per_occupation=args.n_per_occupation)
    val_examples = generator.generate_examples(VAL_OCCUPATIONS, n_per_occupation=max(5, args.n_per_occupation // 4))
    
    print(f"Generated {len(train_examples)} training examples")
    print(f"Generated {len(val_examples)} validation examples")
    
    # Validate
    train_stats = validate_dataset(train_examples)
    print(f"\nTraining set: {train_stats['balanced']}/{train_stats['total']} balanced ({train_stats['balanced']/train_stats['total']*100:.1f}%)")
    
    # Save examples
    pd.DataFrame({"text": train_examples}).to_csv(f"{output_dir}/train_examples.csv", index=False)
    pd.DataFrame({"text": val_examples}).to_csv(f"{output_dir}/val_examples.csv", index=False)
    
    # Load model and tokenizer
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    # Configure LoRA
    print(f"\nConfiguring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,                    # Rank
        lora_alpha=args.lora_alpha,      # Scaling factor
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )
    
    train_dataset = Dataset.from_dict({"text": train_examples})
    val_dataset = Dataset.from_dict({"text": val_examples})
    
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)
    
    print(f"\nTokenized datasets:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=3,
        fp16=False,
        bf16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        logging_dir=f"{output_dir}/logs",
        seed=args.seed,
    )
    
    # Trainer
    print("\nInitializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print(f"\n{'='*70}")
    print(f"Starting Training with LoRA")
    print(f"{'='*70}\n")
    
    train_result = trainer.train()
    
    # Save
    print("\nSaving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save metrics
    metrics = train_result.metrics
    eval_metrics = trainer.evaluate()
    
    training_info = {
        "model": MODEL_NAME,
        "method": "SFT-LoRA",
        "seed": args.seed,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "final_train_loss": metrics.get("train_loss"),
        "final_eval_loss": eval_metrics.get("eval_loss"),
    }
    
    with open(f"{output_dir}/training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ Training Complete!")
    print(f"{'='*70}")
    print(f"Model saved to: {output_dir}")
    print(f"Train loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"Eval loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./sft_lora_output")
    parser.add_argument("--n_per_occupation", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)  # Higher for LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    args = parser.parse_args()
    main(args)

