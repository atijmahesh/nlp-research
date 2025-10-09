#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_sft.py
Supervised Fine-Tuning for LLaMA-3-8B-Instruct

Trains the model on balanced examples containing both agentic and communal terms.
Much simpler than RLHF and works with standard transformers library.

Expected training time: ~4-6 hours on A100
"""

import os
import re
import torch
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict
from dataclasses import dataclass

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
load_dotenv()

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./sft_llama3_8b_output"

OCCUPATIONS = [
    "architect", "artist", "barista", "chef", "counselor",
    "doctor", "electrician", "engineer", "journalist", "lawyer",
    "mechanic", "nurse", "pharmacist", "photographer", "pilot",
    "plumber", "scientist", "teacher", "salesperson", "writer"
]

AGENTIC = [
    "ambitious", "assertive", "bold", "confident", "decisive",
    "independent", "self-reliant", "competitive", "adventurous", "dominant"
]

COMMUNAL = [
    "accommodating", "caring", "cooperative", "empathetic", "friendly",
    "nurturing", "supportive", "compassionate", "helpful", "loyal"
]

# Training examples - balanced sentences with both agentic and communal terms
TRAINING_EXAMPLES = [
    # Each occupation gets multiple balanced examples
    "The architect was confident in her designs and caring toward her clients.",
    "The artist was bold in her vision yet supportive of other creators.",
    "The barista was friendly to customers and decisive during busy hours.",
    "The chef was ambitious in the kitchen and compassionate with the staff.",
    "The counselor was empathetic with clients and confident in her approach.",
    
    "The doctor was decisive in emergencies and caring with patients.",
    "The electrician was independent in his work and helpful to apprentices.",
    "The engineer was competitive in innovation yet cooperative with the team.",
    "The journalist was bold in reporting and accommodating of diverse views.",
    "The lawyer was assertive in court and empathetic with clients.",
    
    "The mechanic was self-reliant in problem-solving and friendly with customers.",
    "The nurse was caring with patients and confident in medical decisions.",
    "The pharmacist was helpful to customers and decisive about medications.",
    "The photographer was adventurous in composition and supportive of subjects.",
    "The pilot was confident in flying and caring about passenger safety.",
    
    "The plumber was independent in repairs and accommodating to homeowners.",
    "The scientist was ambitious in research and cooperative with colleagues.",
    "The teacher was nurturing to students and assertive in classroom management.",
    "The salesperson was friendly to clients and competitive in achieving goals.",
    "The writer was bold in storytelling and empathetic toward characters.",
    
    # More variations for each occupation
    "The architect was ambitious and nurturing toward junior staff members.",
    "The artist was assertive about creative vision and friendly at exhibitions.",
    "The barista was competitive in latte art and caring toward regulars.",
    "The chef was bold with flavors and supportive of kitchen staff.",
    "The counselor was confident in therapy and accommodating of different needs.",
    
    "The doctor was adventurous in treatments and compassionate with families.",
    "The electrician was decisive about safety and helpful during emergencies.",
    "The engineer was dominant in leadership and empathetic with team challenges.",
    "The journalist was independent in investigation and cooperative with editors.",
    "The lawyer was competitive in litigation and caring about justice.",
    
    "The mechanic was confident in diagnostics and supportive of car owners.",
    "The nurse was assertive with doctors and nurturing toward patients.",
    "The pharmacist was ambitious in service and friendly to everyone.",
    "The photographer was bold in style and accommodating to client needs.",
    "The pilot was decisive in navigation and caring about crew welfare.",
    
    "The plumber was adventurous with solutions and cooperative with contractors.",
    "The scientist was self-reliant in experiments and helpful to students.",
    "The teacher was competitive in education and compassionate with struggles.",
    "The salesperson was confident in pitches and empathetic to customer needs.",
    "The writer was independent in craft and supportive of fellow writers.",
]

# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────

def contains_term(text: str, term: str) -> bool:
    """Check if text contains a term."""
    pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    return pattern.search(text) is not None

def has_balanced_terms(text: str) -> bool:
    """Check if text has both agentic and communal terms."""
    has_agentic = any(contains_term(text, term) for term in AGENTIC)
    has_communal = any(contains_term(text, term) for term in COMMUNAL)
    return has_agentic and has_communal

def create_training_texts(examples: List[str], multiplier: int = 50) -> List[str]:
    """
    Create training dataset by repeating balanced examples.
    multiplier=50 gives us ~2500 training samples total.
    """
    # Filter to only balanced examples
    balanced = [ex for ex in examples if has_balanced_terms(ex)]
    print(f"Found {len(balanced)} balanced examples")
    
    # Repeat to create larger dataset
    training_texts = balanced * multiplier
    print(f"Created {len(training_texts)} training samples")
    
    return training_texts

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    """Main supervised fine-tuning."""
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    print(f"Loading model: {MODEL_NAME}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    # Prepare training data
    print("Preparing training data...")
    training_texts = create_training_texts(TRAINING_EXAMPLES, multiplier=50)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )
    
    # Create dataset
    dataset = Dataset.from_dict({"text": training_texts})
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    # Split into train/eval
    split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal language modeling
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # Effective batch size = 16
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=500,
        save_total_limit=3,
        fp16=False,
        bf16=True,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="tensorboard",
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    print("\nInitializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print(f"\n{'='*60}")
    print(f"Starting Supervised Fine-Tuning")
    print(f"Training for {training_args.num_train_epochs} epochs")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"{'='*60}\n")
    
    trainer.train()
    
    # Save final model
    print("\nTraining complete! Saving final model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save training info
    training_info = pd.DataFrame({
        "approach": ["Supervised Fine-Tuning"],
        "model": [MODEL_NAME],
        "training_samples": [len(train_dataset)],
        "epochs": [training_args.num_train_epochs],
        "learning_rate": [training_args.learning_rate],
        "output_dir": [OUTPUT_DIR],
    })
    training_info.to_csv(f"{OUTPUT_DIR}/training_info.csv", index=False)
    
    print(f"\n{'='*60}")
    print(f"✅ Training Complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"{'='*60}\n")
    
    # Print sample generations
    print("Testing with sample prompts...")
    model.eval()
    
    test_prompts = [
        "The engineer was",
        "The nurse was",
        "The pilot was",
    ]
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated}")

if __name__ == "__main__":
    main()

