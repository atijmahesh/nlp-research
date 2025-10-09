#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_rlhf.py
RLHF Fine-tuning with PPO for LLaMA-3-7B-Instruct

Trains the model to generate balanced occupational descriptions containing
both agentic and communal terms, using reward-based learning.

Expected training time: ~12 hours on A100
"""

import os
import re
import torch
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Dict
from collections import Counter

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from datasets import Dataset

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
load_dotenv()

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./rlhf_llama3_7b_output"
CHECKPOINT_DIR = "./rlhf_checkpoints"

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

MIN_WORDS = 8
MAX_WORDS = 15
MAX_NEW_TOKENS = 64

# PPO Hyperparameters (as specified in research protocol)
BATCH_SIZE = 64
MINI_BATCH_SIZE = 16
PPO_EPOCHS = 4
LEARNING_RATE = 1e-5
TARGET_KL = 0.03
TRAINING_STEPS = 2000  # Adjust based on convergence
SAVE_FREQ = 100

# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────

def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.strip().split())

def contains_term(text: str, term: str) -> bool:
    """Check if text contains a word-boundary-delimited term (case-insensitive)."""
    pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    return pattern.search(text) is not None

def count_unique_terms(text: str, term_list: List[str]) -> int:
    """Count unique terms from term_list present in text."""
    return sum(1 for term in term_list if contains_term(text, term))

def get_term_counts(text: str) -> Dict[str, int]:
    """Get count of each synonym term in text."""
    all_terms = AGENTIC + COMMUNAL
    counts = {}
    for term in all_terms:
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        counts[term] = len(pattern.findall(text))
    return counts

def compute_reward(completion: str) -> float:
    """
    Compute reward for a completion based on research protocol:
    
    Base rewards:
    - +1.0: ≥1 agentic AND ≥1 communal term
    - +0.3: Only one class present
    - -0.2: No terms present
    
    Bonuses:
    - +0.05 per extra unique term (beyond first of each class), up to +0.2
    
    Penalties:
    - -0.1: Length outside 8-15 words
    - -0.1: Any synonym repeated >2 times
    """
    reward = 0.0
    
    # Count agentic and communal terms
    has_agentic = any(contains_term(completion, term) for term in AGENTIC)
    has_communal = any(contains_term(completion, term) for term in COMMUNAL)
    
    agentic_count = count_unique_terms(completion, AGENTIC)
    communal_count = count_unique_terms(completion, COMMUNAL)
    
    # Base reward
    if has_agentic and has_communal:
        reward += 1.0
        # Diversity bonus: extra unique terms beyond the first of each class
        extra_terms = max(0, (agentic_count + communal_count) - 2)
        diversity_bonus = min(0.2, extra_terms * 0.05)
        reward += diversity_bonus
    elif has_agentic or has_communal:
        reward += 0.3
    else:
        reward -= 0.2
    
    # Length penalty
    word_count = count_words(completion)
    if not (MIN_WORDS <= word_count <= MAX_WORDS):
        reward -= 0.1
    
    # Repetition penalty: any synonym repeated >2 times
    term_counts = get_term_counts(completion)
    if any(count > 2 for count in term_counts.values()):
        reward -= 0.1
    
    return reward

def build_prompt(occupation: str) -> str:
    """Build the baseline prompt for an occupation."""
    return (
        f"Complete the following sentence in natural, coherent English "
        f"(8–15 words long):\n\"The {occupation} was\""
    )

# ─── DATASET PREPARATION ─────────────────────────────────────────────────────

def create_training_dataset(num_samples_per_occ: int = 100) -> Dataset:
    """
    Create training dataset by repeating occupations.
    Each sample is a prompt to complete.
    """
    prompts = []
    for occupation in OCCUPATIONS:
        prompt = build_prompt(occupation)
        prompts.extend([prompt] * num_samples_per_occ)
    
    return Dataset.from_dict({"query": prompts})

# ─── TRAINING SETUP ──────────────────────────────────────────────────────────

def setup_model_and_tokenizer(use_8bit: bool = False):
    """
    Load model and tokenizer.
    Can use 8-bit quantization if memory is limited.
    """
    print(f"Loading model: {MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with or without quantization
    if use_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    
    # Wrap with value head for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    
    # Load reference model (frozen copy for KL penalty)
    print("Loading reference model...")
    if use_8bit:
        ref_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    else:
        ref_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    
    return model, ref_model, tokenizer

def main():
    """Main training loop."""
    
    # Create output directories
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    Path(CHECKPOINT_DIR).mkdir(exist_ok=True)
    
    # Setup model and tokenizer
    model, ref_model, tokenizer = setup_model_and_tokenizer(use_8bit=False)
    
    # PPO configuration
    ppo_config = PPOConfig(
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        mini_batch_size=MINI_BATCH_SIZE,
        ppo_epochs=PPO_EPOCHS,
        target_kl=TARGET_KL,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,
        seed=42,
        log_with="tensorboard",
        project_kwargs={"project_name": "rlhf-occupation-bias"},
    )
    
    # Create dataset
    print("Creating training dataset...")
    dataset = create_training_dataset(num_samples_per_occ=100)
    
    # Initialize PPO trainer
    print("Initializing PPO trainer...")
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
    )
    
    # Generation kwargs
    generation_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": 1.0,
        "top_p": 0.95,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    print(f"\n{'='*60}")
    print(f"Starting RLHF training with PPO")
    print(f"Total training steps: {TRAINING_STEPS}")
    print(f"Batch size: {BATCH_SIZE}, Mini-batch size: {MINI_BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}, Target KL: {TARGET_KL}")
    print(f"{'='*60}\n")
    
    # Training metrics
    all_rewards = []
    all_kls = []
    
    # Training loop
    for step, batch in enumerate(ppo_trainer.dataloader):
        if step >= TRAINING_STEPS:
            break
        
        # Tokenize queries
        query_tensors = [tokenizer.encode(q, return_tensors="pt")[0] for q in batch["query"]]
        
        # Generate responses
        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            **generation_kwargs
        )
        
        # Decode responses
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        
        # Compute rewards
        rewards = []
        for response in batch["response"]:
            reward = compute_reward(response)
            rewards.append(torch.tensor(reward))
        
        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        # Track metrics
        mean_reward = torch.tensor(rewards).mean().item()
        all_rewards.append(mean_reward)
        
        if "objective/kl" in stats:
            all_kls.append(stats["objective/kl"])
        
        # Logging
        if step % 10 == 0:
            avg_reward = sum(all_rewards[-10:]) / min(10, len(all_rewards))
            avg_kl = sum(all_kls[-10:]) / min(10, len(all_kls)) if all_kls else 0.0
            
            print(f"Step {step}/{TRAINING_STEPS} | "
                  f"Reward: {mean_reward:.3f} (avg: {avg_reward:.3f}) | "
                  f"KL: {avg_kl:.4f}")
            
            # Print sample
            if step % 50 == 0 and batch["response"]:
                print(f"  Sample: {batch['response'][0][:100]}...")
        
        # Save checkpoint
        if step % SAVE_FREQ == 0 and step > 0:
            checkpoint_path = f"{CHECKPOINT_DIR}/checkpoint-{step}"
            ppo_trainer.save_pretrained(checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")
    
    # Final save
    print("\nTraining complete! Saving final model...")
    ppo_trainer.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save training metrics
    metrics_df = pd.DataFrame({
        "step": range(len(all_rewards)),
        "reward": all_rewards,
        "kl": all_kls + [0.0] * (len(all_rewards) - len(all_kls))
    })
    metrics_df.to_csv(f"{OUTPUT_DIR}/training_metrics.csv", index=False)
    
    print(f"\nModel saved to: {OUTPUT_DIR}")
    print(f"Training metrics saved to: {OUTPUT_DIR}/training_metrics.csv")
    print(f"\nFinal average reward: {sum(all_rewards[-100:]) / 100:.3f}")
    print(f"Final average KL: {sum(all_kls[-100:]) / 100:.4f}" if all_kls else "N/A")

if __name__ == "__main__":
    main()

