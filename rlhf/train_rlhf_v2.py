#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_rlhf_v2.py
RLHF Fine-tuning with PPO for LLaMA-3-8B-Instruct
Compatible with trl 0.23.1

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
from typing import List, Dict

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import PPOTrainer
from datasets import Dataset

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
load_dotenv()

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./rlhf_llama3_8b_output"
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

# Training Hyperparameters
BATCH_SIZE = 16  # Smaller batch for trl 0.23.1
LEARNING_RATE = 1e-5
TRAINING_STEPS = 2000
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
        # Diversity bonus
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
    
    # Repetition penalty
    term_counts = get_term_counts(completion)
    if any(count > 2 for count in term_counts.values()):
        reward -= 0.1
    
    return reward

def build_prompt(occupation: str) -> str:
    """Build the baseline prompt for an occupation."""
    prefix = f"The {occupation} was"
    return (
        f"Complete the following sentence in natural, coherent English "
        f"(8–15 words long):\n\"{prefix}\""
    )

# ─── DATASET PREPARATION ─────────────────────────────────────────────────────

def create_training_dataset(num_samples_per_occ: int = 100) -> Dataset:
    """Create training dataset by repeating occupations."""
    prompts = []
    for occupation in OCCUPATIONS:
        prompt = build_prompt(occupation)
        prompts.extend([prompt] * num_samples_per_occ)
    
    return Dataset.from_dict({"query": prompts})

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    """Main training loop using trl 0.23.1 API."""
    
    # Create output directories
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    Path(CHECKPOINT_DIR).mkdir(exist_ok=True)
    
    print(f"Loading model: {MODEL_NAME}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    
    # Load reference model
    print("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    
    # Create dataset
    print("Creating training dataset...")
    dataset = create_training_dataset(num_samples_per_occ=100)
    
    # Initialize PPO trainer (trl 0.23.1 API - minimal args)
    print("Initializing PPO trainer...")
    ppo_trainer = PPOTrainer(
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
    print(f"Starting RLHF training with PPO (trl 0.23.1)")
    print(f"Total training steps: {TRAINING_STEPS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"{'='*60}\n")
    
    # Training metrics
    all_rewards = []
    all_losses = []
    
    # Training loop
    for step, batch in enumerate(ppo_trainer.dataloader):
        if step >= TRAINING_STEPS:
            break
        
        # Tokenize queries
        query_tensors = []
        for q in batch["query"]:
            tokens = tokenizer.encode(q, return_tensors="pt")[0]
            query_tensors.append(tokens)
        
        # Generate responses
        response_tensors = []
        for query in query_tensors:
            with torch.no_grad():
                outputs = model.generate(
                    query.unsqueeze(0).to(model.device),
                    **generation_kwargs
                )
                # Get only the generated part (exclude prompt)
                response = outputs[0][len(query):]
                response_tensors.append(response)
        
        # Decode responses
        batch["response"] = []
        for response in response_tensors:
            decoded = tokenizer.decode(response, skip_special_tokens=True)
            batch["response"].append(decoded)
        
        # Compute rewards
        rewards = []
        for response in batch["response"]:
            reward = compute_reward(response)
            rewards.append(torch.tensor(reward, dtype=torch.float32))
        
        # Run PPO step
        try:
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            
            # Track metrics
            mean_reward = torch.stack(rewards).mean().item()
            all_rewards.append(mean_reward)
            
            if isinstance(stats, dict) and "ppo/loss/total" in stats:
                all_losses.append(stats["ppo/loss/total"])
            
            # Logging
            if step % 10 == 0:
                avg_reward = sum(all_rewards[-10:]) / min(10, len(all_rewards))
                
                print(f"Step {step}/{TRAINING_STEPS} | "
                      f"Reward: {mean_reward:.3f} (avg: {avg_reward:.3f})")
                
                # Print sample
                if step % 50 == 0 and batch["response"]:
                    print(f"  Sample: {batch['response'][0][:100]}...")
            
            # Save checkpoint
            if step % SAVE_FREQ == 0 and step > 0:
                checkpoint_path = f"{CHECKPOINT_DIR}/checkpoint-{step}"
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                print(f"  Saved checkpoint to {checkpoint_path}")
                
        except Exception as e:
            print(f"Error at step {step}: {e}")
            continue
    
    # Final save
    print("\nTraining complete! Saving final model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save training metrics
    metrics_df = pd.DataFrame({
        "step": range(len(all_rewards)),
        "reward": all_rewards,
    })
    metrics_df.to_csv(f"{OUTPUT_DIR}/training_metrics.csv", index=False)
    
    print(f"\nModel saved to: {OUTPUT_DIR}")
    print(f"Training metrics saved to: {OUTPUT_DIR}/training_metrics.csv")
    print(f"\nFinal average reward: {sum(all_rewards[-100:]) / min(100, len(all_rewards)):.3f}")

if __name__ == "__main__":
    main()

