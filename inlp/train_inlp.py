#!/usr/bin/env python3
"""
INLP (Iterative Nullspace Projection) for Gender Debiasing
Paper: Ravfogel et al. (2022) - Null It Out: Guarding Protected Attributes

Removes gender bias by iteratively projecting out gender-encoding directions
from the model's hidden representations.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import pickle

# ============================================================================
# GENDERED WORD PAIRS (for INLP projection)
# ============================================================================

GENDERED_PAIRS = [
    # Pronouns
    ("he", "she"),
    ("him", "her"),
    ("his", "hers"),
    ("himself", "herself"),
    # Family
    ("man", "woman"),
    ("men", "women"),
    ("boy", "girl"),
    ("boys", "girls"),
    ("father", "mother"),
    ("dad", "mom"),
    ("son", "daughter"),
    ("brother", "sister"),
    ("uncle", "aunt"),
    ("nephew", "niece"),
    ("husband", "wife"),
    ("boyfriend", "girlfriend"),
    ("grandfather", "grandmother"),
    ("grandson", "granddaughter"),
    # Professional
    ("actor", "actress"),
    ("waiter", "waitress"),
    ("businessman", "businesswoman"),
    ("spokesman", "spokeswoman"),
    ("chairman", "chairwoman"),
    # Titles
    ("mr", "mrs"),
    ("sir", "madam"),
    ("king", "queen"),
    ("prince", "princess"),
    ("lord", "lady"),
]

# ============================================================================
# INLP PROJECTION
# ============================================================================

def get_projection_matrix(W: np.ndarray) -> np.ndarray:
    """
    Compute projection matrix that removes directions in W.
    Input: W is (k, d) where k is number of directions, d is embedding dim
    Output: P is (d, d) projection matrix
    P = I - W^T (W W^T)^{-1} W
    """
    if len(W.shape) == 1:
        # Single direction - reshape to (1, d)
        W = W.reshape(1, -1)
    
    if W.shape[0] == 0:
        # No directions - return identity
        return np.eye(W.shape[1])
    
    # W is (k, d), W^T is (d, k)
    # W W^T is (k, k)
    # (W W^T)^{-1} is (k, k)
    # W^T (W W^T)^{-1} is (d, k)
    # W^T (W W^T)^{-1} W is (d, d)
    d = W.shape[1]
    P = np.eye(d) - W.T @ np.linalg.pinv(W @ W.T) @ W
    return P

def get_word_embedding(model, tokenizer, word: str, layer_idx: int = -1) -> np.ndarray:
    """Extract embedding for a word at a specific layer."""
    inputs = tokenizer(word, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer_idx]  # (batch, seq, hidden)
        # Take mean over sequence (usually just 1-2 tokens)
        # Convert to float32 for numpy compatibility
        embedding = hidden_states.mean(dim=1).squeeze().float().cpu().numpy()
    
    return embedding

def train_gender_classifier(
    X_male: np.ndarray,
    X_female: np.ndarray,
    n_classifiers: int = 1
) -> list:
    """
    Train linear classifiers to predict gender from embeddings.
    Returns list of classifier weights (the directions to remove).
    """
    # Create training data
    X = np.vstack([X_male, X_female])
    y = np.array([0] * len(X_male) + [1] * len(X_female))
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    classifiers = []
    for i in range(n_classifiers):
        clf = LogisticRegression(max_iter=1000, random_state=42 + i)
        clf.fit(X, y)
        
        # Extract the decision boundary direction (weight vector)
        direction = clf.coef_[0]
        direction = direction / np.linalg.norm(direction)  # Normalize
        classifiers.append(direction)
    
    return classifiers

def apply_inlp(
    model,
    tokenizer,
    gendered_pairs: list,
    n_iterations: int = 300,
    layer_idx: int = -1,
    output_dir: str = "./inlp_projections"
):
    """
    Apply INLP to remove gender information from model representations.
    
    Args:
        model: The language model
        tokenizer: Tokenizer
        gendered_pairs: List of (male_word, female_word) pairs
        n_iterations: Number of INLP iterations
        layer_idx: Which layer to modify (-1 = last layer)
        output_dir: Where to save projection matrices
    """
    print(f"[DEBUG] apply_inlp function called", flush=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] Output directory created: {output_dir}", flush=True)
    
    print(f"\n{'='*70}")
    print(f"INLP: Extracting embeddings from layer {layer_idx}")
    print(f"{'='*70}\n", flush=True)
    
    # Step 1: Extract embeddings for all gendered words
    male_embeddings = []
    female_embeddings = []
    
    for male_word, female_word in gendered_pairs:
        try:
            male_emb = get_word_embedding(model, tokenizer, male_word, layer_idx)
            female_emb = get_word_embedding(model, tokenizer, female_word, layer_idx)
            male_embeddings.append(male_emb)
            female_embeddings.append(female_emb)
        except Exception as e:
            print(f"Warning: Could not process pair ({male_word}, {female_word}): {e}")
    
    X_male = np.array(male_embeddings)
    X_female = np.array(female_embeddings)
    
    print(f"Extracted {len(male_embeddings)} gendered word pairs")
    print(f"Embedding dimension: {X_male.shape[1]}")
    
    # Step 2: Iterative nullspace projection
    print(f"\n{'='*70}")
    print(f"Running {n_iterations} INLP iterations...")
    print(f"{'='*70}\n")
    
    all_directions = []
    
    for iteration in range(n_iterations):
        # Train classifier on current representations
        directions = train_gender_classifier(X_male, X_female, n_classifiers=1)
        direction = directions[0]
        all_directions.append(direction)
        
        # Compute projection matrix
        W = np.array(all_directions)  # Shape: (iteration+1, d)
        P = get_projection_matrix(W)  # Shape: (d, d)
        
        # Project embeddings
        # X_male is (n_samples, d), P is (d, d)
        # We want: X_male_projected = X_male @ P^T = (n, d) @ (d, d) = (n, d)
        X_male = X_male @ P
        X_female = X_female @ P
        
        if (iteration + 1) % 50 == 0:
            print(f"  Iteration {iteration + 1}/{n_iterations}", flush=True)
    
    # Step 3: Save projection matrix
    final_projection = get_projection_matrix(np.array(all_directions))
    
    projection_data = {
        "projection_matrix": final_projection,
        "all_directions": np.array(all_directions),
        "layer_idx": layer_idx,
        "n_iterations": n_iterations,
        "embedding_dim": X_male.shape[1],
    }
    
    output_file = output_dir / f"inlp_projection_layer{layer_idx}.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(projection_data, f)
    
    print(f"\n{'='*70}")
    print(f"✅ INLP Complete!")
    print(f"{'='*70}")
    print(f"Projection matrix saved to: {output_file}")
    print(f"Removed {len(all_directions)} gender directions")
    print(f"{'='*70}\n")
    
    return projection_data

# ============================================================================
# MAIN
# ============================================================================

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("INLP: Iterative Nullspace Projection")
    print("="*70)
    print(f"Model: meta-llama/Meta-Llama-3.1-8B-Instruct")
    print(f"Seed: {args.seed}")
    print(f"Iterations: {args.n_iterations}")
    print(f"Layer: {args.layer_idx}")
    print(f"Output: {output_dir}")
    print("="*70 + "\n")
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded and set to eval mode.", flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        trust_remote_code=True,
    )
    print("Tokenizer loaded.", flush=True)
    
    # Apply INLP
    print("Starting INLP projection computation...", flush=True)
    projection_data = apply_inlp(
        model=model,
        tokenizer=tokenizer,
        gendered_pairs=GENDERED_PAIRS,
        n_iterations=args.n_iterations,
        layer_idx=args.layer_idx,
        output_dir=output_dir,
    )
    
    # Save model identifier (we'll apply projection during generation)
    config = {
        "base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "seed": args.seed,
        "n_iterations": args.n_iterations,
        "layer_idx": args.layer_idx,
    }
    
    config_file = output_dir / "inlp_config.pkl"
    with open(config_file, "wb") as f:
        pickle.dump(config, f)
    
    print(f"Config saved to: {config_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_iterations", type=int, default=300, help="Number of INLP iterations")
    parser.add_argument("--layer_idx", type=int, default=-1, help="Layer to modify (-1 = last)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    
    main(args)

