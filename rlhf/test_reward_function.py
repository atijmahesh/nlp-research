#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_reward_function.py
Quick test script to validate the reward function logic.
"""

import re
from typing import List, Dict

# Test configurations
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

# ─── REWARD FUNCTION (same as in train_rlhf.py) ─────────────────────────────

def count_words(text: str) -> int:
    return len(text.strip().split())

def contains_term(text: str, term: str) -> bool:
    pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    return pattern.search(text) is not None

def count_unique_terms(text: str, term_list: List[str]) -> int:
    return sum(1 for term in term_list if contains_term(text, term))

def get_term_counts(text: str) -> Dict[str, int]:
    all_terms = AGENTIC + COMMUNAL
    counts = {}
    for term in all_terms:
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        counts[term] = len(pattern.findall(text))
    return counts

def compute_reward(completion: str, verbose: bool = False) -> float:
    reward = 0.0
    details = []
    
    # Count agentic and communal terms
    has_agentic = any(contains_term(completion, term) for term in AGENTIC)
    has_communal = any(contains_term(completion, term) for term in COMMUNAL)
    
    agentic_count = count_unique_terms(completion, AGENTIC)
    communal_count = count_unique_terms(completion, COMMUNAL)
    
    # Base reward
    if has_agentic and has_communal:
        reward += 1.0
        details.append(f"Both terms: +1.0")
        # Diversity bonus
        extra_terms = max(0, (agentic_count + communal_count) - 2)
        diversity_bonus = min(0.2, extra_terms * 0.05)
        if diversity_bonus > 0:
            reward += diversity_bonus
            details.append(f"Diversity bonus ({extra_terms} extra): +{diversity_bonus:.2f}")
    elif has_agentic or has_communal:
        reward += 0.3
        term_type = "agentic" if has_agentic else "communal"
        details.append(f"One term ({term_type}): +0.3")
    else:
        reward -= 0.2
        details.append("No terms: -0.2")
    
    # Length penalty
    word_count = count_words(completion)
    if not (MIN_WORDS <= word_count <= MAX_WORDS):
        reward -= 0.1
        details.append(f"Length {word_count} (not 8-15): -0.1")
    
    # Repetition penalty
    term_counts = get_term_counts(completion)
    repeated = [term for term, count in term_counts.items() if count > 2]
    if repeated:
        reward -= 0.1
        details.append(f"Repetition ({', '.join(repeated)}): -0.1")
    
    if verbose:
        print(f"\nText: {completion}")
        print(f"Word count: {word_count}")
        print(f"Agentic terms found ({agentic_count}): {[t for t in AGENTIC if contains_term(completion, t)]}")
        print(f"Communal terms found ({communal_count}): {[t for t in COMMUNAL if contains_term(completion, t)]}")
        print(f"Reward breakdown:")
        for detail in details:
            print(f"  - {detail}")
        print(f"TOTAL REWARD: {reward:.2f}")
    
    return reward

# ─── TEST CASES ──────────────────────────────────────────────────────────────

def run_tests():
    test_cases = [
        {
            "name": "Perfect case (both terms, good length)",
            "text": "The engineer was confident and caring in the workplace environment.",
            "expected_range": (0.9, 1.1)
        },
        {
            "name": "Multiple terms (diversity bonus)",
            "text": "The doctor was confident, decisive, caring, and supportive to patients.",
            "expected_range": (1.1, 1.3)  # 1.0 + up to 0.2 diversity
        },
        {
            "name": "Only agentic term",
            "text": "The pilot was very confident and skilled in flying.",
            "expected_range": (0.25, 0.35)
        },
        {
            "name": "Only communal term",
            "text": "The nurse was caring and attentive to all patients.",
            "expected_range": (0.25, 0.35)
        },
        {
            "name": "No terms",
            "text": "The architect was working on the building project yesterday.",
            "expected_range": (-0.25, -0.15)
        },
        {
            "name": "Both terms but too short",
            "text": "confident and caring",
            "expected_range": (0.85, 0.95)  # 1.0 - 0.1 for length
        },
        {
            "name": "Both terms but too long",
            "text": "The teacher was confident and caring and always went above and beyond to help students succeed.",
            "expected_range": (0.85, 0.95)  # 1.0 - 0.1 for length
        },
        {
            "name": "Repetition penalty",
            "text": "The chef was confident confident confident and caring every day.",
            "expected_range": (0.85, 0.95)  # 1.0 - 0.1 for repetition
        },
        {
            "name": "All penalties",
            "text": "short text",
            "expected_range": (-0.35, -0.25)  # -0.2 (no terms) - 0.1 (length)
        },
        {
            "name": "Max diversity bonus",
            "text": "The journalist was confident, assertive, caring, supportive, and helpful daily.",
            "expected_range": (1.15, 1.25)  # 1.0 + 0.2 max diversity
        }
    ]
    
    print("="*70)
    print("REWARD FUNCTION TEST SUITE")
    print("="*70)
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─'*70}")
        print(f"Test {i}: {test['name']}")
        print(f"{'─'*70}")
        
        reward = compute_reward(test['text'], verbose=True)
        
        min_expected, max_expected = test['expected_range']
        if min_expected <= reward <= max_expected:
            print(f"✓ PASSED (expected: {min_expected:.2f} - {max_expected:.2f})")
            passed += 1
        else:
            print(f"✗ FAILED (expected: {min_expected:.2f} - {max_expected:.2f}, got: {reward:.2f})")
            failed += 1
    
    print(f"\n{'='*70}")
    print(f"TEST RESULTS: {passed}/{passed+failed} passed")
    print(f"{'='*70}\n")
    
    return passed == len(test_cases)

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)

