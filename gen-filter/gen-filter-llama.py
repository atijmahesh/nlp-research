#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen-filter-llama.py
Generate-and-filter for LLaMA models via Together API.

Protocol:
  • 100 raw generations per occupation per model
  • Keep samples containing ≥1 agentic OR ≥1 communal term
  • Cap retained list at 250 per occupation
  • Prompt matches baseline: 8–15 words, "The [OCCUPATION] was"
Outputs:
  • genfilter_llama_raw.csv
  • genfilter_llama_filtered.csv
"""

import os
import re
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from together import Together

# ─── CONFIG ──────────────────────────────────────────────────────────────────────
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

TOGETHER_API_KEY = os.getenv("API-KEY")
print(f"API Key loaded: {bool(TOGETHER_API_KEY)}")

OCCUPATIONS = [
    "architect", "artist", "barista", "chef", "counselor",
    "doctor", "electrician", "engineer", "journalist", "lawyer",
    "mechanic", "nurse", "pharmacist", "photographer", "pilot",
    "plumber", "scientist", "teacher", "salesperson", "writer"
]

MODELS = {
    "llama4-scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "llama3-70b":   "meta-llama/Llama-3.3-70B-Instruct-Turbo"
}

RUNS_PER_OCC = 100
PAUSE_BETWEEN_CALLS = 0.5
MAX_NEW_TOKENS = 64
MIN_WORDS = 8
MAX_WORDS = 15

AGENTIC = [
    "ambitious","assertive","bold","confident","decisive",
    "independent","self-reliant","competitive","adventurous","dominant"
]
COMMUNAL = [
    "accommodating","caring","cooperative","empathetic","friendly",
    "nurturing","supportive","compassionate","helpful","loyal"
]

RAW_CSV      = "genfilter_llama_raw.csv"
FILTERED_CSV = "genfilter_llama_filtered.csv"

client = Together(api_key=TOGETHER_API_KEY)

# ─── HELPERS ─────────────────────────────────────────────────────────────────────
_word_pat_cache = {}
def _compile_word_pat(term: str):
    if term not in _word_pat_cache:
        _word_pat_cache[term] = re.compile(rf"(?<!\\w){re.escape(term)}(?!\\w)", re.IGNORECASE)
    return _word_pat_cache[term]

def contains_any(text: str, terms: list) -> bool:
    return any(_compile_word_pat(t).search(text) for t in terms)

def count_words(text: str) -> int:
    return len(text.strip().split())

def generate_completion(model_key: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODELS[model_key],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_NEW_TOKENS,
        temperature=1.0,
        top_p=0.95,
        stop=["\n"]
    )
    return resp.choices[0].message.content.strip()

# ─── MAIN ────────────────────────────────────────────────────────────────────────
def main():
    raw_rows = []
    filt_rows = []

    total_calls = len(MODELS) * len(OCCUPATIONS) * RUNS_PER_OCC
    call_cnt = 0

    for model_key in MODELS:
        print(f"\n=== Model: {model_key} ===")
        for occ in OCCUPATIONS:
            prefix = f"The {occ} was"
            prompt = (
                "Complete the following sentence in natural, coherent English "
                f"(8–15 words long):\n\"{prefix}\""
            )

            # 1) Generate raw completions
            raw_texts = []
            for run_id in range(1, RUNS_PER_OCC + 1):
                call_cnt += 1
                try:
                    out = generate_completion(model_key, prompt)
                except Exception as e:
                    print(f"Error {model_key}/{occ}#{run_id}: {e}")
                    out = ""
                raw_texts.append((run_id, out))
                time.sleep(PAUSE_BETWEEN_CALLS)
                if call_cnt % 20 == 0 or call_cnt == total_calls:
                    print(f"  Generated {call_cnt}/{total_calls} completions")

            # Save raw
            for run_id, txt in raw_texts:
                raw_rows.append({
                    "Model": model_key,
                    "Occupation": occ,
                    "RunID": run_id,
                    "Text": txt
                })

            # 2) Filter: ≥1 agentic OR ≥1 communal term
            filtered = []
            for run_id, txt in raw_texts:
                if contains_any(txt, AGENTIC) or contains_any(txt, COMMUNAL):
                    filtered.append((run_id, txt))

            # 3) Cap at 250
            filtered = filtered[:250]

            # 4) Record filtered rows
            for run_id, txt in filtered:
                wc = count_words(txt)
                filt_rows.append({
                    "Model": model_key,
                    "Occupation": occ,
                    "RunID": run_id,
                    "Text": txt,
                    "WordCount": wc,
                    "Label": "Gen+Filter"
                })

    # ─── SAVE ────────────────────────────────────────────────────────────────────
    pd.DataFrame(raw_rows).to_csv(RAW_CSV, index=False)
    pd.DataFrame(filt_rows).to_csv(FILTERED_CSV, index=False)

    print(f"\nSaved {len(raw_rows)} raw rows  -> {RAW_CSV}")
    print(f"Saved {len(filt_rows)} filtered rows -> {FILTERED_CSV}")

if __name__ == "__main__":
    main()
