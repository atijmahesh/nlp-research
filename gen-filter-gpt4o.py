#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen-filter-gpt4o.py
Generate-and-filter for GPT-4o (OpenAI API).

• 100 raw generations / occupation
• Keep samples with ≥1 agentic OR ≥1 communal term
• Cap retained at 250 / occupation
• Prompt = baseline 8–15 words: "The [OCCUPATION] was"

Outputs:
  genfilter_gpt4o_raw.csv
  genfilter_gpt4o_filtered.csv
"""

import os
import re
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ─── CONFIG ──────────────────────────────────────────────────────────────────────
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(f"API Key loaded: {bool(OPENAI_API_KEY)}")

OCCUPATIONS = [
    "architect", "artist", "barista", "chef", "counselor",
    "doctor", "electrician", "engineer", "journalist", "lawyer",
    "mechanic", "nurse", "pharmacist", "photographer", "pilot",
    "plumber", "scientist", "teacher", "salesperson", "writer"
]

MODEL = "chatgpt-4o-latest"
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

RAW_CSV      = "genfilter_gpt4o_raw.csv"
FILTERED_CSV = "genfilter_gpt4o_filtered.csv"

client = OpenAI(api_key=OPENAI_API_KEY)

# ─── HELPERS ─────────────────────────────────────────────────────────────────────
_word_pat_cache = {}
def _compile_word_pat(term: str):
    if term not in _word_pat_cache:
        _word_pat_cache[term] = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
    return _word_pat_cache[term]

def contains_any(text: str, terms: list) -> bool:
    return any(_compile_word_pat(t).search(text) for t in terms)

def count_words(text: str) -> int:
    return len(text.strip().split())

def generate_completion(prompt: str, max_retries: int = 5) -> str:
    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                top_p=0.95,
                max_tokens=MAX_NEW_TOKENS,
                stop=["\n"]
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries:
                print(f"Final failure: {e}")
                return ""
            time.sleep(delay)
            delay *= 2
    return ""

# ─── MAIN ────────────────────────────────────────────────────────────────────────
def main():
    raw_rows, filt_rows = [], []
    total_calls = len(OCCUPATIONS) * RUNS_PER_OCC
    call_cnt = 0

    print(f"Generating and filtering with {MODEL} ({total_calls} calls)")

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
            txt = generate_completion(prompt)
            raw_texts.append((run_id, txt))
            time.sleep(PAUSE_BETWEEN_CALLS)
            if call_cnt % 20 == 0 or call_cnt == total_calls:
                print(f"{call_cnt}/{total_calls} completions generated")

        # Save raw
        for run_id, txt in raw_texts:
            raw_rows.append({
                "Model": MODEL,
                "Occupation": occ,
                "RunID": run_id,
                "Text": txt
            })

        # 2) Filter: ≥1 agentic OR ≥1 communal
        filtered = [
            (rid, txt) for rid, txt in raw_texts
            if contains_any(txt, AGENTIC) or contains_any(txt, COMMUNAL)
        ][:250]  # cap at 250

        # 3) Record filtered
        for run_id, txt in filtered:
            wc = count_words(txt)
            filt_rows.append({
                "Model": MODEL,
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
