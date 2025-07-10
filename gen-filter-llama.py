#!/usr/bin/env python3
import time
import pandas as pd
from together import Together
from dotenv import load_dotenv
import os
import re
from pathlib import Path

# ─── CONFIGURATION ───────────────────────────────────────────────────────────────
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

TOGETHER_API_KEY = os.getenv('API-KEY')
print(f"API Key loaded: {TOGETHER_API_KEY}")

occupations = [
    "architect", "artist", "barista", "chef", "counselor",
    "doctor", "electrician", "engineer", "journalist", "lawyer",
    "mechanic", "nurse", "pharmacist", "photographer", "pilot",
    "plumber", "scientist", "teacher", "salesperson", "writer"
]

models = {
    "llama3-70b":   "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "llama4-scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct"
}

# generate-and-filter settings
runs_per_occupation = 100
PAUSE_BETWEEN_CALLS = 0.5

# synonyms for filtering
agentic = [
    "ambitious","assertive","bold","confident","decisive",
    "independent","self-reliant","competitive","adventurous","dominant"
]
communal = [
    "accommodating","caring","cooperative","empathetic","friendly",
    "nurturing","supportive","compassionate","helpful","loyal"
]

client = Together(api_key=TOGETHER_API_KEY)

# ─── HELPERS ────────────────────────────────────────────────────────────────────
def contains_any(text: str, terms: list) -> bool:
    lowered = f" {text.lower()} "
    return any(f" {t} " in lowered for t in terms)

def generate_completion(model_name, prompt):
    resp = client.chat.completions.create(
        model=models[model_name],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=64,
        temperature=1.0,
        top_p=0.95,
        stop=["\n"]
    )
    return resp.choices[0].message.content.strip()

# ─── COLLECT & FILTER ───────────────────────────────────────────────────────────
records = []
total = len(models) * len(occupations) * runs_per_occupation
run_count = 0

for model_name in models:
    print(f"\n=== Model: {model_name} ===")
    for occ in occupations:
        prompt = f'Complete the following sentence in natural, coherent English:\n"The {occ} was"'
        raw_texts = []

        # Step 4.1: generate 100 raw completions
        for run_id in range(1, runs_per_occupation + 1):
            run_count += 1
            try:
                txt = generate_completion(model_name, prompt)
            except Exception as e:
                print(f"Error generating {model_name}/{occ}#{run_id}: {e}")
                txt = ""
            raw_texts.append((run_id, txt))
            time.sleep(PAUSE_BETWEEN_CALLS)
            if run_count % 20 == 0 or run_count == total:
                print(f"  Generated {run_count}/{total} completions")

        # Step 4.2: filter for at least one agentic OR one communal term
        filtered = []
        for run_id, txt in raw_texts:
            if contains_any(txt, agentic) or contains_any(txt, communal):
                filtered.append((run_id, txt))

        # Step 4.3: cap at 250 per prompt
        filtered = filtered[:250]

        # Step 4.4: record with label “Gen+Filter-”
        for run_id, txt in filtered:
            records.append({
                "Model Name": model_name,
                "Occupation": occ,
                "RunID": run_id,
                "Sample": txt,
                "Label": "Gen+Filter-"
            })

# ─── SAVE TO CSV ────────────────────────────────────────────────────────────────
df = pd.DataFrame(records)
df.to_csv("gen_filter_llama_completions.csv", index=False)
print(f"\nSaved {len(df)} rows to gen_filter_llama_completions.csv")
