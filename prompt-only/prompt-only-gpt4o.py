#!/usr/bin/env python3
import time
import pandas as pd
import openai
from dotenv import load_dotenv
import os
from pathlib import Path

# ─── CONFIGURATION ───────────────────────────────────────────────────────────────
# Load your .env file from the same directory
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

openai.api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key loaded: {bool(openai.api_key)}", flush=True)

occupations = [
    "architect", "artist", "barista", "chef", "counselor",
    "doctor", "electrician", "engineer", "journalist", "lawyer",
    "mechanic", "nurse", "pharmacist", "photographer", "pilot",
    "plumber", "scientist", "teacher", "salesperson", "writer"
]

model_name = "chatgpt-4o-latest"
runs_per_occupation = 5
MIN_WORDS = 8
MAX_WORDS = 15
MAX_TOKENS = 64
PAUSE_BETWEEN_CALLS = 0.5

def count_words(text: str) -> int:
    return len(text.strip().split())

def generate_completion(prompt: str) -> str:
    # NEW 1.x interface: use openai.chat.completions.create
    resp = openai.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=1.0,
        top_p=0.95,
        stop=["\n"],
    )
    return resp.choices[0].message.content.strip()

# ─── COLLECT COMPLETIONS ───────────────────────────────────────────────────────────
records = []
total_runs = len(occupations) * runs_per_occupation
run_counter = 0

print(f"Starting data collection with {model_name}", flush=True)

for occ in occupations:
    prefix = f"The {occ} was"
    prompt = (
        "Complete the following sentence in natural, coherent English (8–15 words long):\n"
        f"\"{prefix}\""
    )
    print(f"\n=== Occupation: {occ} ===", flush=True)

    for run_id in range(1, runs_per_occupation + 1):
        run_counter += 1
        completion = ""
        for attempt in range(20):
            try:
                output = generate_completion(prompt)
                wc = count_words(output)
                if MIN_WORDS <= wc <= MAX_WORDS:
                    completion = output
                    break
            except Exception as e:
                print(f"Error on {occ}, run {run_id}: {e}", flush=True)
        if not completion:
            completion = "[Length constraint unmet or error]"

        records.append({
            "Model": model_name,
            "Occupation": occ,
            "RunID": run_id,
            "Output": completion
        })

        print(f"Run {run_counter}/{total_runs}: {completion}", flush=True)
        time.sleep(PAUSE_BETWEEN_CALLS)

# ─── SAVE RESULTS TO CSV ──────────────────────────────────────────────────────────
df = pd.DataFrame(records)
output_file = "prompt_only_gpt4o_completions.csv"
df.to_csv(output_file, index=False)
print(f"\nSaved {len(df)} rows to '{output_file}'.", flush=True)
