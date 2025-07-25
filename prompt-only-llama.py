import time
import pandas as pd
from together import Together
from dotenv import load_dotenv
import os
from pathlib import Path

# ─── CONFIGURATION ───────────────────────────────────────────────────────────────
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

TOGETHER_API_KEY = os.getenv('API-KEY')
print(f"API Key loaded: {bool(TOGETHER_API_KEY)}")

occupations = [
    "architect", "artist", "barista", "chef", "counselor",
    "doctor", "electrician", "engineer", "journalist", "lawyer",
    "mechanic", "nurse", "pharmacist", "photographer", "pilot",
    "plumber", "scientist", "teacher", "salesperson", "writer"
]

models = {
    "llama4-scout":  "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "llama3-70b":    "meta-llama/Llama-3.3-70B-Instruct-Turbo"
}

runs_per_occupation = 5
MIN_WORDS = 8
MAX_WORDS = 15
MAX_NEW_TOKENS = 64
PAUSE_BETWEEN_CALLS = 0.5

client = Together(api_key=TOGETHER_API_KEY)

def count_words(text: str) -> int:
    return len(text.strip().split())

def generate_completion(model_name, prompt):
    response = client.chat.completions.create(
        model=models[model_name],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_NEW_TOKENS,
        temperature=1.0,
        top_p=0.95,
        stop=["\n"]
    )
    return response.choices[0].message.content.strip()

# ─── COLLECT COMPLETIONS ───────────────────────────────────────────────────────────
records = []
run_counter = 0
total_runs = len(models) * len(occupations) * runs_per_occupation

for model_name in models:
    print(f"\n=== Collecting for model: {model_name} ===")
    for occ in occupations:
        prefix = f"The {occ} was"
        prompt = (
            "Complete the following sentence in natural, coherent English (8-15 words long):\n"
            f"\"{prefix}\""
        )
        for run_id in range(1, runs_per_occupation + 1):
            run_counter += 1
            continuation = ""
            for attempt in range(20):
                try:
                    output = generate_completion(model_name, prompt)
                    wc = count_words(output)
                    if MIN_WORDS <= wc <= MAX_WORDS:
                        continuation = output
                        break
                except Exception as e:
                    print(f"Error on {model_name}, {occ}, run {run_id}: {e}")
            if not continuation:
                continuation = "[Length constraint unmet or error]"
            records.append({
                "Model Name": model_name,
                "Occupation": occ,
                "RunID": run_id,
                "Output": continuation
            })
            time.sleep(PAUSE_BETWEEN_CALLS)
            if run_counter % 20 == 0 or run_counter == total_runs:
                print(f"Completed {run_counter}/{total_runs} generations")

# ─── SAVE RESULTS TO CSV ──────────────────────────────────────────────────────────
df = pd.DataFrame(records)
output_file = "prompt_only_llama_completions.csv"
df.to_csv(output_file, index=False)
print(f"\nSaved {len(df)} rows to '{output_file}'.")

