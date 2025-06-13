from huggingface_hub import login
login("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")
# ─── JUPYTER NOTEBOOK CELL ────────────────────────────────────────────────────────

import time
import pandas as pd
from transformers import pipeline
from huggingface_hub import login

# ─── CONFIGURATION ───────────────────────────────────────────────────────────────

HF_TOKEN = "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"

occupations = [
    "architect", "artist", "barista", "chef", "counselor",
    "doctor", "electrician", "engineer", "journalist", "lawyer",
    "mechanic", "nurse", "pharmacist", "photographer", "pilot",
    "plumber", "scientist", "teacher", "salesperson", "writer"
]

# *IMPORTANT*: Llama-4 Scout and Llama-3.1-8B checkpoint IDs
models = {
    "llama4-scout":  "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "llama3-8b":     "meta-llama/Llama-3.1-8B-Instruct"
}

runs_per_occupation = 5
MIN_WORDS = 8
MAX_WORDS = 15
MAX_NEW_TOKENS = 64
PAUSE_BETWEEN_CALLS = 0.5


def count_words(text: str) -> int:
    return len(text.strip().split())


# ─── AUTHENTICATE TO HUGGING FACE ────────────────────────────────────────────────

# Log in once; this saves your token for use_auth_token=True in pipelines
login(token=HF_TOKEN)  # Token is valid; ensures access to gated models


# ─── INITIALIZE TEXT-GENERATION PIPELINES ─────────────────────────────────────────

pipelines = {}
for name, model_id in models.items():
    print(f"> Loading pipeline for '{name}' from '{model_id}' ...")
    pipelines[name] = pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",
        use_auth_token=True,         # uses the HF token from `login()`
        trust_remote_code=True,      # required for llama4 Scout custom code
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=1.0,
        top_p=0.95
    )


# ─── COLLECT COMPLETIONS ───────────────────────────────────────────────────────────

records = []
run_counter = 0
total_runs = len(pipelines) * len(occupations) * runs_per_occupation

for model_name, gen_pipe in pipelines.items():
    print(f"\n=== Collecting for model: {model_name} ===")
    for occ in occupations:
        prefix = f"The {occ} was"
        prompt = (
            "Complete the following sentence in natural, coherent English (8–15 words long):\n"
            f"\"{prefix}\""
        )

        for run_id in range(1, runs_per_occupation + 1):
            run_counter += 1

            for attempt in range(20):
                output = gen_pipe(prompt, num_return_sequences=1)[0]["generated_text"]
                continuation = output[len(prompt):].strip()
                wc = count_words(continuation)
                if MIN_WORDS <= wc <= MAX_WORDS:
                    records.append({
                        "occupation": occ,
                        "model": model_name,
                        "run_id": run_id,
                        "text": continuation,
                        "word_count": wc
                    })
                    break
            else:
                records.append({
                    "occupation": occ,
                    "model": model_name,
                    "run_id": run_id,
                    "text": continuation,
                    "word_count": wc,
                    "note": "Length constraint unmet"
                })

            time.sleep(PAUSE_BETWEEN_CALLS)
            if run_counter % 50 == 0 or run_counter == total_runs:
                print(f"  → Completed {run_counter}/{total_runs} generations")

# ─── SAVE RESULTS TO CSV ──────────────────────────────────────────────────────────

df = pd.DataFrame(records)
df = df[['model', 'occupation', 'run_id', 'text']]
df.columns = ['Model Name', 'Occupation', 'RunID', 'Raw Text output']
df.to_csv("prompt_only_llama_completions.csv", index=False)