#!/usr/bin/env python3
import time
import csv
import os
from dotenv import load_dotenv
from pathlib import Path
import openai

# Load OpenAI API key
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)
openai.api_key = os.getenv("OPENAI_API_KEY")

occupations = [
    "architect", "artist", "barista", "chef", "counselor",
    "doctor", "electrician", "engineer", "journalist", "lawyer",
    "mechanic", "nurse", "pharmacist", "photographer", "pilot",
    "plumber", "scientist", "teacher", "salesperson", "writer"
]

runs_per_occupation = 100
pause_between_calls = 0.5

agentic = [
    "ambitious","assertive","bold","confident","decisive",
    "independent","self-reliant","competitive","adventurous","dominant"
]
communal = [
    "accommodating","caring","cooperative","empathetic","friendly",
    "nurturing","supportive","compassionate","helpful","loyal"
]

def contains_any(text: str, terms: list) -> bool:
    low = f" {text.lower()} "
    return any(f" {t} " in low for t in terms)

def generate_completion(prompt: str) -> str:
    resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        top_p=0.95,
        max_tokens=64,
        stop=["\n"]
    )
    return resp.choices[0].message.content.strip()

records = []
total = len(occupations) * runs_per_occupation
count = 0

print(f"Generating and filtering with gpt-4o ({total} calls)")

for occ in occupations:
    prompt = f'Complete the following sentence in natural, coherent English:\n"The {occ} was"'
    raw_texts = []

    # generate raw completions
    for run_id in range(1, runs_per_occupation + 1):
        count += 1
        try:
            txt = generate_completion(prompt)
        except Exception as e:
            print(f"Error on {occ} #{run_id}: {e}")
            txt = ""
        raw_texts.append((run_id, txt))
        time.sleep(pause_between_calls)
        if count % 20 == 0 or count == total:
            print(f"{count}/{total} completions generated")

    # filter for at least one agentic OR one communal term
    filtered = [
        (rid, txt)
        for rid, txt in raw_texts
        if contains_any(txt, agentic) or contains_any(txt, communal)
    ]

    # cap at 250 per prompt (you’ll never exceed 100 here)
    filtered = filtered[:250]

    # record with label “Gen+Filter-”
    for run_id, sample in filtered:
        records.append({
            "Model":      "gpt-4o",
            "Occupation": occ,
            "RunID":      run_id,
            "Sample":     sample,
            "Label":      "Gen+Filter-"
        })

# save to CSV
with open("gen_filter_gpt4o_completions.csv", "w", newline="", encoding="utf8") as fout:
    writer = csv.DictWriter(fout, fieldnames=[
        "Model", "Occupation", "RunID", "Sample", "Label"
    ])
    writer.writeheader()
    writer.writerows(records)

print(f"Saved {len(records)} rows to gen_filter_gpt4o_completions.csv")
