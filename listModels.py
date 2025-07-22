#!/usr/bin/env python3
import os
from dotenv import load_dotenv
import openai

# Load your API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# List all models
resp = openai.models.list()

# Print each model ID
for mdl in resp.data:
    print(mdl.id)
