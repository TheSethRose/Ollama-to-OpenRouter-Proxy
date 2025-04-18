# Helper functions (e.g., model name processing) will be added here

import re
from typing import Dict, Set

def openrouter_id_to_ollama_name(model_id: str) -> str:
    # Remove vendor prefix (e.g., 'openai/') and append ':latest'
    name = model_id.split("/")[-1]
    return f"{name}:latest"

def build_ollama_to_openrouter_map(models: list) -> Dict[str, str]:
    # models: list of OpenRouter model dicts with 'id'
    mapping = {}
    for m in models:
        ollama_name = openrouter_id_to_ollama_name(m["id"])
        mapping[ollama_name] = m["id"]
    return mapping

def filter_models(models: list, filter_set: Set[str]) -> list:
    # Only keep models whose processed name (no vendor) is in filter_set
    if not filter_set:
        return models
    filtered = []
    for m in models:
        # Get base name with tag (e.g., 'mistral-7b-instruct:free')
        name = m["id"].split("/")[-1]
        if name in filter_set:
            filtered.append(m)
    return filtered
