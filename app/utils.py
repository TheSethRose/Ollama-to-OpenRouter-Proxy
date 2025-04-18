# Helper functions (e.g., model name processing) will be added here

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


def build_openrouter_to_ollama_map(models: list) -> Dict[str, str]:
    # models: list of OpenRouter model dicts with 'id'
    mapping = {}
    for m in models:
        ollama_name = openrouter_id_to_ollama_name(m["id"])
        mapping[m["id"]] = ollama_name  # Swap key/value from previous function
    return mapping


def filter_models(models: list, filter_set: Set[str]) -> list:
    # Only keep models whose processed name (no vendor) is in filter_set
    if not filter_set:
        return models
    filtered = []
    for m in models:
        # Convert OpenRouter ID to Ollama name FIRST
        ollama_name = openrouter_id_to_ollama_name(m["id"])
        # Now check if the Ollama name is in the set
        if ollama_name in filter_set:
            filtered.append(m)
    return filtered
