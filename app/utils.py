# Helper functions (e.g., model name processing) will be added here

from typing import Dict, Set, Tuple, Optional


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


def resolve_model_name(
    requested_name: Optional[str], ollama_map: Dict[str, str]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolves a potentially short/aliased model name (or None) to the full Ollama name
    and its corresponding OpenRouter ID.

    Args:
        requested_name: The model name from the user request.
        ollama_map: The mapping from full Ollama names to OpenRouter IDs.

    Returns:
        A tuple containing (resolved_ollama_name, openrouter_id), or (None, None) if no match or input is None.
    """
    # 0. Handle None input immediately
    if requested_name is None:
        return None, None

    # 1. Check for exact match
    if requested_name in ollama_map:
        return requested_name, ollama_map[requested_name]

    # 2. Check for prefix match (e.g., user provides 'llama3' and map has 'llama3:latest')
    # Ensure the requested name doesn't already contain ':' to avoid ambiguity
    if ":" not in requested_name:
        for ollama_name, openrouter_id in ollama_map.items():
            if ollama_name.startswith(requested_name + ":"):
                return ollama_name, openrouter_id # Return first prefix match

    # 3. No match found
    return None, None
