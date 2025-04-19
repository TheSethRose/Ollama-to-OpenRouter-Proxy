#!/bin/bash

# Script to update models-filter.txt with the top 25 free models from OpenRouter, using Ollama name format

API_URL="https://openrouter.ai/api/v1/models"
FILTER_FILE="models-filter.txt"
# RAW_FILE="models-api-response.json" # No longer saving raw response by default

# Helper: convert OpenRouter model id to Ollama name (remove vendor, add :latest)
openrouter_id_to_ollama_name() {
  local id="$1"
  local name
  name="${id##*/}"
  echo "${name}:latest"
}

echo "Fetching models from $API_URL ..."

# Fetch models, filter for free ones, convert to Ollama name, take top 25, sort alphabetically, and save to filter file
curl -s "$API_URL" | jq -r '.data[] | select(.pricing.prompt == "0" and .pricing.completion == "0") | .id' | \
  head -n 25 | \
  while read -r id; do openrouter_id_to_ollama_name "$id"; done | \
  sort > "$FILTER_FILE"

if [ $? -eq 0 ]; then
  # Check if the filter file is empty (e.g., API error, jq error, no free models found)
  if [ -s "$FILTER_FILE" ]; then
    echo "Successfully updated $FILTER_FILE with top 25 free models (Ollama name format)."
    echo "Found $(wc -l < "$FILTER_FILE") models:"
    cat "$FILTER_FILE"
  else
    echo "Error: Failed to fetch or process models. $FILTER_FILE is empty."
    echo "Please check network connection or API status (https://status.openrouter.ai/)"
    # Optionally keep the old file or handle error differently
    # mv "$FILTER_FILE.bak" "$FILTER_FILE"
    exit 1
  fi
else
  echo "Error executing command pipeline. Check curl/jq execution."
  exit 1
fi

exit 0
