import pytest
from app import utils

# Test openrouter_id_to_ollama_name
@pytest.mark.parametrize("model_id,expected", [
    ("mistralai/mistral-7b-instruct:free", "mistral-7b-instruct:free:latest"),
])
def test_openrouter_id_to_ollama_name(model_id, expected):
    assert utils.openrouter_id_to_ollama_name(model_id) == expected

# Test build_ollama_to_openrouter_map
models = [
    {"id": "mistralai/mistral-7b-instruct:free"},
]
def test_build_ollama_to_openrouter_map():
    mapping = utils.build_ollama_to_openrouter_map(models)
    assert mapping == {
        "mistral-7b-instruct:free:latest": "mistralai/mistral-7b-instruct:free"
    }

# Test filter_models
all_models = [
    {"id": "mistralai/mistral-7b-instruct:free"}
]

def test_filter_models_with_filter():
    filter_set = {"mistral-7b-instruct:free"}
    filtered = utils.filter_models(all_models, filter_set)
    filtered_ids = [m["id"] for m in filtered]
    assert filtered_ids == ["mistralai/mistral-7b-instruct:free"]

def test_filter_models_no_filter():
    filtered = utils.filter_models(all_models, set())
    assert filtered == all_models
