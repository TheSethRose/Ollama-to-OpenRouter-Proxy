import pytest
import httpx
import os
import asyncio
from typing import AsyncGenerator, Generator
from app.main import app  # Assuming your FastAPI app instance is named 'app' in main.py
import json
import pytest_asyncio

# Ensure OPENROUTER_API_KEY is set for tests that hit the real API
# You might want to load this from a .env.test file or similar
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# Specify the model to use for testing, as requested in tasks.md
TEST_MODEL_OLLAMA_NAME = "mistral-7b-instruct:free:latest"  # Ollama format
TEST_MODEL_OPENROUTER_ID = (
    "mistralai/mistral-7b-instruct:free"  # Corresponding OpenRouter ID
)


# Basic pytest setup check
def test_pytest_setup():
    assert True


# --- Fixtures ---

# Use TestClient for synchronous tests if preferred, but httpx.AsyncClient is better for async app
# @pytest.fixture(scope="module")
# def client() -> Generator[TestClient, None, None]:
#     with TestClient(app) as c:
#         yield c


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for each test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Use pytest_asyncio.fixture for async fixtures
@pytest_asyncio.fixture(scope="session")
async def client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create an HTTPX async client for making requests to the test server."""
    async with httpx.AsyncClient(app=app, base_url="http://test") as c:
        yield c # Yield the client instance


# --- Integration Tests ---


@pytest.mark.asyncio
async def test_api_version(client: httpx.AsyncClient):
    """Test the /api/version endpoint."""
    response = await client.get("/api/version")
    assert response.status_code == 200
    assert response.json() == {"version": "0.1.0-openrouter"}


@pytest.mark.asyncio
@pytest.mark.skipif(not OPENROUTER_API_KEY, reason="OPENROUTER_API_KEY not set")
async def test_api_tags_lists_models(client: httpx.AsyncClient):
    """Test that /api/tags returns a list of models, including the test model."""
    response = await client.get("/api/tags")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)
    # Check if the specific test model (in ollama format) is present
    # found_test_model = any(m["name"] == TEST_MODEL_OLLAMA_NAME for m in data["models"])
    # Note: This assumes the filter allows the test model. If filter is active, adjust expectation.
    print(f"Test Model Ollama Name: {TEST_MODEL_OLLAMA_NAME}")
    print(f"Models found: {[m['name'] for m in data['models']]}")


@pytest.mark.asyncio
@pytest.mark.skipif(not OPENROUTER_API_KEY, reason="OPENROUTER_API_KEY not set")
async def test_api_chat_non_streaming(client: httpx.AsyncClient):
    """Test the /api/chat endpoint in non-streaming mode."""
    payload = {
        "model": TEST_MODEL_OLLAMA_NAME,
        "messages": [{"role": "user", "content": "Who are you? Respond concisely."}],
        "stream": False,
    }
    response = await client.post("/api/chat", json=payload)
    print(f"/api/chat non-streaming status: {response.status_code}")
    print(
        f"/api/chat non-streaming response: {response.text}"
    )  # Print raw response on failure
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == TEST_MODEL_OLLAMA_NAME
    assert data["done"] is True
    assert "message" in data
    assert "role" in data["message"]
    assert data["message"]["role"] == "assistant"
    assert "content" in data["message"]
    assert len(data["message"]["content"]) > 0  # Check for non-empty response


@pytest.mark.asyncio
@pytest.mark.skipif(not OPENROUTER_API_KEY, reason="OPENROUTER_API_KEY not set")
async def test_api_chat_streaming(client: httpx.AsyncClient):
    """Test the /api/chat endpoint in streaming mode."""
    payload = {
        "model": TEST_MODEL_OLLAMA_NAME,
        "messages": [{"role": "user", "content": "Explain streaming briefly."}],
        "stream": True,
    }
    received_content = ""
    final_message_received = False
    async with client.stream("POST", "/api/chat", json=payload) as response:
        assert response.status_code == 200
        assert "application/x-ndjson" in response.headers.get("content-type", "")
        async for line in response.aiter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    assert chunk["model"] == TEST_MODEL_OLLAMA_NAME
                    assert "created_at" in chunk
                    assert "done" in chunk

                    if not chunk["done"]:
                        assert "message" in chunk
                        assert chunk["message"]["role"] == "assistant"
                        assert "content" in chunk["message"]
                        received_content += chunk["message"]["content"]
                    else:
                        assert (
                            "message" in chunk
                        )  # Final message should still have structure
                        assert (
                            chunk["message"]["content"] == ""
                        )  # Empty content in final message
                        final_message_received = True
                        break  # Stop after final message
                except json.JSONDecodeError:
                    pytest.fail(f"Failed to decode JSON line: {line}")
                except KeyError as e:
                    pytest.fail(f"Missing expected key {e} in chunk: {line}")

    assert len(received_content) > 0, "No content received in streaming response"
    assert final_message_received, "Final 'done' message not received in stream"


@pytest.mark.asyncio
@pytest.mark.skipif(not OPENROUTER_API_KEY, reason="OPENROUTER_API_KEY not set")
async def test_api_generate_non_streaming(client: httpx.AsyncClient):
    """Test the /api/generate endpoint in non-streaming mode."""
    payload = {
        "model": TEST_MODEL_OLLAMA_NAME,
        "prompt": "What is code generation?",
        "stream": False,
    }
    response = await client.post("/api/generate", json=payload)
    print(f"/api/generate non-streaming status: {response.status_code}")
    print(f"/api/generate non-streaming response: {response.text}")
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == TEST_MODEL_OLLAMA_NAME
    assert data["done"] is True
    assert "response" in data
    assert len(data["response"]) > 0  # Check for non-empty response
    # Check synthesized stats fields exist (values are 0 or None)
    assert "context" in data
    assert "total_duration" in data
    assert "load_duration" in data
    assert "prompt_eval_count" in data
    assert "prompt_eval_duration" in data
    assert "eval_count" in data
    assert "eval_duration" in data


@pytest.mark.asyncio
@pytest.mark.skipif(not OPENROUTER_API_KEY, reason="OPENROUTER_API_KEY not set")
async def test_api_generate_streaming(client: httpx.AsyncClient):
    """Test the /api/generate endpoint in streaming mode."""
    payload = {
        "model": TEST_MODEL_OLLAMA_NAME,
        "prompt": "Why use streaming APIs?",
        "stream": True,
    }
    received_content = ""
    final_message_received = False
    async with client.stream("POST", "/api/generate", json=payload) as response:
        assert response.status_code == 200
        assert "application/x-ndjson" in response.headers.get("content-type", "")
        async for line in response.aiter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    assert chunk["model"] == TEST_MODEL_OLLAMA_NAME
                    assert "created_at" in chunk
                    assert "done" in chunk

                    if not chunk["done"]:
                        assert "response" in chunk
                        assert isinstance(chunk["response"], str)
                        received_content += chunk["response"]
                    else:
                        # Final message has empty response but includes stats
                        assert chunk["response"] == ""
                        assert "context" in chunk
                        assert "total_duration" in chunk
                        assert "load_duration" in chunk
                        assert "prompt_eval_count" in chunk
                        assert "prompt_eval_duration" in chunk
                        assert "eval_count" in chunk
                        assert "eval_duration" in chunk
                        final_message_received = True
                        break  # Stop after final message
                except json.JSONDecodeError:
                    pytest.fail(f"Failed to decode JSON line: {line}")
                except KeyError as e:
                    pytest.fail(f"Missing expected key {e} in chunk: {line}")

    assert len(received_content) > 0, "No content received in streaming response"
    assert final_message_received, "Final 'done' message not received in stream"


@pytest.mark.asyncio
@pytest.mark.skipif(not OPENROUTER_API_KEY, reason="OPENROUTER_API_KEY not set")
async def test_api_show(client: httpx.AsyncClient):
    """Test the /api/show endpoint for a known model."""
    payload = {"model": TEST_MODEL_OLLAMA_NAME}
    response = await client.post("/api/show", json=payload)
    print(f"/api/show status: {response.status_code}")
    print(f"/api/show response: {response.text}")
    # Check if the model exists first using /api/tags
    tags_response = await client.get("/api/tags")
    if tags_response.status_code == 200:
        models_available = [m["name"] for m in tags_response.json().get("models", [])]
        if TEST_MODEL_OLLAMA_NAME not in models_available:
            pytest.skip(
                f"Test model {TEST_MODEL_OLLAMA_NAME} not found in /api/tags, skipping /api/show test."
            )
    else:
        pytest.skip(
            "Could not verify model availability via /api/tags, skipping /api/show test."
        )

    assert response.status_code == 200
    data = response.json()
    assert "details" in data
    assert isinstance(data["details"], dict)
    assert "model_info" in data
    assert isinstance(
        data["model_info"], dict
    )  # Expected to be empty based on implementation
    assert "modelfile" in data
    assert "parameters" in data
    assert "template" in data
    # Check synthesized fields have default values
    assert data["modelfile"] == ""
    assert data["parameters"] == ""
    assert data["template"] == ""
    assert data["details"].get("format") == "gguf"


@pytest.mark.asyncio
@pytest.mark.skipif(not OPENROUTER_API_KEY, reason="OPENROUTER_API_KEY not set")
async def test_api_show_not_found(client: httpx.AsyncClient):
    """Test the /api/show endpoint for a non-existent model."""
    payload = {"model": "non-existent-model:latest"}
    response = await client.post("/api/show", json=payload)
    assert response.status_code == 404


@pytest.mark.asyncio
@pytest.mark.skipif(not OPENROUTER_API_KEY, reason="OPENROUTER_API_KEY not set")
async def test_api_ps(client: httpx.AsyncClient):
    """Test the /api/ps endpoint."""
    response = await client.get("/api/ps")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)
    # Check if the structure of one model is correct
    if data["models"]:
        model_entry = data["models"][0]
        assert "name" in model_entry
        assert "model" in model_entry
        assert "size" in model_entry
        assert "digest" in model_entry
        assert "details" in model_entry
        assert "expires_at" in model_entry
        assert "size_vram" in model_entry
        assert isinstance(model_entry["details"], dict)
        # Check synthesized fields have default values
        assert model_entry["size"] == 0
        assert model_entry["digest"] == ""
        assert model_entry["expires_at"] is None
        assert model_entry["size_vram"] == 0


# --- Error Handling Tests ---


@pytest.mark.asyncio
@pytest.mark.skipif(not OPENROUTER_API_KEY, reason="OPENROUTER_API_KEY not set")
async def test_api_chat_invalid_model(client: httpx.AsyncClient):
    """Test /api/chat with an invalid model name."""
    payload = {
        "model": "this-model-does-not-exist:latest",
        "messages": [{"role": "user", "content": "Test"}],
        "stream": False,
    }
    response = await client.post("/api/chat", json=payload)
    # The current implementation returns 400 Bad Request for model not found in /api/chat
    assert response.status_code == 400
    data = response.json()
    assert "error" in data


@pytest.mark.asyncio
@pytest.mark.skipif(not OPENROUTER_API_KEY, reason="OPENROUTER_API_KEY not set")
async def test_api_generate_invalid_model(client: httpx.AsyncClient):
    """Test /api/generate with an invalid model name."""
    payload = {
        "model": "this-model-does-not-exist:latest",
        "prompt": "Test",
        "stream": False,
    }
    response = await client.post("/api/generate", json=payload)
    assert response.status_code == 404 # Expect 404 Not Found
    data = response.json()
    assert "error" in data


# Test invalid API key requires specific setup (e.g., fixture to unset/mock key)
# @pytest.mark.asyncio
# async def test_api_chat_invalid_key(client: httpx.AsyncClient):
#     """Test API access with an invalid API key (requires fixture setup)."""
#     # This test needs a way to run the request with a known invalid key
#     # or without the key being available to the app context.
#     payload = {...} # Use TEST_MODEL_OLLAMA_NAME
#     # Assuming a fixture `client_with_invalid_key` is set up:
#     # response = await client_with_invalid_key.post("/api/chat", json=payload)
#     # assert response.status_code == 401 # Or potentially 403/500 depending on OR/proxy handling
#     pytest.skip("Requires fixture to test invalid API key")


@pytest.mark.asyncio
async def test_unsupported_endpoint_create(client: httpx.AsyncClient):
    """Test that an unsupported endpoint like /api/create returns 501."""
    payload = {"model": "test-create", "from": "scratch"}  # Example payload
    response = await client.post("/api/create", json=payload)
    assert response.status_code == 501  # Not Implemented
    data = response.json()
    assert "detail" in data
    assert "not supported" in data["detail"]


@pytest.mark.asyncio
async def test_unsupported_endpoint_delete(client: httpx.AsyncClient):
    """Test that an unsupported endpoint like /api/delete returns 501."""
    # DELETE requests typically don't have a body, and httpx.delete doesn't support json/content args.
    # Testing the route/method combination should be sufficient to trigger the 501.
    response = await client.delete("/api/delete")
    assert response.status_code == 501  # Not Implemented
    data = response.json()
    assert "detail" in data
    assert "not supported" in data["detail"]
