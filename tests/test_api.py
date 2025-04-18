import os
import pytest
from fastapi.testclient import TestClient
from app.main import app
import json

client = TestClient(app)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

pytestmark = pytest.mark.skipif(
    not OPENROUTER_API_KEY,
    reason="OPENROUTER_API_KEY must be set to run real OpenRouter integration tests."
)

def test_api_version():
    resp = client.get("/api/version")
    assert resp.status_code == 200
    assert resp.json()["version"].startswith("0.1.0")

def test_api_tags():
    resp = client.get("/api/tags")
    assert resp.status_code == 200
    models = resp.json()["models"]
    # Should include the free Mistral model
    assert any("mistral-7b-instruct:free" in m["name"] for m in models)

def test_api_chat_non_streaming():
    payload = {
        "model": "mistral-7b-instruct:free:latest",
        "messages": [{"role": "user", "content": "Hi!"}],
        "stream": False
    }
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "mistral-7b-instruct:free:latest"
    assert "message" in data and "content" in data["message"]
    assert data["done"] is True

def test_api_chat_streaming():
    payload = {
        "model": "mistral-7b-instruct:free:latest",
        "messages": [{"role": "user", "content": "Hi!"}],
        "stream": True
    }

    buffer = ""
    found_done_false = False
    found_done_true = False
    all_content = ""

    # Use client.stream for streaming requests
    with client.stream("POST", "/api/chat", json=payload) as resp:
        # Iterate over raw bytes chunks
        for chunk_bytes in resp.iter_bytes():
            buffer += chunk_bytes.decode('utf-8') # Decode and add to buffer

            # Process complete lines from buffer
            while '\n' in buffer:
                line_end = buffer.find('\n')
                line = buffer[:line_end].strip() # Extract the full line (should be JSON)
                buffer = buffer[line_end + 1:]

                if not line: # Skip empty lines that might result from \n\n
                    continue

                # REMOVED: Check for line.startswith('data:')
                # The proxy yields complete JSON lines, not SSE data: lines
                try:
                    # Attempt to parse the line directly as JSON
                    data_obj = json.loads(line)

                    if isinstance(data_obj, dict):
                        # Check for errors yielded by the proxy stream
                        if "error" in data_obj:
                            print(f"ERROR received from proxy stream: {data_obj['error']}")
                            # Optionally fail the test here depending on expected behavior
                            # assert False, f"Stream yielded error: {data_obj['error']}"
                            found_done_true = True # Treat errors as stream end? Or handle differently?
                            break

                        if data_obj.get("done") is False:
                            found_done_false = True
                            if "message" in data_obj and isinstance(data_obj["message"], dict):
                                content = data_obj["message"].get("content")
                                if content:
                                    all_content += content
                        elif data_obj.get("done") is True:
                            found_done_true = True
                            break # Stop processing lines once done=True is received

                except json.JSONDecodeError:
                    print(f"Warning: Test could not decode JSON from line: {line!r}")
                    pass # Ignore lines from proxy that aren't valid JSON (shouldn't happen?)
            if found_done_true:
                break

        # Check status code AFTER iterating through the stream
        assert resp.status_code == 200

    # Final assertions after processing the stream
    assert found_done_false, "Stream did not contain any intermediate chunk with 'done': false"
    assert found_done_true, "Stream did not contain the final chunk with 'done': true"
    assert len(all_content) > 0, "Stream did not yield any content"

def test_api_chat_invalid_model():
    payload = {
        "model": "not-a-real-model",
        "messages": [{"role": "user", "content": "Hi!"}],
        "stream": False
    }
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 400 or resp.status_code == 404
    assert "error" in resp.json()
