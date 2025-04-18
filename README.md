# Python Ollama-to-OpenRouter Proxy

## Overview

This project provides a Python-based proxy server that emulates the Ollama REST API but forwards requests to the OpenRouter API. It allows tools designed for Ollama to leverage models available through OpenRouter, supporting chat completions, model listing, and streaming.

- **Ollama API Emulation:** Exposes `/api/tags`, `/api/chat`, and `/api/version` endpoints.
- **OpenRouter Forwarding:** Translates Ollama API requests to OpenRouter format and vice versa.
- **Streaming Support:** Supports streaming chat completions.
- **Model Filtering:** Allows restricting available models via a filter file.

## Architecture

- **FastAPI** for the web server and API validation
- **httpx** for async HTTP requests to OpenRouter
- **Pydantic** for request/response schemas
- **uv** for fast dependency management and running

```
Client (Ollama API) <-> [Proxy] <-> OpenRouter API
```

## Setup Instructions

### Prerequisites
- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (for install/run)

### Local Development
1. **Clone the repo:**
   ```sh
   git clone <repo-url>
   cd ollama-proxy
   ```
2. **Install dependencies:**
   ```sh
   uv pip install -r requirements.txt
   ```
3. **Configure environment:**
   - Copy `.env.example` to `.env` and set your `OPENROUTER_API_KEY`.
   - (Optional) Create `models-filter.txt` to restrict available models (one model name per line, e.g. `gpt-4o`).
4. **Run the server:**
   ```sh
   uv icorn app.main:app --host 0.0.0.0 --port 11434
   ```
   - Use `--api-key`, `--host`, `--port`, or `--models-filter` CLI args to override defaults.

### Docker
1. **Build the image:**
   ```sh
   docker build -t ollama-proxy .
   ```
2. **Run the container:**
   ```sh
   docker run -e OPENROUTER_API_KEY=sk-... -p 11434:11434 ollama-proxy
   ```
   - Mount a custom `models-filter.txt` if needed.

## Configuration

- `OPENROUTER_API_KEY` (env or `--api-key`): **Required**. Your OpenRouter API key.
- `--host`: Host to bind (default: `0.0.0.0`).
- `--port`: Port to listen on (default: `11434`).
- `--models-filter`: Path to model filter file (default: `models-filter.txt`).

## API Endpoints

### `GET /api/version`
Returns the proxy version.
```json
{"version": "0.1.0-openrouter"}
```

### `GET /api/tags`
Lists available models (optionally filtered).
```json
{
  "models": [
    {
      "name": "gpt-4o:latest",
      "modified_at": "2024-06-01T12:00:00Z",
      "size": 0,
      "digest": null,
      "details": {}
    }
  ]
}
```

### `POST /api/chat`
Chat completion (streaming or non-streaming).
**Request:**
```json
{
  "model": "gpt-4o:latest",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "stream": true
}
```
**Response (streaming):**
```
{"model": "gpt-4o:latest", "created_at": "...", "message": {"role": "assistant", "content": "Hi"}, "done": false}
... (chunks)
{"model": "gpt-4o:latest", "created_at": "...", "message": null, "done": true}
```
**Response (non-streaming):**
```json
{
  "model": "gpt-4o:latest",
  "created_at": "...",
  "message": {"role": "assistant", "content": "Hi"},
  "done": true
}
```

## Model Filtering
- If `models-filter.txt` exists, only models whose processed name (e.g., `gpt-4o`) is listed will be available via `/api/tags` and `/api/chat`.
- List one model per line, e.g.:
  ```
  gpt-4o
  mistral-large
  llama-3
  ```

## License
MIT
