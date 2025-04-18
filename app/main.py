import os
import sys
import argparse
from fastapi import FastAPI
from dotenv import load_dotenv
from app.api import router
import httpx
from fastapi.responses import JSONResponse, StreamingResponse
import logging
import json
from starlette.concurrency import iterate_in_threadpool

app = FastAPI()

app.include_router(router)

# --- Configuration & Startup Logic ---


def get_config():
    # Load environment variables from .env if present
    load_dotenv()

    parser = argparse.ArgumentParser(description="Ollama-to-OpenRouter Proxy")
    parser.add_argument("--api-key", type=str, default=None, help="OpenRouter API key")
    parser.add_argument("--host", type=str, default=None, help="Host to bind")
    parser.add_argument("--port", type=int, default=None, help="Port to listen on")
    parser.add_argument(
        "--models-filter", type=str, default=None, help="Model filter file path"
    )
    args, _ = parser.parse_known_args()

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print(
            "[ERROR] OPENROUTER_API_KEY is required (env or --api-key)", file=sys.stderr
        )
        sys.exit(1)

    # Use environment variables for host, port, and models-filter if not set via CLI
    host = args.host or os.getenv("HOST", "0.0.0.0")
    port = args.port or int(os.getenv("PORT", 11434))
    models_filter = args.models_filter or os.getenv(
        "MODELS_FILTER", "models-filter.txt"
    )

    # Load model filter file if present
    filter_set = set()
    if os.path.exists(models_filter):
        with open(models_filter, "r") as f:
            for line in f:
                name = line.strip()
                if name:
                    filter_set.add(name)

    config = {
        "api_key": api_key,
        "host": host,
        "port": port,
        "models_filter_path": models_filter,
        "filter_set": filter_set,
    }
    return config


config = get_config()

# Print loaded config for debugging (do not print API keys)
print("[DEBUG] Proxy config:")
for k, v in config.items():
    if k == "api_key":
        print(f"  {k}: [REDACTED]")
    else:
        print(f"  {k}: {v}")

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("ollama-proxy")

# Log startup configuration (never log API keys)
logger.info(f"Starting Ollama Proxy on {config['host']}:{config['port']}")
if config["models_filter_path"] and config["filter_set"]:
    logger.info(
        f"Model filter loaded: {config['models_filter_path']} ({len(config['filter_set'])} models)"
    )
else:
    logger.info("No model filter applied.")

# Startup logic and route registration will be added here


@app.exception_handler(httpx.HTTPStatusError)
async def openrouter_http_error_handler(request, exc):
    try:
        detail = exc.response.json().get("error", str(exc))
    except Exception:
        detail = str(exc)
    return JSONResponse({"error": detail}, status_code=exc.response.status_code)


@app.exception_handler(Exception)
async def generic_error_handler(request, exc):
    return JSONResponse({"error": str(exc)}, status_code=500)


@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    print(f"[DEBUG] Incoming request: {request.method} {request.url}")
    try:
        response = await call_next(request)

        # Clone the response body to log it without consuming it
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk
        # Reset the iterator for the actual response sending
        response.body_iterator = iterate_in_threadpool(iter([response_body]))

        # Log response details
        logger.info(
            f"Response: {request.method} {request.url.path} {response.status_code}"
        )
        print(
            f"[DEBUG] Outgoing response: {response.status_code} for {request.method} {request.url}"
        )

        # Attempt to log JSON body if media type is application/json and not too large
        if response.media_type == "application/json" and len(response_body) < 10000:
            try:
                body_json = json.loads(response_body.decode('utf-8'))
                print(f"[DEBUG] Response Body (JSON): {json.dumps(body_json, indent=2)}")
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"[DEBUG] Response Body (JSON decode error: {e}): {response_body[:500]}...")
        elif response.media_type == "application/x-ndjson":
             print("[DEBUG] Response Body: StreamingResponse (NDJSON - body not logged)")
        elif len(response_body) >= 10000:
            print(f"[DEBUG] Response Body: Too large to log ({len(response_body)} bytes, type: {response.media_type})")
        else:
             print(f"[DEBUG] Response Body (Other type: {response.media_type}): {response_body[:500]}...")

        return response
    except Exception as exc:
        logger.error(
            f"Error handling request: {request.method} {request.url.path} - {exc}"
        )
        print(f"[DEBUG] Exception: {exc} for {request.method} {request.url}")
        raise
