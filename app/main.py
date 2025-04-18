import os
import sys
import argparse
from fastapi import FastAPI
from dotenv import load_dotenv
from app.api import router
import httpx
from fastapi.responses import JSONResponse
import logging

app = FastAPI()

from app.api import router
app.include_router(router)

# --- Configuration & Startup Logic ---

def get_config():
    # Load environment variables from .env if present
    load_dotenv()

    parser = argparse.ArgumentParser(description="Ollama-to-OpenRouter Proxy")
    parser.add_argument("--api-key", type=str, default=None, help="OpenRouter API key")
    parser.add_argument("--host", type=str, default=None, help="Host to bind")
    parser.add_argument("--port", type=int, default=None, help="Port to listen on")
    parser.add_argument("--models-filter", type=str, default=None, help="Model filter file path")
    args, _ = parser.parse_known_args()

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("[ERROR] OPENROUTER_API_KEY is required (env or --api-key)", file=sys.stderr)
        sys.exit(1)

    # Use environment variables for host, port, and models-filter if not set via CLI
    host = args.host or os.getenv("HOST", "0.0.0.0")
    port = args.port or int(os.getenv("PORT", 11434))
    models_filter = args.models_filter or os.getenv("MODELS_FILTER", "models-filter.txt")

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

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
)
logger = logging.getLogger("ollama-proxy")

# Log startup configuration (never log API keys)
logger.info(f"Starting Ollama Proxy on {config['host']}:{config['port']}")
if config['models_filter_path'] and config['filter_set']:
    logger.info(f"Model filter loaded: {config['models_filter_path']} ({len(config['filter_set'])} models)")
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
    try:
        response = await call_next(request)
        logger.info(f"Response: {request.method} {request.url.path} {response.status_code}")
        return response
    except Exception as exc:
        logger.error(f"Error handling request: {request.method} {request.url.path} - {exc}")
        raise
