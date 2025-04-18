# Client logic for interacting with OpenRouter API will be added here

import httpx
from typing import Dict, Any, AsyncGenerator
from app.models import OpenRouterModelsResponse

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

async def fetch_models(api_key: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{OPENROUTER_BASE_URL}/models", headers=headers)
        resp.raise_for_status()
        return resp.json()

async def chat_completion(api_key: str, payload: Dict[str, Any], stream: bool = False) -> Any:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if stream:
        async def stream_generator():
            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    resp = await client.post(
                        f"{OPENROUTER_BASE_URL}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            except Exception as e:
                print(f"openrouter.py stream_generator error: {e}")
                return
        return stream_generator()
    else:
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()

# Error handling helpers can be added as needed
