# API endpoint implementations will be added here

from fastapi import APIRouter, Request, status, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse, Response
from app import models
from app import openrouter
from app import utils
import time
import json
import httpx
from typing import Dict, Any
import logging
import traceback
from pydantic import ValidationError
import typing

router = APIRouter()


@router.head("/")
async def head_root():
    # Explicitly handle HEAD / to mimic Ollama's 200 OK response
    # Return correct headers but no body
    return Response(status_code=200, media_type="text/plain", headers={"Content-Length": "17"})


@router.get("/")
async def root():
    # Return plain text like standard Ollama server
    # This also implicitly handles HEAD / requests via FastAPI
    return PlainTextResponse("Ollama is running")


@router.get("/api/version")
def api_version():
    return {"version": "0.1.0-openrouter"}


@router.get("/api/tags")
async def api_tags(request: Request):
    # Use models and filter_set from app state
    models_list = request.app.state.all_models
    filter_set = request.app.state.filter_set

    try:
        filtered = utils.filter_models(models_list, filter_set)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        resp_models = []
        for m in filtered:
            # Use hardcoded placeholder details for compatibility
            details = models.OllamaTagDetails(
                format="gguf", # Hardcoded guess
                family="openrouter", # Placeholder
                families=["openrouter"], # Placeholder
                parameter_size="Unknown", # Placeholder
                quantization_level="Unknown" # Placeholder
            )

            resp_models.append(
                models.OllamaTagModel(
                    name=utils.openrouter_id_to_ollama_name(m["id"]),
                    modified_at=now,
                    size=0, # Hardcoded placeholder
                    digest="-", # Hardcoded placeholder
                    details=details, # Hardcoded details
                )
            )
        return models.OllamaTagsResponse(models=resp_models)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/chat")
async def api_chat(request: Request):
    # Access config, maps, and filter set from app state
    api_key = request.app.state.config["api_key"]
    ollama_map = request.app.state.ollama_to_openrouter_map
    filter_set = request.app.state.filter_set

    try:
        body = await request.json()
        req = models.OllamaChatRequest(**body)

        # Resolve model name using the new utility function
        resolved_ollama_name, openrouter_id = utils.resolve_model_name(req.model, ollama_map)

        if not resolved_ollama_name or not openrouter_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{req.model}' not found."
            )

        # Check against filter set if it exists
        if filter_set and resolved_ollama_name not in filter_set:
             raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Model '{resolved_ollama_name}' is not allowed by the filter."
            )

        # Build OpenRouter payload using resolved ID
        payload: Dict[str, Any] = {
            "model": openrouter_id,
            "messages": [m.model_dump() for m in req.messages],
            "stream": req.stream,
        }
        if req.options:
            payload.update(req.options)
        if req.format == "json":
            payload["response_format"] = {"type": "json_object"}
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if req.stream:
            # Define the async generator function for streaming
            async def streamer():
                try:
                    buffer = ""
                    stream_iterator = await openrouter.chat_completion(
                        api_key, payload, stream=True
                    )
                    try:
                        async for raw_chunk in stream_iterator:
                            if raw_chunk:
                                try:
                                    decoded_chunk = raw_chunk.decode("utf-8")
                                    buffer += decoded_chunk
                                except UnicodeDecodeError:
                                    continue

                                while "\n" in buffer:
                                    line_end = buffer.find("\n")
                                    raw_line = buffer[:line_end]
                                    decoded_line = raw_line.strip()
                                    buffer = buffer[line_end + 1 :]

                                    if decoded_line.startswith("data:"):
                                        data_content = decoded_line[
                                            len("data:") :
                                        ].strip()
                                        if data_content == "[DONE]":
                                            continue
                                        try:
                                            chunk = json.loads(data_content)
                                            choice = chunk.get("choices", [{}])[0]
                                            delta = choice.get("delta", {})
                                            finish_reason = choice.get("finish_reason")

                                            if (
                                                delta
                                                and "content" in delta
                                                and delta["content"] is not None
                                            ):
                                                content_to_yield = delta["content"]
                                                yield (
                                                    json.dumps(
                                                        {
                                                            # Use resolved name in response
                                                            "model": resolved_ollama_name,
                                                            "created_at": now,
                                                            "message": {
                                                                "role": "assistant",
                                                                "content": content_to_yield,
                                                            },
                                                            "done": False,
                                                        }
                                                    )
                                                    + "\n"
                                                )
                                            if finish_reason:
                                                yield (
                                                    json.dumps(
                                                        {
                                                            # Use resolved name in response
                                                            "model": resolved_ollama_name,
                                                            "created_at": now,
                                                            "message": {
                                                                "role": "assistant",
                                                                "content": "",
                                                            },
                                                            "done": True,
                                                        }
                                                    )
                                                    + "\n"
                                                )
                                                break
                                        except json.JSONDecodeError:
                                            print(
                                                f"Warning: Could not decode JSON from data line: {data_content!r}"
                                            )  # L22 - Keep Warning
                                            continue
                                        except Exception as e:
                                            print(
                                                f"PROXY: Exception processing chunk: {type(e)} - {e}, Line: {decoded_line!r}"
                                            )  # L23 - Keep Error
                                            continue
                                    else:
                                        pass
                    except Exception:
                        raise
                except httpx.HTTPStatusError as exc:
                    status_code = exc.response.status_code
                    try:
                        error_details = exc.response.json()
                        error_message = error_details.get("error", {}).get(
                            "message", str(error_details)
                        )
                    except Exception:
                        error_details = exc.response.text
                        error_message = error_details
                    yield (
                        json.dumps({"error": error_message, "status_code": status_code})
                        + "\n"
                    )
                except Exception as e:
                    error_message = f"Streaming Error: {str(e)}"
                    yield json.dumps({"error": error_message}) + "\n"

            # Return the StreamingResponse using the generator defined above
            return StreamingResponse(streamer(), media_type="application/x-ndjson")

        else:  # Non-streaming path
            resp = await openrouter.chat_completion(
                api_key, payload, stream=False
            )

            # -- Robustness Checks --
            # 1. Check if OpenRouter returned an error object despite 200 OK
            if isinstance(resp, dict) and "error" in resp:
                error_detail = resp["error"]
                status_code = 500 # Assume internal server error if not specified
                if isinstance(error_detail, dict):
                    # Try to extract code and message if possible
                    status_code = error_detail.get("code", 500) # Use 500 if code missing
                    error_message = error_detail.get("message", str(error_detail))
                    # Attempt to map OpenRouter error codes to HTTP status codes if needed
                    # e.g., if error_detail.get("code") == "invalid_request_error": status_code = 400
                else:
                    error_message = str(error_detail)

                # Try to determine a reasonable status code if possible
                if isinstance(status_code, str): # Handle potential non-integer codes
                    if status_code == 'invalid_request_error':
                        status_code = 400
                    else:
                         # fallback for unknown string codes
                         status_code = 500
                elif not isinstance(status_code, int) or status_code < 400:
                    status_code = 500 # Default to 500 if code is missing or invalid

                raise HTTPException(status_code=status_code, detail=error_message)

            # 2. Check for expected structure: choices list must exist and be non-empty
            if not isinstance(resp, dict) or "choices" not in resp or not resp["choices"]:
                raise HTTPException(
                    status_code=500,
                    detail=f"Unexpected non-streaming response format from OpenRouter: missing or empty 'choices'. Response: {str(resp)[:500]}"
                )

            # 3. Check structure within the first choice
            first_choice = resp["choices"][0]
            if "message" not in first_choice or "content" not in first_choice["message"]:
                 raise HTTPException(
                    status_code=500,
                    detail=f"Unexpected non-streaming response format from OpenRouter: choice missing 'message' or 'content'. Choice: {str(first_choice)[:500]}"
                )
            # -- End Robustness Checks --

            # If checks pass, extract content
            content = first_choice["message"]["content"]

            return models.OllamaChatResponse(
                # Use resolved name in response
                model=resolved_ollama_name,
                created_at=now,
                message=models.OllamaChatMessage(role="assistant", content=content),
                done=True,
            )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/generate")
async def api_generate(request: Request):
    # Access config, maps, and filter set from app state
    api_key = request.app.state.config["api_key"]
    ollama_map = request.app.state.ollama_to_openrouter_map
    filter_set = request.app.state.filter_set

    try:
        body = await request.json()
        req = models.OllamaGenerateRequest(**body)

        # Resolve model name using the new utility function
        resolved_ollama_name, openrouter_id = utils.resolve_model_name(req.model, ollama_map)

        if not resolved_ollama_name or not openrouter_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{req.model}' not found."
            )

        # Check against filter set if it exists
        if filter_set and resolved_ollama_name not in filter_set:
             raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Model '{resolved_ollama_name}' is not allowed by the filter."
            )

        # Build OpenRouter payload (map prompt to messages)
        messages = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.append({"role": "user", "content": req.prompt})

        payload: Dict[str, Any] = {
            "model": openrouter_id,
            "messages": messages,
            "stream": req.stream,
        }
        if req.options:
            payload.update(req.options)
        if req.format == "json":
            payload["response_format"] = {"type": "json_object"}

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if req.stream:

            async def streamer():
                full_response_content = ""
                try:
                    buffer = ""
                    stream_iterator = await openrouter.chat_completion(
                        api_key, payload, stream=True
                    )
                    try:
                        async for raw_chunk in stream_iterator:
                            if raw_chunk:
                                try:
                                    decoded_chunk = raw_chunk.decode("utf-8")
                                    buffer += decoded_chunk
                                except UnicodeDecodeError:
                                    continue

                                while "\n" in buffer:
                                    line_end = buffer.find("\n")
                                    raw_line = buffer[:line_end]
                                    decoded_line = raw_line.strip()
                                    buffer = buffer[line_end + 1 :]

                                    if decoded_line.startswith("data:"):
                                        data_content = decoded_line[
                                            len("data:") :
                                        ].strip()
                                        if data_content == "[DONE]":
                                            continue
                                        try:
                                            chunk = json.loads(data_content)
                                            choice = chunk.get("choices", [{}])[0]
                                            delta = choice.get("delta", {})
                                            finish_reason = choice.get("finish_reason")

                                            if (
                                                delta
                                                and "content" in delta
                                                and delta["content"] is not None
                                            ):
                                                content_to_yield = delta["content"]
                                                full_response_content += (
                                                    content_to_yield
                                                )
                                                yield (
                                                    json.dumps(
                                                        {
                                                            # Use resolved name in response
                                                            "model": resolved_ollama_name,
                                                            "created_at": now,
                                                            "response": content_to_yield,
                                                            "done": False,
                                                        }
                                                    )
                                                    + "\n"
                                                )

                                            if finish_reason:
                                                # Send final done message with empty response and stats
                                                yield (
                                                    json.dumps(
                                                        {
                                                            # Use resolved name in response
                                                            "model": resolved_ollama_name,
                                                            "created_at": now,
                                                            "response": "",
                                                            "done": True,
                                                            # Add synthesized stats here
                                                            "context": req.context or [],
                                                            "total_duration": 0, # Placeholder
                                                            "load_duration": 0, # Placeholder
                                                            "prompt_eval_count": 0, # Placeholder
                                                            "prompt_eval_duration": 0, # Placeholder
                                                            "eval_count": 0, # Placeholder
                                                            "eval_duration": 0 # Placeholder
                                                        }
                                                    )
                                                    + "\n"
                                                )
                                                break
                                        except json.JSONDecodeError:
                                            print(
                                                f"Warning: Could not decode JSON from data line: {data_content!r}"
                                            )
                                            continue
                                        except Exception as e:
                                            print(
                                                f"PROXY: Exception processing chunk: {type(e)} - {e}, Line: {decoded_line!r}"
                                            )
                                            continue
                                    else:
                                        pass  # Ignore non-data lines
                    except Exception:
                        raise

                except httpx.HTTPStatusError as exc:
                    status_code = exc.response.status_code
                    try:
                        error_details = exc.response.json()
                        error_message = error_details.get("error", {}).get(
                            "message", str(error_details)
                        )
                    except Exception:
                        error_details = exc.response.text
                        error_message = error_details
                    yield (
                        json.dumps({"error": error_message, "status_code": status_code})
                        + "\n"
                    )
                except Exception as e:
                    error_message = f"Streaming Error: {str(e)}"
                    yield json.dumps({"error": error_message}) + "\n"

            return StreamingResponse(streamer(), media_type="application/x-ndjson")

        else:  # Non-streaming path
            resp = await openrouter.chat_completion(
                api_key, payload, stream=False
            )
            content = resp["choices"][0]["message"]["content"]
            return models.OllamaGenerateResponse(
                # Use resolved name in response
                model=resolved_ollama_name,
                created_at=now,
                response=content,
                done=True,
                # Synthesized stats
                context=req.context,
                total_duration=0,
                load_duration=0,
                prompt_eval_count=None,
                prompt_eval_duration=0,
                eval_count=len(content.split()),  # Rough token count
                eval_duration=0,
            )

    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/show")
async def api_show(request: Request):
    # Access maps from app state
    ollama_map = request.app.state.ollama_to_openrouter_map
    openrouter_map = request.app.state.openrouter_to_ollama_map

    try:
        body = await request.json()
        req = models.OllamaShowRequest(**body)

        # Prioritize using req.name if provided, otherwise fallback to req.model
        name_to_resolve = req.name if req.name is not None else req.model

        # Use the resolve function which now handles None input
        resolved_ollama_name, _ = utils.resolve_model_name(name_to_resolve, ollama_map)

        if not resolved_ollama_name:
            # Provide a clearer error message indicating which name was attempted
            attempted_name = name_to_resolve if name_to_resolve is not None else "(Not provided)"
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{attempted_name}' not found."
            )

        # Construct the stubbed response
        # Note: We don't fetch specific details from OpenRouter for this endpoint
        details = models.OllamaShowDetails()
        # Attempt to extract family/parameter size from ID/name (basic)
        if resolved_ollama_name and "/" in resolved_ollama_name:
            details.family = resolved_ollama_name.split("/")[0]
        if resolved_ollama_name and ":" in resolved_ollama_name:
            # Very basic parsing, may not always be accurate
            parts = resolved_ollama_name.split(":")[-1]
            if "b" in parts.lower() or "m" in parts.lower():  # e.g., 7b, 8x7b, 1.5m
                details.parameter_size = parts.upper()

        # Fill in details fields with empty string/list if not available
        details.parent_model = ""
        details.format = details.format or ""
        details.family = details.family or ""
        details.families = details.families or []
        details.parameter_size = details.parameter_size or ""
        details.quantization_level = details.quantization_level or ""

        # Synthesize model_info as empty dict (Ollama returns a populated dict, but we can't fetch this info)
        model_info = {}

        # Synthesize tensors as empty list (Ollama returns a large list, but we can't fetch this info)
        tensors = []

        # Synthesize license, modelfile, parameters, template as empty strings
        response_obj = models.OllamaShowResponse(
            license="",
            modelfile="",
            parameters="",
            template="",
            details=details,
            model_info=model_info,
            tensors=tensors,
        )

        return response_obj

    except httpx.HTTPStatusError as exc:
        logging.getLogger("ollama-proxy").error("[ERROR] /api/show HTTPStatusError: %s\n%s", str(exc), traceback.format_exc())
        if exc.response.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{req.name}' not found upstream at OpenRouter.",
            )
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
    except HTTPException as http_exc:  # Re-raise existing HTTPExceptions
        logging.getLogger("ollama-proxy").error("[ERROR] /api/show HTTPException: %s\n%s", str(http_exc), traceback.format_exc())
        raise http_exc
    except Exception as exc:
        logging.getLogger("ollama-proxy").error("[ERROR] /api/show Exception: %s\n%s", str(exc), traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/api/ps")
async def api_ps():
    # This endpoint provides dummy info about running models
    # No need to access config or state for now
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "models": [],
        "created_at": now,
    }


# --- Embeddings Endpoints ---


async def _handle_embeddings(
    request: Request, ollama_model_name: str, input_data: str | list[str]
) -> dict:
    """Helper to handle shared logic for embedding endpoints."""
    # Access config and maps from app state
    api_key = request.app.state.config["api_key"]
    ollama_map = request.app.state.ollama_to_openrouter_map
    filter_set = request.app.state.filter_set

    # Resolve model name
    resolved_ollama_name, openrouter_id = utils.resolve_model_name(
        ollama_model_name, ollama_map
    )

    if not resolved_ollama_name or not openrouter_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{ollama_model_name}' not found."
        )

    # Check against filter set if it exists
    if filter_set and resolved_ollama_name not in filter_set:
         raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Model '{resolved_ollama_name}' is not allowed by the filter."
        )

    # Check if OpenRouter actually supports embeddings for this model ID
    # (We might need a separate check or rely on OpenRouter API error)
    # For now, assume any resolved model might work if the API supports it.

    payload = {"input": input_data, "model": openrouter_id}
    try:
        embedding_response = await openrouter.fetch_embeddings(api_key, payload)
        # Transform OpenRouter response to Ollama format
        # Assuming OpenRouter returns a list of embedding objects with an 'embedding' field
        embeddings = [
            item["embedding"] for item in embedding_response.get("data", [])
        ]
        # Ollama expects a single list for single input, list of lists for multiple
        if isinstance(input_data, str):
            return {"embedding": embeddings[0] if embeddings else []}
        else:
            return {"embeddings": embeddings}

    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/embed")
async def api_embed(request: Request):
    # Uses _handle_embeddings which now takes request object
    try:
        body = await request.json()
        req = models.OllamaEmbedRequest(**body)
        result = await _handle_embeddings(request, req.model, req.input)
        return result
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/embeddings")  # Older endpoint
async def api_embeddings(request: Request):
    # Uses _handle_embeddings which now takes request object
    try:
        body = await request.json()
        req = models.OllamaEmbeddingsRequest(**body)
        # Handle both single prompt (str) and multiple prompts (list)
        result = await _handle_embeddings(request, req.model, req.prompt)
        return result
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# --- Unsupported Endpoints ---


def raise_not_supported():
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="This Ollama API endpoint is not supported by the OpenRouter proxy.",
    )


@router.post("/api/create")
async def api_create(request: Request):
    raise_not_supported()


@router.post("/api/copy")
async def api_copy(request: Request):
    raise_not_supported()


@router.delete("/api/delete")
async def api_delete(request: Request):
    raise_not_supported()


@router.post("/api/pull")
async def api_pull(request: Request):
    raise_not_supported()


@router.post("/api/push")
async def api_push(request: Request):
    raise_not_supported()


@router.post("/api/blobs/{digest}")
async def api_blobs_post(digest: str, request: Request):
    raise_not_supported()


@router.head("/api/blobs/{digest}")
async def api_blobs_head(digest: str, request: Request):
    raise_not_supported()
