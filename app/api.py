# API endpoint implementations will be added here

from fastapi import APIRouter, Request, status, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
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


@router.get("/api/version")
def api_version():
    return {"version": "0.1.0-openrouter"}


@router.get("/api/tags")
async def api_tags():
    from app.main import config

    try:
        data = await openrouter.fetch_models(config["api_key"])
        models_list = data.get("data", [])
        filtered = utils.filter_models(models_list, config["filter_set"])
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        resp_models = [
            models.OllamaTagModel(
                name=utils.openrouter_id_to_ollama_name(m["id"]),
                modified_at=now,
                digest=None,
                details=models.OllamaTagDetails(),
            )
            for m in filtered
        ]
        return models.OllamaTagsResponse(models=resp_models)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/chat")
async def api_chat(request: Request):
    from app.main import config

    try:
        body = await request.json()
        req = models.OllamaChatRequest(**body)
        # Map Ollama model to OpenRouter model id
        data = await openrouter.fetch_models(config["api_key"])
        mapping = utils.build_ollama_to_openrouter_map(data.get("data", []))
        openrouter_id = mapping.get(req.model)
        if not openrouter_id:
            return JSONResponse({"error": "Model not found"}, status_code=400)
        # Build OpenRouter payload
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
                        config["api_key"], payload, stream=True
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
                                                            "model": req.model,
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
                                                            "model": req.model,
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
                config["api_key"], payload, stream=False
            )
            content = resp["choices"][0]["message"]["content"]
            return models.OllamaChatResponse(
                model=req.model,
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
    from app.main import config

    try:
        body = await request.json()
        req = models.OllamaGenerateRequest(**body)

        # Map Ollama model to OpenRouter model id
        data = await openrouter.fetch_models(config["api_key"])
        mapping = utils.build_ollama_to_openrouter_map(data.get("data", []))
        openrouter_id = mapping.get(req.model)
        if not openrouter_id:
            return JSONResponse(
                {
                    "error": f"Model '{req.model}' not found or not supported by the proxy."
                },
                status_code=404,
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
                        config["api_key"], payload, stream=True
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
                                                        models.OllamaGenerateStreamResponse(
                                                            model=req.model,
                                                            created_at=now,
                                                            response=content_to_yield,
                                                            done=False,
                                                        ).model_dump()
                                                    )
                                                    + "\n"
                                                )

                                            if finish_reason:
                                                # Send final done message with empty response and stats
                                                yield (
                                                    json.dumps(
                                                        models.OllamaGenerateStreamResponse(
                                                            model=req.model,
                                                            created_at=now,
                                                            response="",
                                                            done=True,
                                                            # Synthesized stats
                                                            context=req.context,  # Pass context back if provided
                                                            total_duration=0,
                                                            load_duration=0,
                                                            prompt_eval_count=None,  # Cannot get this from OR
                                                            prompt_eval_duration=0,
                                                            eval_count=len(
                                                                full_response_content.split()
                                                            ),  # Rough token count
                                                            eval_duration=0,
                                                        ).model_dump()
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
                config["api_key"], payload, stream=False
            )
            content = resp["choices"][0]["message"]["content"]
            return models.OllamaGenerateResponse(
                model=req.model,
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
    from app.main import config

    try:
        body = await request.json()
        # logging.getLogger("ollama-proxy").info("[DEBUG] /api/show incoming request body: %s", json.dumps(body, indent=2)) # COMMENTED OUT
        req = models.OllamaShowRequest(**body)
        # logging.getLogger("ollama-proxy").info("[DEBUG] /api/show resolved model: %s", req.model) # COMMENTED OUT

        # Fetch all models and find the requested one by Ollama name
        data = await openrouter.fetch_models(config["api_key"])
        models_list = data.get("data", [])
        # Need the reverse mapping to find the OpenRouter ID
        reverse_mapping = utils.build_openrouter_to_ollama_map(models_list)
        # Filter models based on config before searching
        filtered_openrouter_ids = {
            m["id"] for m in utils.filter_models(models_list, config["filter_set"])
        }

        target_openrouter_id = None
        target_model_data = None

        for or_id, ollama_name in reverse_mapping.items():
            if ollama_name == req.model:
                target_openrouter_id = or_id
                break

        if not target_openrouter_id:
            logging.getLogger("ollama-proxy").error("[ERROR] /api/show: Model '%s' not found in reverse mapping", req.model)
            raise HTTPException(
                status_code=404, detail=f"Model '{req.model}' not found."
            )

        # Find the full data for the target model
        for model_data in models_list:
            if model_data["id"] == target_openrouter_id:
                # Check if this model is allowed by the filter
                if target_openrouter_id not in filtered_openrouter_ids:
                    logging.getLogger("ollama-proxy").error("[ERROR] /api/show: Model '%s' filtered out by proxy configuration", req.model)
                    raise HTTPException(
                        status_code=404,
                        detail=f"Model '{req.model}' is available upstream but filtered out by the proxy configuration.",
                    )
                target_model_data = model_data
                break

        if not target_model_data:
            logging.getLogger("ollama-proxy").error("[ERROR] /api/show: Model '%s' (ID: %s) data not found upstream", req.model, target_openrouter_id)
            raise HTTPException(
                status_code=404,
                detail=f"Model '{req.model}' (ID: {target_openrouter_id}) data not found upstream.",
            )

        # Synthesize the OllamaShowResponse
        details = models.OllamaShowDetails()
        # Attempt to extract family/parameter size from ID/name (basic)
        if target_openrouter_id and "/" in target_openrouter_id:
            details.family = target_openrouter_id.split("/")[0]
        if target_openrouter_id and ":" in target_openrouter_id:
            # Very basic parsing, may not always be accurate
            parts = target_openrouter_id.split(":")[-1]
            if "b" in parts.lower() or "m" in parts.lower():  # e.g., 7b, 8x7b, 1.5m
                details.parameter_size = parts.upper()

        # Fill in details fields with empty string/list if not available
        details.parent_model = ""
        details.format = details.format or ""
        details.family = details.family or ""
        details.families = details.families or []
        details.parameter_size = details.parameter_size or ""
        details.quantization_level = details.quantization_level or ""

        # logging.getLogger("ollama-proxy").info("[DEBUG] /api/show synthesized details: %s", json.dumps(details.model_dump(), indent=2)) # COMMENTED OUT

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

        # Type validation check for all fields in the response (on the Pydantic model instance)
        def check_types_instance(obj, model, prefix="response"):
            if hasattr(model, 'model_fields'):  # Pydantic v2
                fields = model.model_fields
            elif hasattr(model, '__fields__'):  # Pydantic v1
                fields = model.__fields__
            else:
                fields = {}
            for k in fields:
                expected_type = fields[k].annotation
                v = getattr(obj, k, None)
                origin = typing.get_origin(expected_type)
                args = typing.get_args(expected_type)
                # For nested models, recurse
                if hasattr(fields[k], 'type_') and hasattr(fields[k].type_, 'model_fields'):
                    if not isinstance(v, fields[k].type_):
                        logging.getLogger("ollama-proxy").error(
                            f"[ERROR] Type mismatch at {prefix}.{k}: expected {fields[k].type_}, got {type(v)} ({repr(v)})"
                        )
                    else:
                        check_types_instance(v, fields[k].type_, f"{prefix}.{k}")
                elif origin is list:
                    if not isinstance(v, list):
                        logging.getLogger("ollama-proxy").error(
                            f"[ERROR] Type mismatch at {prefix}.{k}: expected list, got {type(v)} ({repr(v)})"
                        )
                    else:
                        elem_type = args[0] if args else object
                        for idx, item in enumerate(v):
                            elem_origin = typing.get_origin(elem_type)
                            if elem_origin:
                                if not isinstance(item, elem_origin):
                                    logging.getLogger("ollama-proxy").error(
                                        f"[ERROR] Type mismatch at {prefix}.{k}[{idx}]: expected {elem_origin}, got {type(item)} ({repr(item)})"
                                    )
                                # Optionally recurse for nested generics
                            elif not isinstance(item, elem_type):
                                logging.getLogger("ollama-proxy").error(
                                    f"[ERROR] Type mismatch at {prefix}.{k}[{idx}]: expected {elem_type}, got {type(item)} ({repr(item)})"
                                )
                elif origin is dict:
                    if not isinstance(v, dict):
                        logging.getLogger("ollama-proxy").error(
                            f"[ERROR] Type mismatch at {prefix}.{k}: expected dict, got {type(v)} ({repr(v)})"
                        )
                elif expected_type is not None and origin is None:
                    # Only check non-generic types
                    if v is not None and not isinstance(v, expected_type):
                        logging.getLogger("ollama-proxy").error(
                            f"[ERROR] Type mismatch at {prefix}.{k}: expected {expected_type}, got {type(v)} ({repr(v)})"
                        )

        check_types_instance(response_obj, models.OllamaShowResponse)

        # DEBUG: Print the full response dict for inspection - COMMENTED OUT
        # logging.getLogger("ollama-proxy").info(
        #     "[DEBUG] /api/show response for model '%s': %s",
        #     req.model,
        #     json.dumps(response_obj.model_dump(), indent=2)
        # )
        return response_obj

    except httpx.HTTPStatusError as exc:
        logging.getLogger("ollama-proxy").error("[ERROR] /api/show HTTPStatusError: %s\n%s", str(exc), traceback.format_exc())
        if exc.response.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{req.model}' not found upstream at OpenRouter.",
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
    from app.main import config

    try:
        data = await openrouter.fetch_models(config["api_key"])
        models_list = data.get("data", [])
        filtered_models_data = utils.filter_models(models_list, config["filter_set"])

        running_models = []
        for m_data in filtered_models_data:
            ollama_name = utils.openrouter_id_to_ollama_name(m_data["id"])
            details = models.OllamaShowDetails()  # Create default details
            # Synthesize basic details from ID
            if "/" in m_data["id"]:
                details.family = m_data["id"].split("/")[0]
            if ":" in m_data["id"]:
                parts = m_data["id"].split(":")[-1]
                if "b" in parts.lower() or "m" in parts.lower():
                    details.parameter_size = parts.upper()

            ps_model = models.OllamaPsModel(
                name=ollama_name,
                model=ollama_name,
                digest=None,
                details=details,
                expires_at=None,
            )
            running_models.append(ps_model)

        return models.OllamaPsResponse(models=running_models)

    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# --- Embeddings Endpoints ---


async def _handle_embeddings(
    ollama_model_name: str, input_data: str | list[str]
) -> dict:
    """Core logic for handling embeddings, shared by /api/embed and /api/embeddings."""
    from app.main import config

    # Map Ollama model to OpenRouter model id
    data = await openrouter.fetch_models(config["api_key"])
    mapping = utils.build_ollama_to_openrouter_map(data.get("data", []))
    openrouter_id = mapping.get(ollama_model_name)

    # Check if model exists and is allowed by filter
    if not openrouter_id:
        # Check if it *would* be available but is filtered
        reverse_mapping = utils.build_openrouter_to_ollama_map(data.get("data", []))
        if (
            ollama_model_name in reverse_mapping.values()
        ):  # Check if it exists upstream at all
            raise HTTPException(
                status_code=404,
                detail=f"Model '{ollama_model_name}' is available upstream but filtered out by the proxy configuration.",
            )
        else:
            raise HTTPException(
                status_code=404, detail=f"Model '{ollama_model_name}' not found."
            )
    else:
        # Double check it wasn't filtered out
        models_list = data.get("data", [])
        filtered_openrouter_ids = {
            m["id"] for m in utils.filter_models(models_list, config["filter_set"])
        }
        if openrouter_id not in filtered_openrouter_ids:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{ollama_model_name}' is available upstream but filtered out by the proxy configuration.",
            )

    # OpenRouter expects 'input' to always be a list for embeddings
    if isinstance(input_data, str):
        input_list = [input_data]
    else:
        input_list = input_data

    payload = {"model": openrouter_id, "input": input_list}

    # Call OpenRouter
    openrouter_response = await openrouter.fetch_embeddings(config["api_key"], payload)
    return openrouter_response


@router.post("/api/embed")
async def api_embed(request: Request):
    try:
        body = await request.json()
        req = models.OllamaEmbedRequest(**body)

        openrouter_response = await _handle_embeddings(req.model, req.input)

        # Extract embeddings from OpenRouter response
        # Assumes OpenAI-compatible structure: response['data'][i]['embedding']
        embeddings_list = [
            item["embedding"] for item in openrouter_response.get("data", [])
        ]

        return models.OllamaEmbedResponse(
            model=req.model,  # Return the originally requested Ollama model name
            embeddings=embeddings_list,
        )

    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
    except HTTPException as http_exc:
        raise http_exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/embeddings")  # Older endpoint
async def api_embeddings(request: Request):
    try:
        body = await request.json()
        req = models.OllamaEmbeddingsRequest(**body)

        # Use the shared handler, passing the prompt as a single input string
        openrouter_response = await _handle_embeddings(req.model, req.prompt)

        # Extract the first embedding from the response
        embeddings_list = [
            item["embedding"] for item in openrouter_response.get("data", [])
        ]
        if not embeddings_list:
            # Should not happen if OpenRouter API call succeeded, but handle defensively
            raise HTTPException(
                status_code=500, detail="Upstream API returned no embeddings data."
            )

        return models.OllamaEmbeddingsResponse(embedding=embeddings_list[0])

    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
    except HTTPException as http_exc:
        raise http_exc
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
