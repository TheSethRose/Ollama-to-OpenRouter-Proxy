# API endpoint implementations will be added here

from fastapi import APIRouter, Request, Response, status, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from app import models
from app import openrouter
from app import utils
import time
import json
import asyncio
import httpx

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
                size=0,
                digest=None,
                details=models.OllamaTagDetails(),
            ) for m in filtered
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
        payload = {
            "model": openrouter_id,
            "messages": [m.model_dump() for m in req.messages],
            "stream": req.stream
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
                    buffer = ''
                    stream_iterator = await openrouter.chat_completion(config["api_key"], payload, stream=True)
                    try:
                        async for raw_chunk in stream_iterator:
                            if raw_chunk:
                                try:
                                    decoded_chunk = raw_chunk.decode('utf-8')
                                    buffer += decoded_chunk
                                except UnicodeDecodeError:
                                    continue

                                while '\n' in buffer:
                                    print("PROXY: Entered while '\n' in buffer loop") # L1
                                    line_end = buffer.find('\n')
                                    print(f"PROXY: Found line_end: {line_end}") # L2
                                    raw_line = buffer[:line_end]
                                    decoded_line = raw_line.strip()
                                    print(f"PROXY: Extracted raw_line: {raw_line!r}, decoded_line: {decoded_line!r}") # L3
                                    buffer = buffer[line_end + 1:]
                                    print("PROXY: Updated buffer") # L4

                                    if decoded_line.startswith("data:"):
                                        print("PROXY: Line starts with 'data:'") # L5
                                        data_content = decoded_line[len("data:"):].strip()
                                        print(f"PROXY: Extracted data_content: {data_content!r}") # L6
                                        if data_content == "[DONE]":
                                            print("PROXY: data_content is '[DONE]', continuing loop") # L7
                                            continue
                                        print("PROXY: data_content is not '[DONE]'" ) # L8
                                        try:
                                            print("PROXY: Entering JSON parsing try block") # L9
                                            chunk = json.loads(data_content)
                                            print(f"PROXY: Parsed JSON chunk: {chunk}") # L10
                                            choice = chunk.get("choices", [{}])[0]
                                            print(f"PROXY: Extracted choice: {choice}") # L11
                                            delta = choice.get("delta", {})
                                            print(f"PROXY: Extracted delta: {delta}") # L12
                                            finish_reason = choice.get("finish_reason")
                                            print(f"PROXY: Extracted finish_reason: {finish_reason}") # L13

                                            if delta and "content" in delta and delta["content"] is not None:
                                                print("PROXY: Condition for yielding done:false MET") # L14
                                                content_to_yield = delta['content']
                                                print(f"PROXY: About to yield content: {content_to_yield!r}") # L15
                                                yield json.dumps({
                                                    "model": req.model,
                                                    "created_at": now,
                                                    "message": {"role": "assistant", "content": content_to_yield},
                                                    "done": False,
                                                }) + "\n"
                                                print("PROXY: Finished yielding done:false chunk") # L16
                                            else:
                                                 print("PROXY: Condition for yielding done:false NOT MET") # L17

                                            if finish_reason:
                                                print("PROXY: finish_reason detected") # L18
                                                print("PROXY: About to yield done:true chunk") # L19
                                                yield json.dumps({
                                                    "model": req.model,
                                                    "created_at": now,
                                                    "message": {"role": "assistant", "content": ""},
                                                    "done": True,
                                                }) + "\n"
                                                print("PROXY: Finished yielding done:true chunk, breaking inner loop") # L20
                                                break
                                            else:
                                                 print("PROXY: No finish_reason detected") # L21
                                        except json.JSONDecodeError as e:
                                            print(f"Warning: Could not decode JSON from data line: {data_content!r}") # L22 - Keep Warning
                                            continue
                                        except Exception as e:
                                            print(f"PROXY: Exception processing chunk: {type(e)} - {e}, Line: {decoded_line!r}") # L23 - Keep Error
                                            continue
                                    else:
                                         print("PROXY: Line does NOT start with 'data:'") # L24
                                print("PROXY: Exiting while '\n' in buffer loop") # L25
                    except Exception as e:
                        raise
                except httpx.HTTPStatusError as exc:
                    status_code = exc.response.status_code
                    try:
                        error_details = exc.response.json()
                        error_message = error_details.get("error", {}).get("message", str(error_details))
                    except Exception:
                        error_details = exc.response.text
                        error_message = error_details
                    yield json.dumps({
                        "error": error_message,
                        "status_code": status_code
                    }) + "\n"
                except Exception as e:
                    error_message = f"Streaming Error: {str(e)}"
                    yield json.dumps({"error": error_message}) + "\n"

            # Return the StreamingResponse using the generator defined above
            return StreamingResponse(streamer(), media_type="application/x-ndjson")

        else: # Non-streaming path
            resp = await openrouter.chat_completion(config["api_key"], payload, stream=False)
            print(f"\n--- OpenRouter Non-Streaming Response ---\n{resp}\n-----------------------------------------\n") # DEBUG LOGGING
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
