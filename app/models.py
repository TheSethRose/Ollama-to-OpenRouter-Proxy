from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal

# --- Ollama API Schemas ---

class OllamaChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    images: Optional[List[str]] = None # Note: Image support not planned for v1

class OllamaChatRequest(BaseModel):
    model: str
    messages: List[OllamaChatMessage]
    format: Optional[Literal["json"]] = None
    options: Optional[Dict[str, Any]] = None # e.g., {"temperature": 0.7}
    template: Optional[str] = None # Note: Likely ignored by proxy
    stream: Optional[bool] = False
    keep_alive: Optional[str] = None # Note: Ignored by proxy

class OllamaChatResponseDelta(BaseModel):
    role: Literal["assistant"]
    content: str

class OllamaChatStreamResponse(BaseModel):
    model: str
    created_at: str
    message: Optional[OllamaChatResponseDelta] = None # Null in final 'done' message
    done: bool
    # Ollama includes more stats in the final message, omitted here

class OllamaChatResponse(BaseModel):
    model: str
    created_at: str
    message: OllamaChatMessage # Role is assistant here
    done: bool = True
    # Ollama includes stats here, omitted

class OllamaTagDetails(BaseModel):
    # Placeholder, as we won't have real details from OpenRouter
    format: Optional[str] = None
    family: Optional[str] = None
    families: Optional[List[str]] = None
    parameter_size: Optional[str] = None
    quantization_level: Optional[str] = None

class OllamaTagModel(BaseModel):
    name: str
    modified_at: str
    size: int
    digest: Optional[str] = None
    details: OllamaTagDetails

class OllamaTagsResponse(BaseModel):
    models: List[OllamaTagModel]

# --- OpenRouter Schemas (Simplified, for internal use) ---

class OpenRouterModel(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    pricing: Dict[str, Any]
    context_length: Optional[int] = None
    # Add other fields if needed

class OpenRouterModelsResponse(BaseModel):
    data: List[OpenRouterModel]
