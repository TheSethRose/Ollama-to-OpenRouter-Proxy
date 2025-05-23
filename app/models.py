from pydantic import BaseModel, model_validator, Field
from typing import List, Optional, Dict, Any, Literal

# --- Ollama API Schemas ---


class OllamaChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    images: Optional[List[str]] = None  # Note: Image support not planned for v1


class OllamaChatRequest(BaseModel):
    model: str
    messages: List[OllamaChatMessage]
    format: Optional[Literal["json"]] = None
    options: Optional[Dict[str, Any]] = None  # e.g., {"temperature": 0.7}
    template: Optional[str] = None  # Note: Likely ignored by proxy
    stream: Optional[bool] = False
    keep_alive: Optional[str] = None  # Note: Ignored by proxy


class OllamaChatResponseDelta(BaseModel):
    role: Literal["assistant"]
    content: str


class OllamaChatStreamResponse(BaseModel):
    model: str
    created_at: str
    message: Optional[OllamaChatResponseDelta] = None  # Null in final 'done' message
    done: bool
    # Ollama includes more stats in the final message, omitted here


class OllamaChatResponse(BaseModel):
    model: str
    created_at: str
    message: OllamaChatMessage  # Role is assistant here
    done: bool = True
    # Add missing stats fields (synthesized)
    total_duration: Optional[int] = 0
    load_duration: Optional[int] = 0
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = 0
    eval_count: Optional[int] = 0
    eval_duration: Optional[int] = 0
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
    size: Optional[int] = None
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


# --- /api/generate Schemas ---


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    suffix: str | None = None
    images: list[str] | None = None
    format: str | None = None
    options: dict | None = None
    system: str | None = None
    template: str | None = None
    stream: bool | None = True
    raw: bool | None = False
    keep_alive: str | None = None
    context: list[int] | None = None  # Deprecated, but included for compatibility


class OllamaGenerateStreamResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool = False
    # Include final stats when done=True
    context: list[int] | None = None  # Deprecated
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int | None = None
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0


class OllamaGenerateResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool = True
    context: list[int] | None = None  # Deprecated
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int | None = None
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0


# --- /api/show Schemas ---


class OllamaShowRequest(BaseModel):
    # Allow 'name' to be optional, defaulting to None if missing
    name: Optional[str] = None
    # Keep model field for backward compatibility or direct use if provided
    model: Optional[str] = None

    # NB: Pydantic v2 model_validator replaces root_validator
    @model_validator(mode='before')
    @classmethod
    def check_name_or_model(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # If neither is present, that's an error (handled by framework or subsequent checks)
            if 'name' not in data and 'model' not in data:
                 # Let Pydantic handle the validation error if both are truly missing
                 # Or raise a clearer error if desired:
                 # raise ValueError("Request must contain either 'name' or 'model' field.")
                 pass # Allow Pydantic to potentially catch this based on Optional types
            # If 'name' is present, prioritize it for our logic
            # If 'model' is also present, we can ignore it or keep it
            # If only 'model' is present, we will use it.
            # No data mutation needed here if subsequent code checks req.name or req.model
            pass
        elif data is None:
             raise ValueError("Request body cannot be empty.")
        return data


class OllamaShowDetails(BaseModel):
    parent_model: str = ""
    format: str = ""
    family: str = ""
    families: list[str] = []
    parameter_size: str = ""
    quantization_level: str = ""


class OllamaShowTensor(BaseModel):
    name: str = ""
    type: str = ""
    shape: list[int] = []


class OllamaShowModelInfo(BaseModel):
    # These fields are highly specific to Ollama's internal model representation
    # and cannot be accurately fetched from OpenRouter. Return empty dict.
    pass


class OllamaShowResponse(BaseModel):
    license: str = ""
    modelfile: str = ""
    parameters: str = ""
    template: str = ""
    details: OllamaShowDetails = OllamaShowDetails()
    model_info: dict = {}
    tensors: list[OllamaShowTensor] = []
    # projector REMOVED to match Ollama
    # modified_at is not present in Ollama's /api/show, so remove it

    model_config = {
        "protected_namespaces": (),
    }


# --- /api/ps Schemas --- (List Running Models)


class OllamaPsModel(BaseModel):
    name: str
    model: str
    size: Optional[int] = None
    digest: Optional[str]
    details: OllamaShowDetails
    expires_at: Optional[str] = None
    size_vram: Optional[int] = None


class OllamaPsResponse(BaseModel):
    models: list[OllamaPsModel]


# --- /api/embed Schemas --- (Newer Embeddings Endpoint)


class OllamaEmbedRequest(BaseModel):
    model: str
    input: str | list[str]
    truncate: bool | None = True  # Note: Currently ignored by proxy logic
    options: dict | None = None  # Note: Ignored by proxy
    keep_alive: str | None = None  # Note: Ignored by proxy


class OllamaEmbedResponse(BaseModel):
    model: str
    embeddings: list[list[float]]
    # Ollama includes stats like prompt_eval_count, omitted here


# --- /api/embeddings Schemas --- (Older Embeddings Endpoint)


class OllamaEmbeddingsRequest(BaseModel):
    model: str
    prompt: str  # Equivalent to single input in /api/embed
    options: dict | None = None  # Note: Ignored by proxy
    keep_alive: str | None = None  # Note: Ignored by proxy


class OllamaEmbeddingsResponse(BaseModel):
    embedding: list[float]  # Corresponds to the first embedding in /api/embed response
