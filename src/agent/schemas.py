from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    model: str
    prompt: str
    response: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    raw: dict[str, object] = Field(default_factory=dict)
