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


class ReportSource(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    published: str
    pdf_url: str
    score: float


class ResearchReport(BaseModel):
    question: str
    answer: str
    key_findings: list[str]
    limitations: list[str]
    follow_up_questions: list[str]
    sources: list[ReportSource]
