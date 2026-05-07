from pydantic import BaseModel, Field

from src.agent.schemas import ResearchReport


class ResearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class ResearchResponse(BaseModel):
    report: ResearchReport
