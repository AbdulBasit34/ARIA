from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from src.agent.schemas import ResearchReport, ReportSource
from src.serving.app import create_app


class FakeWorkflow:
    def run(self, question: str) -> ResearchReport:
        return ResearchReport(
            question=question,
            answer="Hybrid retrieval combines dense and sparse search evidence.",
            key_findings=["RRF fuses complementary ranked lists."],
            limitations=["This is a test response."],
            follow_up_questions=["How should retrieval depth be tuned?"],
            sources=[
                ReportSource(
                    paper_id="1234.5678",
                    title="Hybrid Retrieval for QA",
                    authors=["A. Author"],
                    published="2025-01-01T00:00:00+00:00",
                    pdf_url="https://example.com/1234.5678.pdf",
                    score=0.5,
                )
            ],
        )


def test_health_endpoint() -> None:
    app = create_app(
        config=OmegaConf.create({"logging": make_logging_config()}),
        workflow_factory=lambda config: FakeWorkflow(),
    )
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_research_endpoint_returns_report() -> None:
    app = create_app(
        config=OmegaConf.create({"logging": make_logging_config()}),
        workflow_factory=lambda config: FakeWorkflow(),
    )
    client = TestClient(app)

    response = client.post("/research", json={"question": "How does hybrid retrieval help?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["report"]["question"] == "How does hybrid retrieval help?"
    assert payload["report"]["sources"][0]["paper_id"] == "1234.5678"


def test_research_endpoint_validates_question() -> None:
    app = create_app(
        config=OmegaConf.create({"logging": make_logging_config()}),
        workflow_factory=lambda config: FakeWorkflow(),
    )
    client = TestClient(app)

    response = client.post("/research", json={"question": ""})

    assert response.status_code == 422


def make_logging_config() -> dict[str, str]:
    return {
        "level": "INFO",
        "app_log_path": ".test_tmp/serving.log",
        "llm_log_path": ".test_tmp/llm_calls.jsonl",
        "rotation": "10 MB",
        "retention": "1 day",
    }
