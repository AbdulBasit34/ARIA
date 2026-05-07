from datetime import UTC, datetime
import json
from pathlib import Path

from omegaconf import OmegaConf

from src.agent.reporting import build_report_from_llm, parse_report_json
from src.agent.schemas import LLMResponse
from src.agent.workflow import ResearchWorkflow, load_cached_chunks
from src.retrieval.hybrid import RetrievalResult


class FakeRetriever:
    def __init__(self) -> None:
        self.indexed_chunks: list[dict[str, object]] = []

    def index_chunks(self, chunks: list[dict[str, object]]) -> None:
        self.indexed_chunks = chunks

    def retrieve(self, query: str) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk_id="paper:0",
                score=0.5,
                text=f"Hybrid retrieval helps answer: {query}",
                metadata={
                    "paper_id": "paper",
                    "title": "Hybrid Retrieval for QA",
                    "authors": ["A. Author"],
                    "published": "2025-01-01T00:00:00+00:00",
                    "pdf_url": "https://example.com/paper.pdf",
                },
            )
        ]


class FakeLLM:
    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        return LLMResponse(
            model="fake",
            prompt=prompt,
            response='{"answer":"Hybrid retrieval combines dense and sparse evidence.","key_findings":["Dense and sparse retrieval complement each other."],"limitations":["Only one source was retrieved."],"follow_up_questions":["How should RRF weights be tuned?"]}',
            latency_ms=1.0,
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
        )


def test_parse_report_json_handles_plain_json() -> None:
    parsed = parse_report_json('{"answer":"ok"}')

    assert parsed == {"answer": "ok"}


def test_build_report_from_llm_includes_sources() -> None:
    context = RetrievalResult(
        chunk_id="123:0",
        score=0.25,
        text="Evidence text",
        metadata={
            "paper_id": "123",
            "title": "Evidence Paper",
            "authors": ["Researcher"],
            "published": "2024-01-01T00:00:00+00:00",
            "pdf_url": "https://example.com/123.pdf",
        },
    )

    report = build_report_from_llm(
        "Question?",
        '{"answer":"Answer.","key_findings":["Finding."],"limitations":["Limit."],"follow_up_questions":["Next?"]}',
        [context],
    )

    assert report.answer == "Answer."
    assert report.sources[0].paper_id == "123"


def test_workflow_runs_with_injected_dependencies(monkeypatch) -> None:
    config = OmegaConf.create(
        {
            "arxiv": {"use_cache_fallback": False},
            "report": {"max_context_chunks": 3, "max_context_chars": 1000},
            "chunking": {"chunk_size_tokens": 100, "chunk_overlap_tokens": 10},
        }
    )

    class FakePaper:
        paper_id = "paper"
        title = "Hybrid Retrieval for QA"
        abstract = "Dense and sparse retrieval improve grounded answers."
        authors = ["A. Author"]
        published = datetime(2025, 1, 1, tzinfo=UTC)
        updated = datetime(2025, 1, 1, tzinfo=UTC)
        pdf_url = "https://example.com/paper.pdf"
        entry_url = "https://example.com/paper"
        categories = ["cs.IR"]

    monkeypatch.setattr("src.agent.workflow.search_papers", lambda question, cfg: [FakePaper()])

    workflow = ResearchWorkflow(config, retriever=FakeRetriever(), llm_client=FakeLLM())
    report = workflow.run("How does hybrid retrieval help?")

    assert report.question == "How does hybrid retrieval help?"
    assert report.key_findings
    assert report.sources[0].title == "Hybrid Retrieval for QA"


def test_load_cached_chunks_reads_ingestion_cache() -> None:
    cache_path = Path(".test_tmp") / "arxiv_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"chunks": [{"chunk_id": "a:0", "text": "cached evidence", "metadata": {}}]}),
        encoding="utf-8",
    )

    chunks = load_cached_chunks(cache_path)

    assert chunks[0]["chunk_id"] == "a:0"
