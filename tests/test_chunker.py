from datetime import UTC, datetime

from omegaconf import OmegaConf

from src.ingestion.arxiv_client import ArxivPaper
from src.ingestion.chunker import chunk_paper, chunk_text


def test_chunk_text_uses_overlap() -> None:
    chunks = chunk_text("one two three four five six", 4, 2)
    assert chunks == ["one two three four", "three four five six"]


def test_chunk_paper_adds_metadata() -> None:
    config = OmegaConf.create({"chunking": {"chunk_size_tokens": 8, "chunk_overlap_tokens": 2}})
    paper = ArxivPaper(
        paper_id="1234.5678",
        title="Retrieval Augmented Generation",
        abstract="Dense retrieval and sparse retrieval can improve research question answering.",
        authors=["A. Researcher"],
        published=datetime(2024, 1, 1, tzinfo=UTC),
        updated=datetime(2024, 1, 2, tzinfo=UTC),
        pdf_url="https://arxiv.org/pdf/1234.5678",
        entry_url="https://arxiv.org/abs/1234.5678",
        categories=["cs.CL"],
    )

    chunks = chunk_paper(paper, config)

    assert chunks
    assert chunks[0].paper_id == "1234.5678"
    assert chunks[0].metadata["title"] == "Retrieval Augmented Generation"
