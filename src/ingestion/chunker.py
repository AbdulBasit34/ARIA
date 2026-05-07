from dataclasses import dataclass
from typing import Iterable, Sequence

from omegaconf import DictConfig

from src.ingestion.arxiv_client import ArxivPaper


@dataclass(frozen=True)
class PaperChunk:
    chunk_id: str
    paper_id: str
    title: str
    text: str
    metadata: dict[str, object]


def tokenize_text(text: str) -> list[str]:
    return text.split()


def detokenize_text(tokens: Sequence[str]) -> str:
    return " ".join(tokens).strip()


def chunk_text(text: str, chunk_size_tokens: int, chunk_overlap_tokens: int) -> list[str]:
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive")
    if chunk_overlap_tokens < 0:
        raise ValueError("chunk_overlap_tokens cannot be negative")
    if chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

    tokens = tokenize_text(text)
    if not tokens:
        return []

    chunks: list[str] = []
    step = chunk_size_tokens - chunk_overlap_tokens
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size_tokens]
        if not window:
            break
        chunks.append(detokenize_text(window))
        if start + chunk_size_tokens >= len(tokens):
            break
    return chunks


def chunk_paper(paper: ArxivPaper, config: DictConfig) -> list[PaperChunk]:
    source_text = f"{paper.title}\n\n{paper.abstract}"
    chunks = chunk_text(
        source_text,
        chunk_size_tokens=int(config.chunking.chunk_size_tokens),
        chunk_overlap_tokens=int(config.chunking.chunk_overlap_tokens),
    )

    return [
        PaperChunk(
            chunk_id=f"{paper.paper_id}:{index}",
            paper_id=paper.paper_id,
            title=paper.title,
            text=chunk,
            metadata={
                "paper_id": paper.paper_id,
                "title": paper.title,
                "authors": paper.authors,
                "published": paper.published.isoformat(),
                "updated": paper.updated.isoformat(),
                "pdf_url": paper.pdf_url,
                "entry_url": paper.entry_url,
                "categories": paper.categories,
                "chunk_index": index,
            },
        )
        for index, chunk in enumerate(chunks)
    ]


def chunks_to_records(chunks: Iterable[PaperChunk]) -> list[dict[str, object]]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "paper_id": chunk.paper_id,
            "title": chunk.title,
            "text": chunk.text,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]
