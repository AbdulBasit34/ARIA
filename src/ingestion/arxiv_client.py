from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import arxiv
from loguru import logger
from omegaconf import DictConfig


@dataclass(frozen=True)
class ArxivPaper:
    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    published: datetime
    updated: datetime
    pdf_url: str
    entry_url: str
    categories: list[str]


def _sort_by(value: str) -> arxiv.SortCriterion:
    mapping = {
        "relevance": arxiv.SortCriterion.Relevance,
        "last_updated": arxiv.SortCriterion.LastUpdatedDate,
        "submitted": arxiv.SortCriterion.SubmittedDate,
    }
    return mapping.get(value, arxiv.SortCriterion.Relevance)


def _sort_order(value: str) -> arxiv.SortOrder:
    mapping = {
        "ascending": arxiv.SortOrder.Ascending,
        "descending": arxiv.SortOrder.Descending,
    }
    return mapping.get(value, arxiv.SortOrder.Descending)


def _paper_from_result(result: arxiv.Result) -> ArxivPaper:
    paper_id = result.get_short_id()
    return ArxivPaper(
        paper_id=paper_id,
        title=result.title.strip(),
        abstract=result.summary.strip(),
        authors=[author.name for author in result.authors],
        published=result.published,
        updated=result.updated,
        pdf_url=result.pdf_url,
        entry_url=result.entry_id,
        categories=list(result.categories),
    )


def search_papers(question: str, config: DictConfig) -> list[ArxivPaper]:
    client = arxiv.Client(
        page_size=int(config.arxiv.max_results),
        delay_seconds=float(config.arxiv.request_delay_seconds),
        num_retries=3,
    )
    search = arxiv.Search(
        query=question,
        max_results=int(config.arxiv.max_results),
        sort_by=_sort_by(str(config.arxiv.sort_by)),
        sort_order=_sort_order(str(config.arxiv.sort_order)),
    )

    papers = [_paper_from_result(result) for result in client.results(search)]
    logger.info("Fetched {} ArXiv papers for query: {}", len(papers), question)
    return papers


def papers_to_records(papers: Iterable[ArxivPaper]) -> list[dict[str, object]]:
    return [
        {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "abstract": paper.abstract,
            "authors": paper.authors,
            "published": paper.published.isoformat(),
            "updated": paper.updated.isoformat(),
            "pdf_url": paper.pdf_url,
            "entry_url": paper.entry_url,
            "categories": paper.categories,
        }
        for paper in papers
    ]
