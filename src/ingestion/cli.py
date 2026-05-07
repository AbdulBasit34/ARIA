import argparse
import json
from pathlib import Path

from src.ingestion.arxiv_client import papers_to_records, search_papers
from src.ingestion.chunker import chunk_paper, chunks_to_records
from src.utils.config import load_config
from src.utils.logging import setup_logging


def run_search(question: str, output_path: Path, config_path: Path) -> None:
    config = load_config(config_path)
    setup_logging(config)

    papers = search_papers(question, config)
    chunks = [chunk for paper in papers for chunk in chunk_paper(paper, config)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {"papers": papers_to_records(papers), "chunks": chunks_to_records(chunks)},
            indent=2,
        ),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch and chunk ArXiv papers.")
    parser.add_argument("question", type=str)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/ingestion/arxiv_chunks.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_search(args.question, args.output, args.config)


if __name__ == "__main__":
    main()
