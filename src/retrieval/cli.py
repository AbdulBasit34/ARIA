import argparse
import json
from pathlib import Path

from src.retrieval import BM25Index, HybridRetriever, QdrantVectorStore, SentenceTransformerEmbedder
from src.utils.config import load_config
from src.utils.logging import setup_logging


def load_chunks(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError("Input JSON must contain a list at key 'chunks'")
    return [dict(chunk) for chunk in chunks]


def build_retriever(config_path: Path) -> HybridRetriever:
    config = load_config(config_path)
    setup_logging(config)
    return HybridRetriever(
        config=config,
        embedder=SentenceTransformerEmbedder(config),
        vector_store=QdrantVectorStore(config),
        bm25_index=BM25Index(),
    )


def index_chunks(input_path: Path, config_path: Path) -> None:
    retriever = build_retriever(config_path)
    retriever.index_chunks(load_chunks(input_path))


def search(input_path: Path, query: str, config_path: Path) -> None:
    retriever = build_retriever(config_path)
    retriever.index_chunks(load_chunks(input_path))
    results = retriever.retrieve(query)
    print(json.dumps([result.__dict__ for result in results], indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index and search ARIA chunks.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index")
    index_parser.add_argument("--input", type=Path, default=Path("data/ingestion/arxiv_chunks.json"))

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("query", type=str)
    search_parser.add_argument("--input", type=Path, default=Path("data/ingestion/arxiv_chunks.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "index":
        index_chunks(args.input, args.config)
    elif args.command == "search":
        search(args.input, args.query, args.config)


if __name__ == "__main__":
    main()
