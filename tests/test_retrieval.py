from omegaconf import OmegaConf

from src.retrieval.bm25 import BM25Document, BM25Index, tokenize
from src.retrieval.rrf import reciprocal_rank_fusion


def test_tokenize_normalizes_text() -> None:
    assert tokenize("RAG, BM25 + Dense!") == ["rag", "bm25", "dense"]


def test_bm25_returns_matching_document() -> None:
    index = BM25Index()
    index.build(
        [
            BM25Document("a", "graph neural networks for molecules", {"text": "a"}),
            BM25Document("b", "language models for retrieval", {"text": "b"}),
        ]
    )

    results = index.search("retrieval language", limit=1)

    assert results[0][0] == "b"


def test_rrf_fuses_ranked_lists() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], rrf_k=60, limit=3)

    assert fused[0][0] == "b"
    assert {item_id for item_id, _ in fused} == {"a", "b", "c"}


def test_retrieval_config_keeps_embedding_batch_memory_safe() -> None:
    config = OmegaConf.load("configs/config.yaml")

    assert int(config.embeddings.batch_size) <= 32
