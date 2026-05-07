from dataclasses import dataclass
from typing import Protocol

from omegaconf import DictConfig

from src.retrieval.bm25 import BM25Document, BM25Index
from src.retrieval.qdrant_store import QdrantVectorStore
from src.retrieval.rrf import reciprocal_rank_fusion


class Embedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, query: str) -> list[float]:
        ...


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, object]


class HybridRetriever:
    def __init__(
        self,
        config: DictConfig,
        embedder: Embedder,
        vector_store: QdrantVectorStore,
        bm25_index: BM25Index,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    def index_chunks(self, chunks: list[dict[str, object]]) -> None:
        chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]
        texts = [str(chunk["text"]) for chunk in chunks]
        payloads = [dict(chunk["metadata"]) for chunk in chunks]
        vectors = self.embedder.embed_texts(texts)

        self.vector_store.upsert_chunks(chunk_ids, texts, payloads, vectors)
        self.bm25_index.build(
            [
                BM25Document(
                    chunk_id=chunk_id,
                    text=text,
                    payload={"chunk_id": chunk_id, "text": text, **payload},
                )
                for chunk_id, text, payload in zip(chunk_ids, texts, payloads)
            ]
        )

    def retrieve(self, query: str) -> list[RetrievalResult]:
        query_vector = self.embedder.embed_query(query)
        dense_hits = self.vector_store.search(query_vector, int(self.config.retrieval.dense_limit))
        bm25_hits = self.bm25_index.search(query, int(self.config.retrieval.bm25_limit))

        dense_ids = [chunk_id for chunk_id, _, _ in dense_hits]
        bm25_ids = [chunk_id for chunk_id, _ in bm25_hits]
        fused = reciprocal_rank_fusion(
            [dense_ids, bm25_ids],
            rrf_k=int(self.config.retrieval.rrf_k),
            limit=int(self.config.retrieval.final_limit),
        )

        payloads = {chunk_id: payload for chunk_id, _, payload in dense_hits}
        for chunk_id, _ in bm25_hits:
            payload = self.bm25_index.get_payload(chunk_id)
            if payload is not None:
                payloads.setdefault(chunk_id, payload)

        return [
            RetrievalResult(
                chunk_id=chunk_id,
                score=score,
                text=str(payloads.get(chunk_id, {}).get("text", "")),
                metadata={
                    key: value
                    for key, value in payloads.get(chunk_id, {}).items()
                    if key not in {"text"}
                },
            )
            for chunk_id, score in fused
        ]
