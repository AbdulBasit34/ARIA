from uuid import NAMESPACE_URL, uuid5

from loguru import logger
from omegaconf import DictConfig
from qdrant_client import QdrantClient
from qdrant_client.http import models


class QdrantVectorStore:
    def __init__(self, config: DictConfig) -> None:
        self.collection_name = str(config.qdrant.collection_name)
        self.vector_size = int(config.qdrant.vector_size)
        self.distance = _distance(str(config.qdrant.distance))
        self.client = _build_client(config)

    def ensure_collection(self) -> None:
        collections = self.client.get_collections().collections
        exists = any(collection.name == self.collection_name for collection in collections)
        if exists:
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.vector_size, distance=self.distance),
        )
        logger.info("Created Qdrant collection {}", self.collection_name)

    def upsert_chunks(
        self,
        chunk_ids: list[str],
        texts: list[str],
        payloads: list[dict[str, object]],
        vectors: list[list[float]],
    ) -> None:
        if not (len(chunk_ids) == len(texts) == len(payloads) == len(vectors)):
            raise ValueError("chunk_ids, texts, payloads, and vectors must have equal length")

        self.ensure_collection()
        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, chunk_id)),
                vector=vector,
                payload={"chunk_id": chunk_id, "text": text, **payload},
            )
            for chunk_id, text, payload, vector in zip(chunk_ids, texts, payloads, vectors)
        ]
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info("Upserted {} chunks into Qdrant", len(points))

    def search(self, query_vector: list[float], limit: int) -> list[tuple[str, float, dict[str, object]]]:
        results = self._search_points(query_vector, limit)
        output: list[tuple[str, float, dict[str, object]]] = []
        for result in results:
            payload = dict(result.payload or {})
            chunk_id = str(payload.get("chunk_id", result.id))
            output.append((chunk_id, float(result.score), payload))
        return output

    def _search_points(self, query_vector: list[float], limit: int) -> list[models.ScoredPoint]:
        if hasattr(self.client, "search"):
            return self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        return list(response.points)


def _distance(value: str) -> models.Distance:
    mapping = {
        "cosine": models.Distance.COSINE,
        "dot": models.Distance.DOT,
        "euclid": models.Distance.EUCLID,
    }
    return mapping.get(value.lower(), models.Distance.COSINE)


def _build_client(config: DictConfig) -> QdrantClient:
    mode = str(config.qdrant.mode).lower()
    if mode == "local":
        return QdrantClient(path=str(config.qdrant.path))
    if mode == "memory":
        return QdrantClient(location=":memory:")
    return QdrantClient(url=str(config.qdrant.url))
