from typing import Protocol

import torch
from loguru import logger
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, query: str) -> list[float]:
        ...


class SentenceTransformerEmbedder:
    def __init__(self, config: DictConfig) -> None:
        self.batch_size = min(int(config.embeddings.batch_size), 32)
        self.model_name = str(config.embeddings.model_name)
        self.device = self._resolve_device(str(config.embeddings.device))
        self.local_files_only = bool(config.embeddings.local_files_only)
        logger.info("Loading embedding model {} on {}", self.model_name, self.device)
        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            local_files_only=self.local_files_only,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"
