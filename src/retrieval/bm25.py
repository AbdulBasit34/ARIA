import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class BM25Document:
    chunk_id: str
    text: str
    payload: dict[str, object]


class BM25Index:
    def __init__(self) -> None:
        self.documents: list[BM25Document] = []
        self.index: BM25Okapi | None = None

    def build(self, documents: list[BM25Document]) -> None:
        self.documents = documents
        tokenized = [tokenize(document.text) for document in documents]
        self.index = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        if self.index is None or not self.documents:
            return []
        query_tokens = tokenize(query)
        query_token_set = set(query_tokens)
        scores = self.index.get_scores(query_tokens)
        ranked = sorted(
            enumerate(scores),
            key=lambda item: (
                float(item[1]),
                _overlap_count(query_token_set, self.documents[item[0]].text),
            ),
            reverse=True,
        )
        return [
            (self.documents[index].chunk_id, float(score))
            for index, score in ranked[:limit]
        ]

    def get_payload(self, chunk_id: str) -> dict[str, object] | None:
        for document in self.documents:
            if document.chunk_id == chunk_id:
                return document.payload
        return None


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _overlap_count(query_tokens: set[str], document_text: str) -> int:
    return len(query_tokens.intersection(tokenize(document_text)))
