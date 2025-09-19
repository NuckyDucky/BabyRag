"""A lightweight in-memory vector store."""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import List, Sequence

import torch


@dataclass
class RetrievedDocument:
    text: str
    score: float


class VectorStore:
    """Store embeddings and perform cosine-similarity retrieval."""

    def __init__(self) -> None:
        self._documents: list[str] = []
        self._embeddings: list[torch.Tensor] = []
        self._lock = Lock()

    def add(self, texts: Sequence[str], embeddings: Sequence[Sequence[float]]) -> None:
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts and embeddings must match")
        with self._lock:
            for text, embedding in zip(texts, embeddings):
                self._documents.append(text)
                self._embeddings.append(torch.tensor(embedding, dtype=torch.float32))

    def clear(self) -> None:
        with self._lock:
            self._documents.clear()
            self._embeddings.clear()

    def search(self, query_embedding: Sequence[float], top_k: int = 3) -> List[RetrievedDocument]:
        query = torch.tensor(query_embedding, dtype=torch.float32)
        with self._lock:
            if not self._embeddings:
                return []
            matrix = torch.stack(self._embeddings)
            documents = list(self._documents)

        similarities = torch.nn.functional.cosine_similarity(matrix, query.unsqueeze(0), dim=1)
        top_k = min(top_k, similarities.size(0))
        scores, indices = torch.topk(similarities, top_k)
        results = [
            RetrievedDocument(text=documents[idx], score=scores[i].item())
            for i, idx in enumerate(indices.tolist())
        ]
        return results
