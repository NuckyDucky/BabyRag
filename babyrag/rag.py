"""Core retrieval augmented generation pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .embeddings import EmbeddingModel
from .generator import CompletionModel
from .vector_store import RetrievedDocument, VectorStore


@dataclass
class GeneratedAnswer:
    text: str
    contexts: List[RetrievedDocument]


class RAGPipeline:
    """A minimal RAG pipeline that couples embedding, retrieval, and generation."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        completion_model: CompletionModel,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.completion_model = completion_model

    def ingest(self, documents: Iterable[str]) -> None:
        texts = [doc for doc in documents if doc]
        if not texts:
            return
        embeddings = self.embedding_model.embed(texts)
        self.vector_store.add(texts, embeddings)

    def retrieve(self, query: str, *, top_k: int) -> List[RetrievedDocument]:
        query_embedding = self.embedding_model.embed([query])[0]
        return self.vector_store.search(query_embedding, top_k=top_k)

    def generate(
        self,
        prompt: str,
        *,
        top_k: int,
        max_new_tokens: int,
        temperature: float,
    ) -> GeneratedAnswer:
        contexts = self.retrieve(prompt, top_k=top_k)
        if contexts:
            context_block = "\n\n".join(f"- {ctx.text}" for ctx in contexts)
        else:
            context_block = "No supporting context available."
        augmented_prompt = (
            "You are a helpful assistant that answers questions using the provided context.\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {prompt}\nAnswer:"
        )
        completion = self.completion_model.generate(
            augmented_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return GeneratedAnswer(text=completion, contexts=contexts)
