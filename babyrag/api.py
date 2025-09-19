"""FastAPI application exposing the BabyRAG pipeline."""
from __future__ import annotations

from typing import List, Sequence

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .embeddings import EmbeddingModel
from .generator import CompletionModel
from .rag import RAGPipeline
from .settings import Settings
from .vector_store import VectorStore


class EmbeddingRequest(BaseModel):
    input: Sequence[str] | str


class EmbeddingResponseDatum(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    model: str
    data: List[EmbeddingResponseDatum]


class IngestRequest(BaseModel):
    documents: Sequence[str] = Field(default_factory=list)


class CompletionRequest(BaseModel):
    prompt: str
    top_k: int | None = None
    max_new_tokens: int | None = None
    temperature: float | None = None


class CompletionContext(BaseModel):
    text: str
    score: float


class CompletionResponse(BaseModel):
    model: str
    completion: str
    contexts: List[CompletionContext]


def _ensure_sequence(value: Sequence[str] | str) -> List[str]:
    if isinstance(value, str):
        return [value]
    return list(value)


def create_app(settings: Settings) -> FastAPI:
    embedding_model = EmbeddingModel(settings.embedding_model_name, device=settings.resolved_device())
    completion_model = CompletionModel(settings.completion_model_name, device=settings.resolved_device())
    vector_store = VectorStore()
    pipeline = RAGPipeline(embedding_model, vector_store, completion_model)

    app = FastAPI(title="BabyRAG", version="2.0.0")

    @app.post("/ingest", response_model=dict)
    def ingest(request: IngestRequest) -> dict:
        documents = [doc for doc in request.documents if doc]
        pipeline.ingest(documents)
        return {"ingested": len(documents)}

    @app.post("/embeddings", response_model=EmbeddingResponse)
    def embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
        texts = _ensure_sequence(request.input)
        vectors = pipeline.embedding_model.embed(texts)
        data = [
            EmbeddingResponseDatum(embedding=vector, index=index)
            for index, vector in enumerate(vectors)
        ]
        return EmbeddingResponse(model=pipeline.embedding_model.model_name, data=data)

    @app.post("/completions", response_model=CompletionResponse)
    def completions(request: CompletionRequest) -> CompletionResponse:
        top_k = request.top_k or settings.top_k
        max_new_tokens = request.max_new_tokens or settings.max_new_tokens
        temperature = request.temperature or settings.temperature

        if top_k <= 0:
            raise HTTPException(status_code=400, detail="top_k must be positive")
        if max_new_tokens <= 0:
            raise HTTPException(status_code=400, detail="max_new_tokens must be positive")

        answer = pipeline.generate(
            request.prompt,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        contexts = [CompletionContext(text=ctx.text, score=ctx.score) for ctx in answer.contexts]
        return CompletionResponse(
            model=pipeline.completion_model.model_name,
            completion=answer.text,
            contexts=contexts,
        )

    return app
