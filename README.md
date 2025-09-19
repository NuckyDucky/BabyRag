# BabyRAG

BabyRAG is now a lightweight Retrieval Augmented Generation service you can host locally. It exposes HTTP endpoints for
embedding generation, document ingestion, and completions that ground language-model responses on the ingested corpus.

The service is framework agnostic: launch it with `python -m babyrag` and interact with it over plain HTTP. No Gradio UI,
no ONNX runtime – just a minimal stack built on top of PyTorch, Hugging Face Transformers, and FastAPI.

## Features

- **Embeddings on demand** – `POST /embeddings` to transform raw text into normalized vector representations using any
  Hugging Face encoder model.
- **In-memory vector store** – ingest documents at runtime and retrieve the most relevant passages with cosine similarity.
- **RAG completions** – `POST /completions` to blend retrieved context with a causal language model and stream-ready
  completions.
- **Configurable runtime** – choose the embedding model, completion model, device, and generation parameters via CLI flags
  or by wiring up your own `Settings` object.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run will download the default Hugging Face weights (`sentence-transformers/all-MiniLM-L6-v2` for embeddings and
`distilgpt2` for generation). Point the CLI at a local directory if you already have the weights.

## Running the server

```bash
python -m babyrag --host 0.0.0.0 --port 8000 \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --completion-model distilgpt2
```

The server starts a FastAPI app that listens on `/ingest`, `/embeddings`, and `/completions`.

### Ingest documents

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents": ["BabyRAG focuses on local-first RAG workflows.", "It exposes embeddings and completion APIs."]}'
```

### Generate embeddings

```bash
curl -X POST http://localhost:8000/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "What is BabyRAG?"}'
```

### Request a completion

```bash
curl -X POST http://localhost:8000/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain BabyRAG"}'
```

The pipeline retrieves relevant context from the ingested documents and crafts a prompt for the language model to produce
an answer. Tweak `top_k`, `max_new_tokens`, or `temperature` in the request body for finer control.

## Programmatic usage

If you want to embed text or compose a RAG flow inside another project, you can import the building blocks directly:

```python
from babyrag import EmbeddingModel, VectorStore, CompletionModel, RAGPipeline, Settings

settings = Settings()
embeddings = EmbeddingModel(settings.embedding_model_name)
completions = CompletionModel(settings.completion_model_name)
store = VectorStore()

pipeline = RAGPipeline(embeddings, store, completions)
pipeline.ingest(["BabyRAG answers questions about its ingested context."])
answer = pipeline.generate("What does BabyRAG do?", top_k=1, max_new_tokens=64, temperature=0.7)
print(answer.text)
```

## Development

Run the unit tests before committing changes:

```bash
pytest
```

Feel free to replace the default models with local checkpoints optimized for your hardware.
