"""Console entry-point for running the BabyRAG API server."""
from __future__ import annotations

import argparse

import uvicorn

from .api import create_app
from .settings import Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BabyRAG API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name or path",
    )
    parser.add_argument(
        "--completion-model",
        default="distilgpt2",
        help="Causal language model name or path",
    )
    parser.add_argument("--device", default=None, help="Torch device to use (defaults to auto)")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Completion length")
    parser.add_argument("--top-k", type=int, default=3, help="Number of documents to retrieve")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings(
        embedding_model_name=args.embedding_model,
        completion_model_name=args.completion_model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        temperature=args.temperature,
        host=args.host,
        port=args.port,
    )
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
