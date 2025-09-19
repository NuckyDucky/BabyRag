"""Configuration helpers for the BabyRAG service."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


def _detect_device(explicit: Optional[str] = None) -> str:
    """Determine which device the service should use."""
    if explicit:
        return explicit
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Settings:
    """Runtime configuration for the BabyRAG service."""

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    completion_model_name: str = "distilgpt2"
    device: Optional[str] = None
    max_new_tokens: int = 256
    top_k: int = 3
    temperature: float = 0.7

    # FastAPI/uvicorn settings
    host: str = "0.0.0.0"
    port: int = 8000

    def resolved_device(self) -> str:
        """Resolve the device to use for tensor operations."""
        return _detect_device(self.device)


DEFAULT_SETTINGS = Settings()
