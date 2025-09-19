"""Embedding utilities for BabyRAG."""
from __future__ import annotations

from typing import Iterable, List

import torch
from transformers import AutoModel, AutoTokenizer


class EmbeddingModel:
    """Generate normalized embeddings for arbitrary text."""

    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _mean_pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        masked_embeddings = token_embeddings * mask
        summed = masked_embeddings.sum(dim=1)
        counts = mask.sum(dim=1)
        counts = torch.clamp(counts, min=1e-9)
        return summed / counts

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        """Compute embeddings for the provided texts."""
        batch = list(texts)
        if not batch:
            return []

        encoded = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                embeddings = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().tolist()
