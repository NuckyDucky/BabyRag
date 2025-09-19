import pathlib
import sys
from typing import Any

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _FakeEncoder:
    def to(self, device: str) -> "_FakeEncoder":
        return self

    def eval(self) -> "_FakeEncoder":
        return self

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_: Any):
        batch, seq_len = input_ids.shape
        hidden_size = 8
        base = torch.arange(hidden_size, dtype=torch.float32)
        expanded = base.view(1, 1, hidden_size).expand(batch, seq_len, hidden_size)
        last_hidden_state = expanded + input_ids.unsqueeze(-1).float()

        class _Output:
            def __init__(self, hidden: torch.Tensor) -> None:
                self.last_hidden_state = hidden
                self.pooler_output = None

        return _Output(last_hidden_state)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True):
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            texts = [""]
        lengths = [max(len(text), 1) for text in texts]
        max_len = max(lengths)
        input_ids = torch.zeros((len(texts), max_len), dtype=torch.long)
        attention_mask = torch.zeros((len(texts), max_len), dtype=torch.long)
        for row, text in enumerate(texts):
            token_values = [ord(ch) % 13 + 1 for ch in text] or [1]
            length = len(token_values)
            input_ids[row, :length] = torch.tensor(token_values, dtype=torch.long)
            attention_mask[row, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, tokens, skip_special_tokens: bool = True):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return " ".join(str(token) for token in tokens)

    def to(self, device: str) -> "_FakeTokenizer":
        return self


class _FakeGenerator:
    def to(self, device: str) -> "_FakeGenerator":
        return self

    def eval(self) -> "_FakeGenerator":
        return self

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, **_: Any) -> torch.Tensor:
        batch, _ = input_ids.shape
        if max_new_tokens <= 0:
            return input_ids
        additions = torch.arange(1, max_new_tokens + 1, dtype=torch.long).view(1, -1)
        additions = additions.expand(batch, -1)
        return torch.cat([input_ids, additions], dim=1)


@pytest.fixture(autouse=True)
def _mock_transformers(monkeypatch):
    tokenizer = _FakeTokenizer()
    encoder = _FakeEncoder()
    generator = _FakeGenerator()

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *_args, **_kwargs: tokenizer)
    monkeypatch.setattr("transformers.AutoModel.from_pretrained", lambda *_args, **_kwargs: encoder)
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", lambda *_args, **_kwargs: generator
    )

    yield
