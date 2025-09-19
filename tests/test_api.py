from fastapi.testclient import TestClient

from babyrag.api import create_app
from babyrag.settings import Settings


def build_test_app() -> TestClient:
    settings = Settings(
        embedding_model_name="hf-internal-testing/tiny-random-bert",
        completion_model_name="sshleifer/tiny-gpt2",
        max_new_tokens=8,
        top_k=1,
        temperature=0.1,
    )
    app = create_app(settings)
    return TestClient(app)


def test_ingest_and_embedding_endpoint():
    client = build_test_app()
    ingest_response = client.post("/ingest", json={"documents": ["BabyRAG is tiny for tests."]})
    assert ingest_response.status_code == 200
    assert ingest_response.json()["ingested"] == 1

    emb_response = client.post("/embeddings", json={"input": "BabyRAG"})
    body = emb_response.json()
    assert emb_response.status_code == 200
    assert body["model"] == "hf-internal-testing/tiny-random-bert"
    assert len(body["data"]) == 1
    assert len(body["data"][0]["embedding"]) > 0


def test_completions_endpoint_returns_text():
    client = build_test_app()
    client.post("/ingest", json={"documents": ["BabyRAG answers questions about itself."]})
    response = client.post("/completions", json={"prompt": "What does BabyRAG do?"})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "sshleifer/tiny-gpt2"
    assert "completion" in body
    assert isinstance(body["completion"], str)
    assert len(body["contexts"]) == 1
