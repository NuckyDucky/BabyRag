from babyrag.embeddings import EmbeddingModel


def test_embedding_model_returns_vectors():
    model = EmbeddingModel("hf-internal-testing/tiny-random-bert")
    vectors = model.embed(["hello world", "goodbye"])
    assert len(vectors) == 2
    assert all(isinstance(value, float) for value in vectors[0])
    assert len(vectors[0]) > 0
