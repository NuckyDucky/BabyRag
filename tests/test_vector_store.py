from babyrag.vector_store import VectorStore


def test_vector_store_retrieval_orders_by_similarity():
    store = VectorStore()
    store.add(["a", "b"], [[1.0, 0.0], [0.0, 1.0]])
    results = store.search([0.8, 0.2], top_k=2)
    assert [doc.text for doc in results] == ["a", "b"]
    assert results[0].score > results[1].score


def test_vector_store_handles_empty_state():
    store = VectorStore()
    assert store.search([1.0, 0.0], top_k=1) == []
