import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from embeddings.embedding_model import NepaliEmbeddingModel


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_nepali_similarity():
    """Embed a Nepali query and three Nepali documents (two relevant, one irrelevant).
    Verify relevant documents rank higher than the irrelevant one.
    """
    model = NepaliEmbeddingModel()

    query = "नेपालको राजधानी कहाँ हो?"

    doc_relevant_1 = "काठमाडौं नेपालको राजधानी हो। यो बागमती प्रदेशमा पर्दछ।"
    doc_relevant_2 = "नेपालको राजधानी काठमाडौं उपत्यकामा अवस्थित छ। यो शहर ऐतिहासिक र सांस्कृतिक रूपमा महत्त्वपूर्ण छ।"
    doc_irrelevant = "आज मौसम राम्रो छ। बिहानको नाश्तामा चिउरा र दही खाएँ।"

    query_emb = model.embed_query(query)
    doc_embs = model.embed_documents([doc_relevant_1, doc_relevant_2, doc_irrelevant])

    sim_relevant_1 = cosine_similarity(query_emb, doc_embs[0])
    sim_relevant_2 = cosine_similarity(query_emb, doc_embs[1])
    sim_irrelevant = cosine_similarity(query_emb, doc_embs[2])

    print(f"Query: {query}")
    print(f"  Relevant 1 similarity: {sim_relevant_1:.4f}")
    print(f"  Relevant 2 similarity: {sim_relevant_2:.4f}")
    print(f"  Irrelevant similarity:  {sim_irrelevant:.4f}")

    assert sim_relevant_1 > sim_irrelevant, (
        f"Relevant doc 1 ({sim_relevant_1:.4f}) should score higher than irrelevant ({sim_irrelevant:.4f})"
    )
    assert sim_relevant_2 > sim_irrelevant, (
        f"Relevant doc 2 ({sim_relevant_2:.4f}) should score higher than irrelevant ({sim_irrelevant:.4f})"
    )
    print("PASSED: Relevant documents score higher than irrelevant\n")


def test_cross_lingual():
    """Test English query against Nepali documents to verify cross-lingual retrieval."""
    model = NepaliEmbeddingModel()

    query = "What is the capital of Nepal?"

    doc_relevant = "काठमाडौं नेपालको राजधानी हो। यो बागमती प्रदेशमा पर्दछ।"
    doc_irrelevant = "आज मौसम राम्रो छ। बिहानको नाश्तामा चिउरा र दही खाएँ।"

    query_emb = model.embed_query(query)
    doc_embs = model.embed_documents([doc_relevant, doc_irrelevant])

    sim_relevant = cosine_similarity(query_emb, doc_embs[0])
    sim_irrelevant = cosine_similarity(query_emb, doc_embs[1])

    print(f"Query (English): {query}")
    print(f"  Relevant (Nepali) similarity:   {sim_relevant:.4f}")
    print(f"  Irrelevant (Nepali) similarity:  {sim_irrelevant:.4f}")

    assert sim_relevant > sim_irrelevant, (
        f"Cross-lingual relevant ({sim_relevant:.4f}) should score higher than irrelevant ({sim_irrelevant:.4f})"
    )
    print("PASSED: Cross-lingual retrieval works\n")


def test_embedding_shape_and_normalization():
    """Verify embedding dimensions and L2 normalization."""
    model = NepaliEmbeddingModel()

    query_emb = model.embed_query("test query")
    assert query_emb.shape == (model.dimension,), f"Expected ({model.dimension},), got {query_emb.shape}"

    doc_embs = model.embed_documents(["doc one", "doc two"])
    assert doc_embs.shape == (2, model.dimension), f"Expected (2, {model.dimension}), got {doc_embs.shape}"

    norm = np.linalg.norm(query_emb)
    assert abs(norm - 1.0) < 1e-5, f"Query embedding should be L2-normalized, got norm={norm:.6f}"

    for i, emb in enumerate(doc_embs):
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-5, f"Doc {i} embedding should be L2-normalized, got norm={norm:.6f}"

    print(f"Embedding dimension: {model.dimension}")
    print(f"Query embedding norm: {np.linalg.norm(query_emb):.6f}")
    print("PASSED: Shape and normalization correct\n")


if __name__ == "__main__":
    test_embedding_shape_and_normalization()
    test_nepali_similarity()
    test_cross_lingual()
    print("All embedding tests passed")
