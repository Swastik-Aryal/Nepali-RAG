import sys
sys.stdout.reconfigure(encoding="utf-8")

import time
from utils.embeddings.embedding_model import NepaliEmbeddingModel
from utils.vectorstore.qdrant_store import QdrantStore
from utils.retrieval.reranker import Reranker
from utils.retrieval.retriever import Retriever

TEST_COLLECTION = "test_nepali_retrieval"


def run_test():
    print("Loading embedding model")
    embedding_model = NepaliEmbeddingModel()

    chunks = [
        {
            "text": "नेपाल दक्षिण एसियाको एक भूपरिवेष्टित देश हो। यसको उत्तरमा चीन र दक्षिणमा भारत छ।",
            "metadata": {"source": "geography.pdf", "page_number": 1, "chunk_index": 0},
        },
        {
            "text": "काठमाडौं नेपालको राजधानी हो। यो काठमाडौं उपत्यकामा अवस्थित छ। यो नेपालको सबैभन्दा ठूलो शहर पनि हो।",
            "metadata": {"source": "geography.pdf", "page_number": 1, "chunk_index": 1},
        },
        {
            "text": "सगरमाथा संसारको सबैभन्दा अग्लो हिमाल हो। यसको उचाइ ८,८४८.८६ मिटर छ। यो नेपाल र चीनको सीमामा पर्छ।",
            "metadata": {"source": "geography.pdf", "page_number": 2, "chunk_index": 2},
        },
        {
            "text": "नेपालको जनसंख्या करिब तीन करोड छ। नेपालमा विभिन्न जातजाति र भाषाभाषी मानिसहरू बसोबास गर्छन्।",
            "metadata": {"source": "demographics.pdf", "page_number": 1, "chunk_index": 3},
        },
        {
            "text": "पोखरा नेपालको पर्यटकीय शहर हो। फेवा ताल पोखराको मुख्य आकर्षण हो।",
            "metadata": {"source": "tourism.pdf", "page_number": 1, "chunk_index": 4},
        },
        {
            "text": "लुम्बिनी गौतम बुद्धको जन्मस्थल हो। यो नेपालको रूपन्देही जिल्लामा पर्छ। यो युनेस्को विश्व सम्पदा सूचीमा सूचीकृत छ।",
            "metadata": {"source": "culture.pdf", "page_number": 1, "chunk_index": 5},
        },
        {
            "text": "Python is a popular programming language used for data science and machine learning applications worldwide.",
            "metadata": {"source": "tech.pdf", "page_number": 1, "chunk_index": 6},
        },
    ]

    print(f"Computing embeddings for {len(chunks)} chunks")
    texts = [c["text"] for c in chunks]
    embeddings = embedding_model.embed_documents(texts)

    print(f"Connecting to Qdrant, collection: {TEST_COLLECTION}")
    store = QdrantStore(collection_name=TEST_COLLECTION)

    print("Inserting chunks")
    inserted = store.insert_chunks(chunks, embeddings)
    print(f"Inserted {inserted} points")

    time.sleep(2)

    print("\nTesting Reranker in isolation")
    reranker = Reranker()

    sample_query = "नेपालको राजधानी कुन शहर हो?"
    sample_results = [
        {"text": chunks[0]["text"], "metadata": chunks[0]["metadata"]},
        {"text": chunks[1]["text"], "metadata": chunks[1]["metadata"]},
        {"text": chunks[4]["text"], "metadata": chunks[4]["metadata"]},
    ]

    reranked = reranker.rerank(sample_query, sample_results, top_k=3)
    print(f"Query: {sample_query}")
    for i, r in enumerate(reranked):
        print(f"  {i+1}. rerank_score={r['rerank_score']:.4f} | {r['text'][:80]}")

    top_text = reranked[0]["text"]
    assert "काठमाडौं" in top_text, f"Expected capital-related chunk at rank 1, got: {top_text[:60]}"
    print("Reranker isolated test passed: capital chunk ranked first")

    print("\nTesting full Retriever pipeline")
    retriever = Retriever(
        embedding_model=embedding_model,
        store=store,
        reranker=reranker,
    )

    query = "नेपालको राजधानी कुन शहर हो?"
    print(f"Query: {query}")
    results = retriever.retrieve(query, top_k_retrieval=7, top_k_rerank=3)
    print(f"Got {len(results)} reranked results:")
    for i, r in enumerate(results):
        print(f"  {i+1}. hybrid={r['score']:.4f} rerank={r['rerank_score']:.4f} | {r['text'][:80]}")
        print(f"     source={r['metadata'].get('source', 'N/A')}")

    assert len(results) > 0, "No results returned"
    assert "काठमाडौं" in results[0]["text"], "Capital chunk should be ranked first"
    print("Full pipeline test passed")

    print("\nTesting cross-lingual retrieval + reranking")
    en_query = "Where was Gautam Buddha born?"
    print(f"Query: {en_query}")
    en_results = retriever.retrieve(en_query, top_k_retrieval=7, top_k_rerank=3)
    for i, r in enumerate(en_results):
        print(f"  {i+1}. hybrid={r['score']:.4f} rerank={r['rerank_score']:.4f} | {r['text'][:80]}")

    assert len(en_results) > 0, "No cross-lingual results"
    assert "लुम्बिनी" in en_results[0]["text"] or "बुद्ध" in en_results[0]["text"], (
        "Lumbini/Buddha chunk should be ranked first for Buddha birth query"
    )
    print("Cross-lingual retrieval test passed")

    print("\nTesting reranker reorders meaningfully")
    query2 = "सगरमाथाको उचाइ कति हो?"
    print(f"Query: {query2}")
    results2 = retriever.retrieve(query2, top_k_retrieval=7, top_k_rerank=3)
    for i, r in enumerate(results2):
        print(f"  {i+1}. hybrid={r['score']:.4f} rerank={r['rerank_score']:.4f} | {r['text'][:80]}")

    assert "सगरमाथा" in results2[0]["text"], "Everest chunk should rank first for height query"
    print("Reranker reordering test passed")

    print(f"\nCleaning up: deleting collection {TEST_COLLECTION}")
    store.delete_collection()
    print("All retrieval tests passed")


if __name__ == "__main__":
    run_test()
