import sys
sys.stdout.reconfigure(encoding="utf-8")

from utils.embeddings.embedding_model import NepaliEmbeddingModel
from utils.vectorstore.qdrant_store import QdrantStore

TEST_COLLECTION = "test_nepali_vectorstore"


def run_test():
    print("Loading embedding model")
    model = NepaliEmbeddingModel()

    chunks = [
        {"text": "नेपाल एक सुन्दर हिमाली देश हो।", "metadata": {"source": "test", "page_number": 1, "chunk_index": 0}},
        {"text": "काठमाडौं नेपालको राजधानी हो।", "metadata": {"source": "test", "page_number": 1, "chunk_index": 1}},
        {"text": "सगरमाथा संसारको सबैभन्दा अग्लो हिमाल हो।", "metadata": {"source": "test", "page_number": 1, "chunk_index": 2}},
        {"text": "पोखरा फेवा तालको लागि प्रसिद्ध छ।", "metadata": {"source": "test", "page_number": 1, "chunk_index": 3}},
        {"text": "Python is a popular programming language for data science.", "metadata": {"source": "test", "page_number": 2, "chunk_index": 4}},
    ]

    print("Computing embeddings")
    texts = [c["text"] for c in chunks]
    embeddings = model.embed_documents(texts)

    print(f"Connecting to Qdrant Cloud, collection: {TEST_COLLECTION}")
    store = QdrantStore(collection_name=TEST_COLLECTION)

    print(f"Inserting {len(chunks)} chunks")
    inserted = store.insert_chunks(chunks, embeddings)
    print(f"Inserted {inserted} points")

    info = store.get_collection_info()
    print(f"Collection info: {info}")

    import time
    time.sleep(2)

    print("\nDense search: 'नेपालको राजधानी कुन हो?'")
    query = "नेपालको राजधानी कुन हो?"
    q_emb = model.embed_query(query)
    dense_results = store.search_dense(q_emb, top_k=5)
    for i, r in enumerate(dense_results):
        print(f"  {i+1}. score={r['score']:.4f} | {r['text'][:80]}")

    print("\nSparse search: 'काठमाडौं राजधानी'")
    sparse_results = store.search_sparse("काठमाडौं राजधानी", top_k=5)
    for i, r in enumerate(sparse_results):
        print(f"  {i+1}. score={r['score']:.4f} | {r['text'][:80]}")

    print("\nHybrid search: 'नेपालको राजधानी कुन हो?'")
    hybrid_results = store.search_hybrid(q_emb, query, top_k=5)
    for i, r in enumerate(hybrid_results):
        print(f"  {i+1}. score={r['score']:.4f} | {r['text'][:80]}")

    print("\nCross-lingual dense search: 'What is the capital of Nepal?'")
    en_query = "What is the capital of Nepal?"
    en_emb = model.embed_query(en_query)
    cross_results = store.search_dense(en_emb, top_k=5)
    for i, r in enumerate(cross_results):
        print(f"  {i+1}. score={r['score']:.4f} | {r['text'][:80]}")

    print(f"\nCleaning up: deleting collection {TEST_COLLECTION}")
    store.delete_collection()
    print("Done")


if __name__ == "__main__":
    run_test()
