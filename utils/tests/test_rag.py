import sys
sys.stdout.reconfigure(encoding="utf-8")

import time
from utils.embeddings.embedding_model import NepaliEmbeddingModel
from utils.vectorstore.qdrant_store import QdrantStore
from utils.retrieval.reranker import Reranker
from utils.retrieval.retriever import Retriever
from utils.rag.chain import RAGChain

TEST_COLLECTION = "test_nepali_rag_chain"


def run_test():
    print("Loading models")
    embedding_model = NepaliEmbeddingModel()
    reranker = Reranker()

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
            "text": "लुम्बिनी गौतम बुद्धको जन्मस्थल हो। यो नेपालको रूपन्देही जिल्लामा पर्छ।",
            "metadata": {"source": "culture.pdf", "page_number": 1, "chunk_index": 4},
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

    retriever = Retriever(
        embedding_model=embedding_model,
        store=store,
        reranker=reranker,
    )

    print("\n[Test 1] RAG chain in test mode (no LLM)")
    chain = RAGChain(retriever=retriever, llm=None)
    question = "नेपालको राजधानी कुन शहर हो?"
    result = chain.run(question, top_k_retrieval=5, top_k_rerank=3)

    assert "answer" in result, "Missing 'answer' key"
    assert "sources" in result, "Missing 'sources' key"
    assert "context" in result, "Missing 'context' key"

    prompt = result["answer"]
    assert "SYSTEM:" in prompt, "Prompt should contain SYSTEM section"
    assert "USER:" in prompt, "Prompt should contain USER section"
    assert question in prompt, "Prompt should contain the original question"
    assert "Context:" in prompt, "Prompt should contain context section"

    assert len(result["sources"]) > 0, "Sources list should not be empty"
    assert "[1]" in result["context"], "Context should have numbered chunks"

    print(f"Question: {question}")
    print(f"Sources returned: {len(result['sources'])}")
    print(f"Context length: {len(result['context'])} chars")
    print(f"Prompt preview (first 300 chars):\n{prompt[:300]}...")
    print("Test 1 passed: test mode returns well-formed prompt\n")

    print("[Test 2] Verify source metadata in context")
    ctx = result["context"]
    assert "geography.pdf" in ctx, "Context should reference source filename"
    assert "Page:" in ctx or "Page: 1" in ctx, "Context should include page numbers"
    print("Test 2 passed: source metadata present in context\n")

    print("[Test 3] RAG chain with conversation history (test mode)")
    history = [
        {"role": "user", "content": "नेपालबारे बताउनुहोस्।"},
        {"role": "assistant", "content": "नेपाल दक्षिण एसियाको एक देश हो।"},
    ]
    result_with_history = chain.run(
        question="राजधानी कुन हो?",
        conversation_history=history,
        top_k_retrieval=5,
        top_k_rerank=3,
    )
    prompt_h = result_with_history["answer"]
    assert "नेपालबारे बताउनुहोस्" in prompt_h, "History user message should appear in prompt"
    assert "नेपाल दक्षिण एसियाको एक देश हो।" in prompt_h, "History assistant message should appear in prompt"
    assert "SYSTEM:" in prompt_h
    assert "USER:" in prompt_h
    print(f"Prompt with history preview:\n{prompt_h[:400]}...")
    print("Test 3 passed: conversation history included in prompt\n")

    print("[Test 4] Empty retrieval results")
    empty_store = QdrantStore(collection_name="test_empty_rag_chain")
    empty_retriever = Retriever(
        embedding_model=embedding_model,
        store=empty_store,
        reranker=reranker,
    )
    empty_chain = RAGChain(retriever=empty_retriever, llm=None)
    empty_result = empty_chain.run("कुनै प्रश्न")
    assert empty_result["sources"] == [], "Sources should be empty for empty store"
    assert empty_result["context"] == "", "Context should be empty string"
    assert "SYSTEM:" in empty_result["answer"], "Prompt structure should still be valid"
    empty_store.delete_collection()
    print("Test 4 passed: handles empty retrieval gracefully\n")

    print("[Test 5] Cross-lingual query (English question, Nepali context)")
    en_result = chain.run(
        "What is the capital of Nepal?",
        top_k_retrieval=5,
        top_k_rerank=3,
    )
    assert len(en_result["sources"]) > 0
    assert "काठमाडौं" in en_result["context"], "Nepali context should be retrieved for English query"
    print(f"English query sources: {len(en_result['sources'])}")
    print("Test 5 passed: cross-lingual retrieval works in chain\n")

    print(f"Cleaning up: deleting collection {TEST_COLLECTION}")
    store.delete_collection()
    print("All RAG chain tests passed")


if __name__ == "__main__":
    run_test()
