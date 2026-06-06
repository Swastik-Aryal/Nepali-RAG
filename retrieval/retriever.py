from embeddings.embedding_model import NepaliEmbeddingModel
from vectorstore.qdrant_store import QdrantStore
from retrieval.reranker import Reranker
from config.settings import settings


class Retriever:
    """Two-stage retrieval: hybrid search followed by cross-encoder reranking."""

    def __init__(
        self,
        embedding_model: NepaliEmbeddingModel = None,
        store: QdrantStore = None,
        reranker: Reranker = None,
    ):
        self.embedding_model = embedding_model or NepaliEmbeddingModel()
        self.store = store or QdrantStore()
        self.reranker = reranker or Reranker()

    def retrieve(
        self,
        query: str,
        top_k_retrieval: int = None,
        top_k_rerank: int = None,
    ) -> list[dict]:
        """Run the full two-stage retrieval pipeline.

        Stage 1: Hybrid search (dense + sparse) retrieves a broad candidate set.
        Stage 2: Cross-encoder reranker scores each candidate for precision.

        Args:
            query: the user's search query.
            top_k_retrieval: number of candidates from hybrid search (default from settings).
            top_k_rerank: number of final results after reranking (default from settings).

        Returns:
            List of result dicts, each containing:
                text, metadata, score (hybrid), rerank_score.
        """
        if top_k_retrieval is None:
            top_k_retrieval = settings.top_k_retrieval
        if top_k_rerank is None:
            top_k_rerank = settings.top_k_rerank

        query_embedding = self.embedding_model.embed_query(query)

        candidates = self.store.search_hybrid(
            query_embedding=query_embedding,
            query_text=query,
            top_k=top_k_retrieval,
        )

        if not candidates:
            return []

        reranked = self.reranker.rerank(
            query=query,
            results=candidates,
            top_k=top_k_rerank,
        )

        return reranked
