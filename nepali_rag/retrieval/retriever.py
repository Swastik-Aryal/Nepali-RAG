from nepali_rag.embeddings.embedding_model import NepaliEmbeddingModel
from nepali_rag.vectorstore.qdrant_store import QdrantStore
from nepali_rag.retrieval.reranker import Reranker
from nepali_rag.config.settings import settings


class Retriever:
    """Two-stage retrieval: hybrid search followed by cross-encoder reranking.

    Args:
        - embedding_model (NepaliEmbeddingModel, optional): Pre-initialized embedding model for
            generating query vectors. If ``None``, a default ``NepaliEmbeddingModel`` is created.
            Defaults to ``None``.
        - store (QdrantStore, optional): Pre-initialized Qdrant vector store for hybrid search.
            If ``None``, a default ``QdrantStore`` is created. Defaults to ``None``.
        - reranker (Reranker, optional): Pre-initialized cross-encoder reranker. If ``None``, a
            default ``Reranker`` is created. Defaults to ``None``.
    """

    def __init__(
        self,
        embedding_model: NepaliEmbeddingModel = None,
        store: QdrantStore = None,
        reranker: Reranker = None,
    ):
        """Wire up embedding model, vector store, and reranker.
        - embedding_model (NepaliEmbeddingModel, optional): Pre-initialized embedding model for
            generating query vectors. If ``None``, a default ``NepaliEmbeddingModel`` is created.
            Defaults to ``None``.
        - store (QdrantStore, optional): Pre-initialized Qdrant vector store for hybrid search.
            If ``None``, a default ``QdrantStore`` is created. Defaults to ``None``.
        - reranker (Reranker, optional): Pre-initialized cross-encoder reranker. If ``None``, a
            default ``Reranker`` is created. Defaults to ``None``.
        """
        self.embedding_model = embedding_model or NepaliEmbeddingModel()
        self.store = store or QdrantStore()
        self.reranker = reranker or Reranker()

    def retrieve(
        self,
        query: str,
        top_k_retrieval: int = None,
        top_k_rerank: int = None,
    ) -> list[dict]:
        """Retrieve and rerank passages for a query.

        Stage 1: hybrid dense + sparse search in Qdrant.
        Stage 2: cross-encoder reranking for precision.

        Args:
            query: User search query.
            top_k_retrieval: Hybrid-search candidate count (default ``settings.top_k_retrieval``, 20).
            top_k_rerank: Final result count (default ``settings.top_k_rerank``, 5).

        Returns:
            List of dicts with ``text``, ``metadata``, ``score`` (hybrid), and ``rerank_score``.
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
