"""Qdrant vector store with dense + sparse hybrid search.

Stores chunk embeddings alongside term-frequency sparse vectors for reciprocal
rank fusion (RRF) at query time. Supports both Latin and Devanagari tokens.
"""

import uuid
from collections import defaultdict
import hashlib

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    NamedSparseVector,
    NamedVector,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from nepali_rag.config.settings import settings


class QdrantStore:
    """Qdrant client wrapper for chunk upsert and hybrid retrieval.

    Args:
        - collection_name (str, optional): Name of the Qdrant collection to use.
            Defaults to ``settings.collection_name`` (nepali_docs).
        - url (str, optional): URL of the Qdrant server.
            Defaults to ``settings.qdrant_url`` ("").
        - api_key (str, optional): API key for authenticating with the Qdrant server.
            Defaults to ``settings.qdrant_api_key`` ("").
        - client (QdrantClient, optional): A pre-initialized ``QdrantClient`` instance. If
            provided, ``url`` and ``api_key`` are ignored. Defaults to ``None``.
    """

    def __init__(
        self,
        collection_name: str = None,
        url: str = None,
        api_key: str = None,
        client: QdrantClient = None,
    ):
        """Connect to Qdrant and ensure the collection exists.

        - collection_name (str, optional): Name of the Qdrant collection to use.
            Defaults to ``settings.collection_name`` (nepali_docs).
        - url (str, optional): URL of the Qdrant server.
            Defaults to ``settings.qdrant_url`` ("").
        - api_key (str, optional): API key for authenticating with the Qdrant server.
            Defaults to ``settings.qdrant_api_key`` ("").
        - client (QdrantClient, optional): A pre-initialized ``QdrantClient`` instance. If
            provided, ``url`` and ``api_key`` are ignored. Defaults to ``None``.

        """

        self.collection_name = collection_name or settings.collection_name
        if client is not None:
            self.client = client
        else:
            url = url or settings.qdrant_url
            api_key = api_key or settings.qdrant_api_key
            self.client = QdrantClient(url=url, api_key=api_key)
        self._ensure_collection()

    def _ensure_collection(self):
        """Create the dense (1024-dim cosine) + sparse collection if missing."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in collections:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(size=1024, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )

    def _token_to_index(self, token: str) -> int:
        """Map a token to a stable sparse index via MD5 hashing."""
        return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % (2**20)

    def _compute_sparse_vector(self, text: str) -> SparseVector:
        """Build a term-frequency sparse vector (whitespace tokenization, any script)."""
        # Basic whitespace split — works for both scripts
        tokens = text.split()
        term_freq: dict[int, float] = defaultdict(float)
        for token in tokens:
            index = self._token_to_index(token)
            term_freq[index] += 1.0

        indices = sorted(term_freq.keys())
        values = [term_freq[i] for i in indices]
        return SparseVector(indices=indices, values=values)

    def insert_chunks(
        self,
        chunks: list[dict],
        embeddings,
        batch_size: int = 64,
    ) -> int:
        """Upsert chunks with dense embeddings and computed sparse vectors.

        Args:
            chunks: List of dicts with ``text`` and optional ``metadata``.
            embeddings: Dense vectors aligned with *chunks* (numpy array or list).
            batch_size: Points per upsert batch (default 64).

        Returns:
            Total number of points inserted.
        """
        points = []
        for chunk, dense_vec in zip(chunks, embeddings):
            text = chunk["text"]
            metadata = chunk.get("metadata", {})

            dense_list = (
                dense_vec.tolist() if hasattr(dense_vec, "tolist") else list(dense_vec)
            )
            sparse_vec = self._compute_sparse_vector(text)

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_list,
                    "sparse": sparse_vec,
                },
                payload={
                    "text": text,
                    **metadata,
                },
            )
            points.append(point)

        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        return len(points)

    def search_dense(self, query_embedding, top_k: int = None) -> list[dict]:
        """Cosine similarity search on the dense embedding vector.

        Returns:
            List of dicts with ``id``, ``score``, ``text``, and ``metadata``.
        """
        if top_k is None:
            top_k = settings.top_k_retrieval

        dense_list = (
            query_embedding.tolist()
            if hasattr(query_embedding, "tolist")
            else list(query_embedding)
        )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_list,
            using="dense",
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "id": str(point.id),
                "score": point.score,
                "text": point.payload.get("text", ""),
                "metadata": {k: v for k, v in point.payload.items() if k != "text"},
            }
            for point in results.points
        ]

    def search_sparse(self, query_text: str, top_k: int = None) -> list[dict]:
        """Keyword-style search on the sparse term-frequency vector.

        Returns:
            List of dicts with ``id``, ``score``, ``text``, and ``metadata``.
        """
        if top_k is None:
            top_k = settings.top_k_retrieval

        sparse_vec = self._compute_sparse_vector(query_text)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=SparseVector(indices=sparse_vec.indices, values=sparse_vec.values),
            using="sparse",
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "id": str(point.id),
                "score": point.score,
                "text": point.payload.get("text", ""),
                "metadata": {k: v for k, v in point.payload.items() if k != "text"},
            }
            for point in results.points
        ]

    def search_hybrid(
        self,
        query_embedding,
        query_text: str,
        top_k: int = None,
        dense_weight: float = None,
        sparse_weight: float = None,
    ) -> list[dict]:
        """Fuse dense and sparse rankings with weighted reciprocal rank fusion (RRF).

        Args:
            query_embedding: Dense query vector from the embedding model.
            query_text: Raw query string for sparse search.
            top_k: Number of fused results (default ``settings.top_k_retrieval``, 20).
            dense_weight: RRF weight for dense results (default ``settings.dense_weight``, 0.9).
            sparse_weight: RRF weight for sparse results (default ``settings.sparse_weight``, 0.1).

        Returns:
            Fused list sorted by combined RRF score.
        """
        if top_k is None:
            top_k = settings.top_k_retrieval
        if dense_weight is None:
            dense_weight = settings.dense_weight
        if sparse_weight is None:
            sparse_weight = settings.sparse_weight

        dense_results = self.search_dense(query_embedding, top_k=top_k)
        sparse_results = self.search_sparse(query_text, top_k=top_k)

        rrf_k = 60
        scores: dict[str, float] = defaultdict(float)
        result_map: dict[str, dict] = {}

        for rank, result in enumerate(dense_results):
            doc_id = result["id"]
            scores[doc_id] += dense_weight / (rrf_k + rank + 1)
            result_map[doc_id] = result

        for rank, result in enumerate(sparse_results):
            doc_id = result["id"]
            scores[doc_id] += sparse_weight / (rrf_k + rank + 1)
            if doc_id not in result_map:
                result_map[doc_id] = result

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[
            :top_k
        ]

        combined = []
        for doc_id in sorted_ids:
            entry = result_map[doc_id].copy()
            entry["score"] = scores[doc_id]
            combined.append(entry)

        return combined

    def delete_collection(self):
        """Drop the configured collection from Qdrant."""
        self.client.delete_collection(collection_name=self.collection_name)

    def get_collection_info(self) -> dict:
        """Return collection name, point count, and status."""
        info = self.client.get_collection(collection_name=self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "status": info.status.value
            if hasattr(info.status, "value")
            else str(info.status),
        }
