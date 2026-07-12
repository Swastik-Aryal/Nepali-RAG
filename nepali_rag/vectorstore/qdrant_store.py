"""Qdrant vector store with dense + BM25 hybrid search.

Stores chunk embeddings for dense retrieval and uses BM25s for keyword-based
sparse retrieval at query time. Supports both Latin and Devanagari tokens.
"""

import re
import uuid
from collections import defaultdict

import bm25s
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from nepali_rag.config.settings import settings

_NEPALI_SYMBOLS_RE = re.compile(
    r"[\u0964\u0965\u0970"
    r"\u2010-\u2015\u2018\u2019\u201C\u201D\u2022\u2026\u2013\u2014"
    r"!\"#\$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]"
)
_MULTI_SPACE_RE = re.compile(r" {2,}")


def _clean_for_bm25(text: str) -> str:
    """Strip Nepali/Unicode symbols, newlines, and collapse whitespace."""
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("\u00A0", " ")
    text = _NEPALI_SYMBOLS_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


class QdrantStore:
    """Qdrant client wrapper for chunk upsert and hybrid retrieval.

    Dense vectors are stored in Qdrant for cosine similarity search.
    Sparse (keyword) retrieval uses BM25s over all stored chunk texts,
    fetched on-demand via scrolling.

    Args:
        - vector_size (int): Dimensionality of the dense embedding vectors.
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
        vector_size: int,
        collection_name: str = None,
        url: str = None,
        api_key: str = None,
        client: QdrantClient = None,
    ):
        """Connect to Qdrant and ensure the collection exists.

        - vector_size (int): Dimensionality of the dense embedding vectors.
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
        self.vector_size = vector_size
        if client is not None:
            self.client = client
        else:
            url = url or settings.qdrant_url
            api_key = api_key or settings.qdrant_api_key
            self.client = QdrantClient(url=url, api_key=api_key)
        self._ensure_collection()

    def _ensure_collection(self):
        """Create the dense-only collection if missing."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in collections:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(size=self.vector_size, distance=Distance.COSINE),
            },
        )

    def _scroll_all_points(self) -> list:
        """Scroll through all points in the collection, returning payloads and IDs."""
        all_points = []
        offset = None
        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(results)
            if next_offset is None:
                break
            offset = next_offset
        return all_points

    def insert_chunks(
        self,
        chunks: list[dict],
        embeddings,
        batch_size: int = 64,
    ) -> int:
        """Upsert chunks with dense embeddings (no sparse vectors stored).

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

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_list,
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
        """BM25-based keyword search over all stored chunks.

        Scrolls all points from Qdrant, builds a BM25s index in-memory from
        their text payloads, and returns the top-k ranked results.
        Corpus and query texts are cleaned (symbols, newlines, extra spaces
        removed) before tokenization; output ``text`` contains the original.

        Returns:
            List of dicts with ``id``, ``score``, ``text``, and ``metadata``.
        """
        if top_k is None:
            top_k = settings.top_k_retrieval

        all_points = self._scroll_all_points()
        if not all_points:
            return []
        
        corpus_ids = []
        corpus_texts = []
        for pt in all_points:
            text = pt.payload.get("text", "")
            if text:
                corpus_ids.append(str(pt.id))
                corpus_texts.append(text)

        if not corpus_texts:
            return []

        cleaned_corpus = [_clean_for_bm25(t) for t in corpus_texts]
        corpus_tokens = bm25s.tokenize(cleaned_corpus, token_pattern=r"\S+", stemmer=None)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        cleaned_query = _clean_for_bm25(query_text)
        query_tokens = bm25s.tokenize([cleaned_query], token_pattern=r"\S+", stemmer=None)
        k = min(top_k, len(corpus_texts))
        results, scores = retriever.retrieve(query_tokens, corpus=cleaned_corpus, k=k)

        output = []
        for idx in range(results.shape[1]):
            matched_text = results[0, idx]
            score = float(scores[0, idx])
            pos = cleaned_corpus.index(matched_text) if matched_text in cleaned_corpus else -1
            if pos == -1:
                continue
            pt = all_points[pos]
            output.append({
                "id": corpus_ids[pos],
                "score": score,
                "text": corpus_texts[pos],
                "metadata": {k: v for k, v in pt.payload.items() if k != "text"},
            })

        return output

    def search_hybrid(
        self,
        query_embedding,
        query_text: str,
        top_k: int = None,
        dense_weight: float = None,
        sparse_weight: float = None,
    ) -> list[dict]:
        """Fuse dense and BM25 sparse rankings with weighted reciprocal rank fusion (RRF).

        Args:
            query_embedding: Dense query vector from the embedding model.
            query_text: Raw query string for BM25 sparse search.
            top_k: Number of fused results (default ``settings.top_k_retrieval``, 20).
            dense_weight: RRF weight for dense results (default ``settings.dense_weight``, 0.7).
            sparse_weight: RRF weight for sparse results (default ``settings.sparse_weight``, 0.3).

        Returns:
            Fused list sorted by combined RRF score.
            
        Note: If query is *NOT* a Nepali text, then sparse search is skipped, and only dense search results are returned.
        """
        from nepali_rag.preprocessing.normalizer import is_nepali_text

        if top_k is None:
            top_k = settings.top_k_retrieval
        if dense_weight is None:
            dense_weight = settings.dense_weight
        if sparse_weight is None:
            sparse_weight = settings.sparse_weight

        dense_results = self.search_dense(query_embedding, top_k=top_k)

        if not is_nepali_text(query_text):
            return dense_results

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
