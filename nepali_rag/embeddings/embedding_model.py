"""Dense embedding utilities for Nepali RAG retrieval.

Wraps a SentenceTransformer model (default: multilingual-e5-large) with the
required ``passage:`` / ``query:`` prefixes and L2 normalization so vectors
are ready for cosine similarity in Qdrant.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from nepali_rag.config.settings import settings


class NepaliEmbeddingModel:
    """Wrapper around multilingual-e5-large with mandatory prefix handling and L2 normalization."""

    def __init__(self, model_name: str = None):
        """Load the embedding model. Defaults to ``settings.embedding_model`` (``intfloat/multilingual-e5-large``)."""
        if model_name is None:
            model_name = settings.embedding_model
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of document passages.

        Each text is prefixed with ``passage: `` (required by e5 models).

        Returns:
            ``(n, dimension)`` array of L2-normalized embeddings.
        """
        prefixed = [f"passage: {t}" for t in texts]
        embeddings = self.model.encode(prefixed, normalize_embeddings=True)
        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query for retrieval.

        The query is prefixed with ``query: `` (required by e5 models).

        Returns:
            1-D L2-normalized embedding vector.
        """
        prefixed = f"query: {query}"
        embedding = self.model.encode(prefixed, normalize_embeddings=True)
        return np.array(embedding)
