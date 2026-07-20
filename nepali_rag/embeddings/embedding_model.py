"""Dense embedding utilities for Nepali RAG retrieval.

Wraps a SentenceTransformer model with the appropriate encoding strategy
per model family (prefix, prompt_name, or plain) and L2 normalization so
vectors are ready for cosine similarity in Qdrant.

Supported model families:
    - intfloat/multilingual-e5-small / base / large  (prefix: passage/query)
    - intfloat/multilingual-e5-large-instruct  (instruct prefix for queries)
    - Qwen/Qwen3-Embedding-0.6B / 4B / 8B  (prompt_name="query" for queries)
    - jinaai/jina-embeddings-v3  (prompt_name, task ="retrieval.passage" / "retrieval.query")
    - jinaai/jina-embeddings-v5-text-nano / small  (resolves to -retrieval variant)
    - Alibaba-NLP/gte-multilingual-base  (no prefix)
    - BAAI/bge-m3  (no prefix)
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from nepali_rag.config.settings import settings

# Retrieval task description used by e5-instruct models.
_E5_INSTRUCT_TASK = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def _resolve_jina_v5(model_name: str) -> str:
    """Append ``-retrieval`` to bare jina-v5 names when no variant suffix is present."""
    v5_bases = (
        "jinaai/jina-embeddings-v5-text-nano",
        "jinaai/jina-embeddings-v5-text-small",
    )
    for base in v5_bases:
        if model_name == base:
            return f"{base}-retrieval"
    return model_name


def _best_dtype() -> torch.dtype:
    """Return fp16 on CUDA, fp32 on CPU."""
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


# Mapping from exact (lowercased) SentenceTransformer model identifiers to
# the short family tag that drives prefix / prompt_name dispatch.
_KNOWN_MODELS: dict[str, str] = {
    # E5 family
    "intfloat/multilingual-e5-small": "e5",
    "intfloat/multilingual-e5-base": "e5",
    "intfloat/multilingual-e5-large": "e5",
    # E5-instruct
    "intfloat/multilingual-e5-large-instruct": "e5-instruct",
    # Qwen3-Embedding
    "qwen/qwen3-embedding-0.6b": "qwen",
    "qwen/qwen3-embedding-4b": "qwen",
    "qwen/qwen3-embedding-8b": "qwen",
    # Jina v3
    "jinaai/jina-embeddings-v3": "jina-v3",
    # Jina v5 (bare names resolve to -retrieval via _resolve_jina_v5)
    "jinaai/jina-embeddings-v5-text-nano-retrieval": "jina-v5",
    "jinaai/jina-embeddings-v5-text-small-retrieval": "jina-v5",
    # GTE
    "alibaba-nlp/gte-multilingual-base": "gte",
    # BGE-M3
    "baai/bge-m3": "bge",
}


def _model_family(model_name: str) -> str:
    """Return a short family tag used to dispatch encoding behaviour.

    Raises:
        ValueError: If *model_name* is not in the supported model list.
    """
    family = _KNOWN_MODELS.get(model_name.lower())
    if family is None:
        supported = "\n  - ".join(sorted(_KNOWN_MODELS))
        raise ValueError(
            f"Unsupported embedding model: {model_name!r}.\n"
            f"Supported models:\n  - {supported}"
        )
    return family


class NepaliEmbeddingModel:
    """Multi-model embedding wrapper with consistent encode interface.

    Accepts any supported model name and adapts prefix / prompt_name handling
    automatically.  The two public methods (``embed_documents``, ``embed_query``)
    keep the same signature and return types regardless of model.

    Args:
        - model_name (str, optional): Name or path of the SentenceTransformer model to load.
            Defaults to ``settings.embedding_model`` (intfloat/multilingual-e5-large).
        
        Supported models:
            - baai/bge-m3
            - intfloat/multilingual-e5-small
            - intfloat/multilingual-e5-base
            - intfloat/multilingual-e5-large
            - intfloat/multilingual-e5-large-instruct
            - qwen/qwen3-embedding-0.6b
            - qwen/qwen3-embedding-4b
            - qwen/qwen3-embedding-8b
            - jinaai/jina-embeddings-v3
            - jinaai/jina-embeddings-v5-text-nano-retrieval
            - jinaai/jina-embeddings-v5-text-small-retrieval
            - alibaba-nlp/gte-multilingual-base
            
    """

    def __init__(self, model_name: str = None):
        """Load the embedding model.

        - model_name (str, optional): Name or path of the SentenceTransformer model to load.
            Defaults to ``settings.embedding_model`` (``intfloat/multilingual-e5-large``).

        Supported models:
            - baai/bge-m3
            - intfloat/multilingual-e5-small
            - intfloat/multilingual-e5-base
            - intfloat/multilingual-e5-large
            - intfloat/multilingual-e5-large-instruct
            - qwen/qwen3-embedding-0.6b
            - qwen/qwen3-embedding-4b
            - qwen/qwen3-embedding-8b
            - jinaai/jina-embeddings-v3
            - jinaai/jina-embeddings-v5-text-nano-retrieval
            - jinaai/jina-embeddings-v5-text-small-retrieval
            - alibaba-nlp/gte-multilingual-base
        
        """
        if model_name is None:
            model_name = settings.embedding_model

        try:
            model_name = _resolve_jina_v5(model_name)
            self.model_name = model_name
            self.family = _model_family(model_name)
        except ValueError:
            raise

        dtype = _best_dtype()
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": dtype},
        )
        self.dimension = self.model.get_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of document passages.

        Returns:
            ``(n, dimension)`` array of L2-normalized embeddings.
        """
        if self.family == "e5":
            texts = [f"passage: {t}" for t in texts]
            return np.array(self.model.encode(texts, normalize_embeddings=True))

        if self.family == "jina-v3":
            return np.array(
                self.model.encode(
                    texts, task="retrieval.passage", prompt_name="retrieval.passage", normalize_embeddings=True
                )
            )

        if self.family == "jina-v5":
            return np.array(
                self.model.encode(
                    texts, prompt_name="document", normalize_embeddings=True
                )
            )

        # e5-instruct, qwen, gte, generic: no document prefix needed.
        return np.array(self.model.encode(texts, normalize_embeddings=True))

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query for retrieval.

        Returns:
            1-D L2-normalized embedding vector.
        """
        if self.family == "e5":
            text = f"query: {query}"
            return np.array(self.model.encode(text, normalize_embeddings=True))

        if self.family == "e5-instruct":
            text = f"Instruct: {_E5_INSTRUCT_TASK}\nQuery: {query}"
            return np.array(self.model.encode(text, normalize_embeddings=True))

        if self.family == "qwen":
            return np.array(
                self.model.encode(query, prompt_name="query", normalize_embeddings=True)
            )

        if self.family == "jina-v3":
            return np.array(
                self.model.encode(
                    query,task="retrieval.query", prompt_name="retrieval.query", normalize_embeddings=True
                )
            )

        if self.family == "jina-v5":
            return np.array(
                self.model.encode(
                    query, prompt_name="query", normalize_embeddings=True
                )
            )

        # gte, generic: no query prefix needed.
        return np.array(self.model.encode(query, normalize_embeddings=True))
