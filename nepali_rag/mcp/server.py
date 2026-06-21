"""MCP server exposing ingest_document and retrieve as tools for AI agents."""

import asyncio
import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("nepali-rag")
logger = logging.getLogger(__name__)

# Shared model state (lazy-loaded on first use, or eagerly via initialize_models)
_embedding_model = None
_reranker = None

# Cached QdrantClient instances keyed by storage location.
# Local embedded Qdrant locks the storage directory, so only one QdrantClient
# can open a given path at a time. Caching ensures ingest and retrieve share
# the same client instead of fighting over the lock.
_qdrant_clients: dict[str, "QdrantClient"] = {}


def initialize_models():
    """Eagerly load embedding model and reranker at startup."""
    global _embedding_model, _reranker
    from nepali_rag.embeddings.embedding_model import NepaliEmbeddingModel
    from nepali_rag.retrieval.reranker import Reranker
    _embedding_model = NepaliEmbeddingModel()
    _reranker = Reranker()


def _get_embedding_model():
    """Return the cached embedding model, creating it lazily if needed."""
    global _embedding_model
    if _embedding_model is None:
        from nepali_rag.embeddings.embedding_model import NepaliEmbeddingModel
        _embedding_model = NepaliEmbeddingModel()
    return _embedding_model


def _get_reranker():
    """Return the cached reranker, creating it lazily if needed."""
    global _reranker
    if _reranker is None:
        from nepali_rag.retrieval.reranker import Reranker
        _reranker = Reranker()
    return _reranker


def _generate_collection_name() -> str:
    """Generate a unique collection name for a session."""
    import uuid
    return f"session_{uuid.uuid4().hex[:8]}"


def _validate_absolute_path(path: str, param_name: str) -> str:
    """Validate that a path is absolute. Raises ValueError if relative."""
    p = Path(path)
    if not p.is_absolute():
        raise ValueError(
            f"{param_name} must be an absolute path, got relative path: '{path}'. "
            f"Provide a full path like 'C:/Users/.../qdrant_data' (Windows) "
            f"or '/home/.../qdrant_data' (Linux/macOS)."
        )
    return str(p)


def _get_qdrant_client(
    qdrant_path: str | None = None,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
):
    """Return a cached QdrantClient for the given storage location.

    Local embedded Qdrant locks the storage directory so only one client
    can hold it open. This function caches clients by their connection key
    (path for local, url for remote) so all tool calls share one client.
    """
    from qdrant_client import QdrantClient

    if qdrant_url:
        cache_key = f"url:{qdrant_url}"
        if cache_key not in _qdrant_clients:
            _qdrant_clients[cache_key] = QdrantClient(
                url=qdrant_url, api_key=qdrant_api_key or ""
            )
        return _qdrant_clients[cache_key]

    if not qdrant_path:
        raise ValueError(
            "Either qdrant_path or qdrant_url must be provided. "
            "For local storage, provide an absolute path to a directory."
        )
    qdrant_path = _validate_absolute_path(qdrant_path, "qdrant_path")
    cache_key = f"path:{qdrant_path}"
    if cache_key not in _qdrant_clients:
        _qdrant_clients[cache_key] = QdrantClient(path=qdrant_path)
    return _qdrant_clients[cache_key]


def _create_qdrant_store(
    collection_name: str,
    qdrant_path: str | None = None,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
):
    """Create a QdrantStore with a shared (cached) QdrantClient.

    If qdrant_url is provided, connects to a remote server.
    Otherwise, validates qdrant_path and uses local file-based storage.
    The underlying QdrantClient is cached per storage location so that
    ingest and retrieve can both access the same database without locking conflicts.
    """
    from nepali_rag.vectorstore.qdrant_store import QdrantStore

    client = _get_qdrant_client(qdrant_path, qdrant_url, qdrant_api_key)
    return QdrantStore(
        collection_name=collection_name,
        client=client,
    )


def _do_ingest(store, embedding_model, file_paths, chunk_size, chunk_overlap_sentences):
    """Synchronous ingestion logic (runs in thread)."""
    from nepali_rag.pipeline.document_pipeline import DocumentPipeline
    pipeline = DocumentPipeline(
        embedding_model=embedding_model,
        vector_store=store,
        chunk_size=chunk_size,
        chunk_overlap_sentences=chunk_overlap_sentences,
    )
    return pipeline.ingest_batch(file_paths)


def _do_retrieve(store, embedding_model, reranker, query, top_k_retrieval, top_k_rerank):
    """Synchronous retrieval logic (runs in thread)."""
    from nepali_rag.retrieval.retriever import Retriever
    retriever = Retriever(
        embedding_model=embedding_model,
        store=store,
        reranker=reranker,
    )
    return retriever.retrieve(query, top_k_retrieval=top_k_retrieval, top_k_rerank=top_k_rerank)


@mcp.tool()
async def ingest_document(
    file_paths: list[str],
    qdrant_path: str | None = None,
    collection_name: str | None = None,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    chunk_size: int = 256,
    chunk_overlap_sentences: int = 2,
) -> dict:
    """Ingest one or more Nepali language documents into a Qdrant collection for retrival later on.
    IMPORTANT: This MCP server runs locally on the user's machine.
    file_paths are local filesystem paths that this server accesses directly.
    Do NOT ask the user to upload files — pass the path as-is to this tool.
    Example: "C:/Users/Acer/Desktop/xyz/test.pdf" is a valid file_paths entry.

    Data is stored in a local Qdrant database on disk at qdrant_path.
    qdrant_path must be an absolute path to a directory.

    Storage path guidelines:
    - If you have access to the project/workspace directory, use
      <project_root>/qdrant_data (e.g. "C:/Users/me/my-project/qdrant_data").
    - If you do not have a project directory, ask the user where to store data.
      Recommend ~/.nepali_rag/qdrant_data (resolve to absolute path) as default.

    If collection_name is not provided, a unique name is auto-generated
    (e.g. "session_a3f8b2c1"). Both collection_name and qdrant_path are
    returned in the response -- remember them for subsequent calls.

    Recommended: Use the auto-generated unique name.

    Only provide qdrant_url and qdrant_api_key if the user explicitly
    requests a remote Qdrant server. Ask the user before switching to remote mode.
    When qdrant_url is provided, qdrant_path is not needed.

    Don't use the optional arguments in the request unless the user explicitly asks you to or you think it is necessary.

    Args:
        file_paths: List of absolute paths to the document files to ingest.
        qdrant_path: Absolute path for local Qdrant storage directory.
            Required when qdrant_url is not provided. Must be absolute.
        collection_name: Name of the Qdrant collection to use.
            If omitted, a unique name is auto-generated (e.g. "session_a3f8b2c1").
        qdrant_url: URL of a remote Qdrant server.
            When provided, qdrant_path is not needed. Ask the user before using this.
        qdrant_api_key: API key for remote Qdrant. Only used with qdrant_url.
        chunk_size: Maximum number of tokens per chunk (default: 256).
        chunk_overlap_sentences: Number of overlapping sentences between consecutive
            chunks (default: 2).
    """
    try:
        if collection_name is None:
            collection_name = _generate_collection_name()
        store = _create_qdrant_store(collection_name, qdrant_path, qdrant_url, qdrant_api_key)
        embedding_model = _get_embedding_model()
        results = await asyncio.to_thread(
            _do_ingest, store, embedding_model, file_paths, chunk_size, chunk_overlap_sentences
        )
        total_indexed = sum(r["chunks_indexed"] for r in results)
        return {
            "collection_name": collection_name,
            "qdrant_path": qdrant_path,
            "results": results,
            "total_files": len(results),
            "total_chunks_indexed": total_indexed,
        }
    except Exception as e:
        logger.exception("ingest_document failed")
        return {"error": type(e).__name__, "message": str(e)}


@mcp.tool()
async def retrieve(
    query: str,
    collection_name: str,
    qdrant_path: str | None = None,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    top_k_retrieval: int = 20,
    top_k_rerank: int = 5,
) -> dict:
    """Search a Qdrant collection and return reranked results.

    IMPORTANT: If you used any of the reranked reults, add citations in the answer and show the retrived results that you used.

    collection_name is required. Use the name returned by ingest_document,
    or a name the user provides from a previous session.

    qdrant_path must be an absolute path. Use the same path that was used
    (or returned) during ingestion. Not needed when qdrant_url is provided.

    Only provide qdrant_url and qdrant_api_key if the user explicitly
    requests a remote Qdrant server. Ask the user before switching to remote mode.

    Don't use the optional arguments in the request unless the user explicitly asks you to or you think it is necessary.

    Args:
        query: The search query string.
        collection_name: Name of the Qdrant collection to search.
            Use the name returned by ingest_document.
        qdrant_path: Absolute path for local Qdrant storage directory.
            Required when qdrant_url is not provided. Must be absolute.
        qdrant_url: URL of a remote Qdrant server.
            When provided, qdrant_path is not needed. Ask the user before using this.
        qdrant_api_key: API key for remote Qdrant. Only used with qdrant_url.
        top_k_retrieval: Number of candidates to retrieve from hybrid search
            (default: 20).
        top_k_rerank: Number of top results to return after reranking
            (default: 5).
    """
    try:
        store = _create_qdrant_store(collection_name, qdrant_path, qdrant_url, qdrant_api_key)
        embedding_model = _get_embedding_model()
        reranker = _get_reranker()
        results = await asyncio.to_thread(
            _do_retrieve, store, embedding_model, reranker, query, top_k_retrieval, top_k_rerank
        )
        return {"results": results}
    except Exception as e:
        logger.exception("retrieve failed")
        return {"error": type(e).__name__, "message": str(e)}


def run_server(no_init: bool = False):
    """Start the MCP stdio server.

    Args:
        no_init: If True, models are loaded lazily on first tool call.
            If False (default), models are loaded eagerly at startup.
    """
    if not no_init:
        initialize_models()
    mcp.run(transport="streamable-http")
