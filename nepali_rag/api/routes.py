import json
import logging
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException

from nepali_rag.api.schemas import (
    CollectionInfo,
    ConversationDetail,
    ConversationTurn,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceDocument,
    UploadResponse,
    UploadResult,
)
from nepali_rag.ingestion.loader import SUPPORTED_FORMATS
from nepali_rag.config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

CONVERSATIONS_DIR = Path(__file__).resolve().parent / "conversations"


def _load_conversation(conversation_id: str) -> list[dict]:
    """Load turn history from ``api/conversations/<conversation_id>.json``, or ``[]`` if missing."""
    path = CONVERSATIONS_DIR / f"{conversation_id}.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_conversation(conversation_id: str, history: list[dict]) -> None:
    """Persist conversation turn history to ``api/conversations/<conversation_id>.json``."""
    CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = CONVERSATIONS_DIR / f"{conversation_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


@router.post("/upload", response_model=UploadResponse)
async def upload_documents(
    request: Request,
    files: list[UploadFile] = File(...),
    collection_name: str | None = Form(default=None),
    chunk_size: int | None = Form(default=None),
):
    """Upload and index one or more documents via ``DocumentPipeline``.

    Each uploaded file is validated against ``SUPPORTED_FORMATS``, saved to a
    temporary directory, passed through the full ingest pipeline
    (load → normalize → chunk → embed → Qdrant), and then cleaned up.

    Args:
        - request (Request): FastAPI request object carrying ``app.state.embedding_model``.
        - files (list[UploadFile]): One or more files to upload. Must be a supported
            format (``.pdf``, ``.docx``, ``.txt``, ``.md``, ``.html`` etc).
        - collection_name (str, optional): Qdrant collection to index into.
            Defaults to ``None``. If ``None``, ``DocumentPipeline`` auto-generates
            a ``doc_<hex>`` name.
        - chunk_size (int, optional): Maximum number of tokens per chunk.
            Defaults to ``None``. If ``None``, falls back to
            ``settings.chunk_size`` (256), overridden by ``CHUNK_SIZE`` in ENV
            if present.

    Returns:
        JSON dict containing:
            - ``results``: List of per-file dicts, each with ``collection_name``,
              ``filename``, ``pages_extracted``, ``chunks_created``, and ``chunks_indexed``.
            - ``total_files``: Number of files processed.
            - ``total_chunks_indexed``: Sum of all chunks stored in Qdrant.

    Raises:
        400: If any file has an unsupported extension.
        500: If ingestion or indexing fails unexpectedly.

    Example:

        Request (``POST /api/v1/upload``, multipart form-data)::

            files: ["example.pdf"]
            collection_name: "my_collection"
            chunk_size: 512

        Response::

            {
                "results": [
                    {
                        "collection_name": "my_collection",
                        "filename": "/tmp/.../example.pdf",
                        "pages_extracted": 4,
                        "chunks_created": 12,
                        "chunks_indexed": 12
                    }
                ],
                "total_files": 1,
                "total_chunks_indexed": 12
            }
    """
    from nepali_rag.pipeline.document_pipeline import DocumentPipeline

    supported_exts = set(SUPPORTED_FORMATS.keys())
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in supported_exts:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: '{ext}' for file '{f.filename}'. "
                       f"Supported: {', '.join(sorted(supported_exts))}",
            )

    tmp_dir = tempfile.mkdtemp()
    saved_paths = []

    try:
        for f in files:
            dest = os.path.join(tmp_dir, f.filename)
            content = await f.read()
            with open(dest, "wb") as fp:
                fp.write(content)
            saved_paths.append(dest)

        pipeline = DocumentPipeline(
            embedding_model=request.app.state.embedding_model,
            collection_name=collection_name,
            chunk_size=chunk_size,
        )
        raw_results = pipeline.ingest_batch(saved_paths)

    except Exception as exc:
        logger.exception("Upload processing failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        for p in saved_paths:
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

    results = [UploadResult(**r) for r in raw_results]
    total_indexed = sum(r.chunks_indexed for r in results)

    return UploadResponse(
        results=results,
        total_files=len(results),
        total_chunks_indexed=total_indexed,
    )


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: Request, body: QueryRequest):
    """Run a RAG query: retrieve relevant chunks, generate an answer, and persist the conversation.

    The pipeline retrieves candidate passages from Qdrant, optionally reranks
    them, and feeds the context to the configured LLM for answer generation.
    Conversation turns are persisted to ``api/conversations/<conversation_id>.json``
    so subsequent queries can leverage multi-turn context.

    Args:
        - request (Request): FastAPI request object carrying ``app.state.embedding_model``
            and ``app.state.reranker``.
        - body (QueryRequest): JSON request body with the following fields:
            - ``question`` (str): The user's query. Must not be empty.
            - ``conversation_id`` (str, optional): Unique conversation identifier.
                Defaults to ``None``. If ``None``, a random 16-character hex ID is
                generated.
            - ``conversation_history`` (list[ConversationTurn], optional): Explicit turn
                history to replace any file-persisted history. Defaults to ``[]``.
            - ``collection_name`` (str, optional): Qdrant collection to search.
                Defaults to ``None``. If ``None``, falls back to
                ``settings.collection_name`` (``nepali_docs``), overridden by
                ``COLLECTION_NAME`` in ENV if present.
            - ``top_k_retrieval`` (int, optional): Hybrid-search candidate count.
                Defaults to ``None``. If ``None``, falls back to
                ``settings.top_k_retrieval`` (20), overridden by
                ``TOP_K_RETRIEVAL`` in ENV if present.
            - ``top_k_rerank`` (int, optional): Final result count after reranking.
                Defaults to ``None``. If ``None``, falls back to
                ``settings.top_k_rerank`` (5), overridden by ``TOP_K_RERANK``
                in ENV if present.
            - ``llm_provider`` (str, optional): LLM provider override (e.g. ``"ollama"``,
                ``"openai"``, ``"gemini"``, ``"huggingface"``, ``"vllm"``, ``anthropic``).
                Defaults to ``None``. If ``None``, falls back to
                ``settings.llm_provider``, overridden by ``LLM_PROVIDER`` in
                ENV if present.
            - ``llm_model`` (str, optional): Model name override.
                Defaults to ``None``. If ``None``, falls back to the
                provider-specific model in settings (e.g. ``OLLAMA_MODEL``,
                ``OPENAI_MODEL``, ``GEMINI_MODEL``, etc. in ENV).
            - ``temperature`` (float, optional): Sampling temperature.
                Defaults to ``None``. If ``None``, falls back to
                ``settings.llm_temperature`` (0.3), overridden by
                ``LLM_TEMPERATURE`` in ENV if present.
            - ``use_tools`` (bool, optional): Whether the LLM should decide if retrieval
                is needed. Defaults to ``None``. If ``None``, falls back to
                ``settings.use_tools`` (``True``), overridden by ``USE_TOOLS``
                in ENV if present.

    Note:
        LLM credentials and connection details (API keys, base URLs) are not
        passed per-request — they are configured via environment variables /
        ``settings.py``. For example:

        - ``OPENAI_API_KEY`` is **required** for the ``openai`` provider.
        - ``OPENAI_BASE_URL`` is needed when using OpenAI-compatible providers
          that are not actually OpenAI (e.g. Groq, DeepSeek).
        - ``GOOGLE_API_KEY`` is **required** for the ``gemini`` provider.
        - ``HF_TOKEN`` is **required** for the ``huggingface`` provider.
        - ``OLLAMA_BASE_URL`` defaults to ``http://localhost:11434`` for local
          Ollama; no API key needed.

        See ``.env.example`` for the full list of configurable variables.

    Returns:
        JSON dict containing:
            - ``answer`` (str): The generated response from the LLM (or formatted prompt
              in test mode).
            - ``sources`` (list[dict]): Retrieved chunks, each with ``text``,
              ``metadata``, ``score``, and ``rerank_score``.
            - ``context`` (str): The formatted context string fed to the LLM.
            - ``collection_name`` (str): The Qdrant collection that was queried.
            - ``conversation_id`` (str): The conversation ID (generated or provided).
            - ``test_mode`` (bool): ``True`` if the LLM was unavailable and the response
              is a raw prompt.
            - ``used_tool_calling`` (bool): ``True`` if the LLM invoked the
              ``retrieve_documents`` tool.

    Raises:
        400: If ``question`` is empty or blank.
        503: If LLM initialization fails with an unexpected error.
        500: If query processing fails unexpectedly.

    Example:

        Request (``POST /api/v1/query``)::

            {
                "question": "त्यहाँको जनसंख्या कति हो?",
                "collection_name": "nepali_docs",
                "conversation_history": [
                    {"role": "user", "content": "नेपालको राजधानी के हो?"},
                    {"role": "assistant", "content": "नेपालको राजधानी काठमाडौं हो।"}
                ],
                "top_k_retrieval": 10,
                "top_k_rerank": 3
            }

        Response::

            {
                "answer": "नेपालको राजधानी काठमाडौं हो।",
                "sources": [
                    {
                        "text": "काठमाडौं नेपालको राजधानी हो...",
                        "metadata": {"source": "nepal_info.pdf", "page_number": 1, "chunk_index": 0,... },
                        "score": 0.87,
                        "rerank_score": 0.95
                    }
                ],
                "context": "[1](Source: nepal_info.pdf, Page: 1) काठमाडौं नेपालको राजधानी हो...",
                "collection_name": "nepali_docs",
                "conversation_id": "a1b2c3d4e5f67890",
                "test_mode": false,
                "used_tool_calling": true
            }
    """
    from nepali_rag.retrieval.retriever import Retriever
    from nepali_rag.vectorstore.qdrant_store import QdrantStore
    from nepali_rag.rag.chain import RAGChain
    from nepali_rag.llm.base import get_llm

    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    conversation_id = body.conversation_id or uuid.uuid4().hex[:16]
    collection = body.collection_name or settings.collection_name

    if body.conversation_history:
        history = [{"role": t.role, "content": t.content} for t in body.conversation_history]
    else:
        history = _load_conversation(conversation_id)

    try:
        store = QdrantStore(
            vector_size=request.app.state.embedding_model.dimension,
            collection_name=collection)
        retriever = Retriever(
            embedding_model=request.app.state.embedding_model,
            store=store,
            reranker=request.app.state.reranker,
        )

        # Resolve use_tools: per-request override > global setting.
        use_tools = body.use_tools if body.use_tools is not None else settings.use_tools

        llm = None
        try:
            llm_kwargs = {}
            if body.llm_provider:
                llm_kwargs["provider"] = body.llm_provider
            if body.llm_model:
                llm_kwargs["model"] = body.llm_model
            if body.temperature is not None:
                llm_kwargs["temperature"] = body.temperature
            llm = get_llm(**llm_kwargs)
        except (ConnectionError, OSError, TimeoutError) as exc:
            logger.warning("LLM unavailable (%s), operating in test mode", exc)
            llm = None
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"LLM initialization failed: {exc}",
            )

        chain = RAGChain(retriever=retriever, llm=llm, use_tools=use_tools)

        try:
            result = chain.run(
                question=question,
                conversation_history=history or None,
                top_k_retrieval=body.top_k_retrieval,
                top_k_rerank=body.top_k_rerank,
            )
        except Exception as llm_exc:
            # If the LLM was set but failed at invocation time (e.g. connection
            # refused), fall back to test mode instead of crashing.
            if llm is not None:
                logger.warning(
                    "LLM invocation failed (%s), falling back to test mode",
                    llm_exc,
                )
                llm = None
                chain = RAGChain(retriever=retriever, llm=None, use_tools=False)
                result = chain.run(
                    question=question,
                    conversation_history=history or None,
                    top_k_retrieval=body.top_k_retrieval,
                    top_k_rerank=body.top_k_rerank,
                )
            else:
                raise

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Query processing failed")
        raise HTTPException(status_code=500, detail=str(exc))

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": result["answer"]})
    _save_conversation(conversation_id, history)

    sources = []
    for src in result.get("sources", []):
        sources.append(SourceDocument(
            text=src.get("text", ""),
            metadata=src.get("metadata", {}),
            score=src.get("score"),
            rerank_score=src.get("rerank_score"),
        ))

    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        context=result.get("context", ""),
        collection_name=collection,
        conversation_id=conversation_id,
        test_mode=llm is None,
        used_tool_calling=result.get("tool_called", False),
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Return API health status, configured providers, and LLM reachability.

    Attempts to initialize the default LLM to determine reachability. No
    inference call is made — only the connection handshake is tested.

    Returns:
        JSON dict containing:
            - ``status`` (str): Always ``"ok"`` if the API itself is running.
            - ``llm_provider`` (str): The configured LLM provider name in ENV / ``settings.py``
              (e.g. ``"ollama"``, ``"openai"``).
            - ``llm_available`` (bool): ``True`` if llm initialization succeeds without error.
            - ``embedding_model`` (str): The configured embedding model identifier in ENV / ``settings.py``.
            - ``default_collection`` (str): The default Qdrant collection name in ENV / ``settings.py``.

    Example:

        Request (``GET /api/v1/health``):
            No request body required.

        Response::

            {
                "status": "ok",
                "llm_provider": "ollama",
                "llm_available": true,
                "embedding_model": "intfloat/multilingual-e5-large",
                "default_collection": "nepali_docs"
            }
    """
    from nepali_rag.llm.base import get_llm

    llm_available = False
    try:
        get_llm()
        llm_available = True
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        llm_provider=settings.llm_provider,
        llm_available=llm_available,
        embedding_model=settings.embedding_model,
        default_collection=settings.collection_name,
    )


@router.get("/collections", response_model=list[CollectionInfo])
async def list_collections():
    """List all Qdrant collections with their point counts and statuses.

    Connects to the Qdrant instance using ``settings.qdrant_url`` and
    ``settings.qdrant_api_key``, then fetches metadata for every collection.

    Returns:
        list of JSON dicts, one per collection, each containing:
            - ``name`` (str): The collection name.
            - ``points_count`` (int): Number of indexed points (chunks).
            - ``status`` (str): Collection status (e.g. ``"green"``).

    Raises:
        500: If the Qdrant connection or metadata fetch fails.

    Example:

        Request (``GET /api/v1/collections``):
            No request body required.

        Response::

            [
                {
                    "name": "nepali_docs",
                    "points_count": 142,
                    "status": "green"
                },
                {
                    "name": "doc_a1b2c3d4e5f6",
                    "points_count": 38,
                    "status": "green"
                }
            ]
    """
    from qdrant_client import QdrantClient

    try:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        collections = client.get_collections().collections

        result = []
        for col in collections:
            info = client.get_collection(collection_name=col.name)
            status = info.status.value if hasattr(info.status, "value") else str(info.status)
            result.append(CollectionInfo(
                name=col.name,
                points_count=info.points_count,
                status=status,
            ))
        return result

    except Exception as exc:
        logger.exception("Failed to list collections")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/collections/{name}", response_model=CollectionInfo)
async def get_collection(name: str):
    """Return point count and status for a single Qdrant collection.

    Args:
        - name (str): The exact Qdrant collection name to look up.

    Returns:
        JSON dict containing:
            - ``name`` (str): The collection name (echoed back).
            - ``points_count`` (int): Number of indexed points (chunks).
            - ``status`` (str): Collection status (e.g. ``"green"``).

    Raises:
        500: If the collection does not exist or the Qdrant connection fails.

    Example:

        Request (``GET /api/v1/collections/nepali_docs``):
            No request body required.

        Response::

            {
                "name": "nepali_docs",
                "points_count": 142,
                "status": "green"
            }
    """
    from qdrant_client import QdrantClient

    try:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        info = client.get_collection(collection_name=name)
        status = info.status.value if hasattr(info.status, "value") else str(info.status)
        return CollectionInfo(
            name=name,
            points_count=info.points_count,
            status=status,
        )
    except Exception as exc:
        logger.exception("Failed to get collection '%s'", name)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/conversations", response_model=list[str])
async def list_conversations():
    """List all persisted conversation IDs.

    Scans ``api/conversations/`` for ``*.json`` files and returns each file's
    stem (basename without the ``.json`` extension) as a plain string.
    Returns an empty list when the directory does not exist or contains no
    conversation files.

    Returns:
        list[str]: Sorted list of conversation ID strings. Empty if no
        conversations have been persisted yet.

    Example:

        Request (``GET /api/v1/conversations``):
            No request body required.

        Response::

            [
                "061004efc64840f7",
                "a3f8b2e1c9d74501",
                "qwen_fun"
            ]
    """
    if not CONVERSATIONS_DIR.exists():
        return []

    return [p.stem for p in sorted(CONVERSATIONS_DIR.glob("*.json"))]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: str):
    """Return the full turn history for a single persisted conversation.

    Args:
        - conversation_id (str): The bare ID (filename stem) of the conversation,
            e.g. ``061004efc64840f7`` — **without** the ``.json`` suffix.

    Returns:
        JSON dict containing:
            - ``conversation_id`` (str): The requested conversation ID (echoed back).
            - ``turns`` (list[dict]): Ordered list of role/content pairs
              representing the full conversation history.

    Raises:
        404: If no conversation file matching ``conversation_id`` exists.
        500: If the file is present but cannot be parsed as valid JSON.

    Example:
    
        Request (``GET /api/v1/conversations/061004efc64840f7``):
            No request body required.

        Response::

            {
                "conversation_id": "061004efc64840f7",
                "turns": [
                    {"role": "user", "content": "नेपालको राजधानी के हो?"},
                    {"role": "assistant", "content": "नेपालको राजधानी काठमाडौं हो।"}
                ]
            }
    """
    path = CONVERSATIONS_DIR / f"{conversation_id}.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Conversation '{conversation_id}' not found.",
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_turns: list[dict] = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.exception("Failed to read conversation '%s'", conversation_id)
        raise HTTPException(
            status_code=500,
            detail=f"Could not read conversation file: {exc}",
        )

    turns = [ConversationTurn(role=t["role"], content=t["content"]) for t in raw_turns]
    return ConversationDetail(conversation_id=conversation_id, turns=turns)
