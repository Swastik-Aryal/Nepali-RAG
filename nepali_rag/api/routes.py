import json
import logging
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException

from nepali_rag.api.schemas import (
    CollectionInfo,
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
    """Load conversation history from a JSON file, or return empty list."""
    path = CONVERSATIONS_DIR / f"{conversation_id}.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_conversation(conversation_id: str, history: list[dict]) -> None:
    """Save conversation history to a JSON file."""
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
    chunk_overlap_sentences: int | None = Form(default=None),
):
    """Upload one or more documents for ingestion into the vector store.

    Accepts multipart file uploads. Each file is validated against the
    supported format list, processed through the full DocumentPipeline
    (load, normalize, chunk, embed, index), and the temp files are
    cleaned up afterward.
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
            chunk_overlap_sentences=chunk_overlap_sentences,
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
    """Query the RAG pipeline with a question and optional conversation history.

    Conversation persistence: each query is tied to a conversation_id. If not
    provided, a new one is generated. The conversation file in nepali_rag/api/conversations/
    tracks the full history. If the caller sends an explicit conversation_history,
    the file is overwritten with that history plus the new turn. Otherwise the
    existing file history is loaded and appended to.
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
        store = QdrantStore(collection_name=collection)
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
    """Return system health and component availability."""
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
    """List all Qdrant collections with their point counts."""
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
    """Get detailed info for a specific Qdrant collection."""
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
