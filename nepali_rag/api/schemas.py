from typing import Literal
from pydantic import BaseModel, Field


class UploadResult(BaseModel):
    """Per-file summary returned after indexing."""

    collection_name: str
    filename: str
    pages_extracted: int
    chunks_created: int
    chunks_indexed: int


class UploadResponse(BaseModel):
    """Batch upload response with per-file results and totals."""

    results: list[UploadResult]
    total_files: int
    total_chunks_indexed: int


class ConversationTurn(BaseModel):
    """A single user or assistant message in a conversation."""

    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    """RAG query body. ``None`` fields fall back to ``settings`` defaults at request time."""

    question: str
    conversation_id: str | None = None
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    collection_name: str | None = None
    top_k_retrieval: int | None = None
    top_k_rerank: int | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    temperature: float | None = None
    use_tools: bool | None = None


class SourceDocument(BaseModel):
    """A retrieved chunk cited in the query response."""

    text: str
    metadata: dict = Field(default_factory=dict)
    score: float | None = None
    rerank_score: float | None = None


class QueryResponse(BaseModel):
    """RAG answer with sources, context, and conversation metadata."""

    answer: str
    sources: list[SourceDocument]
    context: str
    collection_name: str
    conversation_id: str
    test_mode: bool
    used_tool_calling: bool = False


class HealthResponse(BaseModel):
    """Health check payload with configured models and LLM reachability."""

    status: str
    llm_provider: str
    llm_available: bool
    embedding_model: str
    default_collection: str


class CollectionInfo(BaseModel):
    """Qdrant collection name, indexed point count, and status."""

    name: str
    points_count: int
    status: str
