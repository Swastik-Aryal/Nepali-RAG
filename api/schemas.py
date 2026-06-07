from typing import Literal
from pydantic import BaseModel, Field


class UploadResult(BaseModel):
    collection_name: str
    filename: str
    pages_extracted: int
    chunks_created: int
    chunks_indexed: int


class UploadResponse(BaseModel):
    results: list[UploadResult]
    total_files: int
    total_chunks_indexed: int


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    question: str
    conversation_id: str | None = None
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    collection_name: str | None = None
    top_k_retrieval: int | None = None
    top_k_rerank: int | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    temperature: float | None = None


class SourceDocument(BaseModel):
    text: str
    metadata: dict = Field(default_factory=dict)
    score: float | None = None
    rerank_score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    context: str
    collection_name: str
    conversation_id: str
    test_mode: bool


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    llm_available: bool
    embedding_model: str
    default_collection: str


class CollectionInfo(BaseModel):
    name: str
    points_count: int
    status: str
