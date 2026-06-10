"""Nepali RAG — Retrieval-Augmented Generation for Nepali language documents."""

__version__ = "1.0.0"

from nepali_rag.pipeline.document_pipeline import DocumentPipeline
from nepali_rag.retrieval.retriever import Retriever
from nepali_rag.retrieval.reranker import Reranker
from nepali_rag.rag.chain import RAGChain
from nepali_rag.llm.base import get_llm
from nepali_rag.embeddings.embedding_model import NepaliEmbeddingModel
from nepali_rag.vectorstore.qdrant_store import QdrantStore
from nepali_rag.preprocessing.normalizer import normalize_text, is_nepali_text
from nepali_rag.ingestion.loader import load_document
from nepali_rag.config.settings import settings

__all__ = [
    "__version__",
    "DocumentPipeline",
    "Retriever",
    "Reranker",
    "RAGChain",
    "get_llm",
    "NepaliEmbeddingModel",
    "QdrantStore",
    "normalize_text",
    "is_nepali_text",
    "load_document",
    "settings",
]
