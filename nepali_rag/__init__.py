"""Nepali RAG — Retrieval-Augmented Generation for Nepali language documents."""

from __future__ import annotations

__version__ = "1.0.0"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
    from nepali_rag.evals.rag_output_generator import run_rag
    from nepali_rag.evals.evaluator import run_evaluation

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "DocumentPipeline": ("nepali_rag.pipeline.document_pipeline", "DocumentPipeline"),
    "Retriever": ("nepali_rag.retrieval.retriever", "Retriever"),
    "Reranker": ("nepali_rag.retrieval.reranker", "Reranker"),
    "RAGChain": ("nepali_rag.rag.chain", "RAGChain"),
    "get_llm": ("nepali_rag.llm.base", "get_llm"),
    "NepaliEmbeddingModel": ("nepali_rag.embeddings.embedding_model", "NepaliEmbeddingModel"),
    "QdrantStore": ("nepali_rag.vectorstore.qdrant_store", "QdrantStore"),
    "normalize_text": ("nepali_rag.preprocessing.normalizer", "normalize_text"),
    "is_nepali_text": ("nepali_rag.preprocessing.normalizer", "is_nepali_text"),
    "load_document": ("nepali_rag.ingestion.loader", "load_document"),
    "settings": ("nepali_rag.config.settings", "settings"),
    "run_rag": ("nepali_rag.evals.rag_output_generator", "run_rag"),
    "run_evaluation": ("nepali_rag.evals.evaluator", "run_evaluation"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    *_LAZY_IMPORTS,
]
