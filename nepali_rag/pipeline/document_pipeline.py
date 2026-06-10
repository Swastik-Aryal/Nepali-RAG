import uuid
import logging

from nepali_rag.config.settings import settings
from nepali_rag.ingestion.loader import load_document
from nepali_rag.preprocessing.normalizer import normalize_text
from nepali_rag.chunking.splitter import chunk_pages
from nepali_rag.embeddings.embedding_model import NepaliEmbeddingModel
from nepali_rag.vectorstore.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class DocumentPipeline:
    """Orchestrates ingestion, preprocessing, chunking, and indexing for uploaded documents."""

    def __init__(
        self,
        embedding_model: NepaliEmbeddingModel = None,
        vector_store: QdrantStore = None,
        collection_name: str = None,
        chunk_size: int = None,
        chunk_overlap_sentences: int = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap_sentences = chunk_overlap_sentences or settings.chunk_overlap_sentences
        self.collection_name = collection_name
        self._embedding_model = embedding_model
        self._vector_store = vector_store

    @property
    def embedding_model(self) -> NepaliEmbeddingModel:
        if self._embedding_model is None:
            self._embedding_model = NepaliEmbeddingModel()
        return self._embedding_model

    @property
    def vector_store(self) -> QdrantStore:
        if self._vector_store is None:
            name = self.collection_name or f"doc_{uuid.uuid4().hex[:12]}"
            self.collection_name = name
            self._vector_store = QdrantStore(collection_name=name)
        return self._vector_store

    def ingest(self, file_path: str) -> dict:
        """Run the full pipeline for a single file: load, normalize, chunk, embed, index.

        Returns a summary dict with collection_name, filename, pages_extracted,
        chunks_created, and chunks_indexed.
        """
        logger.info("Loading document: %s", file_path)
        pages = load_document(file_path)
        pages_extracted = len(pages)

        logger.info("Normalizing text for %d page(s)", pages_extracted)
        for page in pages:
            page.text = normalize_text(page.text)

        logger.info("Chunking pages")
        chunks = chunk_pages(
            pages,
            chunk_size=self.chunk_size,
            overlap_sentences=self.chunk_overlap_sentences,
        )
        chunks_created = len(chunks)

        if chunks_created == 0:
            logger.warning("No chunks produced from %s", file_path)
            return {
                "collection_name": self.collection_name or "",
                "filename": file_path,
                "pages_extracted": pages_extracted,
                "chunks_created": 0,
                "chunks_indexed": 0,
            }

        logger.info("Embedding %d chunk(s)", chunks_created)
        texts = [c["text"] for c in chunks]
        embeddings = self.embedding_model.embed_documents(texts)

        logger.info("Indexing into Qdrant collection: %s", self.vector_store.collection_name)
        chunks_indexed = self.vector_store.insert_chunks(chunks, embeddings)

        logger.info(
            "Done: %d pages, %d chunks created, %d indexed",
            pages_extracted, chunks_created, chunks_indexed,
        )
        return {
            "collection_name": self.vector_store.collection_name,
            "filename": file_path,
            "pages_extracted": pages_extracted,
            "chunks_created": chunks_created,
            "chunks_indexed": chunks_indexed,
        }

    def ingest_batch(self, file_paths: list[str]) -> list[dict]:
        """Process multiple files sequentially through the full pipeline."""
        results = []
        for path in file_paths:
            result = self.ingest(path)
            results.append(result)
        return results
