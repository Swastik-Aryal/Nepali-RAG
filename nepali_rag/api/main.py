import logging

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from nepali_rag.api.routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Nepali-RAG API")

    # Load heavy ML models once at startup instead of per-request.
    from nepali_rag.embeddings.embedding_model import NepaliEmbeddingModel
    from nepali_rag.retrieval.reranker import Reranker

    logger.info("Loading embedding model...")
    app.state.embedding_model = NepaliEmbeddingModel()
    logger.info("Embedding model loaded (dim=%d)", app.state.embedding_model.dimension)

    logger.info("Loading reranker model...")
    app.state.reranker = Reranker()
    logger.info("Reranker model loaded")

    yield

    logger.info("Shutting down Nepali-RAG API")


app = FastAPI(
    title="Nepali-RAG API",
    description="Retrieval-Augmented Generation system with deep Nepali language support",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("nepali_rag.api.main:app", host="0.0.0.0", port=8080, reload=False)
