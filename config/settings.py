from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    collection_name: str = "nepali_docs"

    embedding_model: str = "intfloat/multilingual-e5-large"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    llm_model: str = "google/gemma-3-4b-it"
    vllm_base_url: str = "http://localhost:8000/v1"

    tesseract_languages: str = "nep+eng"

    chunk_size: int = 512
    chunk_overlap_sentences: int = 2

    top_k_retrieval: int = 20
    top_k_rerank: int = 5

    dense_weight: float = 0.7
    sparse_weight: float = 0.3

    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024

    data_dir: Path = Path("data")
    sample_docs_dir: Path = Path("data/sample_docs")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
