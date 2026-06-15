from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """
    Default settings for the Nepali RAG system.
    
    Override default values by creating a ``.env`` file with the desired values.
    
    """
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    collection_name: str = "nepali_docs"

    embedding_model: str = "intfloat/multilingual-e5-large"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    llm_provider: str = "ollama"
    vllm_model: str = "google/gemma-3-4b-it"
    vllm_base_url: str = "http://localhost:8000/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    google_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma3:4b"
    hf_model_name: str = "Qwen/Qwen3.6-35B-A3B"
    hf_base_url: str = "https://router.huggingface.co/v1"
    hf_token: str = ""


    tesseract_languages: str = "nep"

    chunk_size: int = 256
    chunk_overlap_sentences: int = 2

    top_k_retrieval: int = 20
    top_k_rerank: int = 5

    dense_weight: float = 0.9
    sparse_weight: float = 0.1

    llm_temperature: float = 0.3
    use_tools: bool = True

    data_dir: Path = Path("data")
    sample_docs_dir: Path = Path("data/sample_docs")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
