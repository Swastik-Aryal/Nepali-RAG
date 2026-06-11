"""Ollama LLM provider for local model inference."""

from langchain_core.language_models import BaseChatModel
from nepali_rag.config.settings import settings


def get_ollama_llm(**kwargs) -> BaseChatModel:
    """Return a ``ChatOllama`` instance pointed at a local Ollama server.

    No API key required. Pull the model first with ``ollama pull <model>``.

    Kwargs default to settings: ``model`` (``settings.ollama_model``, ``gemma3:4b``),
    ``base_url`` (``settings.ollama_base_url``, ``http://localhost:11434``),
    ``temperature`` (``settings.llm_temperature``, 0.3).
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise ImportError(
            "langchain-ollama is required for the Ollama provider. "
            "Install it with: pip install langchain-ollama"
        )

    return ChatOllama(
        model=kwargs.get("model", settings.ollama_model),
        base_url=kwargs.get("base_url", settings.ollama_base_url),
        temperature=kwargs.get("temperature", settings.llm_temperature),
    )
