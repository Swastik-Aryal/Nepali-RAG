from langchain_core.language_models import BaseChatModel
from nepali_rag.config.settings import settings


def get_ollama_llm(**kwargs) -> BaseChatModel:
    """Connect to a local Ollama server via langchain-ollama.

    Ollama runs models locally and exposes a REST API on port 11434.
    No API key is required. Pull a model first with `ollama pull <model>`.
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
