"""vLLM LLM provider via an OpenAI-compatible local endpoint."""

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from nepali_rag.config.settings import settings


def get_vllm_llm(**kwargs) -> BaseChatModel:
    """Return a ``ChatOpenAI`` client pointed at a local vLLM ``/v1`` server.

    No API key required.

    Kwargs default to settings: ``model`` (``settings.vllm_model``, ``google/gemma-4-E4B-it``),
    ``base_url`` (``settings.vllm_base_url``, ``http://localhost:8000/v1``),
    ``temperature`` (``settings.llm_temperature``, 0.3).
    """
    return ChatOpenAI(
        model=kwargs.get("model", settings.vllm_model),
        openai_api_base=kwargs.get("base_url", settings.vllm_base_url),
        openai_api_key=kwargs.get("api_key", "not-needed"),
        temperature=kwargs.get("temperature", settings.llm_temperature),
    )
