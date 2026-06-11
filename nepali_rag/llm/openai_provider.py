"""OpenAI API LLM provider."""

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from nepali_rag.config.settings import settings


def get_openai_llm(**kwargs) -> BaseChatModel:
    """Return a ``ChatOpenAI`` instance for the OpenAI API.

    Requires ``OPENAI_API_KEY`` in ``.env`` or ``api_key`` kwarg.

    Kwargs default to settings: ``model`` (``settings.openai_model``, ``gpt-4o-mini``),
    ``temperature`` (``settings.llm_temperature``, 0.3).
    """
    api_key = kwargs.get("api_key", settings.openai_api_key)
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Set it in your .env file or pass api_key= to get_llm()."
        )

    return ChatOpenAI(
        model=kwargs.get("model", settings.openai_model),
        openai_api_key=api_key,
        temperature=kwargs.get("temperature", settings.llm_temperature),
    )
