"""Anthropic Claude LLM provider."""

from langchain_core.language_models import BaseChatModel
from nepali_rag.config.settings import settings


def get_anthropic_llm(**kwargs) -> BaseChatModel:
    """Return a ``ChatAnthropic`` instance for Anthropic Claude.

    Requires ``ANTHROPIC_API_KEY`` in ``.env`` or ``api_key`` kwarg.

    Kwargs default to settings: ``model`` (``settings.anthropic_model``, ``claude-opus-4-6``),
    ``temperature`` (``settings.llm_temperature``, 0.3).
    """
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError(
            "langchain-anthropic is required for the Anthropic provider. "
            "Install it with: pip install langchain-anthropic"
        )

    api_key = kwargs.get("api_key", settings.anthropic_api_key)
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. "
            "Set it in your .env file or pass api_key= to get_llm()."
        )

    return ChatAnthropic(
        model=kwargs.get("model", settings.anthropic_model),
        api_key=api_key,
        temperature=kwargs.get("temperature", settings.llm_temperature),
    )
